"""Real-model drivers: what a run uses when the config names something other than `scripted`.

A driver renders the context and sends it through a harness's own door. The ones here are thin on purpose:
the conformance suite decides whether a door can be trusted, and the same request bodies are tested
offline. Fixtures' scripted thinking blocks are never replayed to a real provider.
"""
from __future__ import annotations

import json
import time

from eunoe.cost import ListPrice
from eunoe.drivers.base import SpendGate, http_post
from eunoe.editors.claude_cli import ClaudeCliHarness, claude_usage
from eunoe.editors.stateless import (
    ANTHROPIC_URL, BETA_COMPACTION, BETA_CONTEXT_EDITING, BETA_THINKING_CONTROLS, ClaudeApiHarness,
)
from eunoe.interfaces import Block, CallUsage, DriverCaps, ModelTurn, RenderedContext, ToolCall
from eunoe.wire import canonical, to_anthropic


class ClaudeApiDriver:
    """The Messages API, client-side, or with server-side edits when a control asks for them."""

    def __init__(self, model: str = "claude-sonnet-5-5", gate: SpendGate | None = None, transport=http_post,
                 environ=None, ttl: str = "5m") -> None:
        self.harness = ClaudeApiHarness(model, gate or SpendGate(), transport, environ, ttl, ListPrice(model))
        self.capabilities = DriverCaps("claude-api", "account", ("5m", "1h"), server_edits=True, spends=True)
        self.model = model

    def request(self, ctx: RenderedContext) -> tuple:
        control = {"type": "ephemeral"}
        if self.harness.ttl == "1h":
            control["ttl"] = "1h"
        # Automatic caching holds one of the four slots, and this request has one for the frozen region at most.
        points = list(ctx.meta.get("breakpoints", ()))[-3:]
        wire = to_anthropic(ctx, thinking_replay=False, breakpoints=points, cache_control=control)
        body = self.harness.request_body(wire["messages"])
        for key in ("system", "tools"):
            if key in wire:
                body[key] = wire[key]
        betas = []
        if ctx.meta.get("thinking_policy") == "drop_block":
            body["thinking"] = {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
            betas.append(BETA_THINKING_CONTROLS)
        management = ctx.meta.get("context_management")
        if management:
            body["context_management"] = management
            kinds = {edit["type"] for edit in management["edits"]}
            betas.append(BETA_COMPACTION if "compact_20260112" in kinds else BETA_CONTEXT_EDITING)
        return self.harness.headers(*betas), body

    def call(self, request: RenderedContext) -> ModelTurn:
        self.harness.gate.check("claude-api")
        headers, body = self.request(request)
        started = time.monotonic()
        status, data = self.harness.transport(ANTHROPIC_URL, headers, body)
        latency = time.monotonic() - started
        if status != 200:
            return ModelTurn(error=f"{status} {str(data.get('error'))[:200]}", latency_s=latency)
        text, calls = "", []
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(block.get("id", ""), block.get("name", ""), block.get("input") or {}))
        usage = claude_usage(data.get("usage") or {})
        self.harness.gate.record("claude-api", self.model, usage, self.harness._price(usage))
        return ModelTurn(text, calls, usage, latency, raw={"stop_reason": data.get("stop_reason")})


class ClaudeCliDriver:
    """The CLI on its own login, keeping one session the way a real one is kept.

    New events are appended to the session file, and a pacing turn resumes in place, so its entries stay readable. A
    probe or a task forks, so asking never teaches the model the fact, and the session behind it is untouched. Only an
    edit rewrites the file, and that restarts the cache at the floor, which is what an edit costs on this door.

    Writing a new file for every call would look equivalent and cost far more: a new session's opening rows land
    somewhere else each time, so no request can read the last one's entry. Measured: 3,100 read against 5.9 million
    written over 38 calls.

    Tools are not offered through this door, so a lookup comes back as text: a line `TOOL <name> <JSON arguments>`.
    """

    def __init__(self, model: str = "claude-haiku-5-5", gate=None, ttl: str = "1h", environ=None) -> None:
        self.harness = ClaudeCliHarness(model, environ=environ)
        self.capabilities = DriverCaps("claude-cli", "account", ("1h",), spends=False)
        self.ttl = ttl
        self.model = model
        self.written: list = []
        self.rewrites = 0

    def _sync(self, blocks: list) -> None:
        """Bring the session file up to `blocks`: append if nothing earlier changed, rewrite if something did."""
        keys = [canonical(b) for b in blocks]
        if self.harness.current and keys[:len(self.written)] == self.written:
            fresh = blocks[len(self.written):]
            if fresh:
                self.harness.append_rows(RenderedContext("", fresh))
        else:
            self.harness.apply(None, RenderedContext("", list(blocks)))
            self.rewrites += 1
        self.written = keys

    def call(self, request: RenderedContext) -> ModelTurn:
        if request.system:
            self.harness.extra_args = ["--append-system-prompt", request.system]
        prompt = request.blocks[-1].text if request.blocks else ""
        history = [b for b in request.blocks[:-1] if b.event_id]
        lookups = [b for b in request.blocks[:-1] if not b.event_id]
        forked = bool(request.meta.get("mode")) or bool(lookups)
        if lookups:
            prompt = "\n".join(f"Lookup result:\n{b.text}" for b in lookups if b.type == "tool_result") + "\n\n" + prompt
        if request.tools:
            prompt += _TEXT_TOOLS.format(names="; ".join(f"{t.name} {_arg_names(t)}" for t in request.tools))
        started = time.monotonic()
        if history or self.harness.current:
            self._sync(history)
            turn = self.harness.fork(prompt) if forked else self.harness.resume(prompt)
        else:
            turn = self.harness.start(prompt)
        latency = time.monotonic() - started
        if turn.error:
            return ModelTurn(error=turn.error, latency_s=latency)
        if not forked:
            # The CLI wrote this turn into the session itself, so it counts as written.
            self.written += [canonical(Block("user", "text", text=request.blocks[-1].text if request.blocks else "")),
                             canonical(Block("assistant", "text", text=turn.text or " "))]
        return _from_text(turn.text, turn.usage, latency)

    def close(self) -> None:
        self.harness.cleanup()


_TEXT_TOOLS = ("\n\nNo tools are callable here. To use one, reply with one line only: "
               "TOOL <name> <JSON arguments>. Available, with their arguments: {names}.")


def _arg_names(tool) -> str:
    """An example object: shown a bare list of names, a model answers with a bare list of values."""
    return json.dumps({name: "..." for name in sorted(tool.schema.get("properties") or {})})


def _from_text(text: str, usage: CallUsage, latency: float) -> ModelTurn:
    for i, line in enumerate(text.splitlines()):
        line = line.strip().strip("`").strip()
        if line.startswith("TOOL "):
            parts = line.split(" ", 2)
            if len(parts) == 3:
                try:
                    args = json.loads(parts[2])
                except ValueError:
                    continue
                if isinstance(args, dict):
                    return ModelTurn("", [ToolCall(f"text-{i}", parts[1], args)], usage, latency)
    return ModelTurn(text, [], usage, latency)


def make_driver(name: str, *, model: str, gate: SpendGate, ttl: str):
    if name == "claude-api":
        return ClaudeApiDriver(model, gate, ttl=ttl)
    if name == "claude-cli":
        return ClaudeCliDriver(model, gate, ttl)
    raise ValueError(f"no live driver named {name!r} yet: claude-api and claude-cli run sessions; every other "
                     "harness has its conformance probe")
