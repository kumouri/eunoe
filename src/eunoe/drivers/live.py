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
from eunoe.interfaces import CallUsage, DriverCaps, ModelTurn, RenderedContext, ToolCall
from eunoe.wire import to_anthropic


class ClaudeApiDriver:
    """The Messages API, client-side, or with server-side edits when a control asks for them."""

    def __init__(self, model: str = "claude-sonnet-5-5", gate: SpendGate | None = None, transport=http_post,
                 environ=None, ttl: str = "5m") -> None:
        self.harness = ClaudeApiHarness(model, gate or SpendGate(), transport, environ, ttl, ListPrice(model))
        self.capabilities = DriverCaps("claude-api", "account", ("5m", "1h"), server_edits=True, spends=True)
        self.model = model

    def request(self, ctx: RenderedContext) -> tuple:
        wire = to_anthropic(ctx, thinking_replay=False)
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
    """The CLI on its own login: each call writes the rendered context into a session, then resumes it.

    The prompt is the context's last block. Tools are not offered through this door, so a lookup comes back
    as text: a line `RECALL: <query>`, which the driver turns into a tool call. An `apply` is the same.
    """

    def __init__(self, model: str = "claude-haiku-5-5", gate=None, ttl: str = "1h", environ=None) -> None:
        self.harness = ClaudeCliHarness(model, environ=environ)
        self.capabilities = DriverCaps("claude-cli", "account", ("1h",), spends=False)
        self.ttl = ttl
        self.model = model

    def call(self, request: RenderedContext) -> ModelTurn:
        if request.system:
            self.harness.extra_args = ["--append-system-prompt", request.system]
        prompt = request.blocks[-1].text if request.blocks else ""
        history = RenderedContext(request.system, list(request.blocks[:-1]), request.tools, request.meta)
        if request.tools:
            prompt += _TEXT_TOOLS.format(names="; ".join(f"{t.name} {_arg_names(t)}" for t in request.tools))
        started = time.monotonic()
        if history.blocks:
            self.harness.apply(None, history)
            turn = self.harness.resume(prompt)
        else:
            turn = self.harness.start(prompt)
        latency = time.monotonic() - started
        if turn.error:
            return ModelTurn(error=turn.error, latency_s=latency)
        return _from_text(turn.text, turn.usage, latency)

    def close(self) -> None:
        self.harness.cleanup()


_TEXT_TOOLS = ("\n\nNo tools are callable here. To use one, reply with one line only: "
               "TOOL <name> <JSON arguments>. Available, with their arguments: {names}.")


def _arg_names(tool) -> str:
    return json.dumps(sorted((tool.schema.get("properties") or {})))


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
                return ModelTurn("", [ToolCall(f"text-{i}", parts[1], args)], usage, latency)
    return ModelTurn(text, [], usage, latency)


def make_driver(name: str, *, model: str, gate: SpendGate, ttl: str):
    if name == "claude-api":
        return ClaudeApiDriver(model, gate, ttl=ttl)
    if name == "claude-cli":
        return ClaudeCliDriver(model, gate, ttl)
    raise ValueError(f"no live driver named {name!r} yet: claude-api and claude-cli run sessions; every other "
                     "harness has its conformance probe")
