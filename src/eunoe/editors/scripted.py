"""A harness with no model and no network, so the suite itself runs anywhere, in CI included.

It holds a message list and answers with a small fixed reader. The prefix cache is the same simulation
the scripted driver uses. `binding=True` makes it behave like an account where editing history under a
thinking block is refused, so the failure path of the suite is exercised too.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

from eunoe.cachesim import PrefixCache, ThinkingBindingError
from eunoe.editors.base import Harness, Turn, replace_in_json
from eunoe.interfaces import Block, RenderedContext, SessionRef
from eunoe.levels import estimate_tokens

_CODEWORD_RE = re.compile(r"[Cc]odeword: ([A-Z]+-\d+)")


class ScriptedHarness(Harness):
    name = "scripted"
    default_model = "scripted"
    apply_supported = True

    def __init__(self, model=None, gate=None, binding: bool = False, ttl: str = "5m") -> None:
        super().__init__(model, gate)
        self.binding = binding
        self.cache = PrefixCache(ttl, binding)
        self.blocks: list = []
        self.clock = 0.0
        self.counter = 0
        self.session = "s0"
        self.thinking = False

    def version(self) -> str:
        return "offline"

    def _send(self, prompt: str) -> Turn:
        self.clock += 1.0
        self.blocks.append(Block("user", "text", text=prompt))
        reply = self._reply(prompt)
        ctx = RenderedContext("", list(self.blocks), [], {})
        try:
            usage = self.cache.call(ctx, self.clock, estimate_tokens(reply))
        except ThinkingBindingError as exc:
            self.blocks.pop()
            return Turn(error=f"400 invalid_request_error: {exc}", session_id=self.session)
        if self.thinking:
            self.counter += 1
            self.blocks.append(Block("assistant", "thinking", signature=f"sig{self.counter}", tokens_hint=40))
        self.blocks.append(Block("assistant", "text", text=reply))
        return Turn(reply, usage, self.session, self.thinking_block_count())

    def _reply(self, prompt: str) -> str:
        if "only the word OK" in prompt:
            return "OK"
        if "only the word READY" in prompt:
            return "READY"
        if "rollback token" in prompt:
            return " ".join(_shaped(self.blocks))
        if "only the codeword" in prompt:
            found = None
            for block in self.blocks:
                match = _CODEWORD_RE.search(block.text or "")
                if match:
                    found = match.group(1)
            return found or "UNKNOWN"
        return "42"

    def start(self, prompt: str, thinking: bool = False) -> Turn:
        self.counter += 1
        self.session = f"s{self.counter}"
        self.thinking = thinking
        self.blocks = []
        self.cache = PrefixCache(self.cache.ttl, self.binding)
        return self._send(prompt)

    def resume(self, prompt: str) -> Turn:
        return self._send(prompt)

    def replace(self, old: str, new: str) -> int:
        total, out = 0, []
        for block in self.blocks:
            text, n = replace_in_json(block.text, old, new)
            total += n
            out.append(Block(block.role, block.type, text, block.name, block.input, block.tool_id, block.event_id,
                             block.signature, block.tokens_hint))
        self.blocks = out
        return total

    def fork(self, prompt: str) -> Turn:
        keep = copy.deepcopy(self.blocks), self.session, self.cache
        turn = self._send(prompt)
        turn.session_id = f"{self.session}-fork"
        self.blocks, self.session, self.cache = keep
        return turn

    def apply(self, session, rendered: RenderedContext) -> SessionRef:
        self.counter += 1
        self.session = f"s{self.counter}"
        self.thinking = False
        self.blocks = list(rendered.blocks)
        self.cache = PrefixCache(self.cache.ttl, self.binding)
        return SessionRef(self.name, self.session)

    def snapshot(self) -> str:
        data = json.dumps([[b.role, b.type, b.text] for b in self.blocks], sort_keys=True)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def thinking_block_count(self) -> int:
        return sum(1 for b in self.blocks if b.type == "thinking")


def _shaped(blocks: list) -> list:
    out = []
    for pattern in (r"RB-[0-9A-Z]+", r"GREEN-\d+"):
        for block in blocks:
            match = re.search(pattern, block.text or "")
            if match:
                out.append(match.group(0))
                break
    return out
