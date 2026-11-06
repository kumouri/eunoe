"""The scripted driver: a fixture standing in for a model, with no LLM and nothing to spend.

**Its numbers prove the machinery and say nothing about any hypothesis.** It can only answer from the
rendered context it was handed, so the plumbing can't cheat: an evicted fact really is gone from what
it reads. How often it looks things up and whether it guesses are settings, written down, not findings.
A real model answers these questions in a scored run.

What it does with a probe, in order:

1. If the context states "<key> is <value>", it answers with the newest one.
2. If a block naming the subject holds exactly one token shaped like a value, it answers with that.
3. If the context still shows a trace of the subject, it looks up with a `lookup_rate` chance, or
   always when an explicit marker names the event.
4. With no trace at all it looks up at `blind_lookup_rate`, then abstains or, at `guess_rate`, invents.
"""
from __future__ import annotations

import re

from eunoe.cachesim import PrefixCache, ThinkingBindingError
from eunoe.interfaces import CallUsage, DriverCaps, ModelTurn, RenderedContext, ToolCall
from eunoe.levels import estimate_tokens
from eunoe.rng import Rng


class ScriptedDriver:
    def __init__(self, seed: int = 0, ttl: str = "5m", binding: bool = False, lookup_rate: float = 0.6,
                 blind_lookup_rate: float = 0.1, guess_rate: float = 0.15, reply_tokens: int = 30) -> None:
        self.capabilities = DriverCaps("scripted", "bound" if binding else "none", ("5m", "1h"),
                                       replays_scripted_thinking=True)
        self.cache = PrefixCache(ttl, binding)
        self.ttl = ttl
        self.seed = seed
        self.lookup_rate, self.blind_lookup_rate, self.guess_rate = lookup_rate, blind_lookup_rate, guess_rate
        self.reply_tokens = reply_tokens

    def call(self, request: RenderedContext) -> ModelTurn:
        now = float(request.meta.get("now", 0.0))
        turn = self._decide(request)
        try:
            usage = self.cache.call(request, now, estimate_tokens(turn.text) + self.reply_tokens)
        except ThinkingBindingError as exc:
            return ModelTurn(error=f"400: {exc}")
        turn.usage = usage
        return turn

    def _decide(self, request: RenderedContext) -> ModelTurn:
        meta = request.meta
        if "key" not in meta:
            return ModelTurn(text="OK")
        key, subject, prefix = meta["key"], meta["subject"], meta.get("prefix", "")
        blocks = [b for b in request.blocks if b.type != "thinking"]
        answer = _newest_statement(blocks, key)
        if answer is None and prefix:
            answer = _single_shaped_token(blocks, subject, prefix)
        if answer is not None:
            return self._answer(meta, answer)
        has_recall = any(t.name == "recall" for t in request.tools)
        if has_recall and not meta.get("looked_up"):
            trace = [b for b in blocks if subject in (b.text or "") or subject in str(b.input or "")]
            marked = any("[more: recall" in (b.text or "") for b in trace)
            rate = 1.0 if marked else (self.lookup_rate if trace else self.blind_lookup_rate)
            if self._draw(meta, "lookup") < rate:
                return ModelTurn(tool_calls=[ToolCall(f"lookup-{meta.get('id', '')}", "recall", {"query": subject})])
        if self._draw(meta, "guess") >= self.guess_rate:
            return self._answer(meta, "UNKNOWN")
        return self._answer(meta, f"{prefix or 'X-'}{Rng(self.seed, meta.get('id'), 'value').randint(1000, 9999)}")

    def _draw(self, meta: dict, purpose: str) -> float:
        """Keyed by question, not by position in a stream, so every arm is paired on the same draw."""
        return Rng("scripted-driver", self.seed, meta.get("id", ""), purpose).random()

    def _answer(self, meta: dict, value: str) -> ModelTurn:
        if meta.get("mode") == "task" and value != "UNKNOWN":
            return ModelTurn(tool_calls=[ToolCall(f"apply-{meta.get('id', '')}", "apply",
                                                  {"subject": meta["subject"], "value": value})])
        return ModelTurn(text=f"ANSWER: {value}")


def _newest_statement(blocks: list, key: str) -> str | None:
    pattern = re.compile(re.escape(key) + r" is ([^\n;,]+?)(?:[.;,]\s|\.?\s*$)", re.MULTILINE)
    answer = None
    for block in blocks:
        for match in pattern.finditer(block.text or ""):
            answer = match.group(1).strip()
    return answer


def _single_shaped_token(blocks: list, subject: str, prefix: str) -> str | None:
    """The digest keeps tokens without their sentence. One token of the right shape can be read."""
    token = re.compile(re.escape(prefix) + r"[A-Za-z0-9._/-]+")
    found = None
    for block in blocks:
        text = block.text or ""
        if subject not in text and subject not in str(block.input or ""):
            continue
        hits = {m.group(0).rstrip(".") for m in token.finditer(text)}
        if len(hits) == 1:
            found = hits.pop()
        elif len(hits) > 1:
            return None
    return found


__all__ = ["ScriptedDriver", "CallUsage"]
