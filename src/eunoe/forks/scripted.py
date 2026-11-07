"""A door with no model behind it: it answers only from the text it was handed, so an evicted fact really is gone.

Its numbers prove the machinery and say nothing about any hypothesis, the same rule as the scripted driver. It looks a
question up by the words either side of a blank, or by the `<key> is <value>` a generated fixture states, and if it finds
nothing it asks the archive once.
"""
from __future__ import annotations

import re

from eunoe.editors.base import Turn
from eunoe.forks.probes import BLANK
from eunoe.forks.transcript import held_text
from eunoe.interfaces import CallUsage, RenderedContext
from eunoe.levels import estimate_tokens
from eunoe.wire import to_text

_ITEM_RE = re.compile(r"^(\d+)\.\s(.*)$", re.MULTILINE)
_RESULT_RE = re.compile(r"^(\d+)\. Results:\n(.*?)(?=^\d+\. Results:|\Z)", re.MULTILINE | re.DOTALL)


class ScriptedDoor:
    def __init__(self, reads: bool = True) -> None:
        self.text = ""
        self.sessions = {}
        self.reads = reads
        self.next_id = 0

    def available(self) -> tuple:
        return True, ""

    def scratch(self) -> str:
        return "scripted"

    def open_rows(self, rows: list) -> None:
        self.text = held_text(rows)

    def open_context(self, ctx: RenderedContext) -> None:
        self.text = to_text(ctx)

    def _usage(self, reply: str, cached: bool) -> CallUsage:
        size = estimate_tokens(self.text)
        return CallUsage(cache_read=size if cached else 0, cache_write_1h=0 if cached else size, output=estimate_tokens(reply))

    def pace(self, prompt: str) -> Turn:
        return Turn("OK", self._usage("OK", False), "paced")

    def fork(self, prompt: str) -> Turn:
        lines = []
        for number, ask in _ITEM_RE.findall(prompt):
            value = self._find(ask, self.text)
            lines.append(f"{number}. ANSWER: {value}" if value else f"{number}. LOOKUP: {_words(ask)}")
        reply = "\n".join(lines) or "A reply naming nothing."
        self.next_id += 1
        session = f"fork-{self.next_id}"
        self.sessions[session] = prompt
        return Turn(reply, self._usage(reply, True), session)

    def follow(self, session_id: str, prompt: str) -> Turn:
        lines = []
        asked = dict(_ITEM_RE.findall(self.sessions.get(session_id, "")))
        for number, found in _RESULT_RE.findall(prompt):
            value = self._find(asked.get(number, ""), found)
            lines.append(f"{number}. ANSWER: {value or 'UNKNOWN'}")
        reply = "\n".join(lines)
        return Turn(reply, self._usage(reply, True), session_id)

    def one_shot(self, prompt: str) -> Turn:
        ids = re.findall(r"^(e\d+) ", prompt, re.MULTILINE)
        reply = "\n".join(f"{event_id} {sum(map(ord, event_id)) % 10}" for event_id in ids)
        return Turn(reply, CallUsage(input=estimate_tokens(prompt), output=estimate_tokens(reply)), "rate")

    def end_condition(self) -> None:
        self.sessions.clear()

    def close(self) -> None:
        self.text = ""

    @staticmethod
    def _find(ask: str, text: str) -> str:
        low = text
        m = re.match(r"What is the (.+)\?", ask)
        if m:
            found = re.findall(re.escape(m.group(1)) + r" is ([^\n]+?)\.(?:\s|$)", low, re.IGNORECASE)
            return found[-1] if found else ""
        if BLANK in ask:
            before, _, after = ask.partition(BLANK)
            pattern = re.escape(before[-24:].lower()) + r"(\S{3,120}?)" + re.escape(after[:12].lower())
            found = re.search(pattern, low.lower())
            return found.group(1) if found else ""
        return ""


def _words(ask: str) -> str:
    m = re.match(r"What is the (.+)\?", ask)
    return m.group(1) if m else " ".join(ask.replace(BLANK, " ").split()[:5])


__all__ = ["ScriptedDoor"]
