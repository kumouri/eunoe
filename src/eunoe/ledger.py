"""The event ledger: every event with a stable id, its verbatim copy, and the level it is held at.

An event is one thing that happened: a tool call with its result, a user message, a system note, an
assistant turn, a loaded file. The verbatim copy never changes, and a level change only changes what
the renderer produces, so nothing is lost by evicting and a lookup can always reach the original.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field

from eunoe.levels import Level, estimate_tokens, find_details

USER = "user"
ASSISTANT = "assistant"
THINKING = "thinking"
SYSTEM_NOTE = "system_note"
FILE = "file"
TOOL_PREFIX = "tool:"
#: A read tool's result is the file, so the file is the archive.
RECONSTRUCTIBLE_TOOLS = ("tool:read_file", "tool:read_runbook", "tool:read_contract", "tool:fetch_paper")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*")


@dataclass
class Event:
    id: str
    kind: str
    index: int
    call: int
    at: float
    text: str = ""
    input: dict | None = None
    tool_id: str = ""
    source: str = ""
    signature: str = ""
    tokens_hint: int = 0
    details: tuple = field(default_factory=tuple)

    @property
    def reconstructible(self) -> bool:
        return self.kind == FILE or self.kind in RECONSTRUCTIBLE_TOOLS

    @property
    def is_tool(self) -> bool:
        return self.kind.startswith(TOOL_PREFIX)

    @property
    def tool_name(self) -> str:
        return self.kind[len(TOOL_PREFIX):] if self.is_tool else ""


class Ledger:
    """One session's events, in order. Holds no opinion about what to evict."""

    def __init__(self, renderer=None, marker: str = "M0") -> None:
        from eunoe.renderer import DefaultRenderer

        self.renderer = renderer or DefaultRenderer(marker=marker)
        self.events: list = []
        self.by_id: dict = {}
        self.levels: dict = {}
        self.lookups: collections.Counter = collections.Counter()
        self.references: collections.defaultdict = collections.defaultdict(list)
        self._token_cache: dict = {}
        self._mentions: collections.defaultdict = collections.defaultdict(list)
        self.call = 0
        self.now = 0.0

    # ---- appending

    def append(self, kind: str, text: str = "", *, input: dict | None = None, tool_id: str = "",
               source: str = "", signature: str = "", tokens_hint: int = 0, at: float | None = None) -> Event:
        index = len(self.events)
        event_id = f"e{index:05d}"
        body = text if not input else f"{text}\n{_flatten(input)}"
        details = tuple(sorted({d for d, _, _ in find_details(body)})) if kind != THINKING else ()
        event = Event(event_id, kind, index, self.call, self.now if at is None else at, text, input,
                      tool_id or f"t{index:05d}", source, signature, tokens_hint, details)
        self.events.append(event)
        self.by_id[event_id] = event
        self.levels[event_id] = Level.L0
        for detail in details:
            earlier = self._mentions[detail]
            for other in earlier[-8:]:
                self.references[other].append(index)
            earlier.append(event_id)
        return event

    # ---- levels and sizes

    def level(self, event_id: str) -> Level:
        return self.levels[event_id]

    def set_level(self, event_id: str, level: int) -> None:
        self.levels[event_id] = Level(level)

    def tokens_at(self, event: Event, level: int) -> int:
        key = (event.id, int(level))
        cached = self._token_cache.get(key)
        if cached is None:
            cached = sum(block_tokens(b) for b in self.renderer.render(event, Level(level)))
            self._token_cache[key] = cached
        return cached

    def tokens_held(self) -> int:
        return sum(self.tokens_at(e, self.levels[e.id]) for e in self.events)

    def held(self) -> list:
        return [e for e in self.events if self.levels[e.id] < Level.L4]

    # ---- the archive

    def archive_get(self, event_id: str) -> Event | None:
        event = self.by_id.get(event_id)
        if event is not None:
            self.lookups[event_id] += 1
        return event

    def search(self, query: str, limit: int = 3) -> list:
        """Lexical search over the verbatim archive, every word required, newest first."""
        words = [w.lower() for w in _WORD_RE.findall(query or "")]
        if not words:
            return []
        hits = []
        for event in reversed(self.events):
            if event.kind == THINKING:
                continue
            body = f"{event.text}\n{_flatten(event.input)}".lower() if event.input else event.text.lower()
            if all(w in body for w in words):
                hits.append(event)
                self.lookups[event.id] += 1
                if len(hits) >= limit:
                    break
        return hits

    def reference_count(self, event: Event, upto_index: int | None = None) -> int:
        """Later events naming this event's identifiers, so far."""
        refs = self.references.get(event.id, ())
        if upto_index is None:
            return len(refs)
        return sum(1 for r in refs if r <= upto_index)


def block_tokens(block) -> int:
    if block.tokens_hint:
        return block.tokens_hint
    size = estimate_tokens(block.text)
    if block.input:
        size += estimate_tokens(_flatten(block.input)) + 4
    return size + 3


def _flatten(value) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    return "" if value is None else str(value)
