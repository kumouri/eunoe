"""(event, level) to blocks: a pure function of its inputs, so the bytes are stable call to call.

Byte stability is what the cache depends on. A cached prefix survives up to its first changed byte,
so a renderer that produced different bytes for the same event and level would break every read after
it for nothing. `test_renderer` pins this.

A tool call and its result move together, because an API needs the pair valid: they render as a
`tool_use` and a `tool_result` through L2, and at L3 they merge into one line of assistant text.
"""
from __future__ import annotations

from eunoe import levels
from eunoe.interfaces import Block, RenderedContext
from eunoe.ledger import ASSISTANT, FILE, SYSTEM_NOTE, THINKING, USER, Event, Ledger
from eunoe.levels import Level

MARKERS = ("M0", "M1", "M2")


class DefaultRenderer:
    """The default ladder. `marker` is how the model is told there's more: nothing, `…`, or an id."""

    def __init__(self, marker: str = "M0") -> None:
        if marker not in MARKERS:
            raise ValueError(f"unknown marker {marker!r}, one of {MARKERS}")
        self.marker = marker

    def _mark(self, event: Event, text: str, level: Level) -> str:
        if level == Level.L0 or self.marker == "M0":
            return text
        if self.marker == "M1":
            return f"{text} …"
        return f"{text} [more: recall {event.id}]"

    def render(self, event: Event, level: Level) -> list:
        level = Level(level)
        if level == Level.L4:
            return []
        if event.kind == THINKING:
            return [Block("assistant", "thinking", text=event.text, signature=event.signature,
                          event_id=event.id, tokens_hint=event.tokens_hint)]
        if event.is_tool:
            return self._tool(event, level)
        role = "assistant" if event.kind == ASSISTANT else "user"
        return [Block(role, "text", text=self._mark(event, self._prose(event, level), level), event_id=event.id)]

    def _prose(self, event: Event, level: Level) -> str:
        text = event.text
        if level == Level.L0:
            if event.kind == FILE:
                return f"{event.source}\n{text}"
            return text
        if event.kind == FILE:
            return levels.reconstructible_stub(event.source, text)
        if event.kind == SYSTEM_NOTE:
            return levels.first_sentence(text)
        if event.kind == USER:
            if level == Level.L1:
                return text
            return levels.key_sentences(text) if level == Level.L2 else levels.first_sentence(text)
        if event.kind == ASSISTANT:
            if level == Level.L1:
                return levels.first_last_paragraphs(text)
            return levels.key_sentences(text) if level == Level.L2 else levels.first_sentence(text)
        return levels.trim(text) if level == Level.L1 else levels.headline(text)

    def _tool(self, event: Event, level: Level) -> list:
        name = event.tool_name
        if level == Level.L3:
            return [Block("assistant", "text", event_id=event.id,
                          text=self._mark(event, _tool_headline(event), level))]
        if level == Level.L0:
            args, body = event.input or {}, event.text
        else:
            args = levels.trim_input(event.input or {}, 200 if level == Level.L1 else 60)
            if event.reconstructible:
                body = levels.reconstructible_stub(_source(event), event.text)
            elif level == Level.L1:
                body = levels.trim(event.text)
            else:
                body = levels.digest(event.text)
            body = self._mark(event, body, level)
        return [
            Block("assistant", "tool_use", name=name, input=args, tool_id=event.tool_id, event_id=event.id),
            Block("user", "tool_result", text=body, tool_id=event.tool_id, event_id=event.id),
        ]


def _source(event: Event) -> str:
    args = event.input or {}
    return str(event.source or args.get("path") or args.get("id") or event.tool_name)


def _tool_headline(event: Event) -> str:
    args = event.input or {}
    target = ""
    for value in args.values():
        if isinstance(value, (str, int, float)):
            target = str(value)[:60]
            break
    return f"{event.tool_name} {target}: {levels.headline(event.text)}".strip()


def render_context(ledger: Ledger, system: str = "", tools=None, meta=None) -> RenderedContext:
    """Every event at its current level, in order. The floor (`system`, `tools`) is never edited."""
    blocks = []
    for event in ledger.events:
        blocks.extend(ledger.renderer.render(event, ledger.levels[event.id]))
    return RenderedContext(system, blocks, list(tools or []), dict(meta or {}))
