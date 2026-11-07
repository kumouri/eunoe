"""A Claude Code session file as a checkpoint, and an edited copy of it to resume.

Reads a session's rows up to a turn boundary into a ledger, with each event mapped to the blocks that hold it. Writing
goes through the kit's renderer, so a real session gets the same levels a generated one does. The format is undocumented,
so this keeps only the rows a resume needs (`user`, `assistant`, `attachment`) and chains them again. It writes text
nowhere but the session file it was asked to write, and `run` removes that when a condition is done.

A row can hold other copies of its text (`toolUseResult`, an attachment's own fields), and those are dropped or
replaced too, because an edit has to reach every copy.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field

from eunoe.ledger import ASSISTANT, FILE, THINKING, USER, Ledger
from eunoe.levels import Level
from eunoe.renderer import DefaultRenderer

MESSAGE_ROWS = ("user", "assistant", "attachment")
#: Fields that copy a row's text, or that a resume rebuilds for itself.
DROP_FIELDS = ("toolUseResult", "sourceToolAssistantUUID", "requestRef")
_THINKING_TYPES = ("thinking", "redacted_thinking")


def read_rows(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def blocks_of(row: dict) -> list:
    content = (row.get("message") or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content or [] if isinstance(b, dict)]


def result_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
        elif isinstance(block, dict) and "content" in block:
            parts.append(result_text(block["content"]))
    return "\n".join(parts)


def rendered_text(row: dict) -> str:
    rendered = row.get("rendered")
    if isinstance(rendered, str):
        return rendered
    return "\n".join(b.get("text", "") for b in rendered or [] if isinstance(b, dict))


def turn_boundaries(rows: list) -> list:
    """`(row index, context before it)` where a real user turn starts and no tool call is waiting, before any compaction."""
    out, pending, context = [], set(), 0
    for i, row in enumerate(rows):
        kind = row.get("type")
        if row.get("isSidechain"):
            continue
        if kind == "system" and row.get("subtype") == "compact_boundary":
            break
        if kind == "assistant":
            usage = (row.get("message") or {}).get("usage") or {}
            context = (int(usage.get("input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0)
                       + int(usage.get("cache_creation_input_tokens") or 0)) or context
            pending |= {b.get("id") for b in blocks_of(row) if b.get("type") == "tool_use"}
        elif kind == "user":
            blocks = blocks_of(row)
            pending -= {b.get("tool_use_id") for b in blocks if b.get("type") == "tool_result"}
            if not pending and not row.get("isMeta") and any(b.get("type") == "text" for b in blocks) and \
                    not any(b.get("type") == "tool_result" for b in blocks):
                out.append((i, context))
    return out


@dataclass
class Checkpoint:
    id: str
    rows: list
    ledger: Ledger
    where: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    frozen_upto: int = -1
    after: list = field(default_factory=list)

    def label(self, event) -> str:
        return self.labels.get(event.id, event.kind)


def load_checkpoint(rows: list, cut: int, checkpoint_id: str = "cp", classify=None, marker: str = "M0") -> Checkpoint:
    """Rows before `cut` as a checkpoint. Rows after it are kept to draw probes from, and never written to a fork.

    `classify(text, row)` may return a kind for a user block, so a caller can split a person's words from what a program
    put in the same channel. By default all of it is `USER`. Everything before the first model call is the floor, which
    `frozen_upto` names, so a condition can leave it alone.
    """
    pending, tools = [], {}
    call, seen_ids, floor_ends = 0, set(), None
    for i, row in enumerate(rows[:cut]):
        kind = row.get("type")
        if row.get("isSidechain") or kind not in MESSAGE_ROWS:
            continue
        if kind == "attachment":
            text = rendered_text(row)
            if text:
                att = row.get("attachment") or {}
                source = str(att.get("displayPath") or att.get("path") or att.get("type"))
                pending.append(dict(kind=FILE, text=text, call=call, source=source,
                                    label=f"attachment:{att.get('type')}", where=[(i, None)]))
            continue
        if kind == "assistant":
            mid = (row.get("message") or {}).get("id")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                call += 1
                if floor_ends is None:
                    floor_ends = len(pending)
            for b, block in enumerate(blocks_of(row)):
                btype = block.get("type")
                if btype == "text":
                    pending.append(dict(kind=ASSISTANT, text=block.get("text") or "", call=call, where=[(i, b)]))
                elif btype in _THINKING_TYPES:
                    pending.append(dict(kind=THINKING, text="", call=call, where=[(i, b)],
                                        signature=str(block.get("signature") or "")))
                elif btype == "tool_use":
                    entry = dict(kind=f"tool:{block.get('name')}", text="", input=block.get("input") or {}, call=call,
                                 where=[(i, b)], tool_id=block.get("id") or "")
                    tools[block.get("id")] = entry
                    pending.append(entry)
            continue
        for b, block in enumerate(blocks_of(row)):
            btype = block.get("type")
            if btype == "tool_result":
                entry = tools.get(block.get("tool_use_id"))
                if entry is not None:
                    entry["text"] += result_text(block.get("content"))
                    entry["where"].append((i, b))
            elif btype == "text":
                text = block.get("text") or ""
                pending.append(dict(kind=(classify(text, row) if classify else None) or USER, text=text, call=call,
                                    where=[(i, b)]))
    ledger = Ledger(renderer=DefaultRenderer(marker=marker))
    where, labels, frozen = {}, {}, -1
    for n, entry in enumerate(pending):
        ledger.call = entry["call"]
        event = ledger.append(entry["kind"], entry["text"], input=entry.get("input"), source=entry.get("source", ""),
                              tool_id=entry.get("tool_id", ""), signature=entry.get("signature", ""))
        where[event.id] = entry["where"]
        labels[event.id] = entry.get("label", entry["kind"])
        if floor_ends is not None and n < floor_ends:
            frozen = event.index
    ledger.call = call
    return Checkpoint(checkpoint_id, rows[:cut], ledger, where, labels, frozen, rows[cut:])


def _edits_for(cp: Checkpoint, levels: dict, strip_thinking: bool) -> dict:
    """`{(row, block): None to remove, or the fields to change}`. An attachment's block is `None`."""
    ledger, edits = cp.ledger, {}
    for event in ledger.events:
        spots = cp.where.get(event.id) or []
        if event.kind == THINKING:
            if strip_thinking:
                edits.update({spot: None for spot in spots})
            continue
        level = Level(levels.get(event.id, Level.L0))
        if level == Level.L0:
            continue
        blocks = ledger.renderer.render(event, level)
        if not blocks:
            edits.update({spot: None for spot in spots})
        elif not event.is_tool:
            edits.update({spot: dict(text=blocks[0].text or " ") for spot in spots})
        elif level == Level.L3:
            # The renderer merges a call and its result into one line of assistant text.
            edits[spots[0]] = dict(replace={"type": "text", "text": blocks[0].text or " "})
            edits.update({spot: None for spot in spots[1:]})
        else:
            edits[spots[0]] = dict(input=blocks[0].input)
            edits.update({spot: dict(content=blocks[1].text or " ") for spot in spots[1:]})
    return edits


def render_rows(cp: Checkpoint, levels: dict, strip_thinking: bool, session_id: str, cwd: str) -> list:
    """The checkpoint's rows with `levels` applied, chained again, ready to write as a session file."""
    edits = _edits_for(cp, levels, strip_thinking)
    surviving = _surviving_messages(cp.rows, edits)
    out, remap = [], {}
    for i, row in enumerate(cp.rows):
        new = _edit_row(i, row, edits, surviving)
        rid = row.get("uuid")
        parent = _resolve(row.get("parentUuid"), remap)
        if new is None:
            if rid:
                remap[rid] = parent
            continue
        new = {k: v for k, v in new.items() if k not in DROP_FIELDS}
        new.update(parentUuid=parent, sessionId=session_id, cwd=cwd)
        out.append(new)
    return out


def _resolve(parent, remap: dict):
    while parent in remap:
        parent = remap[parent]
    return parent


def _surviving_messages(rows: list, edits: dict) -> set:
    """Message ids with something other than thinking left. A message of only thinking can't stand alone."""
    out = set()
    for i, row in enumerate(rows):
        if row.get("type") == "assistant":
            for b, block in enumerate(blocks_of(row)):
                if block.get("type") not in _THINKING_TYPES and edits.get((i, b), block) is not None:
                    out.add((row.get("message") or {}).get("id"))
    return out


def _edit_row(i: int, row: dict, edits: dict, surviving: set):
    kind = row.get("type")
    if row.get("isSidechain") or kind not in MESSAGE_ROWS:
        return None
    if kind == "attachment":
        if (i, None) not in edits:
            return row if row.get("rendered") else None
        change = edits[(i, None)]
        if change is None:
            return None
        att = dict(row.get("attachment") or {})
        for name in ("content", "text", "snippet", "context"):
            if isinstance(att.get(name), str):
                att[name] = change["text"]
        return {**row, "rendered": [{"type": "text", "text": change["text"]}], "attachment": att}
    message = row.get("message") or {}
    if kind == "assistant" and message.get("id") not in surviving:
        return None
    blocks = blocks_of(row)
    if not any((i, b) in edits for b in range(len(blocks))):
        return row
    kept = []
    for b, block in enumerate(blocks):
        if (i, b) not in edits:
            kept.append(block)
        elif edits[(i, b)] is not None:
            change = edits[(i, b)]
            kept.append(change["replace"] if "replace" in change else {**block, **change})
    return {**row, "message": {**message, "content": kept}} if kept else None


def held_text(rows: list) -> str:
    """Everything in written rows a model could read, as one lowercase string, to see if a detail survived."""
    parts = []
    for row in rows:
        kind = row.get("type")
        if kind == "attachment":
            parts.append(rendered_text(row))
        elif kind in ("user", "assistant"):
            for block in blocks_of(row):
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text") or "")
                elif btype == "tool_use":
                    parts.append(json.dumps(block.get("input") or {}, ensure_ascii=False))
                elif btype == "tool_result":
                    parts.append(result_text(block.get("content")))
    return "\n".join(parts).lower()


def write_session(rows: list, folder: str, session_id: str | None = None) -> tuple:
    """Write rows as a session file. Returns `(session_id, path)`."""
    session_id = session_id or str(uuid.uuid4())
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{session_id}.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps({**row, "sessionId": session_id}, ensure_ascii=False, separators=(",", ":")) + "\n")
    return session_id, path


__all__ = ["Checkpoint", "blocks_of", "held_text", "load_checkpoint", "read_rows", "render_rows", "rendered_text",
           "result_text", "turn_boundaries", "write_session"]
