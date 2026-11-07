"""Probes from a real session: details its later turns used that only the context before the checkpoint held.

A later turn wrote a detail (an id, a slug, a path, a file name). If nothing arrived after the checkpoint that carried it,
that turn reached back. The line it wrote, with the detail blanked, is the question. The bias is the one every demand
measure has: that session ran with everything in context, so this undercounts what a lean one would have looked for.
"""
from __future__ import annotations

import json
import re

from eunoe.forks.probes import Item, clean_batch, cloze, detail_key, strong_details
from eunoe.forks.transcript import blocks_of, rendered_text, result_text
from eunoe.ledger import THINKING
from eunoe.levels import estimate_tokens
from eunoe.rng import Rng
from eunoe.signals import age_band

TASK = "Give the value that goes in each blank."
MAX_SOURCES = 8
#: A probe never quotes a line that names a credential, whatever else it says.
_SECRETISH = re.compile(r"(?i)\b(?:token|password|passwd|secret|api[_ -]?key|credential|oauth|bearer|cookie)s?\b|\.env\b")


def is_prose(line: str) -> bool:
    """A sentence someone would write, not a command. Blanking a value in a command tests little: most are guessable or
    can't be told apart, and a request to fill in a path inside a shell line can look like something else to a classifier."""
    if re.search(r"(?:\||&&|;\s|\$\(|\bsed\b|\bgrep\b|\bcat\b|\bpython\b|\bgit\b|=\s*open\(|\bcd\s|2>|\bbash\b|\bcurl\b)", line):
        return False
    words = re.findall(r"[A-Za-z]{2,}", line)
    return len(words) >= 8 and sum(len(w) for w in words) >= 0.55 * len(line)


_key = detail_key


def _size_band(tokens: int) -> str:
    if tokens < 100:
        return "under 100"
    if tokens < 1000:
        return "100-999"
    if tokens < 5000:
        return "1k-5k"
    if tokens < 20000:
        return "5k-20k"
    return "over 20k"


def _strings(value) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def source_index(cp) -> tuple:
    """`(details by key to their events, keys held in the floor)`. Thinking is stored empty, so it never counts."""
    index, floor = {}, set()
    for event in cp.ledger.events:
        if event.kind == THINKING:
            continue
        body = "\n".join([event.text] + _strings(event.input))
        for detail, _, _ in strong_details(body):
            key = _key(detail)
            if event.index <= cp.frozen_upto:
                floor.add(key)
            elif not index.get(key) or index[key][-1] is not event:
                index.setdefault(key, []).append(event)
    return index, floor


def candidates(cp, horizon_calls: int = 300) -> list:
    """Every later first use of a detail that only the checkpoint held, oldest first. `(detail, class, line, calls later)`."""
    index, floor = source_index(cp)
    arrived, used, out = set(), set(), []
    calls, seen = 0, set()
    for row in cp.after:
        kind = row.get("type")
        if row.get("isSidechain") or kind not in ("user", "assistant", "attachment"):
            continue
        if kind == "attachment":
            arrived |= {_key(d) for d, _, _ in strong_details(rendered_text(row))}
            continue
        if kind == "user":
            for block in blocks_of(row):
                text = block.get("text") if block.get("type") == "text" else result_text(block.get("content"))
                arrived |= {_key(d) for d, _, _ in strong_details(text or "")}
            continue
        mid = (row.get("message") or {}).get("id")
        if mid not in seen:
            seen.add(mid)
            calls += 1
        if calls > horizon_calls:
            break
        for block in blocks_of(row):
            if block.get("type") == "text":
                texts = [block.get("text") or ""]
            elif block.get("type") == "tool_use":
                texts = _strings(block.get("input") or {})
            else:
                continue
            for text in texts:
                for line in text.splitlines():
                    if _SECRETISH.search(line) or not is_prose(line):
                        continue
                    for detail, cls, _ in strong_details(line):
                        key = _key(detail)
                        if key in used or key in arrived or key in floor or key not in index:
                            continue
                        if len(index[key]) > MAX_SOURCES:
                            continue
                        used.add(key)
                        out.append((detail, cls, line, calls))
    return out


def items_from(cp, limit: int = 40, seed: int = 0, horizon_calls: int = 300) -> tuple:
    """A stratified draw of cloze items, and the census of every candidate by label and band, for a demand density."""
    index, _ = source_index(cp)
    pool: dict = {}
    census: dict = {}
    for detail, cls, line, later in candidates(cp, horizon_calls):
        ask = cloze(line, detail)
        if not ask:
            continue
        sources = index[_key(detail)]
        newest = sources[-1]
        tags = dict(cls=cls, label=cp.label(newest), band=age_band(cp.ledger.call - newest.call),
                    size=_size_band(estimate_tokens(newest.text)), sources=len(sources), later_calls=later,
                    source_kinds=sorted({cp.label(s) for s in sources}))
        census[(tags["label"], tags["band"])] = census.get((tags["label"], tags["band"]), 0) + 1
        pool.setdefault(tags["label"], []).append(Item("", ask, detail, (), "cloze", [s.id for s in sources], tags))
    rng = Rng("eunoe-demand", cp.id, seed)
    for rows in pool.values():
        rng.shuffle(rows)
    drawn: list = []
    labels = sorted(pool)
    while labels and len(drawn) < limit * 2:
        for label in list(labels):
            if pool[label]:
                drawn.append(pool[label].pop())
            else:
                labels.remove(label)
    items = clean_batch(drawn)[:limit]
    for i, item in enumerate(items):
        item.id = f"c{i:03d}"
    return items, [dict(label=k[0], band=k[1], demand=v) for k, v in sorted(census.items())]


def next_message(cp) -> tuple:
    """The session's real next user text, and the strong details the turn after it wrote, for a continuation measure."""
    from eunoe.forks.run import _hash

    prompt, details, answered = "", set(), False
    for row in cp.after:
        kind = row.get("type")
        if kind == "user" and any(b.get("type") == "text" for b in blocks_of(row)):
            if answered:
                break
            if not prompt:
                prompt = "\n".join(b.get("text") or "" for b in blocks_of(row) if b.get("type") == "text")
        elif kind == "assistant":
            answered = True
            for block in blocks_of(row):
                body = block.get("text") if block.get("type") == "text" else json.dumps(block.get("input") or {})
                details |= {_hash(d) for d, _, _ in strong_details(body or "")}
    return prompt, tuple(sorted(details))


def source_tags(cp) -> dict:
    """`{hashed detail: where the checkpoint held it}`, for reading a reply's details back to their sources. No text."""
    from eunoe.forks.run import _hash

    index, floor = source_index(cp)
    out = {}
    for key, sources in index.items():
        if key in floor:
            continue
        newest = sources[-1]
        out[_hash(key)] = dict(label=cp.label(newest), band=age_band(cp.ledger.call - newest.call),
                               size=_size_band(estimate_tokens(newest.text)), sources=len(sources))
    return out


def next_messages(cp, count: int = 3, min_words: int = 0) -> list:
    """The session's next few user messages, each the question a fork can be asked instead of the next turn it really had."""
    out = []
    for row in cp.after:
        if row.get("type") == "user" and not row.get("isMeta"):
            blocks = blocks_of(row)
            if any(b.get("type") == "text" for b in blocks) and not any(b.get("type") == "tool_result" for b in blocks):
                text = "\n".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
                if len(text.split()) >= min_words:
                    out.append(text)
                if len(out) >= count:
                    break
    return out


__all__ = ["TASK", "candidates", "items_from", "next_message", "next_messages", "source_index", "source_tags"]
