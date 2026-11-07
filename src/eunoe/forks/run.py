"""Drive conditions through a door and write one row per call, per fork and per item.

The flow for one condition, which is what keeps it to one rewrite:

1. Write the edited context as a session file.
2. **Pace:** resume it in place with a line that asks nothing. This pays the rewrite once and leaves a cache entry
   that later forks can read, and its usage says how much context this condition really holds.
3. **Batch:** fork, and ask every item at once.
4. **Lookup:** if the batch asked to search, continue that fork with what the archive returned, and score again.
5. **Replay:** optionally fork again and send the session's real next message, for a continuation measure.

Rows hold counts, tags and hashes, never text, unless a caller asks for `keep_text` on a run it owns. A row is
appended as it happens, so a run that stops keeps what it measured, and running again skips conditions already done.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass

from eunoe.editors.base import Turn
from eunoe.editors.claude_cli import ClaudeCliHarness, _project_slug
from eunoe.forks.conditions import Condition
from eunoe.forks.probes import batch_prompt, detail_key, excerpt, gold_forms, lookup_prompt, parse_batch, score_answer, strong_details
from eunoe.forks.transcript import Checkpoint, held_text, render_rows, write_session
from eunoe.interfaces import CallUsage, RenderedContext
from eunoe.ledger import THINKING
from eunoe.levels import Level, estimate_tokens, find_details
from eunoe.renderer import render_context
from eunoe.wire import to_text

PACE = "Continue. Reply with only the word OK."
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*")


def _hash(detail: str) -> str:
    return hashlib.sha256(detail.lower().encode("utf-8")).hexdigest()[:10]


@dataclass
class Replay:
    """The session's real next message, and the details its real next turn went on to use, hashed."""

    prompt: str
    real_details: tuple = ()


class CliDoor:
    """The Claude CLI on one login: a written session, forked. Its session files go when each condition ends."""

    def __init__(self, model: str, timeout: float = 900.0, environ=None, config_dir: str | None = None, runner=None,
                 extra_args=()) -> None:
        kwargs = {"runner": runner} if runner else {}
        self.h = ClaudeCliHarness(model, environ=environ, config_dir=config_dir, **kwargs)
        self.h.timeout = timeout
        self.h.extra_args = list(extra_args)
        self.model = model

    def available(self) -> tuple:
        return self.h.available()

    def _folder(self) -> str:
        return os.path.join(self.h.config_dir, "projects", _project_slug(self.h._dir()))

    def open_rows(self, rows: list) -> None:
        session_id, _ = write_session(rows, self._folder())
        self.h.sessions.append(session_id)
        self.h.current = session_id

    def open_context(self, ctx: RenderedContext) -> None:
        self.h.apply(None, RenderedContext(ctx.system, ctx.blocks, [], {}))

    def scratch(self) -> str:
        return self.h._dir()

    def pace(self, prompt: str) -> Turn:
        return self.h.resume(prompt)

    def fork(self, prompt: str) -> Turn:
        return self.h.fork(prompt)

    def follow(self, session_id: str, prompt: str) -> Turn:
        return self.h._run(prompt, "--resume", session_id)

    def one_shot(self, prompt: str) -> Turn:
        return self.h.start(prompt)

    def end_condition(self) -> None:
        for session_id in self.h.sessions:
            path = self.h.session_path(session_id)
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
                shutil.rmtree(path[:-len(".jsonl")], ignore_errors=True)
        self.h.sessions.clear()
        self.h.current = ""

    def close(self) -> None:
        self.h.cleanup()


def archive_search(ledger, query: str, limit: int = 2) -> str:
    """Every word if anything has them all, otherwise whatever has most of them. Verbatim excerpts, newest first."""
    words = [w.lower() for w in _WORD_RE.findall(query or "")]
    if not words:
        return ""
    scored = []
    for event in reversed(ledger.events):
        if event.kind == THINKING:
            continue
        body = f"{event.text} {json.dumps(event.input or {}, ensure_ascii=False)}".lower()
        hits = sum(1 for w in words if w in body)
        if hits * 2 >= len(words) and hits:
            scored.append((hits, event))
    scored.sort(key=lambda pair: -pair[0])
    return "\n---\n".join(f"{e.id} {e.kind}: {excerpt(e.text, words)}" for _, e in scored[:limit])


def write_condition(door, checkpoint, condition: Condition) -> str:
    """Open the condition as a session on this door. Returns what it holds, lowercase, to check an item against."""
    if isinstance(checkpoint, Checkpoint):
        rows = render_rows(checkpoint, condition.levels, condition.strip_thinking, "pending", door.scratch())
        door.open_rows(rows)
        return held_text(rows)
    ledger = checkpoint.ledger
    saved = dict(ledger.levels)
    try:
        ledger.levels.update(condition.levels)
        ctx = render_context(ledger, checkpoint.system)
        ctx.blocks = [b for b in ctx.blocks if b.type != "thinking"]
    finally:
        ledger.levels.clear()
        ledger.levels.update(saved)
    door.open_context(ctx)
    return to_text(ctx).lower()


class Sink:
    """Appends rows to a JSONL file and knows which conditions already finished."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("row") == "done":
                        self.done.add((row["checkpoint"], row["condition"]))
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, **row) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source_level(checkpoint, condition: Condition, item) -> int:
    levels = [int(condition.levels.get(s, Level.L0)) for s in item.sources]
    return min(levels) if levels else 0


def run_condition(door, checkpoint, condition: Condition, items: list, sink: Sink, cost, task: str, run_id: str,
                  replay: Replay | None = None, keep_text: bool = False, lookup: bool = True, intro: str | None = None) -> dict:
    """One condition, start to finish. Returns its totals. Raises nothing for a model's error: it records it."""
    key = dict(run=run_id, checkpoint=checkpoint.id, condition=condition.name)
    totals = CallUsage()
    errors = []

    def note(stage: str, turn: Turn, started: float) -> Turn:
        usage = turn.usage or CallUsage()
        for name in ("input", "cache_read", "cache_write_5m", "cache_write_1h", "output"):
            setattr(totals, name, getattr(totals, name) + getattr(usage, name))
        if turn.error:
            errors.append("flagged" if "safeguards" in turn.error else stage)
        sink.write(row="call", stage=stage, input=usage.input, read=usage.cache_read, write_5m=usage.cache_write_5m,
                   write_1h=usage.cache_write_1h, output=usage.output, context=usage.context,
                   seconds=round(time.monotonic() - started, 1), list_usd=round(cost.price(usage, "1h"), 5),
                   error=(turn.error[:160] if turn.error and keep_text else bool(turn.error)), **key)
        return turn

    try:
        held = write_condition(door, checkpoint, condition)
        started = time.monotonic()
        paced = note("pace", door.pace(PACE), started)
        sink.write(row="fork", held_tokens=(paced.usage or CallUsage()).context, changed=condition.changed(),
                   strip_thinking=condition.strip_thinking, levels=_level_counts(condition), **key)
        if paced.error:
            return {"usage": totals, "errors": errors}
        if items:
            started = time.monotonic()
            asked = note("batch", door.fork(batch_prompt(items, task, lookup, intro)), started)
            answers = parse_batch(asked.text) if not asked.error else {}
            first = {}
            wanted = {}
            for number, item in enumerate(items, 1):
                verb, value = answers.get(number, ("ANSWER", ""))
                first[number] = (verb, value, score_answer("" if verb == "LOOKUP" else value, item))
                if verb == "LOOKUP":
                    wanted[number] = value
            final = {}
            if wanted and asked.session_id:
                found = {n: archive_search(checkpoint.ledger, q) for n, q in wanted.items()}
                started = time.monotonic()
                after = note("lookup", door.follow(asked.session_id, lookup_prompt(found)), started)
                for number, (verb, value) in (parse_batch(after.text) if not after.error else {}).items():
                    if number in wanted and verb == "ANSWER":
                        final[number] = score_answer(value, items[number - 1])
            for number, item in enumerate(items, 1):
                verb, value, scored = first[number]
                done = final.get(number, scored)
                sink.write(row="item", item=item.id, kind=item.kind, answerable=item.answerable,
                           held=any(f and f in held for f in gold_forms(item.gold)) if item.gold else False,
                           source_level=_source_level(checkpoint, condition, item), tags=item.tags, answered=number in answers,
                           looked_up=verb == "LOOKUP", first=scored, final=done, **key)
                if keep_text:
                    sink.write(row="said", item=item.id, verb=verb, value=value[:200], **key)
        replays = [] if replay is None else (replay if isinstance(replay, list) else [replay])
        for n, one in enumerate(replays):
            started = time.monotonic()
            said = note("replay", door.fork(one.prompt), started)
            details = sorted({_hash(detail_key(d)) for d, _, _ in strong_details(said.text or "")})
            every = sorted({_hash(d) for d, _, _ in find_details(said.text or "")})
            sink.write(row="replay", n=n, details=details, every=every, real=list(one.real_details), words=len((said.text or "").split()),
                       output=(said.usage or CallUsage()).output, error=bool(said.error), **key)
    finally:
        door.end_condition()
    sink.write(row="done", errors=errors, list_usd=round(cost.price(totals, "1h"), 5), read=totals.cache_read,
               write=totals.cache_write_5m + totals.cache_write_1h, output=totals.output, input=totals.input, **key)
    return {"usage": totals, "errors": errors}


def _level_counts(condition: Condition) -> dict:
    counts: dict = {}
    for level in condition.levels.values():
        counts[f"L{int(level)}"] = counts.get(f"L{int(level)}", 0) + 1
    return counts


def rate_events(door, ledger, label=None, chunk: int = 120, chars: int = 160, keep_recent: int = 10) -> tuple:
    """S7: a small model rates each event once, from its first characters, in one batch per `chunk` events.

    Returns `(ratings, usage_rows)`. A real write-time rating would see each event alone; this is its cheap stand-in,
    and it never sees anything later than the checkpoint, so it can't read the future.
    """
    events = [e for e in ledger.events if e.kind != THINKING][:-keep_recent or None]
    ratings, rows = {}, []
    for start in range(0, len(events), chunk):
        part = events[start:start + chunk]
        lines = [f"{e.id} {(label(e) if label else e.kind)}: {' '.join((e.text or json.dumps(e.input or {}))[:chars].split())}"
                 for e in part]
        prompt = ("Below are events from one long working session, oldest first, each cut short. For each, rate from 0 to 9 "
                  "how likely it is that a later turn would need its exact details. Reply with one line per event and "
                  "nothing else: <id> <score>.\n\n" + "\n".join(lines))
        turn = door.one_shot(prompt)
        rows.append(turn.usage or CallUsage())
        for line in (turn.text or "").splitlines():
            parts = line.replace(":", " ").split()
            if len(parts) >= 2 and parts[0] in ledger.by_id and parts[1][:1].isdigit():
                ratings[parts[0]] = float(parts[1][:1])
        door.end_condition()
    return ratings, rows


def cell_tokens(checkpoint) -> list:
    """Tokens by label and age band, estimated: a demand density needs a denominator."""
    from eunoe.signals import age_band

    ledger = checkpoint.ledger
    label = checkpoint.label if hasattr(checkpoint, "label") else (lambda e: e.kind)
    cells: dict = {}
    for event in ledger.events:
        if event.kind == THINKING or event.index <= getattr(checkpoint, "frozen_upto", -1):
            continue
        cell = (label(event), age_band(ledger.call - event.call))
        cells[cell] = cells.get(cell, 0) + estimate_tokens(event.text) + 1
    return [dict(label=k[0], band=k[1], tokens=v) for k, v in sorted(cells.items())]


__all__ = ["CliDoor", "PACE", "Replay", "Sink", "archive_search", "cell_tokens", "rate_events", "run_condition",
           "write_condition"]

