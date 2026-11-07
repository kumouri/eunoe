"""Probes as one numbered batch: asked once per condition, scored deterministically against gold.

A batch makes a condition cheap, because a rewrite is the expensive part and each item only costs a line of output.
Items in a batch can leak into each other, so `clean_batch` refuses an item whose wording shows another item's gold.

On a real session a probe is a cloze. The later turns of that session used a detail that was only in context from
before the checkpoint, so the line they wrote it in, with the detail blanked, asks for exactly what they needed. No
model writes the question and no model judges the answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from eunoe.judge import normalize
from eunoe.levels import find_details

#: The classes where a later use reliably means a real reach back. A date is left out: most contexts can guess one.
STRONG_CLASSES = frozenset(("id", "slug", "path", "file"))
BLANK = "_____"
CONTEXT_CHARS = 280
LOOKUP_RESULT_CHARS = 1_500
_LINE_RE = re.compile(r"^\W*(\d+)\W+(ANSWER|LOOKUP)\s*:\s*(.*)$", re.IGNORECASE)
_ABSTAIN_RE = re.compile(r"(?i)\b(?:unknown|don'?t know|do not know|not sure|cannot tell|can'?t tell|no record|not available)\b")

BATCH_INTRO = """This is a check of what you can still tell from this session. No tools are available for this turn.

{task}

Reply with exactly one line per item, in order, and nothing else:
<number>. ANSWER: <the value>
Give your best answer from what you hold, even if you are not certain. {escape}

"""
ESCAPE_LOOKUP = ("Only if nothing you hold points to it: reply <number>. LOOKUP: <a few search words> to search a verbatim "
                 "archive of this session, or <number>. ANSWER: UNKNOWN if you would not know what to search for.")
ESCAPE_NONE = "Only if nothing you hold points to it: reply <number>. ANSWER: UNKNOWN."
AFTER_LOOKUP = """Search results from the archive follow. Now answer those items, one line each, in the same shape, with no \
further LOOKUP: <number>. ANSWER: <the value>, or ANSWER: UNKNOWN.

"""


@dataclass
class Item:
    """One probe. `sources` are the ids of events holding `gold`; `tags` is what a table groups it by."""

    id: str
    ask: str
    gold: str = ""
    stale: tuple = ()
    kind: str = "recall"
    sources: list = field(default_factory=list)
    tags: dict = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return bool(self.gold)


def gold_forms(gold: str) -> tuple:
    """A path may come back with less of its directory, so its last part also counts."""
    forms = [normalize(gold)]
    parts = [p for p in re.split(r"[\\/]+", gold) if p]
    if len(parts) > 1 and "." in parts[-1]:
        forms.append(normalize(parts[-1]))
    return tuple(forms)


def score_answer(said: str, item: Item) -> dict:
    """`said` is the text after `ANSWER:`. Exactly one of correct, abstained, stale or hallucinated is true."""
    text = normalize(said)
    abstained = not text or bool(_ABSTAIN_RE.search(text))
    if item.gold:
        correct = any(form and form in text for form in gold_forms(item.gold))
    else:
        correct = abstained
    if correct and item.gold:
        abstained = False
    stale = not correct and not abstained and any(normalize(v) in text for v in item.stale)
    return {"correct": correct, "abstained": abstained and not correct, "stale": stale,
            "hallucinated": not correct and not abstained and not stale}


CLOZE_INTRO = """Fill in the blanks. Each numbered line below comes from later in this same piece of work, with one value replaced \
by _____. {task}

Reply with one line per item, in order, and nothing else:
<number>. ANSWER: <the value>
Give your best answer from this session, even if you are not certain. If nothing here points to it, reply <number>. ANSWER: UNKNOWN.

"""


def batch_prompt(items: list, task: str, lookup: bool = True, intro: str | None = None) -> str:
    if intro is not None:
        lines = [intro.format(task=task)]
    else:
        lines = [BATCH_INTRO.format(task=task, escape=ESCAPE_LOOKUP if lookup else ESCAPE_NONE)]
    lines.extend(f"{i}. {item.ask}" for i, item in enumerate(items, 1))
    return "\n".join(lines)


def parse_batch(text: str) -> dict:
    """`{number: (verb, value)}`. A later line for the same number wins, because a model can correct itself."""
    out = {}
    for line in (text or "").splitlines():
        m = _LINE_RE.match(line.strip())
        if m:
            out[int(m.group(1))] = (m.group(2).upper(), m.group(3).strip())
    return out


def lookup_prompt(found: dict) -> str:
    """`found` maps an item's number to what the archive returned for its search."""
    parts = [AFTER_LOOKUP]
    for number, text in sorted(found.items()):
        parts.append(f"{number}. Results:\n{text or 'no match'}\n")
    return "\n".join(parts)


def excerpt(text: str, words: list, limit: int = LOOKUP_RESULT_CHARS) -> str:
    """The part of an archived event around the first search word, so a lookup can't dump a whole log."""
    if len(text) <= limit:
        return text
    low = text.lower()
    hits = [low.find(w.lower()) for w in words]
    hits = [h for h in hits if h >= 0]
    at = min(hits) if hits else 0
    start = max(0, at - limit // 2)
    return text[start:start + limit]


def clean_batch(items: list) -> list:
    """Drop an item whose wording holds another kept item's gold. Earlier items win."""
    kept: list = []
    for item in items:
        gold = normalize(item.gold) if item.gold else ""
        clashes = any(k.gold and normalize(k.gold) in normalize(item.ask) for k in kept)
        clashes = clashes or any(gold and gold in normalize(k.ask) for k in kept)
        if not clashes:
            kept.append(item)
    return kept


def cloze(line: str, detail: str, limit: int = CONTEXT_CHARS) -> str:
    """`line` with every copy of `detail` blanked, cut around the first blank. Empty if too little is left to ask with."""
    pattern = re.compile(re.escape(detail), re.IGNORECASE)
    m = pattern.search(line)
    if not m:
        return ""
    start = max(0, m.start() - limit // 2)
    window = pattern.sub(BLANK, line[start:start + limit + len(detail)])
    window = " ".join(window.split())
    if BLANK not in window or len(re.findall(r"[A-Za-z]{3,}", window)) < 4:
        return ""
    return window


def detail_key(detail: str) -> str:
    """One detail however it is written: a path by its last part, everything else lowercase."""
    if "/" in detail or "\\" in detail:
        return detail.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
    return detail.lower()


def strong_details(text: str) -> list:
    """Details a later use really means a reach back: long enough, and a path only if it ends in a file."""
    out = []
    for detail, cls, pos in find_details(text):
        if cls not in STRONG_CLASSES or len(detail) < 6:
            continue
        if cls == "path" and "." not in re.split(r"[\\/]+", detail)[-1]:
            continue
        out.append((detail, cls, pos))
    return out


__all__ = ["Item", "STRONG_CLASSES", "BLANK", "batch_prompt", "clean_batch", "cloze", "detail_key", "excerpt", "gold_forms",
           "lookup_prompt", "parse_batch", "score_answer", "strong_details"]
