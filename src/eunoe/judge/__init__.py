"""Deterministic scoring against gold. No model judges anything here.

A probe's answer is read off its `ANSWER:` line. Correct means the gold value is in it. A stale answer
gave a superseded value. An abstention admitted it could not tell. A hallucination gave a specific
wrong value, which `evicted` then says was about a fact no longer held.
"""
from __future__ import annotations

import re

from eunoe.interfaces import BlindItem, Grade, Probe, ProbeScore

_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.*)", re.IGNORECASE)
_ABSTAIN_RE = re.compile(r"(?i)\b(?:unknown|don'?t know|do not know|not sure|cannot tell|can'?t tell|no record|not available)\b")


def normalize(value: str) -> str:
    return re.sub(r"[\s`'\"*]+", " ", str(value)).strip(" .,;:").casefold()


def parse_answer(text: str) -> str:
    matches = _ANSWER_RE.findall(text or "")
    return matches[-1].strip() if matches else (text or "").strip()


def score_probe(text: str, probe: Probe, looked_up: bool = False, evicted: bool = False) -> ProbeScore:
    answer = parse_answer(text)
    said = normalize(answer)
    abstained = not said or bool(_ABSTAIN_RE.search(said))
    if probe.gold:
        correct = normalize(probe.gold) in said
        stale = not correct and any(normalize(v) in said for v in probe.stale)
    else:
        correct, stale = abstained, False
    hallucinated = not correct and not abstained and not stale
    return ProbeScore(probe.id, probe.kind, correct, abstained, hallucinated, stale, looked_up, answer, evicted)


class GoldJudge:
    """Implements `Judge`. A blind item with gold, graded with no idea which arm produced it."""

    def grade(self, item: BlindItem) -> Grade:
        said = normalize(parse_answer(item.answer))
        ok = bool(item.gold) and normalize(item.gold) in said
        return Grade(item.id, 1.0 if ok else 0.0, "correct" if ok else "wrong")
