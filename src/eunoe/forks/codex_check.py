"""Why does Codex's answer sometimes not hold after an edit under reasoning items? A paired test of what causes it.

The conformance suite found the edit accepted and the answer held 7 of 11 times, against 6 of 6 unedited, with no error.
This builds one session the same way, then asks for the same codeword from four copies of its rollout:

* `kept`: nothing changed.
* `edited`: the padding removed from the first user turn, reasoning items left in. The case the suite saw.
* `edited_stripped`: the same edit, with reasoning items removed.
* `stripped`: no edit, with reasoning items removed.
* `swapped`: the padding replaced with different padding of the same length, reasoning left in. An edit that changes no meaning.
* `never_padded`: a session of its own that never had the padding. No edit at all, the same words as `edited`.

If `edited` fails and `edited_stripped` doesn't, the items are what the edit breaks, and dropping them on an edit is the
fix. If `stripped` fails too, removing them does the harm. If `never_padded` fails like `edited`, the edit isn't the cause:
the words left behind are. If nothing separates, the suite's gap was noise. Synthetic, so the answers are kept to read.
"""
from __future__ import annotations

import json

from eunoe.editors.conformance import SECOND_CODEWORD, make_padding

ARMS = ("kept", "edited", "edited_stripped", "stripped", "swapped", "never_padded")
FIRST = ("Remember this codeword: {codeword}. The rest of this message is padding, ignore it: {padding} . Without using any "
         "tools, work out carefully, step by step, how many of the integers from 1 to 200 are divisible by 3 or 7 but not by "
         "both, and reply with only that number.")
SECOND = "Without using any tools, work out carefully how many of those are prime, and reply with only that number."
ASK = "What is the codeword? Reply with only the codeword."


def without_reasoning(lines: list) -> list:
    kept = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            kept.append(line)
            continue
        if row.get("type") == "response_item" and (row.get("payload") or {}).get("type") == "reasoning":
            continue
        kept.append(line)
    return kept


def classify(text: str, error: str, codeword: str = SECOND_CODEWORD) -> str:
    if error:
        return "error"
    if codeword in (text or ""):
        return "held"
    low = (text or "").lower()
    low = low.replace("\u2019", "'")
    if any(word in low for word in ("sorry", "can't", "cannot", "unable", "don't", "not able", "no codeword", "not sure")):
        return "declined"
    return "wrong"


def trial(harness, forks: int = 2) -> dict:
    padding = make_padding()
    out = dict(error="", arms={})
    first = harness.start(FIRST.format(codeword=SECOND_CODEWORD, padding=padding), thinking=True)
    if first.error:
        return {**out, "error": first.error[:200]}
    second = harness.resume(SECOND)
    if second.error:
        return {**out, "error": second.error[:200]}
    out["reasoning_items"] = harness.thinking_block_count()
    base, totals = harness.thread, harness.totals
    try:
        for arm in ARMS:
            if arm == "never_padded":
                continue
            rows = []
            for _ in range(forks):
                harness.thread = base
                harness.thread = harness._copy_rollout(without_reasoning if "stripped" in arm else None)
                harness.totals = (0, 0, 0)
                copies = 0
                if arm.startswith("edited"):
                    copies = harness.replace(padding, "")
                elif arm == "swapped":
                    copies = harness.replace(padding, " ".join(reversed(padding.split())))
                turn = harness.resume(ASK)
                usage = (turn.usage.raw or {}) if turn.usage else {}
                rows.append(dict(result=classify(turn.text, turn.error), said=(turn.text or turn.error or "")[:200],
                                 input=int(usage.get("input_tokens") or 0), cached=int(usage.get("cached_input_tokens") or 0),
                                 reasoning=int(usage.get("reasoning_output_tokens") or 0), copies=copies))
            out["arms"][arm] = rows
        rows = []
        harness.totals = (0, 0, 0)
        fresh = harness.start(FIRST.format(codeword=SECOND_CODEWORD, padding=""), thinking=True)
        if not fresh.error and not harness.resume(SECOND).error:
            own = harness.thread
            for _ in range(forks):
                harness.thread = own
                harness.thread = harness._copy_rollout()
                harness.totals = (0, 0, 0)
                turn = harness.resume(ASK)
                usage = (turn.usage.raw or {}) if turn.usage else {}
                rows.append(dict(result=classify(turn.text, turn.error), said=(turn.text or turn.error or "")[:200],
                                 input=int(usage.get("input_tokens") or 0), cached=int(usage.get("cached_input_tokens") or 0),
                                 reasoning=int(usage.get("reasoning_output_tokens") or 0), copies=0))
        out["arms"]["never_padded"] = rows
    finally:
        harness.thread, harness.totals = base, totals
    return out


def summarize(trials: list) -> dict:
    """Per arm: how each fork came out, and the mean input a fork reported."""
    out = {"trials": len(trials), "errors": sum(1 for t in trials if t.get("error"))}
    for arm in ARMS:
        rows = [r for t in trials for r in t.get("arms", {}).get(arm, [])]
        tally: dict = {}
        for row in rows:
            tally[row["result"]] = tally.get(row["result"], 0) + 1
        out[arm] = dict(forks=len(rows), **tally, mean_input=round(sum(r["input"] for r in rows) / len(rows)) if rows else 0)
    return out


__all__ = ["ARMS", "classify", "summarize", "trial", "without_reasoning"]
