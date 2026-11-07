"""What does a session lose when thinking goes? A puzzle whose answer takes work, then a follow-up. Gold by construction.

A transcript stores thinking as an empty string and a signature, so no replay can see what it carried. Asking a model to
repeat its reasoning isn't the way to find out: that is what the provider's safeguards exist to refuse, and on the first
version of this check they did. So this measures an ordinary follow-up instead. The model solves a small knapsack and
writes only the total. A later turn asks which items made that total. With its reasoning it can read them off; without
it, it has to solve the puzzle again, or get it wrong. The arms:

* `native`: the session as the CLI left it.
* `rewritten`: the same rows written back through `transcript.render_rows`, to show the rewrite itself loses nothing.
* `stripped`: every thinking block removed, which is what `strip_on_edit` does.
* `edited`: thinking kept and an earlier user turn changed, so every block fails the prefix check.

The measures are whether the follow-up is right, what it cost in output and thinking tokens, and how much context each
arm holds, which shows whether thinking really reached the model. Any safeguards flag stops the check and is reported.
"""
from __future__ import annotations

import itertools
import json
import re

from eunoe.forks.transcript import load_checkpoint, read_rows, render_rows, write_session
from eunoe.rng import Rng

ARMS = ("native", "rewritten", "stripped", "edited")
PACE = "Reply with only the word OK."
FILLER = "Notes for this session: alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november."
_FLAG = "safeguards"


def puzzle(seed: int, items: int = 14) -> dict:
    """A knapsack with one best answer. Regenerated until the best subset is unique."""
    rng = Rng("eunoe-thinking-check", seed)
    while True:
        rows = [(f"K{i:02d}", rng.randint(3, 40), rng.randint(5, 99)) for i in range(items)]
        limit = sum(w for _, w, _ in rows) // 2
        best, ties = [], 0
        top = -1
        for size in range(items + 1):
            for combo in itertools.combinations(rows, size):
                if sum(w for _, w, _ in combo) <= limit:
                    value = sum(v for _, _, v in combo)
                    if value > top:
                        top, best, ties = value, combo, 0
                    elif value == top:
                        ties += 1
        if not ties:
            table = "\n".join(f"{name}: weight {w}, value {v}" for name, w, v in rows)
            return dict(table=table, limit=limit, total=top, ids=sorted(name for name, _, _ in best))


def first_prompt(p: dict) -> str:
    return (f"{FILLER}\n\nHere are some items:\n{p['table']}\n\nChoose the subset with the highest total value whose total "
            f"weight is at most {p['limit']}. Work it out carefully, without writing your working. Reply with only the total value.")


FOLLOW_UP = ("Which items made up that total? Reply on one line as ANSWER: followed by their names, separated by spaces, with "
             "no other text.")


def score(text: str, p: dict) -> bool:
    m = re.search(r"ANSWER\s*:(.*)", text or "", re.IGNORECASE)
    return bool(m) and sorted(set(re.findall(r"K\d\d", m.group(1)))) == p["ids"]


def _thinking_tokens(turn) -> int:
    details = (turn.usage.raw or {}).get("output_tokens_details") or {}
    return int(details.get("thinking_tokens") or 0)


def _rewrite(door, rows: list, strip: bool, edit: bool) -> None:
    if edit:
        rows = json.loads(json.dumps(rows).replace("foxtrot", "foxglove"))
    cp = load_checkpoint(rows, len(rows), "thinking-check")
    written = render_rows(cp, {}, strip, "pending", door.scratch())
    session_id, _ = write_session(written, door._folder())
    door.h.sessions.append(session_id)
    door.h.current = session_id


def trial(door, seed: int, forks: int = 2) -> dict:
    """One puzzle, its follow-up asked in `forks` forks under each arm. The session files go when it ends."""
    harness = door.h
    p = puzzle(seed)
    out = dict(seed=seed, error="", flagged=False, arms={})
    try:
        first = harness.start(first_prompt(p), thinking=True)
        if first.error:
            return {**out, "error": first.error[:200], "flagged": _FLAG in first.error}
        out.update(first_right=str(p["total"]) in (first.text or ""), thinking_tokens=_thinking_tokens(first),
                   thinking_blocks=harness.thinking_block_count(), first_output=first.usage.output)
        base = harness.current
        rows = read_rows(harness.session_path())
        for arm in ARMS:
            if arm == "native":
                harness.current = base
            else:
                _rewrite(door, rows, strip=arm == "stripped", edit=arm == "edited")
            paced = harness.fork(PACE) if arm == "native" else harness.resume(PACE)
            if arm == "native" and paced.session_id:
                harness.current = paced.session_id
            results = []
            for _ in range(forks):
                turn = harness.fork(FOLLOW_UP)
                if turn.error:
                    results.append(dict(error=True))
                    if _FLAG in turn.error:
                        out["flagged"] = True
                        out["arms"][arm] = dict(context=paced.usage.context, results=results)
                        return out
                    continue
                results.append(dict(right=score(turn.text, p), output=turn.usage.output, thinking=_thinking_tokens(turn),
                                    context=turn.usage.context))
            out["arms"][arm] = dict(context=paced.usage.context, results=results, error=bool(paced.error))
    finally:
        door.end_condition()
    return out


def summarize(trials: list, only_first_right: bool = True) -> dict:
    """Per arm, over trials whose first answer was right: follow-ups right, mean output and thinking tokens, mean context."""
    out = {"trials": len(trials), "flagged": sum(1 for t in trials if t.get("flagged")),
           "first_right": sum(1 for t in trials if t.get("first_right")),
           "with_thinking": sum(1 for t in trials if t.get("thinking_blocks"))}
    for arm in ARMS:
        rows = [t["arms"][arm] for t in trials if arm in t.get("arms", {}) and (t.get("first_right") or not only_first_right)]
        done = [r for row in rows for r in row["results"] if "right" in r]
        contexts = [row["context"] for row in rows if row.get("context")]
        out[arm] = dict(forks=len(done), right=sum(1 for r in done if r["right"]),
                        mean_output=round(sum(r["output"] for r in done) / len(done), 1) if done else 0,
                        mean_thinking=round(sum(r["thinking"] for r in done) / len(done), 1) if done else 0,
                        mean_context=round(sum(contexts) / len(contexts)) if contexts else 0)
    return out


__all__ = ["ARMS", "puzzle", "score", "summarize", "trial"]
