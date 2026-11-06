"""`eunoe replay`: the cost surface for your own session files. Prints counts and ratios, no text."""
from __future__ import annotations

import json

from eunoe.replay.surface import rates, real_cost, replay_cost, schedule_grid
from eunoe.replay.usage import read_usage


def main(args) -> int:
    price = rates(args.model)
    grid = schedule_grid(args.budget, args.post_compact_tokens)
    out = []
    for index, path in enumerate(args.transcripts):
        usage = read_usage(path)
        calls = usage["calls"]
        if not calls:
            out.append({"session": index, "calls": 0})
            continue
        real = real_cost(calls, price)
        row = {"session": index, "calls": len(calls), "rows": usage["rows"], "compactions": usage["boundaries"],
               "peak_context": max(c.ctx for c in calls), "real_cost_usd": round(real, 4), "ratio_to_real": {}}
        for name, schedule in grid.items():
            result = replay_cost(calls, schedule, price, static=args.static_tokens)
            row["ratio_to_real"][name] = round(result["cost"] / real, 3) if real else None
        out.append(row)
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"Yardstick {args.model}, budget {args.budget}. Behaviour held fixed: this bounds cost and says nothing about quality.")
    for row in out:
        if not row["calls"]:
            print(f"session {row['session']}: no usable calls")
            continue
        print(f"session {row['session']}: {row['calls']} calls from {row['rows']} rows, peak {row['peak_context']}, "
              f"{row['compactions']} compactions, {row['real_cost_usd']} USD at list")
        for name, ratio in row["ratio_to_real"].items():
            print(f"  {ratio:>7}  {name}")
    return 0

