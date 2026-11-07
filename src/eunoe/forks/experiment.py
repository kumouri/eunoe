"""`eunoe ablate`: fork ablation on generated sessions. Train S8 on some seeds, compare signals on others.

Two stages, on separate seeds, so a signal is never scored on what trained it:

* `train`: what each level and kind costs. Ends in S8's table.
* `test`: every signal at the same budget, plus one step to a stub. The signal table.

The seeds a scored grid uses have to be different again, because choosing here spends these. A run on the scripted door
proves the machinery. A run on a real model states what it read and wrote. Nothing here is a scored result.
"""
from __future__ import annotations

import json
import os

from eunoe.cost import ListPrice
from eunoe.forks import analysis, conditions as C
from eunoe.forks.generated import TASK, build
from eunoe.forks.run import CliDoor, Sink, cell_tokens, rate_events, run_condition
from eunoe.forks.scripted import ScriptedDoor
from eunoe.interfaces import CallUsage
from eunoe.signals import AblationEstimate
from eunoe.tasks.generator import DOMAINS

SHARE = 0.5
TRAIN_SEEDS = (1001,)
TEST_SEEDS = (2001, 2002, 2003)
TEST_SIGNALS = ("S0", "S1", "S3", "S6", "S7", "S8", "S9")
TRAIN_KINDS = ("tool", "read", "assistant", "user", "system_note", "file")


class GroupedEstimate(AblationEstimate):
    """S8 with a table keyed by kind group and age band, as `analysis.train_s8` writes it."""

    def score(self, event, ledger, now):
        from eunoe.signals import age_band

        key = f"{analysis.group_kind(event.kind)}|{age_band(ledger.call - event.call)}"
        return float(self.table.get(key, self.default))


def census_of(cp) -> list:
    """Generated facts are the demand: each one is a later use of the event that holds it."""
    out: dict = {}
    for item in cp.items:
        if item.answerable and item.tags.get("label"):
            cell = (item.tags["label"], item.tags["band"])
            out[cell] = out.get(cell, 0) + 1
    return [dict(label=k[0], band=k[1], demand=v) for k, v in sorted(out.items())]


def train_conditions(cp) -> list:
    led = cp.ledger
    out = [C.ctrl(), C.ctrl("ctrl2")]
    out += [C.at_level(led, level) for level in (1, 2, 3)]
    for group in TRAIN_KINDS:
        cond = C.kind_at(led, group, 3, label=lambda e: analysis.group_kind(e.kind))
        if cond.levels:
            out.append(cond)
    return out


def test_conditions(cp, rater, table: dict) -> list:
    led = cp.ledger
    out = [C.ctrl()]
    for name in TEST_SIGNALS:
        if name == "S7":
            ratings = rater(cp) if rater else {}
            signal = _rated(ratings)
        elif name == "S8":
            signal = GroupedEstimate(table, default=0.0)
        else:
            signal = name
        out.append(C.by_signal(led, signal, f"signal:{name}", SHARE))
    out.append(C.by_signal(led, "S9", "stub:S9", SHARE, stub=True))
    return out


def _rated(ratings: dict):
    from eunoe.signals import ModelRating

    return ModelRating(ratings, default=4.5)


def run_stage(stage: str, args, door=None, rate_door=None) -> int:
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"forks-{stage}.jsonl")
    sink = Sink(path)
    cost = ListPrice(args.model)
    scripted = getattr(args, "door", "cli") == "scripted"
    door = door or (ScriptedDoor() if scripted else CliDoor(args.model))
    table = {}
    if stage == "test":
        with open(os.path.join(args.out, "s8.json"), encoding="utf-8") as fh:
            table = json.load(fh)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (TRAIN_SEEDS if stage == "train" else TEST_SEEDS)
    domains = args.domains.split(",") if args.domains else list(DOMAINS)
    only = set(args.only.split(",")) if args.only else None
    meta = open(os.path.join(args.out, f"cells-{stage}.jsonl"), "a", encoding="utf-8")
    flagged = 0
    try:
        for seed in seeds:
            for domain in domains:
                cp = build(domain, seed, marker=args.marker)
                meta.write(json.dumps(dict(checkpoint=cp.id, cells=cell_tokens(cp), census=census_of(cp))) + "\n")
                meta.flush()
                rater = None
                if stage == "test":
                    rate_door = rate_door or (ScriptedDoor() if scripted else CliDoor(args.rate_model))
                    rater = _rater(rate_door, sink, ListPrice(args.rate_model), args)
                conds = train_conditions(cp) if stage == "train" else test_conditions(cp, rater, table)
                for cond in conds:
                    if only and cond.name not in only:
                        continue
                    if (cp.id, cond.name) in sink.done:
                        continue
                    out = run_condition(door, cp, cond, cp.items, sink, cost, TASK, args.run_id)
                    print(cp.id, cond.name, "changed", cond.changed(), "errors", out["errors"], flush=True)
                    if out["errors"]:
                        flagged += 1
                        if flagged >= args.max_errors:
                            print(f"stopping: {flagged} conditions had an error, which is the limit set")
                            return 3
    finally:
        meta.close()
        door.close()
        if rate_door is not None:
            rate_door.close()
    print(json.dumps(analysis.usage_totals(analysis.read(path))))
    return 0


def _rater(door, sink, cost, args):
    def rate(cp):
        kept = os.path.join(args.out, f"ratings-{cp.id}.json")
        if os.path.exists(kept):
            with open(kept, encoding="utf-8") as fh:
                return json.load(fh)
        ratings, usages = rate_events(door, cp.ledger)
        with open(kept, "w", encoding="utf-8") as fh:
            json.dump(ratings, fh)
        total = CallUsage()
        for usage in usages:
            for name in ("input", "cache_read", "cache_write_5m", "cache_write_1h", "output"):
                setattr(total, name, getattr(total, name) + getattr(usage, name))
        sink.write(row="rating", checkpoint=cp.id, events=len(ratings), output=total.output, read=total.cache_read,
                   write=total.cache_write, input=total.input, list_usd=round(cost.price(total, "1h"), 5))
        return ratings

    return rate


def fit(args) -> int:
    """Write S8's table from a finished `train` stage, and the tables, to the same folder."""
    rows = analysis.read(os.path.join(args.out, "forks-train.jsonl"))
    cells, census = [], []
    with open(os.path.join(args.out, "cells-train.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cells += row["cells"]
            census += row["census"]
    table = analysis.train_s8(rows, cells, census)
    with open(os.path.join(args.out, "s8.json"), "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
    report(args, "train")
    return 0


def report(args, stage: str) -> dict:
    rows = analysis.read(os.path.join(args.out, f"forks-{stage}.jsonl"))
    out = dict(stage=stage, usage=analysis.usage_totals(rows), first=analysis.condition_table(rows),
               after_lookup=analysis.condition_table(rows, stage="final"))
    if stage == "train":
        out["importance"] = {axis: analysis.importance_table(rows, axis) for axis in ("group", "band", "size", "fact_kind")}
    else:
        names = sorted({r["condition"] for r in rows if r.get("row") == "fork"})
        out["lookups"] = [analysis.lookups_per_correct(rows, n) for n in names]
    with open(os.path.join(args.out, f"report-{stage}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    for row in out["first"]:
        print(row)
    return out


def main(args) -> int:
    if args.stage == "fit":
        return fit(args)
    if args.stage == "report":
        report(args, "test")
        return 0
    return run_stage(args.stage, args)


__all__ = ["GroupedEstimate", "main", "report", "run_stage", "test_conditions", "train_conditions"]
