"""Rows to tables: what each condition lost against its control, an importance table, and S8's numbers.

Every comparison is paired within a checkpoint and resampled by checkpoint, because items in one checkpoint aren't
independent. Only items the control got right can show a loss, so a loss is a share of those. Input and output hold no text.
"""
from __future__ import annotations

import json

from eunoe.analysis import paired_bootstrap
from eunoe.rng import Rng


def read(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def group_kind(kind: str) -> str:
    """Kinds folded for a table that has to generalise. A tool's name is too specific; what it does is not."""
    if kind.startswith("attachment:"):
        return "file"
    if not kind.startswith("tool:"):
        return kind
    name = kind[5:]
    if name in ("Bash", "PowerShell"):
        return "shell"
    if name in ("Read",) or name.startswith(("read_", "fetch_paper")):
        return "read"
    if name in ("Write", "Edit", "NotebookEdit", "apply"):
        return "write"
    return "tool"


def _items(rows: list) -> dict:
    """`{(checkpoint, condition): {item: row}}`, answerable items a model really answered in that condition."""
    out: dict = {}
    for row in rows:
        if row.get("row") == "item" and row.get("answerable"):
            out.setdefault((row["checkpoint"], row["condition"]), {})[row["item"]] = row
    return out


def condition_table(rows: list, control: str = "ctrl", stage: str = "first") -> list:
    """Per condition: recall, hallucination, abstention, how much it holds, and the paired recall difference from control."""
    items = _items(rows)
    forks = {(r["checkpoint"], r["condition"]): r for r in rows if r.get("row") == "fork"}
    done = {(r["checkpoint"], r["condition"]): r for r in rows if r.get("row") == "done"}
    names = sorted({c for _, c in items})
    table = []
    for name in names:
        checkpoints = sorted(cp for cp, c in items if c == name and items[(cp, name)])
        treat, base, loss = [], [], []
        pool = []
        for cp in checkpoints:
            here = items[(cp, name)]
            answered = [r for r in here.values() if r["answered"]]
            if not answered:
                continue
            pool.extend(answered)
            ctrl_rows = items.get((cp, control), {})
            common = [i for i, r in here.items() if r["answered"] and i in ctrl_rows and ctrl_rows[i]["answered"]]
            if common:
                treat.append(sum(here[i][stage]["correct"] for i in common) / len(common))
                base.append(sum(ctrl_rows[i][stage]["correct"] for i in common) / len(common))
                right = [i for i in common if ctrl_rows[i][stage]["correct"]]
                if right:
                    loss.append(sum(not here[i][stage]["correct"] for i in right) / len(right))
        ci = paired_bootstrap(treat, base) if len(treat) >= 2 else {"mean": (sum(treat) - sum(base)) if treat else 0.0, "low": None, "high": None, "n": len(treat)}
        n = len(pool) or 1
        contexts = [forks[(cp, name)]["held_tokens"] for cp in checkpoints if (cp, name) in forks]
        cost = [done[(cp, name)]["list_usd"] for cp in checkpoints if (cp, name) in done]
        table.append(dict(
            condition=name, checkpoints=len(treat), items=len(pool),
            recall=round(sum(r[stage]["correct"] for r in pool) / n, 3),
            recall_diff=round(ci["mean"], 3), low=_round(ci["low"]), high=_round(ci["high"]),
            lost_of_control_right=round(sum(loss) / len(loss), 3) if loss else None,
            hallucinated=round(sum(r[stage]["hallucinated"] for r in pool) / n, 3),
            abstained=round(sum(r[stage]["abstained"] for r in pool) / n, 3),
            gold_held=round(sum(r["held"] for r in pool) / n, 3),
            recall_when_held=_share([r for r in pool if r["held"]], stage),
            recall_when_gone=_share([r for r in pool if not r["held"]], stage),
            lookups=sum(r["looked_up"] for r in pool),
            mean_held_tokens=round(sum(contexts) / len(contexts)) if contexts else 0,
            mean_list_usd=round(sum(cost) / len(cost), 3) if cost else 0))
    return table


def _round(value):
    return None if value is None else round(value, 3)


def _share(rows: list, stage: str):
    return round(sum(r[stage]["correct"] for r in rows) / len(rows), 3) if rows else None


def lookups_per_correct(rows: list, condition: str) -> dict:
    """H7's measure: lookups spent per correct recall, after lookups."""
    pool = [r for r in rows if r.get("row") == "item" and r.get("answerable") and r["condition"] == condition and r["answered"]]
    right = sum(r["final"]["correct"] for r in pool)
    looked = sum(r["looked_up"] for r in pool)
    return dict(condition=condition, items=len(pool), correct=right, lookups=looked,
                lookups_per_correct=round(looked / right, 3) if right else None,
                lookups_that_helped=sum(1 for r in pool if r["looked_up"] and r["final"]["correct"]))


def importance_table(rows: list, axis: str, conditions: tuple = ("all:L1", "all:L2", "all:L3"), control: str = "ctrl",
                     stage: str = "first", resamples: int = 2000) -> list:
    """Of items the control got right, the share each level lost, by `axis` of the item's newest source, with a CI."""
    items = _items(rows)
    checkpoints = sorted({cp for cp, c in items if c == control})
    table = []
    keys = sorted({_axis(r, axis) for (cp, c), here in items.items() if c == control for r in here.values()}, key=str)
    for key in keys:
        row = dict(axis=axis, band=key)
        for name in conditions:
            per = []
            for cp in checkpoints:
                base = items.get((cp, control), {})
                here = items.get((cp, name), {})
                right = [i for i, r in base.items() if _axis(r, axis) == key and r[stage]["correct"] and i in here and here[i]["answered"]]
                if right:
                    per.append((sum(not here[i][stage]["correct"] for i in right), len(right)))
            lost, total = sum(p[0] for p in per), sum(p[1] for p in per)
            row[f"{name}_n"] = total
            row[f"{name}_lost"] = round(lost / total, 3) if total else None
            low, high = _cluster_ci(per, resamples)
            row[f"{name}_low"], row[f"{name}_high"] = low, high
        table.append(row)
    return table


def _axis(row: dict, axis: str):
    tags = row.get("tags") or {}
    if axis == "group":
        return group_kind(str(tags.get("label")))
    return tags.get(axis)


def _cluster_ci(per: list, resamples: int) -> tuple:
    if len(per) < 2:
        return None, None
    rng = Rng("eunoe-cluster", resamples, len(per))
    values = []
    for _ in range(resamples):
        draw = [per[rng.randint(0, len(per) - 1)] for _ in per]
        total = sum(d[1] for d in draw)
        values.append(sum(d[0] for d in draw) / total if total else 0.0)
    values.sort()
    return round(values[int(0.025 * resamples)], 3), round(values[min(resamples - 1, int(0.975 * resamples))], 3)


def train_s8(rows: list, cells: list, census: list, condition: str = "all:L3", control: str = "ctrl",
             stage: str = "first", prior: float = 1.0) -> dict:
    """S8's table: expected answers lost per token freed, by kind group and age band. Lower scores go first.

    The chance a demand falls in a cell, per token it holds, comes from the census of later uses. The chance a demand is
    lost when that cell goes down a level comes from the forks. `prior` keeps a small cell from reading as exactly zero.
    """
    items = _items(rows)
    tally: dict = {}
    for (cp, name), base in items.items():
        if name != control:
            continue
        here = items.get((cp, condition), {})
        for i, row in base.items():
            if not row[stage]["correct"] or i not in here or not here[i]["answered"]:
                continue
            cell = (group_kind(str(row["tags"].get("label"))), row["tags"].get("band"))
            lost, total = tally.get(cell, (0, 0))
            tally[cell] = (lost + (not here[i][stage]["correct"]), total + 1)
    tokens: dict = {}
    for row in cells:
        cell = (group_kind(row["label"]), row["band"])
        tokens[cell] = tokens.get(cell, 0) + row["tokens"]
    demand: dict = {}
    for row in census:
        cell = (group_kind(row["label"]), row["band"])
        demand[cell] = demand.get(cell, 0) + row["demand"]
    all_lost = sum(v[0] for v in tally.values())
    all_total = sum(v[1] for v in tally.values())
    pooled = (all_lost + prior) / (all_total + 2 * prior) if all_total else 0.5
    table = {}
    for cell, size in tokens.items():
        if not size:
            continue
        lost, total = tally.get(cell, (0, 0))
        flip = (lost + prior * pooled) / (total + prior)
        table[f"{cell[0]}|{cell[1]}"] = round(flip * (demand.get(cell, 0) + 0.5) / size * 1000, 5)
    return table


def replay_overlap(rows: list, control: str = "ctrl", field: str = "details") -> list:
    """How much of what the control's reply named each condition's reply also named, per condition.

    Replies are paired by checkpoint and by which next message they answered. `field` is `details` for the strong classes,
    or `every` for every class, which is noisier and sees more of ordinary prose.
    """
    replies = {(r["checkpoint"], r["condition"], r.get("n", 0)): r for r in rows if r.get("row") == "replay" and not r.get("error")}
    names = sorted({c for _, c, _ in replies})
    out = []
    for name in names:
        per_checkpoint: dict = {}
        words, output = [], []
        for (cp, c, n), row in replies.items():
            if c != name:
                continue
            words.append(row["words"])
            output.append(row["output"])
            base = replies.get((cp, control, n))
            if base and base.get(field):
                share = len(set(row.get(field) or ()) & set(base[field])) / len(base[field])
                per_checkpoint.setdefault(cp, []).append(share)
        means = {cp: sum(v) / len(v) for cp, v in per_checkpoint.items()}
        out.append(dict(condition=name, replies=len(words), checkpoints=len(means),
                        overlap_with_control=round(sum(means.values()) / len(means), 3) if means else None,
                        per_checkpoint=means,
                        mean_words=round(sum(words) / len(words)) if words else 0, mean_output=round(sum(output) / len(output)) if output else 0))
    return out


def reply_details(rows: list, sources: dict, control: str, conditions: tuple) -> list:
    """Each detail a control's reply named that the checkpoint held, and whether each condition's reply named it too.

    One row per detail per next message: a replay's answer to "what did this condition lose", with the detail's source to
    group it by. Sources are hashed, so this holds no text. A detail the reply brought from nowhere in the checkpoint isn't
    counted, and neither is one the floor already held.
    """
    replies = {(r["checkpoint"], r["condition"], r.get("n", 0)): r for r in rows if r.get("row") == "replay" and not r.get("error")}
    out = []
    for (cp, cond, n), base in replies.items():
        if cond != control:
            continue
        table = sources.get(cp, {})
        for detail in base.get("details") or ():
            tags = table.get(detail)
            if tags is None:
                continue
            row = dict(checkpoint=cp, n=n, detail=detail, **tags)
            for name in conditions:
                other = replies.get((cp, name, n))
                row[name] = None if other is None else detail in (other.get("details") or ())
            out.append(row)
    return out


def reply_table(details: list, axis: str, conditions: tuple, resamples: int = 2000) -> list:
    """Of the details a control's reply named, by `axis` of their source, the share each condition's reply still named."""
    def key_of(row):
        return group_kind(str(row.get("label"))) if axis == "group" else row.get(axis)

    table = []
    for key in sorted({key_of(r) for r in details}, key=str):
        row = dict(axis=axis, band=key)
        for name in conditions:
            per: dict = {}
            for r in details:
                if key_of(r) == key and r.get(name) is not None:
                    kept, total = per.get(r["checkpoint"], (0, 0))
                    per[r["checkpoint"]] = (kept + bool(r[name]), total + 1)
            kept, total = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
            low, high = _cluster_ci(list(per.values()), resamples)
            row.update({f"{name}_n": total, f"{name}_named": round(kept / total, 3) if total else None,
                        f"{name}_low": low, f"{name}_high": high})
        table.append(row)
    return table


def overlap_table(details: list, conditions: tuple, control: str) -> list:
    """Per condition, details still named, paired by checkpoint and resampled by checkpoint, against the repeat that is
    the noise floor."""
    table = []
    for name in conditions:
        per: dict = {}
        for r in details:
            if r.get(name) is not None:
                kept, total = per.get(r["checkpoint"], (0, 0))
                per[r["checkpoint"]] = (kept + bool(r[name]), total + 1)
        kept, total = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
        low, high = _cluster_ci(list(per.values()), 2000)
        table.append(dict(condition=name, control=control, checkpoints=len(per), details=total,
                          still_named=round(kept / total, 3) if total else None, low=low, high=high,
                          per_checkpoint={k: round(v[0] / v[1], 3) for k, v in sorted(per.items())}))
    return table


def paired_against(details: list, name: str, baseline: str) -> dict:
    """The paired difference in details still named, one condition against another, by checkpoint."""
    a, b = {}, {}
    for r in details:
        if r.get(name) is not None and r.get(baseline) is not None:
            for store, key in ((a, name), (b, baseline)):
                kept, total = store.get(r["checkpoint"], (0, 0))
                store[r["checkpoint"]] = (kept + bool(r[key]), total + 1)
    cps = sorted(set(a) & set(b))
    return paired_bootstrap([a[c][0] / a[c][1] for c in cps], [b[c][0] / b[c][1] for c in cps]) if len(cps) >= 2 else {"n": len(cps)}


def train_s8_from_replies(details: list, cells: list, condition: str = "all:L3", prior: float = 1.0) -> dict:
    """S8 from replies: the chance a named detail stops being named when its cell goes down, times its density per token."""
    tally: dict = {}
    for r in details:
        if r.get(condition) is None:
            continue
        cell = (group_kind(str(r.get("label"))), r.get("band"))
        lost, total = tally.get(cell, (0, 0))
        tally[cell] = (lost + (not r[condition]), total + 1)
    tokens: dict = {}
    for row in cells:
        cell = (group_kind(row["label"]), row["band"])
        tokens[cell] = tokens.get(cell, 0) + row["tokens"]
    all_lost, all_total = sum(v[0] for v in tally.values()), sum(v[1] for v in tally.values())
    pooled = (all_lost + prior) / (all_total + 2 * prior) if all_total else 0.5
    table = {}
    for cell, size in tokens.items():
        if size:
            lost, total = tally.get(cell, (0, 0))
            table[f"{cell[0]}|{cell[1]}"] = round((lost + prior * pooled) / (total + prior) * (total + 0.5) / size * 1000, 5)
    return table


def usage_totals(rows: list) -> dict:
    calls = [r for r in rows if r.get("row") == "call"]
    return dict(calls=len(calls), forks=sum(1 for r in rows if r.get("row") == "fork"),
                read=sum(r["read"] for r in calls), written=sum(r["write_1h"] + r["write_5m"] for r in calls),
                output=sum(r["output"] for r in calls), input=sum(r["input"] for r in calls),
                list_usd=round(sum(r["list_usd"] for r in calls), 2), errors=sum(1 for r in calls if r["error"]))


__all__ = ["condition_table", "group_kind", "importance_table", "lookups_per_correct", "overlap_table", "paired_against", "read", "replay_overlap", "reply_details", "reply_table", "train_s8",
           "train_s8_from_replies",
           "usage_totals"]
