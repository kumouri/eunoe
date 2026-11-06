"""The cost surface: what a real session's calls would have cost under another schedule.

Same calls, same output, same new input. A schedule says when an edit lands and to what size. `static` and
`post_compact` are assumptions you set, not measurements this kit carries: the unedited prefix, and the context a
compaction leaves. Measure your own and pass them. Dollars are at list, for comparing schedules, not a bill.

The "half frozen" row assumes a cache entry exists at the frozen region's end. A lookup checks only 20 positions back,
so that takes an explicit breakpoint there, which an API driver can place and a CLI can't. Read that row as an API
figure. On a CLI, expect an edit to fall back to the floor, as the conformance suite shows.
"""
from __future__ import annotations

from dataclasses import dataclass

from eunoe.cost import load_prices, DEFAULT_FILE

TTL_1H, TTL_5M = 3600, 300


@dataclass
class Edit:
    ctx: int
    frozen: int = 0
    free_write: bool = False
    model_call: int = 0


def rates(model: str, price_file: str = DEFAULT_FILE) -> dict:
    return load_prices(price_file)["models"][model]


def growth(calls: list) -> list:
    """New tokens before each call, real compaction drops excluded."""
    out = [0] * len(calls)
    for k in range(1, len(calls)):
        out[k] = max(0, calls[k].ctx - calls[k - 1].ctx)
    return out


def natural_misses(calls: list, ttl: int) -> set:
    """Calls where the cache is already dead: a gap past the TTL, or a real miss."""
    out = set()
    for k in range(1, len(calls)):
        cur, prev = calls[k], calls[k - 1]
        if cur.ts - prev.ts > ttl or (cur.read < 0.5 * cur.ctx and cur.ctx > 0.5 * prev.ctx):
            out.add(k)
    return out


def real_cost(calls: list, price: dict) -> float:
    return sum(c.input * price["input"] + c.read * price["read"] + c.w5 * price["write_5m"]
               + (c.write - c.w5) * price["write_1h"] + c.out * price["output"] for c in calls) / 1e6


def replay_cost(calls: list, schedule, price: dict, *, static: int, ttl: int = TTL_1H) -> dict:
    """Dollars under a schedule `(pre, since, free) -> Edit | None`, with the rewrites it caused."""
    r_p, o_p = price["read"], price["output"]
    w_p = price["write_1h"] if ttl == TTL_1H else price["write_5m"]
    grow, nat = growth(calls), natural_misses(calls, ttl)
    ctx = calls[0].ctx
    cost = (max(0, ctx - static) * w_p + min(ctx, static) * r_p + calls[0].out * o_p) / 1e6
    cached, since, rewrites, ctx_sum, peak = ctx, 0, 0, ctx, ctx
    for k in range(1, len(calls)):
        new = grow[k]
        pre = cached + new
        since += 1
        free = k in nat
        edit = schedule(pre, since, free)
        now = None
        if edit is not None and edit.ctx < pre:
            if edit.free_write:
                cost += (edit.ctx * r_p + new * w_p) / 1e6
            else:
                keep = min(edit.ctx, static if free else static + edit.frozen)
                cost += (keep * r_p + (edit.ctx - keep) * w_p) / 1e6
                rewrites += 1
            if edit.model_call:
                cost += (pre * r_p + edit.model_call * o_p) / 1e6
            now, since = edit.ctx, 0
        if now is None:
            if free:
                head = min(pre, static)
                cost += (head * r_p + (pre - head) * w_p) / 1e6
                rewrites += 1
            else:
                cost += (cached * r_p + new * w_p) / 1e6
            now = pre
        cost += calls[k].out * o_p / 1e6
        cached = now
        ctx_sum += now
        peak = max(peak, now)
    return {"cost": cost, "rewrites": rewrites, "mean_ctx": ctx_sum / len(calls), "peak": peak}


def schedule_grid(budget: int, post_compact: int, summary_tokens: int = 10_000) -> dict:
    """Named schedules around a budget, the shapes a run can choose between."""
    return {
        "no management": lambda pre, since, free: None,
        "compaction at 2x budget": lambda pre, since, free:
            Edit(post_compact, model_call=summary_tokens) if pre > 2 * budget else None,
        "every call": lambda pre, since, free: Edit(budget) if since >= 1 and pre > budget else None,
        "every 100 calls": lambda pre, since, free: Edit(budget) if since >= 100 and pre > budget else None,
        "batch 1.5x to 1x": lambda pre, since, free: Edit(budget) if pre > 1.5 * budget else None,
        "batch 2.5x to 1.25x": lambda pre, since, free: Edit(int(1.25 * budget)) if pre > 2.5 * budget else None,
        "expiry keep 50%": lambda pre, since, free: Edit(max(post_compact, int(pre * 0.5))) if free else None,
        "same batch, half frozen": lambda pre, since, free:
            Edit(budget, frozen=int(0.5 * budget)) if pre > 1.5 * budget else None,
        "free edits (no rewrite cost)": lambda pre, since, free: Edit(budget, free_write=True) if pre > budget else None,
    }
