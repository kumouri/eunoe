"""A provider's prefix cache, simulated, so the scripted driver and CI see the real shape of the cost.

Three documented rules make editing expensive, and every price in this kit multiplies them [D]:

* A cache entry is written at a breakpoint, for the whole prefix up to it, and is good only while its TTL holds.
* A later request reads an entry only if its own prefix matches byte for byte up to there.
* The lookup checks **at most 20 positions back from each breakpoint**, so a request that lands more than 20
  positions past the last entry misses it, even with every byte unchanged. Consecutive `tool_use` blocks collapse to
  one position, and so do consecutive `tool_result` blocks. A second breakpoint starts a second window.

The binding check models the provider rule that editing an earlier turn fails a check on every later thinking block
it already issued: on an account where that is enforced, an edit before a kept one raises.
"""
from __future__ import annotations

import hashlib

from eunoe.interfaces import CallUsage, RenderedContext
from eunoe.ledger import block_tokens
from eunoe.levels import estimate_tokens
from eunoe.schedules import TTL_SECONDS
from eunoe.wire import canonical

LOOKBACK_POSITIONS = 20
MAX_BREAKPOINTS = 4


class ThinkingBindingError(Exception):
    """What a bound account gets: a 400 for edited history under a kept thinking block."""


def positions(blocks: list) -> list:
    """Each block's lookup position. A run of tool_use, or of tool_result, counts as one."""
    out, position, previous = [], -1, None
    for block in blocks:
        if not (block.type in ("tool_use", "tool_result") and block.type == previous):
            position += 1
        out.append(position)
        previous = block.type
    return out


class PrefixCache:
    def __init__(self, ttl: str = "5m", binding: bool = False, lookback: int = LOOKBACK_POSITIONS) -> None:
        self.ttl = ttl
        self.binding = binding
        self.lookback = lookback
        self._keys: list = []
        self._entries: dict = {}
        self._floor: tuple | None = None
        self._last_at: float | None = None
        self._seen_thinking: set = set()

    def _alive(self, at: float, now: float) -> bool:
        return now - at <= TTL_SECONDS[self.ttl]

    def call(self, ctx: RenderedContext, now: float, output_tokens: int = 0) -> CallUsage:
        keys = [canonical(b) for b in ctx.blocks]
        sizes = [block_tokens(b) for b in ctx.blocks]
        self._check_binding(ctx, keys, now)
        chain, digest = [], hashlib.sha256()
        for key in keys:
            digest.update(key.encode("utf-8"))
            chain.append(digest.hexdigest())
        spots = positions(ctx.blocks)
        breakpoints = self._breakpoints(ctx, len(keys))
        best = -1
        for point in breakpoints:
            for i in range(point, -1, -1):
                if spots[point] - spots[i] >= self.lookback:
                    break
                if chain[i] in self._entries and self._alive(self._entries[chain[i]], now):
                    best = max(best, i)
                    break
        floor_size = estimate_tokens(ctx.system) + sum(estimate_tokens(t.description) + 40 for t in ctx.tools)
        floor_key = (ctx.system, tuple(t.name for t in ctx.tools))
        floor_alive = self._floor is not None and self._floor[0] == floor_key and self._alive(self._floor[1], now)
        read = sum(sizes[:best + 1]) + (floor_size if floor_alive else 0)
        write = sum(sizes[best + 1:]) + (0 if floor_alive else floor_size)
        # A read refreshes what it reads, and each breakpoint writes an entry.
        if best >= 0:
            self._entries[chain[best]] = now
        for point in breakpoints:
            self._entries[chain[point]] = now
        self._floor = (floor_key, now)
        self._keys, self._last_at = keys, now
        self._seen_thinking.update(b.signature for b in ctx.blocks if b.type == "thinking" and b.signature)
        usage = CallUsage(output=output_tokens, cache_read=read)
        if self.ttl == "1h":
            usage.cache_write_1h = write
        else:
            usage.cache_write_5m = write
        return usage

    def _breakpoints(self, ctx: RenderedContext, count: int) -> list:
        if not count:
            return []
        extra = [i for i in ctx.meta.get("breakpoints", ()) if 0 <= i < count - 1]
        return sorted({count - 1, *extra[-(MAX_BREAKPOINTS - 1):]}, reverse=True)

    def _check_binding(self, ctx: RenderedContext, keys: list, now: float) -> None:
        if not self.binding or self._last_at is None or not self._alive(self._last_at, now):
            return
        common = 0
        for old, new in zip(self._keys, keys):
            if old != new:
                break
            common += 1
        if common < len(self._keys):
            for block in ctx.blocks[common:]:
                # A block new to this request is fine: only one the provider already issued is bound.
                if block.type == "thinking" and block.signature in self._seen_thinking:
                    raise ThinkingBindingError(f"history edited before a kept thinking block at {block.event_id}")
