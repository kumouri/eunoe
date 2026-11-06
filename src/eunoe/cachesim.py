"""A provider's prefix cache, simulated, so the scripted driver and CI see the real shape of the cost.

A cached prefix is good up to its first changed block, and only while its TTL holds. That one rule is
what makes editing an old event expensive, and it is the rule every price in this kit multiplies. The
binding check models the provider rule that editing an earlier turn fails a check on every later
thinking block: an edit before a kept thinking block raises, on an account where the check is enforced.
"""
from __future__ import annotations

from eunoe.interfaces import CallUsage, RenderedContext
from eunoe.ledger import block_tokens
from eunoe.levels import estimate_tokens
from eunoe.schedules import TTL_SECONDS
from eunoe.wire import canonical


class ThinkingBindingError(Exception):
    """What a bound account gets: a 400 for edited history under a kept thinking block."""


class PrefixCache:
    def __init__(self, ttl: str = "5m", binding: bool = False) -> None:
        self.ttl = ttl
        self.binding = binding
        self._keys: list = []
        self._last_at: float | None = None
        self._seen_thinking: set = set()

    def call(self, ctx: RenderedContext, now: float, output_tokens: int = 0) -> CallUsage:
        keys = [canonical(b) for b in ctx.blocks]
        sizes = [block_tokens(b) for b in ctx.blocks]
        floor = estimate_tokens(ctx.system) + sum(estimate_tokens(t.description) + 40 for t in ctx.tools)
        alive = self._last_at is not None and now - self._last_at <= TTL_SECONDS[self.ttl]
        common = 0
        if alive:
            for old, new in zip(self._keys, keys):
                if old != new:
                    break
                common += 1
        if self.binding and alive and common < len(self._keys):
            for block in ctx.blocks[common:]:
                # A block new to this request is fine: only one the provider already issued is bound.
                if block.type == "thinking" and block.signature in self._seen_thinking:
                    raise ThinkingBindingError(f"history edited before a kept thinking block at {block.event_id}")
        read = sum(sizes[:common]) + (floor if alive else 0)
        write = sum(sizes[common:]) + (0 if alive else floor)
        self._keys, self._last_at = keys, now
        self._seen_thinking.update(b.signature for b in ctx.blocks if b.type == "thinking" and b.signature)
        usage = CallUsage(output=output_tokens)
        usage.cache_read = read
        if self.ttl == "1h":
            usage.cache_write_1h = write
        else:
            usage.cache_write_5m = write
        return usage
