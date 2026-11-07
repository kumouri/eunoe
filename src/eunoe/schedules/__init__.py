"""Schedules: when an edit lands, and to what size. Knows TTLs and prices, never importance.

Spell a schedule as a string in a config, with token counts in plain numbers:
`every:100:200000`, `threshold:300000:200000`, `expiry:0.5`, `expiry_threshold:0.5:300000:200000`,
`generational:300000:200000:0.5`, `every_call:200000`, `never`.
"""
from __future__ import annotations

from eunoe.interfaces import Budget, CallUsage, ScheduleState
from eunoe.ledger import Ledger

TTL_SECONDS = {"5m": 300, "1h": 3600}


class _Base:
    frozen_share = 0.0

    def __init__(self) -> None:
        self.state = ScheduleState()

    def target(self, ledger: Ledger, budget: Budget) -> int:
        return budget.target_tokens

    def note_edit(self, reached: int, wanted: int, pre: int) -> None:
        """A target it could not reach must not fire again next call: wait for a batch of growth."""
        self.state.edits += 1
        self.state.calls_since_edit = 0
        self.state.last_edit_tokens = reached
        self.state.hold_until_tokens = reached + max(0, pre - wanted)

    def expired(self, gap_s: float, ttl: str) -> bool:
        return gap_s > TTL_SECONDS.get(ttl, 3600)


class Never(_Base):
    def due(self, ledger: Ledger, last: CallUsage, gap_s: float) -> bool:
        return False


class EveryN(_Base):
    """Every `n` calls, down to a cap, when the context is over that cap."""

    def __init__(self, n: int, cap: int) -> None:
        super().__init__()
        self.n, self.cap = n, cap

    def due(self, ledger: Ledger, last: CallUsage, gap_s: float) -> bool:
        return (self.state.calls_since_edit >= self.n and ledger.tokens_held() > self.cap
                and ledger.tokens_held() >= self.state.hold_until_tokens)

    def target(self, ledger: Ledger, budget: Budget) -> int:
        return self.cap


class EveryCall(EveryN):
    """Literal continuity. A negative control: every edit restarts the cache."""

    def __init__(self, cap: int) -> None:
        super().__init__(1, cap)


class Threshold(_Base):
    """A batch: when the context passes `trigger`, cut to `target_tokens`."""

    def __init__(self, trigger: int, target_tokens: int) -> None:
        super().__init__()
        self.trigger, self.target_tokens = trigger, target_tokens

    def due(self, ledger: Ledger, last: CallUsage, gap_s: float) -> bool:
        held = ledger.tokens_held()
        return held > self.trigger and held >= self.state.hold_until_tokens

    def target(self, ledger: Ledger, budget: Budget) -> int:
        return self.target_tokens


class Expiry(_Base):
    """Edit only when the cache has already expired, keeping a share of what is held."""

    def __init__(self, keep: float, ttl: str = "1h", floor: int = 0) -> None:
        super().__init__()
        self.keep, self.ttl, self.floor = keep, ttl, floor

    def due(self, ledger: Ledger, last: CallUsage, gap_s: float) -> bool:
        return self.expired(gap_s, self.ttl)

    def target(self, ledger: Ledger, budget: Budget) -> int:
        return max(self.floor, int(ledger.tokens_held() * self.keep))


class ExpiryThreshold(Expiry):
    """Both at once: at expiry keep a share, and past the trigger cut to the target anyway."""

    def __init__(self, keep: float, trigger: int, target_tokens: int, ttl: str = "1h", floor: int = 0) -> None:
        super().__init__(keep, ttl, floor)
        self.trigger, self.target_tokens = trigger, target_tokens
        self._at_expiry = False

    def due(self, ledger: Ledger, last: CallUsage, gap_s: float) -> bool:
        self._at_expiry = self.expired(gap_s, self.ttl)
        if self._at_expiry:
            return True
        held = ledger.tokens_held()
        return held > self.trigger and held >= self.state.hold_until_tokens

    def target(self, ledger: Ledger, budget: Budget) -> int:
        if self._at_expiry:
            return max(self.floor, int(ledger.tokens_held() * self.keep))
        return self.target_tokens


class Generational(Threshold):
    """Edits land only past a frozen region, which grows at each batch, so reads survive before it."""

    def __init__(self, trigger: int, target_tokens: int, frozen_share: float = 0.5) -> None:
        super().__init__(trigger, target_tokens)
        self.frozen_share = frozen_share


def parse_schedule(spec: str):
    """A schedule from its string form. See the module docstring."""
    name, *args = spec.split(":")
    try:
        if name == "never":
            return Never()
        if name == "every":
            return EveryN(int(args[0]), int(args[1]))
        if name == "every_call":
            return EveryCall(int(args[0]))
        if name == "threshold":
            return Threshold(int(args[0]), int(args[1]))
        if name == "expiry":
            return Expiry(float(args[0]), *(args[1:2] or ["1h"]))
        if name == "expiry_threshold":
            return ExpiryThreshold(float(args[0]), int(args[1]), int(args[2]), *(args[3:4] or ["1h"]))
        if name == "generational":
            return Generational(int(args[0]), int(args[1]), float(args[2]) if len(args) > 2 else 0.5)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"bad schedule {spec!r}: {exc}") from None
    raise ValueError(f"unknown schedule {spec!r}")


#: The live-run grid, by name. Token counts are scaled to a session size in `grid_for`.
def grid_for(budget: int) -> dict:
    """The six schedule factors of the pre-registered design, at a given budget."""
    return {
        "every_100": f"every:100:{budget}",
        "batch_1.5x_to_1x": f"threshold:{int(budget * 1.5)}:{budget}",
        "batch_2.5x_to_1.25x": f"threshold:{int(budget * 2.5)}:{int(budget * 1.25)}",
        "expiry_keep_50": "expiry:0.5:1h",
        "expiry_plus_batch": f"expiry_threshold:0.5:{int(budget * 1.5)}:{budget}:1h",
        "generational": f"generational:{int(budget * 1.5)}:{budget}:0.5",
    }
