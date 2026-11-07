"""A condition: which events go to which level, and what happens to thinking. Each is a pure function of a ledger.

Named conditions, so a table can be read off their names:

* `ctrl`: nothing changes. `ctrl2` is the same again, which measures how much a repeat moves on its own.
* `think`: only thinking goes. The real effect of `strip_on_edit` on a real session.
* `all:L1`, `all:L2`, `all:L3`: everything older than the recent tail at that level. What a level carries.
* `kind:<kind>:L<n>`: one kind at that level, the rest verbatim. A collateral effect shows on items it doesn't source.
* `signal:<S>`: the ladder in that signal's order, down to a budget. `stub:<S>`: one step to a stub, same budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from eunoe.interfaces import Budget, ScheduleState
from eunoe.ledger import THINKING, Ledger
from eunoe.levels import Level
from eunoe.policies import LadderPolicy, StubPolicy


@dataclass
class Condition:
    name: str
    levels: dict = field(default_factory=dict)
    strip_thinking: bool = True

    def changed(self) -> int:
        return len(self.levels)


def _tail_start(ledger: Ledger, keep_recent: int) -> int:
    """The index of the first event in the recent tail, which no condition touches."""
    return max(0, len(ledger.events) - keep_recent)


def _older(ledger: Ledger, keep_recent: int, frozen_upto: int) -> list:
    limit = _tail_start(ledger, keep_recent)
    return [e for e in ledger.events if frozen_upto < e.index < limit and e.kind != THINKING]


def ctrl(name: str = "ctrl") -> Condition:
    return Condition(name, {}, strip_thinking=False)


def strip_only() -> Condition:
    return Condition("think", {}, strip_thinking=True)


def at_level(ledger: Ledger, level: int, keep_recent: int = 10, frozen_upto: int = -1, strip: bool = True) -> Condition:
    return Condition(f"all:L{level}{'' if strip else ':allow'}",
                     {e.id: Level(level) for e in _older(ledger, keep_recent, frozen_upto)}, strip)


def kind_at(ledger: Ledger, kind: str, level: int, keep_recent: int = 10, frozen_upto: int = -1, label=None,
            strip: bool = True) -> Condition:
    def belongs(event) -> bool:
        return (label(event) if label else event.kind) == kind

    return Condition(f"kind:{kind}:L{level}",
                     {e.id: Level(level) for e in _older(ledger, keep_recent, frozen_upto) if belongs(e)}, strip)


class _Offered:
    """A ledger whose recent tail isn't on offer, without changing the policy's rules."""

    def __init__(self, ledger: Ledger, tail: int) -> None:
        self._ledger, self._tail = ledger, tail

    def __getattr__(self, name):
        return getattr(self._ledger, name)

    @property
    def events(self) -> list:
        return [e for e in self._ledger.events if e.index < self._tail]

    def tokens_held(self) -> int:
        return self._ledger.tokens_held()


def by_signal(ledger: Ledger, signal, name: str, share: float = 0.5, stub: bool = False, keep_recent: int = 10,
              frozen_upto: int = -1, strip: bool = True) -> Condition:
    """The real policy, on the real ledger, down to `share` of what it holds now. `signal` is a name or an instance."""
    held = ledger.tokens_held()
    policy = (StubPolicy if stub else LadderPolicy)(signal)
    state = ScheduleState(frozen_upto=frozen_upto)
    # A budget's recent tail counts calls, and a checkpoint has none left to run, so here the tail goes by position.
    offered = _Offered(ledger, _tail_start(ledger, keep_recent))
    ledger.call += 1
    try:
        changes = policy.plan(offered, Budget(int(held * share), 0), state)
    finally:
        ledger.call -= 1
    return Condition(name, {c.event_id: Level(c.level) for c in changes}, strip)


__all__ = ["Condition", "at_level", "by_signal", "ctrl", "kind_at", "strip_only"]
