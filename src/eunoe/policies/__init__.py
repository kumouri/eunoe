"""Eviction policies: given the ledger and a budget, which events go down a level.

The ladder steps one level at a time. An event's priority is its rank in importance order plus half a
point for every level it has already gone down, so detail goes from many events before any one event
is reduced to a headline, and an event leaves context only after every other option is used.
"""
from __future__ import annotations

import heapq

from eunoe.interfaces import Budget, LevelChange, ScheduleState
from eunoe.ledger import THINKING, USER, Event, Ledger
from eunoe.levels import Level
from eunoe.signals import get_signal

LADDER_STEP = 0.5
#: User messages stay at L0 longest. A default a signal may override.
USER_PENALTY = 1.0
LAST_RESORT = 10.0


def _candidates(ledger: Ledger, budget: Budget, sched: ScheduleState, only_tools: bool = False) -> list:
    limit = ledger.call - budget.keep_recent_calls
    out = []
    for event in ledger.events:
        if event.call > limit or event.index <= sched.frozen_upto:
            continue
        if event.kind == THINKING or ledger.levels[event.id] >= Level.L4:
            continue
        if only_tools and not event.is_tool:
            continue
        out.append(event)
    return out


def _ranks(ledger: Ledger, candidates: list, signal) -> dict:
    scored = sorted(candidates, key=lambda e: (signal.score(e, ledger, ledger.now), e.index))
    total = float(max(len(scored), 1))
    return {e.id: pos / total for pos, e in enumerate(scored)}


class LadderPolicy:
    """The contracting ladder, in the order a signal gives."""

    def __init__(self, signal="S3", **signal_args) -> None:
        self.signal = get_signal(signal, **signal_args) if isinstance(signal, str) else signal

    def next_level(self, event: Event, level: Level) -> int | None:
        if level >= Level.L4:
            return None
        if event.reconstructible:
            return Level.L3 if level == Level.L0 else Level.L4
        return level + 1

    def plan(self, ledger: Ledger, budget: Budget, sched: ScheduleState) -> list:
        candidates = _candidates(ledger, budget, sched)
        ranks = _ranks(ledger, candidates, self.signal)
        now = ledger.tokens_held()
        levels = {e.id: ledger.levels[e.id] for e in candidates}
        heap = []
        for event in candidates:
            nxt = self.next_level(event, levels[event.id])
            if nxt is not None:
                heap.append((self._priority(event, ranks[event.id], levels[event.id], nxt), event.index, event))
        heapq.heapify(heap)
        while heap and now > budget.target_tokens:
            _, _, event = heapq.heappop(heap)
            old = levels[event.id]
            nxt = self.next_level(event, old)
            if nxt is None:
                continue
            now -= ledger.tokens_at(event, old) - ledger.tokens_at(event, nxt)
            levels[event.id] = Level(nxt)
            following = self.next_level(event, Level(nxt))
            if following is not None:
                heapq.heappush(heap, (self._priority(event, ranks[event.id], Level(nxt), following), event.index, event))
        return [LevelChange(e.id, levels[e.id]) for e in candidates if levels[e.id] != ledger.levels[e.id]]

    def _priority(self, event: Event, rank: float, level: Level, nxt: int) -> float:
        priority = rank + LADDER_STEP * int(level)
        if event.kind == USER:
            priority += USER_PENALTY
        if nxt == Level.L4:
            priority += LAST_RESORT
        return priority


class StubPolicy(LadderPolicy):
    """One step to a stub, no ladder. Isolates the ladder (roughly the VISTA and LETHE shape)."""

    def next_level(self, event: Event, level: Level) -> int | None:
        if level >= Level.L4:
            return None
        return Level.L3 if level == Level.L0 else Level.L4


class MaskPolicy:
    """Observation masking, oldest first: a tool result becomes one line. The honest simple baseline."""

    def __init__(self, signal="S0") -> None:
        self.signal = get_signal(signal) if isinstance(signal, str) else signal

    def plan(self, ledger: Ledger, budget: Budget, sched: ScheduleState) -> list:
        candidates = [e for e in _candidates(ledger, budget, sched, only_tools=True) if ledger.levels[e.id] == Level.L0]
        ordered = sorted(candidates, key=lambda e: (self.signal.score(e, ledger, ledger.now), e.index))
        now = ledger.tokens_held()
        changes = []
        for event in ordered:
            if now <= budget.target_tokens:
                break
            now -= ledger.tokens_at(event, Level.L0) - ledger.tokens_at(event, Level.L3)
            changes.append(LevelChange(event.id, Level.L3))
        return changes


POLICIES = {"ladder": LadderPolicy, "stub": StubPolicy, "mask": MaskPolicy}


def get_policy(name: str, **kwargs):
    try:
        return POLICIES[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown policy {name!r}, one of {sorted(POLICIES)}") from None
