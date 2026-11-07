"""The arms and controls of the experiment, by name, each a `Strategy` the runner can drive.

| Arm | What it is |
|---|---|
| C0 | No management, up to the model's limit |
| C1 | Compaction at a trigger: provider-side on a driver that has it, a labelled emulation otherwise |
| C2 | A handoff-style restart: a fresh session holding a code-built block |
| C3 | Observation masking, oldest first: the honest simple baseline |
| C4 | Clearing old tool results, oldest first: a provider's own eviction, emulated where it has none |
| T1 | The contracting ladder in oldest-first order. A control only |
| T2 | The contracting ladder in importance order, no marker |
| T3 | T2 with an explicit marker naming the event |
| T4 | T2's order, one step to a stub, no ladder |
| lethe | LETHE's own engine, if installed (`eunoe.arms.lethe`) |
| lcm | hermes-lcm's own engine, run in its host's environment (`eunoe.arms.hermes_lcm`) |

Budgets match within a comparison, and a manifest records what each arm really held.
"""
from __future__ import annotations

from eunoe.interfaces import Budget
from eunoe.arms.strategies import (
    CompactionStrategy, ContractingStrategy, HandoffStrategy, MaskingStrategy, NoManagement, RECALL_TOOL, Strategy,
    ClearToolUsesStrategy,
)
from eunoe.policies import LadderPolicy, StubPolicy
from eunoe.schedules import parse_schedule

CONTROLS = ("C0", "C1", "C2", "C3", "C4")
TREATMENTS = ("T1", "T2", "T3", "T4")
EXTERNAL = ("lethe", "lcm")
ARMS = CONTROLS + TREATMENTS + EXTERNAL
#: A default until an importance study chooses. The design says a signal is chosen by measurement.
DEFAULT_SIGNAL = "S9"


def build_arm(name: str, *, budget: int, system: str = "", tools=None, schedule: str | None = None,
              signal: str = DEFAULT_SIGNAL, thinking: str = "strip_on_edit", keep_recent: int = 6,
              server_side: bool = False) -> Strategy:
    """An arm at a budget. `schedule` defaults to a batch from 1.5 times the budget down to it."""
    bud = Budget(budget, keep_recent)
    sched = parse_schedule(schedule or f"threshold:{int(budget * 1.5)}:{budget}")
    common = dict(system=system, tools=tools or [], budget=bud, thinking=thinking)
    if name == "C0":
        return NoManagement("C0", thinking=thinking, system=system, tools=tools or [], budget=bud)
    if name == "C1":
        return CompactionStrategy("C1", trigger=int(budget * 1.5), server_side=server_side, **common)
    if name == "C2":
        return HandoffStrategy("C2", trigger=int(budget * 1.5), **common)
    if name == "C3":
        return MaskingStrategy("C3", schedule=sched, **common)
    if name == "C4":
        return ClearToolUsesStrategy("C4", trigger=int(budget * 1.5), server_side=server_side, **common)
    if name == "T1":
        return ContractingStrategy("T1", policy=LadderPolicy("S0"), schedule=sched, marker="M0", **common)
    if name == "T2":
        return ContractingStrategy("T2", policy=LadderPolicy(signal), schedule=sched, marker="M0", **common)
    if name == "T3":
        return ContractingStrategy("T3", policy=LadderPolicy(signal), schedule=sched, marker="M2", **common)
    if name == "T4":
        return ContractingStrategy("T4", policy=StubPolicy(signal), schedule=sched, marker="M0", **common)
    if name == "lethe":
        from eunoe.arms.lethe import LetheStrategy
        return LetheStrategy(**common)
    if name == "lcm":
        from eunoe.arms.hermes_lcm import HermesLcmStrategy
        return HermesLcmStrategy(**common)
    raise ValueError(f"unknown arm {name!r}, one of {ARMS}")


__all__ = ["ARMS", "CONTROLS", "EXTERNAL", "TREATMENTS", "DEFAULT_SIGNAL", "RECALL_TOOL", "Strategy", "build_arm"]
