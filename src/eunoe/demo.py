"""`eunoe demo`: every built-in arm on generated sessions, offline, with no key and no model."""
from __future__ import annotations

from eunoe.agents import RECALL_NOTE, DefaultAgent
from eunoe.arms import CONTROLS, TREATMENTS, build_arm
from eunoe.cost import ListPrice
from eunoe.drivers.scripted import ScriptedDriver
from eunoe.runner import SessionRun, aggregate
from eunoe.tasks.generator import DOMAINS, generate_session, get_profile

NOTICE = ("The scripted driver is a fixture, not a model: it answers only from the context it is handed, under "
          "written-down settings.\nThese numbers show the machinery working. They say nothing about any hypothesis.")


def run_arms(arms, profile="demo", seeds=(1,), domains=None, budget=20_000, ttl="5m", thinking="strip_on_edit",
             model="claude-sonnet-5-5", schedule=None, driver_factory=None) -> dict:
    """Arm name to `(aggregate, results)`. Every arm sees exactly the same sessions."""
    prof = get_profile(profile) if isinstance(profile, str) else profile
    out = {}
    for arm in arms:
        results = []
        for seed in seeds:
            for name in domains or DOMAINS:
                session = generate_session(name, seed, prof)
                agent = DefaultAgent(DOMAINS[name].brief, session.tool_specs)
                system = agent.instructions + (RECALL_NOTE if arm.startswith("T") else "")
                strategy = build_arm(arm, budget=budget, system=system, tools=agent.tools(), thinking=thinking,
                                     schedule=schedule if arm.startswith(("T", "C3")) else None)
                driver = driver_factory(seed) if driver_factory else ScriptedDriver(seed=f"{seed}:{name}", ttl=ttl)
                results.append(SessionRun(session, strategy, driver, ListPrice(model), ttl).run())
        out[arm] = (aggregate(results), results)
    return out


def format_table(rows: dict) -> str:
    header = ("arm", "recall", "halluc.", "abstain", "stale", "tasks", "lookups", "cost $", "mean ctx", "edits")
    lines = [" ".join(f"{h:>8}" for h in header)]
    for arm, (agg, _) in rows.items():
        lines.append(" ".join(f"{v:>8}" for v in (
            arm, f"{agg['probe_recall']:.2f}", f"{agg['hallucination_on_evicted']:.2f}",
            f"{agg['correct_abstention']:.2f}", f"{agg['staleness']:.2f}", f"{agg['task_success']:.2f}",
            agg["lookups"], f"{agg['cost_usd']:.3f}", f"{agg['mean_context'] / 1000:.1f}k", agg["edits"])))
    return "\n".join(lines)


def main(quick: bool = False, seeds: int = 1, budget: int = 20_000, print_=print) -> int:
    profile = "tiny" if quick else "demo"
    arms = CONTROLS + TREATMENTS
    use_budget = 3_000 if quick else budget
    rows = run_arms(arms, profile, tuple(range(1, seeds + 1)), budget=use_budget)
    print_(NOTICE)
    print_("")
    print_(f"profile={profile} seeds={seeds} domains={','.join(DOMAINS)} budget={use_budget} ttl=5m yardstick=claude-sonnet-5-5")
    print_("")
    print_(format_table(rows))
    print_("")
    print_("C0 is no management. T2 is the ladder in importance order. `eunoe conform` shows what your own harness can run.")
    return 0
