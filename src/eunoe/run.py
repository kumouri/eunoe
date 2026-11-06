"""`eunoe run`: a config of arms, seeds and a driver, with a stated cost estimate first.

A config is TOML. Results go to `--out`, outside the repo by default through `.gitignore`, with a manifest
naming the seeds, versions, fixture hashes and price file, so a number can be recomputed and a stranger can
see what produced it. A scored run also needs a frozen pre-registration: see `PREREGISTRATION.md`.
"""
from __future__ import annotations

import json
import os
import time
import tomllib

from eunoe import __version__
from eunoe.agents import RECALL_NOTE, DefaultAgent
from eunoe.analysis import paired_bootstrap
from eunoe.arms import build_arm
from eunoe.cost import DEFAULT_FILE, ListPrice
from eunoe.demo import format_table
from eunoe.drivers.base import SpendGate, SpendRefused
from eunoe.drivers.scripted import ScriptedDriver
from eunoe.runner import SessionRun, aggregate
from eunoe.tasks.generator import DOMAINS, generate_session, get_profile


def load_config(path: str) -> dict:
    with open(path, "rb") as fh:
        config = tomllib.load(fh)
    config.setdefault("arms", ["C0", "T2"])
    config.setdefault("seeds", 1)
    config.setdefault("profile", "demo")
    config.setdefault("budget", 20_000)
    config.setdefault("ttl", "5m")
    config.setdefault("thinking", "strip_on_edit")
    config.setdefault("driver", "scripted")
    config.setdefault("model", "claude-sonnet-5-5")
    config.setdefault("domains", list(DOMAINS))
    return config


def build_driver(config: dict, seed, gate: SpendGate):
    name = config["driver"]
    if name == "scripted":
        return ScriptedDriver(seed=seed, ttl=config["ttl"])
    from eunoe.drivers.live import make_driver

    return make_driver(name, model=config["model"], gate=gate, ttl=config["ttl"])


def estimate(config: dict, seeds: list) -> dict:
    """List price of the run with no management: an upper bound on what it reads, from the fixture itself."""
    prof = get_profile(config["profile"], **config.get("profile_overrides", {}))
    rows = []
    for name in config["domains"]:
        session = generate_session(name, seeds[0], prof)
        strategy = build_arm("C0", budget=config["budget"], system="", tools=session.tool_specs)
        driver = ScriptedDriver(seed=seeds[0], ttl=config["ttl"])
        rows.append(SessionRun(session, strategy, driver, ListPrice(config["model"]), config["ttl"]).run())
    per_seed = sum(r.cost_usd for r in rows)
    sessions = len(config["arms"]) * len(seeds) * len(config["domains"])
    return {"sessions": sessions, "calls_per_session": rows[0].calls, "upper_bound_usd": round(
        per_seed * len(config["arms"]) * len(seeds), 4)}


def main(args) -> int:
    config = load_config(args.config)
    count = args.seeds if args.seeds is not None else config["seeds"]
    seeds = list(range(1, count + 1))
    plan = estimate(config, seeds)
    print(f"{plan['sessions']} sessions, about {plan['calls_per_session']} calls each, on {config['driver']}.")
    print(f"List-price upper bound on {config['model']}: {plan['upper_bound_usd']} USD. Scripted runs cost nothing.")
    if args.estimate_only:
        return 0
    if config["driver"] != "scripted" and not args.yes:
        print("This uses a real model. Pass --yes to go ahead.")
        return 2
    os.makedirs(args.out, exist_ok=True)
    gate = SpendGate(allow=args.allow_spend, cap_usd=args.spend_cap, ledger_path=os.path.join(args.out, "spend.jsonl"))
    prof = get_profile(config["profile"], **config.get("profile_overrides", {}))
    rows, fingerprints = {}, {}
    try:
        for arm in config["arms"]:
            results = []
            for seed in seeds:
                for name in config["domains"]:
                    session = generate_session(name, seed, prof)
                    fingerprints[session.id] = session.fingerprint()
                    agent = DefaultAgent(DOMAINS[name].brief, session.tool_specs)
                    system = agent.instructions + (RECALL_NOTE if arm.startswith("T") else "")
                    strategy = build_arm(arm, budget=config["budget"], system=system, tools=agent.tools(),
                                         thinking=config["thinking"], schedule=config.get("schedule"))
                    driver = build_driver(config, f"{seed}:{name}", gate)
                    log = _call_log(os.path.join(args.out, "calls.jsonl"), arm, session.id)
                    try:
                        results.append(SessionRun(session, strategy, driver, ListPrice(config["model"]), config["ttl"],
                                                  spend_log=log).run())
                    finally:
                        if hasattr(driver, "close"):
                            driver.close()
            rows[arm] = (aggregate(results), results)
    except SpendRefused as exc:
        print(f"Stopped: {exc}")
        return 3
    print(format_table(rows))
    baseline = config["arms"][0]
    comparisons = {}
    for arm in config["arms"][1:]:
        comparisons[f"{arm} - {baseline}"] = paired_bootstrap(_session_recall(rows[arm][1]), _session_recall(rows[baseline][1]))
    for label, ci in comparisons.items():
        print(f"recall, {label}: {ci['mean']:+.3f} [{ci['low']:+.3f}, {ci['high']:+.3f}]")
    manifest = {
        "kit_version": __version__, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": config,
        "seeds": seeds, "price_file": DEFAULT_FILE, "fixtures": fingerprints,
        "arms": {arm: {"aggregate": agg, "manifests": [r.manifest for r in res]} for arm, (agg, res) in rows.items()},
        "comparisons": comparisons, "api_usd_spent": round(gate.spent_usd, 6),
    }
    path = os.path.join(args.out, f"run-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"Manifest: {path}. API dollars spent: {round(gate.spent_usd, 6)}.")
    return 0


def _call_log(path: str, arm: str, session: str):
    """One row per call as it happens, so a run that is cut short still keeps what it measured."""

    def write(usage, usd: float) -> None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"arm": arm, "session": session, "input": usage.input, "cache_read": usage.cache_read,
                                 "cache_write": usage.cache_write, "output": usage.output, "list_usd": round(usd, 6)}) + "\n")

    return write


def _session_recall(results: list) -> list:
    return [r.rate(lambda s: s.correct, lambda s: s.kind != "unanswerable") for r in results]
