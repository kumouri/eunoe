"""The `eunoe` command line. Offline unless a command says otherwise, and nothing spends unasked."""
from __future__ import annotations

import argparse
import json
import os
import sys

from eunoe import __version__

DEFAULT_OUT = "eunoe-results"


def _cmd_demo(args) -> int:
    from eunoe import demo

    return demo.main(quick=args.quick, seeds=args.seeds, budget=args.budget)


def _cmd_conform(args) -> int:
    from eunoe.cost import ListPrice
    from eunoe.drivers.base import SpendGate
    from eunoe.editors import HARNESSES, SUBSCRIPTION_OR_FREE, get_harness
    from eunoe.editors.conformance import Suite, format_report, report_to_json

    if args.list:
        for name in HARNESSES:
            print(name)
        return 0
    names = SUBSCRIPTION_OR_FREE if args.harness == "all" else (args.harness,)
    os.makedirs(args.out, exist_ok=True)
    worst = 0
    for name in names:
        gate = SpendGate(allow=args.allow_spend, cap_usd=args.spend_cap, ledger_path=os.path.join(args.out, "spend.jsonl"))
        harness = get_harness(name, model=args.model, gate=gate)
        if name == "claude-api-server":
            from eunoe.editors.conformance import make_padding

            report = harness.run_suite(make_padding())
        else:
            cost = None
            if args.model is None and name in ("scripted", "claude-cli", "claude-agent-sdk", "claude-api"):
                cost = ListPrice("claude-haiku-5-5")
            report = Suite(harness, skip_thinking=args.skip_thinking, cost=cost).run()
        print(format_report(report))
        print()
        path = os.path.join(args.out, f"conform-{name.replace(':', '-').replace('.', '-')}.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report_to_json(report))
        if not report.available:
            worst = max(worst, 2 if args.harness != "all" else 0)
        elif not report.conforms:
            worst = max(worst, 1)
    return worst


def _cmd_arms(args) -> int:
    from eunoe.arms import ARMS, __doc__ as doc

    print(doc)
    print("names:", ", ".join(ARMS))
    return 0


def _cmd_gen(args) -> int:
    from eunoe.tasks.generator import generate_session, get_profile

    session = generate_session(args.domain, args.seed, get_profile(args.profile))
    summary = {"id": session.id, "fingerprint": session.fingerprint(), "steps": len(session.steps),
               "facts": len(session.facts), "probes": len(session.probes), "tasks": len(session.tasks)}
    print(json.dumps(summary, indent=2))
    if args.out_file:
        with open(args.out_file, "w", encoding="utf-8") as fh:
            fh.write(session.to_json())
    return 0


def _cmd_run(args) -> int:
    from eunoe.run import main as run_main

    return run_main(args)


def _cmd_replay(args) -> int:
    from eunoe.replay.cli import main as replay_main

    return replay_main(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eunoe", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="every arm on generated sessions, offline, with no key and no model")
    demo.add_argument("--quick", action="store_true", help="a tiny profile, for a smoke test")
    demo.add_argument("--seeds", type=int, default=1)
    demo.add_argument("--budget", type=int, default=20_000)
    demo.set_defaults(func=_cmd_demo)

    conform = sub.add_parser("conform", help="can this harness and account really run a contracting arm?")
    conform.add_argument("--harness", default="scripted",
                         help="a built-in name, all, or package.module:ClassName. See --list")
    conform.add_argument("--list", action="store_true")
    conform.add_argument("--model", default=None)
    conform.add_argument("--out", default=DEFAULT_OUT)
    conform.add_argument("--skip-thinking", action="store_true")
    conform.add_argument("--allow-spend", action="store_true", help="permit metered API calls; each one is recorded")
    conform.add_argument("--spend-cap", type=float, default=1.0, help="USD at list, per harness")
    conform.set_defaults(func=_cmd_conform)

    arms = sub.add_parser("arms", help="the arms and controls")
    arms.set_defaults(func=_cmd_arms)

    gen = sub.add_parser("gen", help="generate one session and print its fingerprint")
    gen.add_argument("--domain", default="oncall")
    gen.add_argument("--seed", type=int, default=1)
    gen.add_argument("--profile", default="demo")
    gen.add_argument("--out-file", default="")
    gen.set_defaults(func=_cmd_gen)

    run = sub.add_parser("run", help="run a config: arms times seeds, with a stated cost estimate first")
    run.add_argument("--config", required=True)
    run.add_argument("--seeds", type=int, default=None)
    run.add_argument("--out", default=DEFAULT_OUT)
    run.add_argument("--allow-spend", action="store_true")
    run.add_argument("--spend-cap", type=float, default=1.0)
    run.add_argument("--estimate-only", action="store_true")
    run.add_argument("--yes", action="store_true", help="skip the confirmation before a run that uses a real model")
    run.set_defaults(func=_cmd_run)

    replay = sub.add_parser("replay", help="read-only: cost surface and usage from your own CLI transcripts")
    replay.add_argument("transcripts", nargs="+", help="session JSONL files")
    replay.add_argument("--budget", type=int, default=200_000)
    replay.add_argument("--model", default="claude-opus-5-5")
    replay.add_argument("--static-tokens", type=int, default=15_000, help="assumed unedited prefix")
    replay.add_argument("--post-compact-tokens", type=int, default=90_000, help="assumed context after a compaction")
    replay.add_argument("--json", action="store_true")
    replay.set_defaults(func=_cmd_replay)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
