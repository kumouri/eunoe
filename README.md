# Eunoe

**Contracting context for LLM agents, as an experiment you can run yourself.**

Long-running agents hit a wall: a cached prefix has to stay byte-identical, so context only ever
grows until something compacts all of it at once. Eunoe tests a different shape. Detail falls out of
context one level at a time, least important first, and the event itself stays until last, so the
model can still see that there is more to look up.

The questions, plainly: does contracting beat compaction, on what measures, and at what cost? Is there
a region where performance rises and cost falls? What is acceptable to forget?

**Status: phase 1, the kit.** The machinery below works and is tested. No scored result exists yet, and
nothing here is a finding. The design is [`docs/design.md`](docs/design.md).

## Run it

Python 3.11 or later, and nothing else. It installs no dependencies and needs no key.

```bash
git clone https://github.com/kumouri/eunoe
cd eunoe
python -m pip install .
python -m eunoe demo
python -m eunoe conform
python -m unittest discover -s tests -t .
```

`demo` takes under a minute: every arm on generated sessions, with a scripted driver standing in for a
model. It answers only from the context it was handed, so an evicted fact really is gone, but its
numbers show the machinery working and say nothing about any hypothesis.

`conform` runs the conformance suite on an offline harness. To find out whether your own harness and
account can really run a contracting arm, name it. It costs a few cents of your plan at most:

```bash
python -m eunoe conform --list
python -m eunoe conform --harness claude-cli
python -m eunoe conform --harness codex
python -m eunoe conform --harness agy
python -m eunoe conform --harness ollama --model <a model you have>
python -m eunoe conform --harness all          # everything that cannot spend API credit
```

Exit 0 means it conforms, 1 that a required step failed, and 2 that it can't be reached, with the
reason. Results from this repo's own runs are in
[`docs/harness-conformance.md`](docs/harness-conformance.md).

## What it does

An **event** is one thing that happened: a tool call with its result, a message, a loaded file. Each one
has a stable id and an untouched copy in an archive, and is held at a level:

| Level | What stays | How it is made |
|---|---|---|
| L0 Verbatim | Everything | As it arrived |
| L1 Trimmed | Head and tail, counts, error lines, exit code | Code |
| L2 Digest | Names, numbers, paths, decisions, outcome | Code |
| L3 Headline | One line: what, and how it came out | Code |
| L4 Gone | Nothing in context; the archive still has it | The last resort |

The **arms** (`python -m eunoe arms`): no management, compaction, a handoff restart, observation
masking and provider clearing as controls; the ladder in oldest-first order, in importance order, with an
explicit marker, and one step to a stub. LETHE and hermes-lcm run as arms through their own engines
([`docs/try-first.md`](docs/try-first.md)).

## Swap anything

Everything an experimenter changes sits behind one protocol in
[`src/eunoe/interfaces.py`](src/eunoe/interfaces.py), each with a registry and a default:

| Protocol | What it is |
|---|---|
| `Agent` | Instructions, tools, loop style |
| `Driver` | Model, provider, billing. The only thing that can spend |
| `ContextEditor` | How an edit really reaches the next call on a harness |
| `TaskSet` | Fixtures, probes, gold. A private set implements the same thing |
| `EvictionPolicy` | Given the ledger and a budget, which event goes down a level |
| `ImportanceSignal` | How much an event matters. Never reads the schedule or the cache |
| `Schedule` | When edits land. Knows TTLs and prices, never importance |
| `Renderer` | `(event, level)` to bytes: a pure function, so the bytes are stable |
| `Judge` | Deterministic gold first; a human or a model only where a run declares it |
| `CostModel` | Usage times a dated price file |

To test your own harness, subclass `eunoe.editors.base.Harness`, implement `start`, `resume` and
`replace`, and run `python -m eunoe conform --harness your.module:YourHarness`.

To run your own comparison, write a config like [`configs/example.toml`](configs/example.toml):

```bash
python -m eunoe run --config configs/example.toml --seeds 3
```

It states a cost estimate first, and a run on a real model waits for `--yes`. To read your own Claude
Code sessions, usage and a cost surface only, with no text kept:

```bash
python -m eunoe replay path/to/session.jsonl --budget 200000
```

## Nothing spends unasked

CI can never spend. A subscription harness runs with every metered key scrubbed from its environment. A
paid harness reads its key only from a variable named for this kit (`EUNOE_ANTHROPIC_API_KEY`,
`EUNOE_OPENAI_API_KEY`, `EUNOE_GEMINI_API_KEY`), needs `--allow-spend`, stops at `--spend-cap`, and writes
every call to `spend.jsonl`. With `EUNOE_OFFLINE=1` nothing leaves the machine.

Results go to `eunoe-results/`, which git ignores. A conformance report holds no path, session id or
account name.

## Licence

MIT for code, CC BY 4.0 for fixtures and docs. See [`LICENSE`](LICENSE) and [`LICENSE-DATA`](LICENSE-DATA).
Cite it with [`CITATION.cff`](CITATION.cff).
