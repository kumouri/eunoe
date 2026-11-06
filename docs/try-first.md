# Try-first arms: other people's work, attempted before building ours

Three published or shipped systems are close to this idea. Each was attempted on 2026-11-06.

**These runs prove an arm runs on the kit's fixtures. They are not comparisons.** The scripted driver is not a
model, each arm below stands in a code-built digest for its summary model, and nothing here says one is better.

## LETHE: runs as an arm

[LETHE](https://github.com/JesusGarcia9009/lethe-engineering), Unlicense, tried at commit `4f10077`
(`lethe-llm-context` 0.7.0).

It installs cleanly and its core is stdlib. Arm `lethe` (`src/eunoe/arms/lethe.py`) runs LETHE's real
`ContextManager`, `Session` and `Scheduler`: its relevance scoring picks what goes, its compactor folds cold runs
into notes, its archivist pages blocks out and leaves a stub, and its recall answers the lookup tool.

| | |
|---|---|
| What is LETHE's | Which blocks go, when, and what stub is left |
| What is not | The text of a note. Its `Compactor` asks an adapter for a summary, and the default adapter is a code-built digest, so no model call happens. Pass your own `adapter` to change that |
| Install | `pip install eunoe[lethe]`. Needs Python 3.12 or later, one above the kit's floor |
| Left out of | `eunoe demo`, because the package is not stdlib |

## LCM, through hermes-lcm: runs as an arm, with its host

[hermes-lcm](https://github.com/stephenschoettler/hermes-lcm), MIT, tried at commit `a13e709`. It implements
[LCM](https://arxiv.org/html/2605.04050v1) as a plugin for Hermes Agent, and imports the host's packages, so it
can't be a dependency of a stdlib kit.

Arm `lcm` (`src/eunoe/arms/hermes_lcm.py`) starts a worker in an environment that has both, and talks to it over
JSON lines. The worker drives the real `LCMEngine`: its summary DAG, fresh tail, thresholds and search tool.

```bash
python -m venv hermes-venv
hermes-venv/bin/pip install hermes-agent
git clone https://github.com/stephenschoettler/hermes-lcm
export EUNOE_HERMES_PYTHON=$PWD/hermes-venv/bin/python
export EUNOE_HERMES_LCM_DIR=$PWD/hermes-lcm
```

| | |
|---|---|
| What is LCM's | When it compacts, the DAG, what stays in the tail, and how a lookup searches |
| What is not | The summary text. The host's `call_llm` is replaced in the worker by a digest of identifiers, unless `EUNOE_HERMES_REAL_LLM=1` |
| Catch | Its leaf-chunk threshold defaults far above a small session, so the arm sets `LCM_LEAF_CHUNK_TOKENS` from the budget, or nothing compacts. That setting is part of the arm's configuration and has to be reported with it |

## VISTA: nothing to run yet

[VISTA](https://arxiv.org/html/2606.30005) (arXiv 2606.30005v5). The paper's full text, read on 2026-11-06, has
no code-availability statement and no repository link, and a search found no release. It stays in the design as
a citation and a comparison of designs, to be retried when the pre-registration's prior-art search re-runs.

## What happens next

Whether LETHE and LCM join the scored grid is decided at the pilot, by whether they adapt cleanly with a real
model behind their summary seams. A control that would have won under a better configuration gets reported rerun
under it, and the same goes for these.
