# Contributing

Git Flow: branch from `develop`, open a pull request into `develop`. `main` holds releases.

**Before pushing:**

```bash
python -m pip install .
python -m unittest discover -s tests -t .
python -m eunoe demo --quick
python -m eunoe conform
```

The tests never leave the machine, and CI can never spend.

**The kit stays stdlib.** `pyproject.toml`'s `dependencies` stays empty. An optional arm or harness gets an extra,
and a test pins the rest.

**Nothing private, ever.** This repo is public from its first commit. No message text, tool arguments, file names,
people, session ids or real transcripts, in any commit, fixture, test or doc. Fixtures are generated. Results stay
out of the tree, in `eunoe-results/`, which git ignores. Publishing an aggregate from a private run needs its
owner's review of the exact numbers first.

**A new harness** is a subclass of `eunoe.editors.base.Harness`, a row in `docs/harness-conformance.md` once it has
run, and offline tests against a recorded shape, as in `tests/test_harnesses.py`.

**A new arm** states what it is and why it is there, and its best documented configuration goes in writing before
any run compares it.
