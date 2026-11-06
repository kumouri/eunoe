"""Paired bootstrap confidence intervals and the Holm correction, stdlib and seeded."""
from __future__ import annotations

from eunoe.rng import Rng

RESAMPLES = 10_000
SEED = 20261105


def paired_bootstrap(treatment: list, control: list, resamples: int = RESAMPLES, seed: int = SEED,
                     alpha: float = 0.05) -> dict:
    """The mean paired difference and its percentile interval, resampling seeds. Deterministic."""
    if len(treatment) != len(control):
        raise ValueError("a paired comparison needs one value per seed on each side")
    diffs = [t - c for t, c in zip(treatment, control)]
    if not diffs:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    rng = Rng("bootstrap", seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randint(0, n - 1)] for _ in range(n)) / n for _ in range(resamples))
    return {"mean": sum(diffs) / n, "low": means[int(alpha / 2 * resamples)],
            "high": means[min(resamples - 1, int((1 - alpha / 2) * resamples))], "n": n}


def holm(pvalues: dict, alpha: float = 0.05) -> dict:
    """Which hypotheses survive, by name: step down through the sorted p values and stop at the first miss."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    out, still = {}, True
    for rank, (name, p) in enumerate(ordered):
        still = still and p <= alpha / (len(ordered) - rank)
        out[name] = still
    return out
