"""Importance signals. Lower scores go first. A signal never reads the schedule or the cache.

Oldest-first is a control, never a candidate. S4 is an oracle: it reads the future, so it runs only in
a replay and a run that asks for it has to pass `oracle=True`. S7 and S8 take their numbers from
outside: a small model's rating at write time, and a table an ablation study produced.
"""
from __future__ import annotations

from eunoe.ledger import ASSISTANT, FILE, SYSTEM_NOTE, THINKING, USER, Event, Ledger
from eunoe.rng import Rng

#: A taxonomy prior, hand-set. Reported as a default a study can replace, not as a measurement.
KIND_PRIOR = (
    (USER, 1.0), (ASSISTANT, 0.7), ("tool:write", 0.6), ("tool:apply", 0.6), (FILE, 0.2),
    ("tool:read", 0.2), (SYSTEM_NOTE, 0.15), ("tool:", 0.5),
)


def kind_prior(kind: str) -> float:
    for prefix, value in KIND_PRIOR:
        if kind.startswith(prefix):
            return value
    return 0.3


class OldestFirst:
    """S0. A control: position alone."""

    def score(self, event: Event, ledger: Ledger, now: float) -> float:
        return float(event.index)


class RandomOrder:
    """S1. A control: one seeded draw per event, stable for the run."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def score(self, event: Event, ledger: Ledger, now: float) -> float:
        return Rng(self.seed, event.id).random()


class LargestFirst:
    """S2. By size, largest going first."""

    def score(self, event: Event, ledger: Ledger, now: float) -> float:
        return -float(ledger.tokens_at(event, 0))


class KindPrior:
    """S3. A taxonomy of event kinds."""

    def score(self, event: Event, ledger: Ledger, now: float) -> float:
        return kind_prior(event.kind)


class Hindsight:
    """S4. Whether later turns used it. An oracle ceiling, never deployable."""

    def __init__(self, future_uses: dict, oracle: bool = False) -> None:
        if not oracle:
            raise ValueError("S4 reads the future: it runs only in a replay, with oracle=True")
        self.future_uses = future_uses

    def score(self, event: Event, ledger: Ledger, now: float) -> float:
        return float(sum(1 for call in self.future_uses.get(event.id, ()) if call > ledger.call))


class AccessCount:
    """S5. Lookups so far."""

    def score(self, event: Event, ledger: Ledger, now: float) -> float:
        return float(ledger.lookups[event.id])


class ReferenceDensity:
    """S6. Later events naming its identifiers, so far."""

    def score(self, event: Event, ledger: Ledger, now: float) -> float:
        return float(ledger.reference_count(event))


class ModelRating:
    """S7. A small model's rating at write time, with its own cost line. `ratings` maps id to score."""

    def __init__(self, ratings: dict | None = None, default: float = 0.5) -> None:
        self.ratings = ratings if ratings is not None else {}
        self.default = default

    def score(self, event: Event, ledger: Ledger, now: float) -> float:
        return float(self.ratings.get(event.id, self.default))


class AblationEstimate:
    """S8. A table from a counterfactual ablation study, keyed by kind and age band."""

    def __init__(self, table: dict | None = None, default: float = 0.5) -> None:
        self.table = table if table is not None else {}
        self.default = default

    def score(self, event: Event, ledger: Ledger, now: float) -> float:
        band = age_band(ledger.call - event.call)
        return float(self.table.get((event.kind, band), self.table.get(event.kind, self.default)))


class FrequencyWeight:
    """S9. Frequency times weight, so a rare and heavy event survives. A public stand-in."""

    def score(self, event: Event, ledger: Ledger, now: float) -> float:
        return (1 + ledger.reference_count(event)) * kind_prior(event.kind)


def age_band(calls_old: int) -> str:
    if calls_old < 10:
        return "0-9"
    if calls_old < 50:
        return "10-49"
    if calls_old < 200:
        return "50-199"
    return "200+"


SIGNALS = {
    "S0": OldestFirst, "S1": RandomOrder, "S2": LargestFirst, "S3": KindPrior, "S4": Hindsight,
    "S5": AccessCount, "S6": ReferenceDensity, "S7": ModelRating, "S8": AblationEstimate, "S9": FrequencyWeight,
}


def get_signal(name: str, **kwargs):
    try:
        return SIGNALS[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown signal {name!r}, one of {sorted(SIGNALS)}") from None


__all__ = ["SIGNALS", "get_signal", "kind_prior", "age_band", "THINKING"]
