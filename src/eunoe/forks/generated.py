"""A generated session as a checkpoint: every event held, every planted fact asked at the end, gold by construction.

The fixture's own probes are spaced through a session so an arm has to survive to them. Here the session is frozen at
its end instead, so one rewrite answers all of them. A fact a later line superseded is asked for its newest value, and
answering with the old one scores as stale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from eunoe.forks.probes import Item
from eunoe.ledger import THINKING, Ledger
from eunoe.renderer import DefaultRenderer
from eunoe.signals import age_band
from eunoe.tasks.generator import DOMAINS, generate_session, get_profile

TASK = "Answer each question from this session."
#: Sized so a fork fits a sensible share of a plan: about 120k real tokens before anything is evicted.
FORK_PROFILE = dict(steps=300, mean_result_chars=2000)


def _size_band(tokens: int) -> str:
    if tokens < 100:
        return "under 100"
    if tokens < 1000:
        return "100-999"
    if tokens < 5000:
        return "1k-5k"
    return "5k+"


@dataclass
class GeneratedCheckpoint:
    id: str
    ledger: Ledger
    system: str = ""
    items: list = field(default_factory=list)
    frozen_upto: int = -1

    def label(self, event) -> str:
        return event.kind


def build(domain: str, seed: int, profile: str = "standard", marker: str = "M0", **overrides) -> GeneratedCheckpoint:
    overrides = {**FORK_PROFILE, **overrides}
    session = generate_session(domain, seed, get_profile(profile, **overrides))
    ledger = Ledger(renderer=DefaultRenderer(marker=marker))
    for step in session.steps:
        if step.kind == "event" and step.event.kind != THINKING:
            spec = step.event
            ledger.call += 1
            ledger.append(spec.kind, spec.text, input=spec.input, source=spec.source)
    cp = GeneratedCheckpoint(f"{domain}-{seed}", ledger, DOMAINS[domain].brief)
    for fact in session.facts:
        sources = [e for e in ledger.events if fact.value in e.text]
        newest = sources[-1] if sources else None
        tags = dict(fact_kind=fact.kind, sources=len(sources))
        if newest is not None:
            tags.update(label=newest.kind, band=age_band(ledger.call - newest.call),
                        size=_size_band(ledger.tokens_at(newest, 0)))
        cp.items.append(Item(f"f{len(cp.items):03d}", f"What is the {fact.key}?", fact.value, tuple(fact.stale), fact.kind,
                             [e.id for e in sources], tags))
    for probe in (p for p in session.probes if not p.gold):
        cp.items.append(Item(probe.id, f"What is the {probe.key}?", "", (), "unanswerable", [], {"fact_kind": "unanswerable"}))
    return cp


__all__ = ["FORK_PROFILE", "GeneratedCheckpoint", "TASK", "build"]
