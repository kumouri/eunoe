"""Fork ablation: step part of a held context down a level, ask, and compare with the unedited fork.

A *checkpoint* is a context frozen at one point, as a ledger of events: a generated session or a real session file.
A *condition* says which events go to which level. Each condition is written once as its own session and asked one
batch of probes in forks, so a condition costs one rewrite and a few reads. Nothing from a probe enters a session
the next condition sees, and every condition answers the same probes, so differences are paired.

`transcript` reads a Claude Code session file and writes an edited copy. `generated` makes a checkpoint from the
fixture generator. `probes` builds and scores the batches, `conditions` builds the level maps, `run` drives them
through a door, and `analysis` turns the rows into an importance table and trains S8.
"""
from eunoe.forks.conditions import Condition
from eunoe.forks.probes import Item

__all__ = ["Condition", "Item"]
