"""A seeded generator that gives the same stream on every Python version and platform.

Built on SHA-256 rather than `random.Random`, because a fixture's bytes are part of what a run's
manifest pins, and a change in a standard-library method would silently change the fixture.
"""
from __future__ import annotations

import hashlib


class Rng:
    def __init__(self, *seed) -> None:
        self._seed = "\x00".join(str(s) for s in seed).encode("utf-8")
        self._counter = 0

    def _next(self) -> int:
        self._counter += 1
        digest = hashlib.sha256(self._seed + self._counter.to_bytes(8, "big")).digest()
        return int.from_bytes(digest[:8], "big")

    def random(self) -> float:
        return self._next() / 2 ** 64

    def randint(self, low: int, high: int) -> int:
        """`low` to `high`, both included."""
        return low + self._next() % (high - low + 1)

    def choice(self, items):
        return items[self._next() % len(items)]

    def shuffle(self, items: list) -> list:
        for i in range(len(items) - 1, 0, -1):
            j = self._next() % (i + 1)
            items[i], items[j] = items[j], items[i]
        return items

    def child(self, *name) -> "Rng":
        return Rng(self._seed, *name)
