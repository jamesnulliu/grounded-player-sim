"""A tiny seeded PRNG with no third-party dependency.

We avoid ``numpy`` here so the Phase-0 / eval path runs on a bare Python.
We avoid the global ``random`` module's *shared* state so that experiments
are reproducible regardless of import order: every stochastic component owns
its own :class:`LCG` seeded explicitly.

This is a standard 64-bit linear congruential generator (the constants are
Knuth's MMIX). It is *not* cryptographic and not meant to be -- it only has
to be deterministic and decently uniform for simulation.
"""

from __future__ import annotations


class LCG:
    """Deterministic linear congruential generator."""

    _A = 6364136223846793005
    _C = 1442695040888963407
    _MASK = (1 << 64) - 1

    def __init__(self, seed: int = 0) -> None:
        self._state = seed & self._MASK

    def _next_u64(self) -> int:
        self._state = (self._state * self._A + self._C) & self._MASK
        return self._state

    def random(self) -> float:
        """Float in [0, 1)."""
        # Use the top 53 bits for a double-precision fraction.
        return (self._next_u64() >> 11) / float(1 << 53)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.random()

    def randint(self, lo: int, hi: int) -> int:
        """Integer in [lo, hi] inclusive."""
        if hi < lo:
            raise ValueError("randint: hi < lo")
        span = hi - lo + 1
        return lo + (self._next_u64() % span)

    def choice(self, seq):
        return seq[self.randint(0, len(seq) - 1)]

    def categorical(self, weights: list[float]) -> int:
        """Sample an index from non-negative, unnormalised ``weights``."""
        total = sum(weights)
        if total <= 0:
            raise ValueError("categorical: weights sum to <= 0")
        r = self.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r < acc:
                return i
        return len(weights) - 1
