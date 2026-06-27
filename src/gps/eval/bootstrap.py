"""Bootstrap confidence intervals over independent units (e.g. players).

The Milestone-A decision rule (``documents/milestone_a.md`` section 5) is
explicit: significance must be assessed by **bootstrapping over players, not
over moves** -- moves within a player are correlated, so resampling moves
would badly understate the uncertainty. This module is the reusable tool for
that: give it one value per independent unit (a per-player ``D - B``, a
per-player metric, ...) and it returns a percentile confidence interval on the
statistic plus the fraction of resamples on each side of zero.

numpy-only (already a base dependency); deterministic given ``seed`` so a
reported CI replays exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class BootstrapCI:
    """A bootstrap estimate of a statistic over independent units."""

    point: float  # statistic on the observed sample
    low: float  # lower percentile bound
    high: float  # upper percentile bound
    p_below_zero: float  # fraction of bootstrap statistics < 0
    n_units: int  # number of independent units resampled
    confidence: float  # e.g. 0.95

    @property
    def excludes_zero(self) -> bool:
        """True iff the whole CI is on one side of 0 (a 'significant' sign)."""
        return self.high < 0.0 or self.low > 0.0


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the *mean* of ``values``.

    ``values`` is one number per independent unit (e.g. per player). Resamples
    the units with replacement ``n_resamples`` times, takes the mean of each
    resample, and reports the central ``confidence`` percentile interval. Also
    reports ``p_below_zero`` -- the share of resample means below 0 -- a handy
    one-sided read on a signed effect like ``D - B``.
    """
    import numpy as np

    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    # [n_resamples, n] indices, then mean over the unit axis.
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(
        point=point,
        low=float(low),
        high=float(high),
        p_below_zero=float((means < 0).mean()),
        n_units=int(n),
        confidence=confidence,
    )
