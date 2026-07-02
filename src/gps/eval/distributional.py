"""Population-level distributional metrics (Milestone F).

Generated players have **no pointwise** ground truth (an invented person has no
"true" move), so a generative population is judged *distributionally* against
the real held-out population. This module is the reusable toolbox:

* :func:`wasserstein_1d` / :func:`js_divergence` -- distance between two 1-D
  behavioral-stat distributions (per-player accuracy, blunder rate, ...).
* :func:`precision_recall` -- the generative-model precision/recall of
  Kynkaanniemi et al. (2019): **precision** = share of generated samples that
  land on the *real* manifold (plausibility), **recall** = share of real
  samples covered by the *generated* manifold (diversity/coverage). This is
  what exposes the "positive average person": a mean-collapsing model has high
  precision but ~zero recall.

Pure numpy (a base dependency); works in any dimension (the k-NN form), so the
same code serves 1-D behavioral stats now and latent-vector populations later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def wasserstein_1d(a: Sequence[float], b: Sequence[float]) -> float:
    """1-D Wasserstein-1 between two samples (equal or unequal length)."""
    import numpy as np

    sa = np.sort(np.asarray(a, float))
    sb = np.sort(np.asarray(b, float))
    n = max(len(sa), len(sb))
    # Resample both CDFs on a common grid of quantiles, then integrate |F-G|.
    q = (np.arange(n) + 0.5) / n
    ia = np.interp(q, (np.arange(len(sa)) + 0.5) / len(sa), sa)
    ib = np.interp(q, (np.arange(len(sb)) + 0.5) / len(sb), sb)
    return float(np.abs(ia - ib).mean())


def js_divergence(
    a: Sequence[float], b: Sequence[float], *, bins: int = 12
) -> float:
    """Jensen-Shannon divergence (log base 2, in [0,1]) of two 1-D samples."""
    import numpy as np

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    m = 0.5 * (pa + pb)

    def _kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


@dataclass
class PrecisionRecall:
    precision: float  # share of generated on the real manifold (plausibility)
    recall: float  # share of real covered by generated (diversity)
    k: int


def _to_2d(x):
    import numpy as np

    arr = np.asarray(x, float)
    return arr.reshape(len(arr), -1)


def _knn_radii(points, k: int):
    """k-th nearest-neighbour distance for each point (within its own set)."""
    import numpy as np

    n = len(points)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    d[np.arange(n), np.arange(n)] = np.inf  # exclude self
    kk = min(k, n - 1)
    return np.sort(d, axis=1)[:, kk - 1]


def precision_recall(
    real: Sequence, generated: Sequence, *, k: int = 3
) -> PrecisionRecall:
    """Generative precision/recall (Kynkaanniemi 2019), any dimension.

    A generated point is "precise" if it lies within the k-NN radius of some
    real point; a real point is "covered" (recall) if it lies within the k-NN
    radius of some generated point.
    """
    import numpy as np

    r = _to_2d(real)
    g = _to_2d(generated)
    if len(r) < 2 or len(g) < 2:
        return PrecisionRecall(precision=0.0, recall=0.0, k=k)
    r_rad = _knn_radii(r, k)
    g_rad = _knn_radii(g, k)
    cross = np.linalg.norm(g[:, None, :] - r[None, :, :], axis=-1)  # [G, R]
    precision = float((cross <= r_rad[None, :]).any(axis=1).mean())
    recall = float((cross.T <= g_rad[None, :]).any(axis=1).mean())
    return PrecisionRecall(precision=precision, recall=recall, k=k)
