"""State-recovery probes (proposal RQ2 / Phase-0 P0.1).

Question: does the learned latent ``z_t`` recover known behavioural
phenomena -- time-pressure degradation, post-loss tilt, fatigue -- per
individual? We answer it by fitting a *linear* probe from the model's latent
snapshots to an engineered indicator (or, in Phase-0, to the known hidden
mechanism), and reporting how well it predicts.

Caveat baked into the interface (see proposal critique): a probe recovering
an indicator shows the information is *present* in z_t, not that the policy
*uses* it. Causal/intervention checks (clamp a dimension, measure prediction
change) belong elsewhere; this module measures presence only and says so.

Linear least squares via the normal equations, plain Python, no numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class StateRecoveryResult:
    """Fit quality of a linear probe latent -> target indicator."""

    n: int
    r2: float  # coefficient of determination on the fit
    weights: list[float]  # probe weights (incl. bias as last entry)
    target_name: str


def state_recovery_probe(
    latents: Sequence[Sequence[float]],
    targets: Sequence[float],
    target_name: str = "indicator",
) -> StateRecoveryResult:
    """Fit ``target ~ W . latent + b`` by least squares; report R^2.

    Parameters
    ----------
    latents:
        Per-step latent snapshots (e.g. ``StepResult.latent_probe`` or the
        injector's ``probe_vector``). All rows must share a length.
    targets:
        The indicator to recover at each step (same length as ``latents``).
    """
    if len(latents) != len(targets):
        raise ValueError("latents and targets must align 1:1")
    if len(latents) < 2:
        raise ValueError("need >= 2 samples for a probe")

    dim = len(latents[0])
    if any(len(x) != dim for x in latents):
        raise ValueError("all latent rows must share a length")

    # Design matrix with bias column.
    x = [list(row) + [1.0] for row in latents]
    p = dim + 1

    # Normal equations: (X^T X) w = X^T y, solved via Gaussian elimination
    # with a tiny ridge term for numerical stability on degenerate columns.
    ridge = 1e-8
    xtx = [[0.0] * p for _ in range(p)]
    xty = [0.0] * p
    for row, t in zip(x, targets):
        for a in range(p):
            xty[a] += row[a] * t
            for b in range(p):
                xtx[a][b] += row[a] * row[b]
    for a in range(p):
        xtx[a][a] += ridge

    w = _solve(xtx, xty)

    # R^2 on the training fit (presence test, not generalisation).
    mean_t = sum(targets) / len(targets)
    ss_tot = sum((t - mean_t) ** 2 for t in targets)
    ss_res = 0.0
    for row, t in zip(x, targets):
        pred = sum(row[a] * w[a] for a in range(p))
        ss_res += (t - pred) ** 2
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return StateRecoveryResult(
        n=len(targets), r2=r2, weights=w, target_name=target_name
    )


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve ``a x = b`` via Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-12:
            continue  # singular column; leave as-is (ridge should prevent)
        piv = m[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col] / piv
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    return [
        m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)
    ]
