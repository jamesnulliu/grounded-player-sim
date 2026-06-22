"""Move- and timing-prediction metrics (proposal section 5, Phase 4).

All metrics consume the simulator's :class:`~gps.simulator.StepResult`s
paired with the ground-truth :class:`~gps.latent.base.Observation`s, so the
same eval works for any backbone/injector combination.

Implemented in plain Python (no numpy) to keep the eval path dependency-free.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from gps.latent.base import Observation
from gps.simulator import StepResult


@dataclass
class MoveMetrics:
    """Likelihood/calibration summary for move prediction."""

    n: int
    nll: float  # mean negative log-likelihood of the played move (nats)
    brier: float  # multiclass Brier over legal moves
    ece: float  # expected calibration error of the top move
    top1_acc: float  # argmax agreement with the human move
    top3_acc: float  # human move in top-3
    perplexity: float


@dataclass
class TimingMetrics:
    """Likelihood/correlation summary for think-time prediction."""

    n: int
    nll: float  # mean NLL of observed think-time (nats)
    spearman: float  # rank corr of predicted-median vs. actual (per-player)


def move_metrics(
    results: Sequence[StepResult],
    observations: Sequence[Observation],
    n_bins: int = 10,
) -> MoveMetrics:
    """Compute move-prediction metrics over aligned results/observations."""
    if len(results) != len(observations):
        raise ValueError("results and observations must align 1:1")
    if not results:
        raise ValueError("no results to score")

    nll_sum = 0.0
    brier_sum = 0.0
    top1 = 0
    top3 = 0
    conf_correct: list[tuple[float, bool]] = []

    for r, obs in zip(results, observations):
        dist = r.prediction.moves
        played = obs.move

        nll_sum += -dist.logprob_of(played)

        # Multiclass Brier over the legal set: sum (p_i - 1{i==played})^2.
        b = 0.0
        for m in r.decision.legal_actions:
            p = dist.probs.get(m, 0.0)
            target = 1.0 if m == played else 0.0
            b += (p - target) ** 2
        brier_sum += b

        ranked = [m for m, _ in dist.top_k(3)]
        if ranked and ranked[0] == played:
            top1 += 1
        if played in ranked:
            top3 += 1

        top_move, top_p = dist.top_k(1)[0]
        conf_correct.append((top_p, top_move == played))

    n = len(results)
    mean_nll = nll_sum / n
    return MoveMetrics(
        n=n,
        nll=mean_nll,
        brier=brier_sum / n,
        ece=expected_calibration_error(conf_correct, n_bins),
        top1_acc=top1 / n,
        top3_acc=top3 / n,
        perplexity=math.exp(mean_nll),
    )


def expected_calibration_error(
    conf_correct: Sequence[tuple[float, bool]], n_bins: int = 10
) -> float:
    """ECE of the top-1 prediction.

    ``conf_correct`` is a list of ``(confidence, was_correct)``. Standard
    equal-width binning over [0,1]; returns the sample-weighted gap between
    confidence and accuracy.
    """
    if not conf_correct:
        return 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in conf_correct:
        idx = min(n_bins - 1, int(conf * n_bins))
        bins[idx].append((conf, correct))

    total = len(conf_correct)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        acc = sum(1 for _, ok in b if ok) / len(b)
        ece += (len(b) / total) * abs(avg_conf - acc)
    return ece


def timing_metrics(
    results: Sequence[StepResult],
    observations: Sequence[Observation],
) -> TimingMetrics:
    """Compute think-time NLL and per-player rank correlation.

    Skips steps lacking a timing prediction or an observed time.
    """
    pred_median: list[float] = []
    actual: list[float] = []
    nll_sum = 0.0
    n = 0
    for r, obs in zip(results, observations):
        tp = r.prediction.timing
        if tp is None or obs.time_spent is None:
            continue
        nll_sum += -tp.logpdf(obs.time_spent)
        pred_median.append(tp.median_seconds)
        actual.append(obs.time_spent)
        n += 1

    if n == 0:
        return TimingMetrics(n=0, nll=float("nan"), spearman=float("nan"))
    return TimingMetrics(
        n=n,
        nll=nll_sum / n,
        spearman=_spearman(pred_median, actual),
    )


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation (ties via average ranks)."""
    if len(xs) < 2:
        return float("nan")
    rx = _rank(xs)
    ry = _rank(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _rank(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based), handling ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average of 1-based ranks i..j
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks
