"""Tests for metrics, probes, and temporal splits."""

import math

import pytest

from gps.eval.metrics import (
    expected_calibration_error,
    move_metrics,
    timing_metrics,
)
from gps.eval.probes import state_recovery_probe
from gps.eval.splits import temporal_split
from gps.interface import (
    DecisionPoint,
    EngineReference,
    Game,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import Observation
from gps.prediction import MoveDistribution, Prediction, TimingPrediction
from gps.simulator import StepResult


def _step(probs, played_meta=None, timing=None, latent=None):
    dp = DecisionPoint(
        game=Game.CHESS,
        player_id="p",
        state="s",
        legal_actions=tuple(probs.keys()),
        engine_reference=EngineReference(
            candidate_values=dict.fromkeys(probs, 1.0)
        ),
        time_signal=TimeSignal(),
        recent_outcomes=OutcomeStream(),
    )
    pred = Prediction(
        moves=MoveDistribution(probs=probs),
        timing=timing,
        latent=latent,
    )
    return StepResult(decision=dp, prediction=pred, latent_probe=latent)


def test_move_metrics_perfect_prediction():
    steps = [_step({"a": 0.99, "b": 0.01}) for _ in range(5)]
    obs = [Observation(move="a") for _ in range(5)]
    m = move_metrics(steps, obs)
    assert m.top1_acc == 1.0
    assert m.top3_acc == 1.0
    assert m.nll < 0.02
    assert m.perplexity < 1.05
    assert 0.0 <= m.ece <= 1.0


def test_move_metrics_rewards_calibration():
    # Confident-and-right should beat uniform on NLL.
    confident = [_step({"a": 0.9, "b": 0.1}) for _ in range(10)]
    uniform = [_step({"a": 0.5, "b": 0.5}) for _ in range(10)]
    obs = [Observation(move="a") for _ in range(10)]
    assert move_metrics(confident, obs).nll < move_metrics(uniform, obs).nll


def test_ece_bounds_and_extremes():
    assert expected_calibration_error([]) == 0.0
    # Perfectly calibrated: confidence 1.0, always correct -> ECE 0.
    assert expected_calibration_error([(1.0, True)] * 10) == 0.0
    # Overconfident: confidence 1.0, always wrong -> ECE 1.
    assert math.isclose(expected_calibration_error([(1.0, False)] * 10), 1.0)


def test_timing_metrics_rank_correlation():
    steps = []
    obs = []
    for i in range(1, 6):
        steps.append(
            _step(
                {"a": 1.0},
                timing=TimingPrediction(mu=math.log(float(i)), sigma=0.3),
            )
        )
        obs.append(Observation(move="a", time_spent=float(i)))
    tm = timing_metrics(steps, obs)
    assert tm.n == 5
    # Predicted medians are monotone in actual times -> spearman == 1.
    assert math.isclose(tm.spearman, 1.0, rel_tol=1e-6)


def test_state_recovery_probe_recovers_linear_signal():
    # latent = [x], target = 2x + 1 -> R^2 ~ 1.
    latents = [[float(i)] for i in range(20)]
    targets = [2.0 * i + 1.0 for i in range(20)]
    res = state_recovery_probe(latents, targets, target_name="linear")
    assert res.r2 > 0.999
    assert res.n == 20


def test_temporal_split_orders_and_rejects_empty():
    s = temporal_split(list(range(10)), train_frac=0.6, val_frac=0.2)
    assert s.train == [0, 1, 2, 3, 4, 5]
    assert s.val == [6, 7]
    assert s.test == [8, 9]
    with pytest.raises(ValueError):
        temporal_split([1, 2], train_frac=0.6, val_frac=0.2)
