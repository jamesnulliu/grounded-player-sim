"""Tests for the shared decision-point interface + prediction objects."""

import math

import pytest

from gps.interface import (
    DecisionPoint,
    EngineReference,
    Game,
    Outcome,
    OutcomeStream,
    TimeSignal,
)
from gps.prediction import MoveDistribution, TimingPrediction


def test_engine_reference_loss():
    ref = EngineReference(
        candidate_values={"a": 100.0, "b": 60.0, "c": 100.0},
        best_value=100.0,
        best_move="a",
    )
    assert ref.loss_of("a") == 0.0
    assert ref.loss_of("b") == 40.0
    assert ref.loss_of("missing") is None


def test_outcome_stream_recent_win_rate():
    s = OutcomeStream(
        recent=[
            Outcome(won=True),
            Outcome(won=False),
            Outcome(won=True),
        ]
    )
    assert s.recent_win_rate(k=2) == 0.5
    assert s.last().won is True
    assert OutcomeStream().recent_win_rate() is None


def test_time_signal_in_time_trouble():
    assert TimeSignal(time_remaining=5.0).in_time_trouble is True
    assert TimeSignal(time_remaining=120.0).in_time_trouble is False
    assert TimeSignal(time_remaining=None).in_time_trouble is False


def test_decision_point_requires_legal_actions():
    with pytest.raises(ValueError):
        DecisionPoint(
            game=Game.CHESS,
            player_id="x",
            state="s",
            legal_actions=(),
            engine_reference=None,
            time_signal=TimeSignal(),
            recent_outcomes=OutcomeStream(),
        )


def test_move_distribution_normalises_and_floors():
    d = MoveDistribution(probs={"a": 2.0, "b": 2.0})
    assert math.isclose(d.probs["a"], 0.5)
    assert d.prob_of("missing") == 1e-9
    assert d.argmax() in {"a", "b"}
    assert len(d.top_k(1)) == 1


def test_timing_logpdf_is_finite_and_peaks_near_median():
    tp = TimingPrediction(mu=math.log(10.0), sigma=0.5)
    assert math.isclose(tp.median_seconds, 10.0, rel_tol=1e-6)
    near = tp.logpdf(10.0)
    far = tp.logpdf(0.5)
    assert near > far
    assert math.isfinite(tp.logpdf(0.0))  # guarded non-positive input
