"""Tests for session segmentation + CPU-only parts of GPU backbones.

The GPU backbones' ``predict`` needs hardware, but their prompt/message
construction is pure-Python and must stay correct -- so we test those, and
assert the heavy paths fail with an *informative* error rather than silently.
"""

import pytest

from gps.data.sessions import segment_sessions
from gps.interface import (
    DecisionPoint,
    Game,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import Injection, InjectionKind
from gps.policy.api_backbone import APIBackbone
from gps.policy.sglang_backbone import SGLangBackbone


def _dp():
    return DecisionPoint(
        game=Game.CHESS,
        player_id="magnus",
        state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        legal_actions=("e2e4", "d2d4", "g1f3"),
        engine_reference=None,
        time_signal=TimeSignal(time_remaining=42.0),
        recent_outcomes=OutcomeStream(),
    )


def test_segment_sessions_splits_on_gap():
    # Two games close together, then a 1-hour gap, then one more.
    games = [(0.0, 100.0), (200.0, 300.0), (4000.0, 4100.0)]
    sessions = segment_sessions(games, gap_threshold_seconds=1800.0)
    assert sessions == [[0, 1], [2]]


def test_segment_sessions_empty():
    assert segment_sessions([]) == []


def test_segment_sessions_threshold_is_tunable():
    games = [(0.0, 100.0), (200.0, 300.0)]
    # Tiny threshold -> every gap splits.
    assert segment_sessions(games, gap_threshold_seconds=1.0) == [[0], [1]]


def test_sglang_prompt_includes_legal_moves_and_verbal_latent():
    bb = SGLangBackbone()
    inj = Injection(
        kind=InjectionKind.VERBAL, text="Current player state: tilted."
    )
    prompt = bb.build_prompt(_dp(), inj)
    assert "e2e4" in prompt and "g1f3" in prompt
    assert "tilted" in prompt
    assert "magnus" in prompt


def test_api_messages_carry_profile():
    bb = APIBackbone(provider="openai")
    inj = Injection(kind=InjectionKind.VERBAL, text="plays aggressively")
    msgs = bb.build_messages(_dp(), inj)
    assert msgs[0]["role"] == "system"
    assert "aggressively" in msgs[0]["content"]
    assert "e2e4" in msgs[1]["content"]


def test_sglang_predict_raises_informative_without_gpu():
    bb = SGLangBackbone()
    with pytest.raises((ImportError, NotImplementedError)):
        bb.predict(_dp())


def test_api_default_accepts_verbal_only():
    assert APIBackbone().accepts == (InjectionKind.VERBAL,)
