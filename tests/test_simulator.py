"""Tests for the simulator loop + injector/backbone compatibility."""

import pytest

from gps.interface import (
    DecisionPoint,
    EngineReference,
    Game,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import InjectionKind
from gps.latent.structured import DIMENSIONS, StructuredInjector
from gps.policy.api_backbone import APIBackbone
from gps.policy.base import PolicyBackbone
from gps.policy.mock_backbone import MockBackbone
from gps.prediction import MoveDistribution, Prediction
from gps.simulator import IncompatiblePairingError, Simulator


def _dp(time_remaining=60.0, stream=None):
    return DecisionPoint(
        game=Game.CHESS,
        player_id="p",
        state="s",
        legal_actions=("m0", "m1", "m2"),
        engine_reference=EngineReference(
            candidate_values={"m0": 100.0, "m1": 50.0, "m2": 0.0},
            best_move="m0",
            best_value=100.0,
        ),
        time_signal=TimeSignal(time_remaining=time_remaining),
        recent_outcomes=stream or OutcomeStream(),
    )


def test_simulator_runs_trajectory_and_threads_latent():
    inj = StructuredInjector(kind=InjectionKind.HIDDEN)
    sim = Simulator(inj, MockBackbone())
    decisions = [_dp() for _ in range(4)]
    results = sim.run_trajectory("p", decisions)
    assert len(results) == 4
    # Probe vector has one entry per anchored dimension.
    assert len(results[0].latent_probe) == len(DIMENSIONS)


def test_no_latent_simulator_is_allowed():
    sim = Simulator(None, MockBackbone(use_latent=False))
    results = sim.run_trajectory("p", [_dp()])
    assert results[0].latent_probe is None


def test_incompatible_pairing_rejected():
    # Hidden-only injector with a verbal-only backbone -> error at construct.
    hidden_inj = StructuredInjector(kind=InjectionKind.HIDDEN)
    with pytest.raises(IncompatiblePairingError):
        Simulator(hidden_inj, APIBackbone())  # API accepts VERBAL only


def test_observations_must_align():
    sim = Simulator(None, MockBackbone(use_latent=False))
    with pytest.raises(ValueError):
        sim.run_trajectory("p", [_dp(), _dp()], observations=[])


def test_latent_changes_predictions_under_degradation():
    """Injecting a 'degraded' latent must move the move distribution.

    This is the property Phase-0 leans on: the latent is not inert.
    """
    backbone = MockBackbone(use_latent=True)
    dp = _dp(time_remaining=2.0)  # time pressure
    # Build a degraded vs. neutral hidden injection by hand.
    from gps.latent.base import Injection

    neutral = Injection(kind=InjectionKind.HIDDEN, vector=[0, 0, 0, 0])
    degraded = Injection(
        kind=InjectionKind.HIDDEN, vector=[1.0, 1.0, 1.0, 0.0]
    )
    p_neutral = backbone.predict(dp, neutral).moves.probs["m0"]
    p_degraded = backbone.predict(dp, degraded).moves.probs["m0"]
    # Under degradation the policy is noisier -> less mass on the best move.
    assert p_degraded < p_neutral


def test_custom_backbone_subclass_contract():
    class Const(PolicyBackbone):
        accepts = ()

        def predict(self, dp, injection=None):
            return Prediction(
                moves=MoveDistribution(
                    probs=dict.fromkeys(dp.legal_actions, 1.0)
                )
            )

    sim = Simulator(None, Const())
    out = sim.run_trajectory("p", [_dp()])
    assert abs(sum(out[0].prediction.moves.probs.values()) - 1.0) < 1e-9
