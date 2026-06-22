"""End-to-end Phase-0 tests: the falsifiable core.

These assert the *direction* of the central claims under known ground truth
(proposal section 5, P0.1-P0.2): the dynamic latent should help where
dynamics exist, and the latent snapshot should carry recoverable signal
about the injected mechanism.
"""

import pytest

from gps.experiments.phase0 import run_phase0
from gps.latent.base import InjectionKind
from gps.synthetic.players import TiltPlayer
from gps.synthetic.toy_game import ToyGame


@pytest.mark.parametrize("kind", ["tilt", "time_pressure", "fatigue"])
def test_phase0_mechanism_actually_fires(kind):
    # Guard against the degenerate case (a player who never triggers the
    # mechanism), which would make every downstream claim vacuous.
    res = run_phase0(player_kind=kind, n_games=24, seed=0)
    assert res.mechanism_fired_frac > 0.1


@pytest.mark.parametrize("kind", ["tilt", "time_pressure", "fatigue"])
def test_phase0_oracle_beats_static(kind):
    # P0.2: a model that *knows* the true dynamic state must beat the static
    # one wherever the mechanism fires. This is the non-circular check that
    # dynamics carry signal the eval can see.
    res = run_phase0(player_kind=kind, n_games=24, seed=0)
    assert res.oracle.nll < res.static.nll


def test_phase0_state_recovery_has_signal():
    # P0.1: the structured latent should linearly carry degradation info.
    res = run_phase0(player_kind="time_pressure", n_games=30, seed=2)
    assert res.recovery.r2 > 0.2


def test_phase0_verbal_and_hidden_both_run():
    for kind in (InjectionKind.VERBAL, InjectionKind.HIDDEN):
        res = run_phase0("tilt", n_games=16, injector_kind=kind, seed=3)
        assert res.static.n > 0 and res.oracle.n > 0


def test_synthetic_session_is_deterministic():
    g1 = ToyGame(seed=7)
    g2 = ToyGame(seed=7)
    s1 = TiltPlayer("p", g1, seed=7).play_session(5)
    s2 = TiltPlayer("p", g2, seed=7).play_session(5)
    moves1 = [o.move for g in s1 for o in g.observations]
    moves2 = [o.move for g in s2 for o in g.observations]
    assert moves1 == moves2  # reproducible without wall-clock randomness
