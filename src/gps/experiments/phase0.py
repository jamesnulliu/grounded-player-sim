"""Phase-0 experiment: do dynamics matter, and are they recoverable?

Runs entirely on CPU with the mock backbone, so the central claims are
exercisable today (proposal section 5, Phase 0). Three arms over the *same*
synthetic trajectory (deterministic given the seed), so differences are due
only to the latent:

* ``static``   -- no latent (the no-personalization / static-individual foil).
* ``heuristic``-- the hand-specified :class:`StructuredInjector`. It has no
  trained parameters, so it is NOT guaranteed to beat ``static`` -- closing
  that gap is exactly what the *trained* injector must earn. We report it,
  we do not assert it.
* ``oracle``   -- the :class:`OracleInjector`, which reads the true injected
  degradation. By construction it should beat ``static`` wherever the
  mechanism fires; this is the non-circular check that (a) dynamics carry
  predictive signal and (b) the eval can detect it (P0.2), and it
  upper-bounds the achievable gain.

Plus the state-recovery probe (P0.1): can a linear probe recover the true
degradation from the heuristic latent's snapshots?
"""

from __future__ import annotations

from dataclasses import dataclass

from gps.eval.metrics import MoveMetrics, move_metrics
from gps.eval.probes import StateRecoveryResult, state_recovery_probe
from gps.latent.base import InjectionKind, LatentStateInjector
from gps.latent.structured import OracleInjector, StructuredInjector
from gps.policy.mock_backbone import MockBackbone
from gps.simulator import Simulator
from gps.synthetic.players import SyntheticPlayer
from gps.synthetic.toy_game import ToyGame


@dataclass
class Phase0Result:
    """Outcome of one Phase-0 comparison (one synthetic mechanism)."""

    player_kind: str
    static: MoveMetrics
    heuristic: MoveMetrics
    oracle: MoveMetrics
    recovery: StateRecoveryResult
    mechanism_fired_frac: float

    @property
    def oracle_helps(self) -> bool:
        """Does knowing the true dynamic state beat the static model?"""
        return self.oracle.nll < self.static.nll

    @property
    def heuristic_helps(self) -> bool:
        """Reported, not asserted: untrained heuristic vs. static."""
        return self.heuristic.nll < self.static.nll

    def summary(self) -> str:
        return (
            f"[{self.player_kind}] "
            f"static NLL={self.static.nll:.4f} | "
            f"heuristic NLL={self.heuristic.nll:.4f} "
            f"(helps: {self.heuristic_helps}) | "
            f"oracle NLL={self.oracle.nll:.4f} "
            f"(helps: {self.oracle_helps}) | "
            f"recovery R^2={self.recovery.r2:.3f} | "
            f"mechanism fired {self.mechanism_fired_frac:.0%} of plies"
        )


def _build_session(player: SyntheticPlayer, n_games: int):
    """Generate a session as one continuous trajectory.

    Concatenating the whole session into ONE trajectory lets the latent
    state persist across games -- that is what carries cross-game tilt /
    fatigue. Per-game trajectories would reset z_t and discard the dynamics.
    """
    games = player.play_session(n_games=n_games)
    decisions = [dp for g in games for dp in g.decisions]
    observations = [o for g in games for o in g.observations]
    true_degr = [
        float(dp.context.get("true_degradation", 0.0)) for dp in decisions
    ]
    return decisions, observations, true_degr


def _run_arm(
    player_kind: str,
    n_games: int,
    seed: int,
    injector: LatentStateInjector | None,
    injector_kind: InjectionKind,
):
    """Run one arm; rebuild the player so every arm sees the same games."""
    game = ToyGame(branching=5, plies=20, seed=seed)
    player = _make_player(player_kind, game, seed)
    decisions, observations, true_degr = _build_session(player, n_games)

    backbone = MockBackbone(use_latent=injector is not None)
    sim = Simulator(injector, backbone)
    results = sim.run_trajectory(player.player_id, decisions, observations)
    return results, observations, true_degr


def run_phase0(
    player_kind: str = "tilt",
    n_games: int = 24,
    seed: int = 0,
    injector_kind: InjectionKind = InjectionKind.HIDDEN,
) -> Phase0Result:
    """Run the three-arm Phase-0 comparison for one mechanism."""
    static_res, static_obs, true_degr = _run_arm(
        player_kind, n_games, seed, None, injector_kind
    )
    heur_res, heur_obs, _ = _run_arm(
        player_kind,
        n_games,
        seed,
        StructuredInjector(kind=injector_kind),
        injector_kind,
    )
    orac_res, orac_obs, _ = _run_arm(
        player_kind,
        n_games,
        seed,
        OracleInjector(kind=injector_kind),
        injector_kind,
    )

    # RQ2 / P0.1 probe: recover true degradation from the heuristic latent.
    latents = [r.latent_probe or [0.0] for r in heur_res]
    recovery = state_recovery_probe(
        latents, true_degr, target_name=f"{player_kind}-degradation"
    )

    fired = sum(1 for d in true_degr if d > 0) / max(1, len(true_degr))

    return Phase0Result(
        player_kind=player_kind,
        static=move_metrics(static_res, static_obs),
        heuristic=move_metrics(heur_res, heur_obs),
        oracle=move_metrics(orac_res, orac_obs),
        recovery=recovery,
        mechanism_fired_frac=fired,
    )


def _make_player(kind: str, game: ToyGame, seed: int) -> SyntheticPlayer:
    from gps.synthetic.players import (
        FatiguePlayer,
        TiltPlayer,
        TimePressurePlayer,
    )

    pid = f"synthetic-{kind}"
    # base_beta=4 gives a strong-but-imperfect player who loses often enough
    # for the post-loss mechanism to fire.
    if kind == "tilt":
        return TiltPlayer(pid, game, seed=seed, base_beta=4.0)
    if kind == "time_pressure":
        return TimePressurePlayer(pid, game, seed=seed, base_beta=4.0)
    if kind == "fatigue":
        return FatiguePlayer(pid, game, seed=seed, base_beta=4.0)
    raise ValueError(f"unknown player kind: {kind}")
