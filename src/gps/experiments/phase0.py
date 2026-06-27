"""Phase-0 experiment: do dynamics matter, and are they recoverable?

Runs entirely on CPU with the mock backbone, so the central claims are
exercisable today (proposal section 5, Phase 0). All arms run over the *same*
synthetic trajectory (deterministic given the seed), so differences are due
only to the injector:

* ``static``    -- no latent (the no-personalization / static-individual
  foil).
* ``history``   -- the :class:`HistoryConditionedInjector`: the *same*
  engineered history features as ``heuristic``, but **memoryless** (no
  evolving latent). This is the Milestone-A control that answers the #1
  desk-reject objection: "isn't the dynamic latent just an expressive
  history-conditioned policy?" The load-bearing comparison is
  ``heuristic`` vs. ``history`` at equal inputs (E-A1).
* ``heuristic`` -- the hand-specified :class:`StructuredInjector`, which
  *accumulates* those same features into an evolving ``z_t``. It has no
  trained parameters, so it is NOT guaranteed to beat ``static`` or
  ``history`` -- closing that gap is exactly what the *trained* injector must
  earn. We report it, we do not assert it.
* ``oracle``    -- the :class:`OracleInjector`, which reads the true injected
  degradation. By construction it should beat ``static`` wherever the
  mechanism fires; this is the non-circular check that (a) dynamics carry
  predictive signal and (b) the eval can detect it (P0.2), and it
  upper-bounds the achievable gain.

Plus the state-recovery probe (P0.1): can a linear probe recover the true
degradation from the heuristic latent's snapshots?

Caveat (recorded so it is not over-read): the CPU ``history`` vs.
``heuristic`` contrast is *parameter-free* -- both share the mock backbone,
so it isolates "accumulated vs. instantaneous features" but NOT capacity. The
capacity-matched version of this comparison (E-C2) trains
:class:`~gps.policy.history_conditioned.HistoryConditionedBackbone` against
the neural injector on GPU; see ``documents/milestone_a.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from gps.eval.metrics import MoveMetrics, move_metrics
from gps.eval.probes import StateRecoveryResult, state_recovery_probe
from gps.latent.base import InjectionKind, LatentStateInjector
from gps.latent.structured import (
    HistoryConditionedInjector,
    OracleInjector,
    StructuredInjector,
)
from gps.policy.mock_backbone import MockBackbone
from gps.simulator import Simulator
from gps.synthetic.players import SyntheticPlayer
from gps.synthetic.toy_game import ToyGame


@dataclass
class Phase0Result:
    """Outcome of one Phase-0 comparison (one synthetic mechanism)."""

    player_kind: str
    static: MoveMetrics
    history: MoveMetrics
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

    @property
    def dynamic_beats_history(self) -> bool:
        """The decisive Milestone-A direction: does *accumulating* the same
        features into an evolving latent beat consuming them memorylessly?

        Reported, not asserted for the untrained heuristic -- the trained
        injector is what must win this at equal capacity (E-C2). A False here
        for the heuristic is informative, not a failure.
        """
        return self.heuristic.nll < self.history.nll

    def summary(self) -> str:
        return (
            f"[{self.player_kind}] "
            f"static NLL={self.static.nll:.4f} | "
            f"history NLL={self.history.nll:.4f} | "
            f"heuristic NLL={self.heuristic.nll:.4f} "
            f"(>history: {self.dynamic_beats_history}) | "
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
    """Run the four-arm Phase-0 comparison for one mechanism."""
    static_res, static_obs, true_degr = _run_arm(
        player_kind, n_games, seed, None, injector_kind
    )
    hist_res, hist_obs, _ = _run_arm(
        player_kind,
        n_games,
        seed,
        HistoryConditionedInjector(kind=injector_kind),
        injector_kind,
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
        history=move_metrics(hist_res, hist_obs),
        heuristic=move_metrics(heur_res, heur_obs),
        oracle=move_metrics(orac_res, orac_obs),
        recovery=recovery,
        mechanism_fired_frac=fired,
    )


def _make_player(kind: str, game: ToyGame, seed: int) -> SyntheticPlayer:
    from gps.synthetic.players import (
        FatiguePlayer,
        HysteresisTiltPlayer,
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
    if kind == "hysteresis":
        # The Milestone-A mechanism: a hidden leaky integral of losses that
        # history_features cannot reconstruct (see the class docstring and
        # documents/milestone_a.md section 6).
        return HysteresisTiltPlayer(pid, game, seed=seed, base_beta=4.0)
    raise ValueError(f"unknown player kind: {kind}")
