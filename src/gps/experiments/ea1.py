"""E-A1: does a *trained* evolving latent beat the memoryless control?

This is the trained, GPU-capable form of the Milestone-A test (the CPU Phase-0
arms only compare an *untrained* heuristic). It pits two injectors that are
**identical in architecture, parameters, and inputs** and differ in exactly one
bit -- whether the recurrence carries state across steps:

* **arm D** (proposed): :class:`NeuralInjector` with ``persist=True`` -- the
  evolving latent.
* **arm B** (control): the *same* :class:`NeuralInjector` with
  ``persist=False`` -- the hidden state is zeroed every step, so it is
  memoryless at identical capacity.

Both are trained by SFT against the same synthetic players and read the same
:func:`history_features`; both feed the same differentiable head
(:class:`DiffMovePolicy`). The headline number is **D - B** on a strict
temporal split: train on each player's earlier games, score move-NLL on their
*later* games (RQ3 in miniature). The mechanism is
:class:`HysteresisTiltPlayer`, whose hidden leaky-loss accumulator a memoryless
reader provably cannot reconstruct -- so this is where the evolving latent
*should* win; if it cannot win here it never will (``milestone_a.md`` §5-6).

Resources: tiny models; trains in seconds on the RTX 4060 (or CPU). Each arm
is a tracked training run (mandatory W&B), so run with ``WANDB_API_KEY`` set.
"""

from __future__ import annotations

from dataclasses import dataclass

from gps.eval.bootstrap import BootstrapCI, bootstrap_ci
from gps.latent.base import InjectionKind
from gps.latent.neural import NeuralInjector
from gps.policy.diff_policy import DiffMovePolicy
from gps.synthetic.players import HysteresisTiltPlayer
from gps.synthetic.toy_game import ToyGame
from gps.train.base import TrainConfig, Trajectory, TrajectoryDataset
from gps.train.sft import EvalSpec, SFTTrainer


@dataclass
class EA1Result:
    """Outcome of one E-A1 run: the D-vs-B comparison on the future split.

    Carries the *per-player* held-out move-NLLs (paired: same players, same
    games) and a bootstrap CI over players on ``D - B`` -- the significance
    read the Milestone-A decision rule (``milestone_a.md`` section 5) requires.
    """

    d_per_player: list[float]
    b_per_player: list[float]
    diff_per_player: list[float]  # D - B, one per player
    ci: BootstrapCI
    d_params: int
    b_params: int
    d_summary: dict
    b_summary: dict

    @property
    def d_val_move_nll(self) -> float:
        return sum(self.d_per_player) / len(self.d_per_player)

    @property
    def b_val_move_nll(self) -> float:
        return sum(self.b_per_player) / len(self.b_per_player)

    @property
    def d_minus_b(self) -> float:
        """Mean per-player D - B; negative means the evolving latent predicts
        the future better than the memoryless control."""
        return self.ci.point

    @property
    def frac_players_d_wins(self) -> float:
        n = len(self.diff_per_player)
        return sum(1 for d in self.diff_per_player if d < 0) / n

    def verdict(self) -> str:
        if self.ci.point >= 0:
            return "NO WIN (memoryless ties/beats)"
        if self.ci.high < 0:
            return "SIGNIFICANT WIN (95% CI excludes 0)"
        return "positive but NOT significant (CI includes 0)"

    def summary(self) -> str:
        cap = ""
        if self.b_params != self.d_params:
            cap = f" [B={self.b_params / self.d_params:.0f}x D capacity]"
        return (
            f"[E-A1] held-out move-NLL, n_players={self.ci.n_units} | "
            f"D (persist)={self.d_val_move_nll:.4f} ({self.d_params}p) | "
            f"B (memoryless)={self.b_val_move_nll:.4f} "
            f"({self.b_params}p){cap}\n"
            f"        per-player D-B: mean={self.ci.point:+.4f} "
            f"95% CI [{self.ci.low:+.4f}, {self.ci.high:+.4f}] | "
            f"D wins {self.frac_players_d_wins:.0%} of players | "
            f"P(D-B<0)={self.ci.p_below_zero:.2f}\n"
            f"        VERDICT: {self.verdict()}"
        )


def _build_trajectories(n_players: int, n_games: int, base_seed: int):
    """One full-session trajectory per synthetic hysteresis player."""
    full: list[Trajectory] = []
    plies = None
    n_actions = None
    for i in range(n_players):
        seed = base_seed + i
        game = ToyGame(seed=seed)
        plies = game.plies
        n_actions = game.branching
        player = HysteresisTiltPlayer(
            f"hysteresis-{i}", game, seed=seed, base_beta=4.0
        )
        games = player.play_session(n_games=n_games)
        decisions = [dp for g in games for dp in g.decisions]
        observations = [o for g in games for o in g.observations]
        full.append(Trajectory(player.player_id, decisions, observations))
    return full, plies, n_actions


def _per_player_move_nll(injector, backbone, eval_ds, window):
    """Held-out per-player move-NLL ``[n_players]`` from a trained arm."""
    import torch

    device = next(backbone.parameters()).device
    backbone._build().eval()
    injector._build().eval()
    batch = backbone.encode_batch(eval_ds.trajectories).to(device)
    a, b = window
    with torch.no_grad():
        latent = injector.latent_trajectory(batch.feats)
        nll = backbone.per_traj_move_nll(latent[a:b], batch.window(a, b))
    return [float(x) for x in nll.cpu().tolist()]


def _train_arm(persist, latent_dim, n_actions, train_ds, eval_spec, cfg):
    injector = NeuralInjector(
        kind=InjectionKind.HIDDEN,
        latent_dim=latent_dim,
        seed=cfg.seed,
        persist=persist,
    )
    backbone = DiffMovePolicy(latent_dim=latent_dim, n_actions=n_actions)
    trainer = SFTTrainer(injector, backbone, cfg)
    summary = trainer.fit(train_ds, eval_spec=eval_spec)
    return injector, backbone, summary


def run_ea1(
    n_players: int = 32,
    n_games: int = 20,
    train_frac: float = 0.7,
    latent_dim: int = 16,
    epochs: int = 400,
    lr: float = 1e-2,
    seed: int = 0,
    b_latent_dim: int | None = None,
    bootstrap_n: int = 2000,
) -> EA1Result:
    """Train arms D and B and bootstrap ``D - B`` over players.

    ``b_latent_dim`` lets arm B be *wider* than D (a capacity-robustness check:
    does D still win when the memoryless control is given more parameters?).
    The default (``None``) makes B exactly capacity-matched to D.
    """
    full, plies, n_actions = _build_trajectories(n_players, n_games, seed)
    n_train_games = max(1, round(train_frac * n_games))
    boundary = n_train_games * plies
    total_steps = n_games * plies
    window = (boundary, total_steps)

    train_ds = TrajectoryDataset(
        [
            Trajectory(
                t.player_id,
                t.decisions[:boundary],
                t.observations[:boundary],
            )
            for t in full
        ]
    )
    eval_ds = TrajectoryDataset(full)
    eval_spec = EvalSpec(dataset=eval_ds, window=window)

    arms = {"D": (True, latent_dim), "B": (False, b_latent_dim or latent_dim)}
    summaries: dict[str, dict] = {}
    per_player: dict[str, list[float]] = {}
    for label, (persist, ld) in arms.items():
        cfg = TrainConfig(
            epochs=epochs,
            lr=lr,
            seed=seed,
            experiment=f"E-A1-{label}",
            extra={
                "timing_lambda": 0.5,
                "arm": label,
                "persist": persist,
                "n_players": n_players,
                "n_games": n_games,
                "train_games": n_train_games,
                "latent_dim": ld,
            },
        )
        injector, backbone, summaries[label] = _train_arm(
            persist, ld, n_actions, train_ds, eval_spec, cfg
        )
        per_player[label] = _per_player_move_nll(
            injector, backbone, eval_ds, window
        )

    diffs = [d - b for d, b in zip(per_player["D"], per_player["B"])]
    ci = bootstrap_ci(diffs, n_resamples=bootstrap_n, seed=seed)
    return EA1Result(
        d_per_player=per_player["D"],
        b_per_player=per_player["B"],
        diff_per_player=diffs,
        ci=ci,
        d_params=summaries["D"]["total_params"],
        b_params=summaries["B"]["total_params"],
        d_summary=summaries["D"],
        b_summary=summaries["B"],
    )


def run_ea1_capacity_sweep(
    latent_dim: int = 16,
    b_mults: tuple[int, ...] = (1, 2, 4),
    **kwargs,
) -> dict[int, EA1Result]:
    """Re-run E-A1 giving arm B 1x / 2x / 4x arm D's width.

    A win that survives B being *bigger* than D is not a capacity artifact
    (``milestone_a.md`` section 6). Returns one result per multiplier.
    """
    results: dict[int, EA1Result] = {}
    for m in b_mults:
        results[m] = run_ea1(
            latent_dim=latent_dim, b_latent_dim=latent_dim * m, **kwargs
        )
    return results
