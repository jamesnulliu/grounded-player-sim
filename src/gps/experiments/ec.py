"""E-C2: does the evolving latent beat the memoryless control on *real* chess?

The Milestone-A result (E-A1) settled this on a synthetic player whose hidden
state was, by construction, unreconstructable from the history features. E-C2
is the same airtight contrast carried onto **real Lichess trajectories** and a
**board-native move model** (no engine oracle, no synthetic mechanism):

* **arm D** (proposed): :class:`~gps.latent.neural.NeuralInjector` with
  ``persist=True`` -- the evolving latent.
* **arm B** (control): the *same* injector with ``persist=False`` -- the
  memoryless twin at *identical* parameters and inputs.

Both feed the same :class:`~gps.policy.board_native.BoardNativeBackbone`
(FEN -> factored from/to move logits; see that module for why it, not
:class:`~gps.policy.diff_policy.DiffMovePolicy`, is the right backbone for
oracle-free real-chess move-NLL). Both read the same
:func:`~gps.latent.structured.history_features`. The only difference is whether
the recurrence carries state -- so any gap is the evolving state, not capacity
or inputs (the #1 reviewer objection, on real data this time).

The headline is **D - B** on a strict, **per-player** temporal split: train on
each player's earlier moves, score held-out move-NLL on their later moves
(variable-length, masked; see :class:`~gps.train.sft.EvalSpec`). Significance
is a bootstrap over **players** (the independent unit), per the design.md
section-5 decision rule -- identical to E-A1, so the two results compare
directly.

Each arm is a tracked training run (mandatory W&B). Runs on CPU; a GPU only
speeds the per-epoch full-batch step.
"""

from __future__ import annotations

from dataclasses import dataclass

from gps.eval.bootstrap import BootstrapCI, bootstrap_ci
from gps.eval.probes import state_recovery_probe
from gps.latent.base import InjectionKind
from gps.latent.neural import NeuralInjector
from gps.policy.board_native import BoardNativeBackbone
from gps.train.base import TrainConfig, Trajectory, TrajectoryDataset
from gps.train.sft import EvalSpec, SFTTrainer


def _session_split_index(traj: Trajectory, train_frac: float) -> int:
    """Per-player boundary at a **session** edge: hold out the later sessions.

    The decisive RQ3 form (E-C3): train on a player's earlier *sessions* and
    predict their *later* sessions -- what separates real dynamics from a habit
    memorized within one sitting (design.md / proposal RQ3). The crude
    alternative is a move-fraction split (``split_indices``).

    Sessions are recovered from the persisted decisions: the game index is
    ``len(recent_outcomes.recent)`` (completed games so far) and the first game
    of each session has ``session_position == 0``, so the set of game indices
    with ``session_position == 0`` are the session starts. Falls back to the
    move-fraction split when a player has <2 sessions.
    """
    n = len(traj.decisions)
    if n < 2:
        return n
    game_idx = [len(d.recent_outcomes.recent) for d in traj.decisions]
    starts = sorted(
        {
            g
            for g, d in zip(game_idx, traj.decisions)
            if d.recent_outcomes.session_position == 0
        }
    )
    n_sess = len(starts)
    if n_sess < 2:  # one sitting -> no session split possible
        return min(n - 1, max(1, round(train_frac * n)))
    n_train = max(1, min(n_sess - 1, round(train_frac * n_sess)))
    boundary_game = starts[n_train]  # first game of the first held-out session
    for i, g in enumerate(game_idx):
        if g >= boundary_game:
            return max(1, min(n - 1, i))
    return n


def session_split_indices(
    trajectories: list[Trajectory], train_frac: float = 0.7
) -> list[int]:
    """Per-player session-aware split boundaries (hold out later sessions)."""
    return [_session_split_index(t, train_frac) for t in trajectories]


@dataclass
class ECResult:
    """Outcome of one E-C2 run: D-vs-B on the per-player future split.

    Mirrors :class:`~gps.experiments.ea1.EA1Result` (same
    bootstrap-over-players decision rule) so the real-chess number sits beside
    the synthetic one.
    """

    d_per_player: list[float]
    b_per_player: list[float]
    diff_per_player: list[float]  # D - B, one per player
    ci: BootstrapCI
    d_params: int
    b_params: int
    d_summary: dict
    b_summary: dict
    label: str = "E-C2"
    # Timing (think-time NLL) counterparts -- the E-C6 differentiator. Empty
    # when the cohort is unclocked (e.g. pre-2017 archives).
    d_timing_per_player: list[float] | None = None
    b_timing_per_player: list[float] | None = None
    timing_ci: BootstrapCI | None = None

    @property
    def d_val_move_nll(self) -> float:
        return sum(self.d_per_player) / len(self.d_per_player)

    @property
    def b_val_move_nll(self) -> float:
        return sum(self.b_per_player) / len(self.b_per_player)

    @property
    def d_minus_b(self) -> float:
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
            f"[{self.label}] held-out move-NLL, n_players={self.ci.n_units} | "
            f"D (persist)={self.d_val_move_nll:.4f} ({self.d_params}p) | "
            f"B (memoryless)={self.b_val_move_nll:.4f} "
            f"({self.b_params}p){cap}\n"
            f"        per-player D-B: mean={self.ci.point:+.4f} "
            f"95% CI [{self.ci.low:+.4f}, {self.ci.high:+.4f}] | "
            f"D wins {self.frac_players_d_wins:.0%} of players | "
            f"P(D-B<0)={self.ci.p_below_zero:.2f}\n"
            f"        VERDICT: {self.verdict()}" + self._timing_summary()
        )

    def _timing_summary(self) -> str:
        if self.timing_ci is None:
            return ""
        c = self.timing_ci
        d = sum(self.d_timing_per_player) / len(self.d_timing_per_player)
        b = sum(self.b_timing_per_player) / len(self.b_timing_per_player)
        return (
            f"\n        timing-NLL: D={d:.4f} B={b:.4f} | "
            f"D-B mean={c.point:+.4f} 95% CI [{c.low:+.4f}, {c.high:+.4f}] | "
            f"P(D-B<0)={c.p_below_zero:.2f}"
        )


def _per_player_nlls(injector, backbone, full, splits, batch_size=16):
    """Held-out per-player (move-NLL, timing-NLL) lists from a trained arm.

    Warms the latent over each player's whole session (no grad) and scores
    only their held-out tail (``splits`` -> eval mask) -- one number per
    player, the unit the bootstrap resamples. Minibatched (so it scales like
    training); results are returned in the original ``full`` player order.
    """
    import torch

    device = next(backbone.parameters()).device
    backbone._build().eval()
    injector._build().eval()
    n = len(full.trajectories)
    move = [0.0] * n
    timing = [0.0] * n
    # Length-sort for tight padding, but remember the original index so the
    # returned lists line up with full.trajectories (and thus the splits).
    indexed = sorted(
        enumerate(zip(full.trajectories, splits)),
        key=lambda kv: len(kv[1][0].decisions),
        reverse=True,
    )
    for i in range(0, len(indexed), batch_size):
        chunk = indexed[i : i + batch_size]
        idxs = [k for k, _ in chunk]
        trajs = [t for _, (t, _) in chunk]
        sp = [s for _, (_, s) in chunk]
        batch = backbone.encode_batch(trajs).to(device)
        _, eval_mask = backbone.train_eval_masks(batch, sp)
        with torch.no_grad():
            latent = injector.latent_trajectory(
                batch.feats, player_ids=batch.player_ids
            )
            mv = backbone.per_traj_move_nll(latent, batch, step_mask=eval_mask)
            tm = backbone.per_traj_timing_nll(
                latent, batch, step_mask=eval_mask
            )
        for j, idx in enumerate(idxs):
            move[idx] = float(mv[j].cpu())
            timing[idx] = float(tm[j].cpu())
    return move, timing


def _train_arm(injector, latent_dim, hidden_dim, full, splits, cfg):
    # Same seed for both arms -> identical backbone init, so the held-out gap
    # is the injector difference, not initialization luck (each injector is
    # likewise seeded with cfg.seed via its own forked RNG).
    backbone = BoardNativeBackbone(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        seed=cfg.seed,
        timing_model=cfg.extra.get("timing_model", "lognormal"),
        trunk=cfg.extra.get("trunk", "mlp"),
    )
    eval_spec = EvalSpec(dataset=full, splits=splits)
    trainer = SFTTrainer(injector, backbone, cfg)
    summary = trainer.fit(full, eval_spec=eval_spec)
    move_pp, timing_pp = _per_player_nlls(
        injector, backbone, full, splits, batch_size=cfg.batch_size
    )
    return injector, backbone, summary, move_pp, timing_pp


def run_ec(
    dataset: TrajectoryDataset,
    *,
    train_frac: float = 0.7,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    epochs: int = 300,
    lr: float = 1e-2,
    seed: int = 0,
    b_latent_dim: int | None = None,
    bootstrap_n: int = 2000,
    batch_size: int = 16,
    split_mode: str = "fraction",
    control: str = "memoryless",
    timing_lambda: float = 0.5,
    timing_model: str = "lognormal",
    trunk: str = "mlp",
    label: str | None = None,
) -> ECResult:
    """Train arms D and B on real trajectories; bootstrap ``D - B``/player.

    ``dataset`` is the *full* per-player sessions (e.g. from
    :func:`gps.data.store.load_dataset`). The per-player temporal boundary is a
    move-fraction split (``train_frac``); a session-aware split is the
    documented upgrade. ``b_latent_dim`` lets arm B be wider than D for the
    capacity-robustness check (default: exactly capacity-matched).
    ``batch_size`` players per minibatch (the masked trainer minibatches over
    players so large cohorts fit; <= n_players reproduces full-batch).
    ``split_mode`` is ``"fraction"`` (last 1-train_frac of *moves*, E-C2) or
    ``"session"`` (last sessions held out, the decisive E-C3 form).
    ``control`` picks arm B: ``"memoryless"`` (history-conditioned no-latent,
    the E-C2/E-C3 control) or ``"static"`` (per-player embedding, no dynamics,
    the E-C1 static-individual baseline B2).
    """
    if label is None:
        if control == "static":
            label = "E-C1"
        else:
            label = "E-C3" if split_mode == "session" else "E-C2"
    # One boundary per player (shared by both arms so the comparison is paired
    # on identical train/eval steps).
    if split_mode == "session":
        splits = session_split_indices(
            dataset.trajectories, train_frac=train_frac
        )
    else:
        splits = BoardNativeBackbone.split_indices(
            dataset.trajectories, train_frac=train_frac
        )

    bld = b_latent_dim or latent_dim
    players = [t.player_id for t in dataset.trajectories]

    def _make_injector(name):
        if name == "D":
            return NeuralInjector(
                kind=InjectionKind.HIDDEN,
                latent_dim=latent_dim,
                seed=seed,
                persist=True,
            )
        if control == "static":  # B2: per-player embedding, no dynamics
            from gps.latent.static_individual import StaticIndividualInjector

            return StaticIndividualInjector(
                players,
                kind=InjectionKind.HIDDEN,
                latent_dim=bld,
                seed=seed,
            )
        return NeuralInjector(  # memoryless history-conditioned twin
            kind=InjectionKind.HIDDEN,
            latent_dim=bld,
            seed=seed,
            persist=False,
        )

    summaries: dict[str, dict] = {}
    per_player: dict[str, list[float]] = {}
    timing_pp: dict[str, list[float]] = {}
    for name, ld in (("D", latent_dim), ("B", bld)):
        cfg = TrainConfig(
            epochs=epochs,
            lr=lr,
            seed=seed,
            batch_size=batch_size,
            experiment=f"{label}-{name}",
            extra={
                "timing_lambda": timing_lambda,
                "timing_model": timing_model,
                "arm": name,
                "control": control if name == "B" else "dynamic",
                "n_players": len(dataset.trajectories),
                "train_frac": train_frac,
                "split_mode": split_mode,
                "latent_dim": ld,
                "hidden_dim": hidden_dim,
                "trunk": trunk,
            },
        )
        _, _, summaries[name], per_player[name], timing_pp[name] = _train_arm(
            _make_injector(name), ld, hidden_dim, dataset, splits, cfg
        )

    diffs = [d - b for d, b in zip(per_player["D"], per_player["B"])]
    ci = bootstrap_ci(diffs, n_resamples=bootstrap_n, seed=seed)

    # Timing is only meaningful when the cohort is clocked (some think-time is
    # non-trivial). Skip the timing CI on unclocked data (all ~1e-3).
    timing_ci = None
    clocked = any(t > 1e-2 for t in timing_pp["D"] + timing_pp["B"])
    if clocked:
        tdiffs = [d - b for d, b in zip(timing_pp["D"], timing_pp["B"])]
        timing_ci = bootstrap_ci(tdiffs, n_resamples=bootstrap_n, seed=seed)

    return ECResult(
        d_per_player=per_player["D"],
        b_per_player=per_player["B"],
        diff_per_player=diffs,
        ci=ci,
        d_params=summaries["D"]["total_params"],
        b_params=summaries["B"]["total_params"],
        d_summary=summaries["D"],
        b_summary=summaries["B"],
        label=label,
        d_timing_per_player=timing_pp["D"],
        b_timing_per_player=timing_pp["B"],
        timing_ci=timing_ci,
    )


# --------------------------------------------------------------------------- #
# E-C4 (RQ2): does the learned latent *encode* the hidden behavioural state?
# --------------------------------------------------------------------------- #


@dataclass
class StateRecovery:
    """R^2 of a linear probe latent -> ground-truth hidden state, D vs B.

    Held-out R^2 (probe fit on train steps, scored on the temporal-split tail)
    so a high number means the *information is present and generalises*, not
    that a wide latent memorised the fit. (Presence, not use -- a causal
    clamp check is the complementary test; see ``probes.py``.)
    """

    d_r2: float
    b_r2: float
    d_r2_train: float
    b_r2_train: float
    target: str
    n_eval: int

    def summary(self) -> str:
        better = self.d_r2 > self.b_r2
        verdict = "D recovers it better" if better else "no D advantage"
        return (
            f"[E-C4 state-recovery] target={self.target} n_eval={self.n_eval} "
            f"| held-out R^2: D(persist)={self.d_r2:.3f} "
            f"B(memoryless)={self.b_r2:.3f} (train R^2 "
            f"{self.d_r2_train:.3f}/{self.b_r2_train:.3f})\n"
            f"        verdict: {verdict} "
            f"(delta R^2 = {self.d_r2 - self.b_r2:+.3f})"
        )


def _r2_with_weights(latents, targets, w) -> float:
    p = len(w)
    mean_t = sum(targets) / len(targets)
    ss_tot = sum((t - mean_t) ** 2 for t in targets)
    if ss_tot <= 0:
        return 0.0
    ss_res = 0.0
    for row, t in zip(latents, targets):
        pred = sum(row[a] * w[a] for a in range(p - 1)) + w[p - 1]
        ss_res += (t - pred) ** 2
    return 1.0 - ss_res / ss_tot


def _probe_arm(injector, backbone, dataset, splits, target_key):
    """Held-out + train R^2 of a linear probe latent_t -> target_t."""
    import torch

    device = next(backbone.parameters()).device
    backbone._build().eval()
    injector._build().eval()
    batch = backbone.encode_batch(dataset.trajectories).to(device)
    train_mask, eval_mask = backbone.train_eval_masks(batch, splits)
    with torch.no_grad():
        latent = injector.latent_trajectory(
            batch.feats, player_ids=batch.player_ids
        )
    lat = latent.cpu().tolist()  # [T][B][L]
    tr_x, tr_y, ev_x, ev_y = [], [], [], []
    for b, traj in enumerate(dataset.trajectories):
        for t, dp in enumerate(traj.decisions):
            target = dp.context.get(target_key)
            if target is None:
                continue
            row = lat[t][b]
            if bool(train_mask[t][b]):
                tr_x.append(row)
                tr_y.append(float(target))
            elif bool(eval_mask[t][b]):
                ev_x.append(row)
                ev_y.append(float(target))
    fit = state_recovery_probe(tr_x, tr_y, target_name=target_key)
    r2_eval = _r2_with_weights(ev_x, ev_y, fit.weights) if ev_x else 0.0
    return fit.r2, r2_eval, len(ev_y)


def run_state_recovery(
    dataset: TrajectoryDataset,
    *,
    target_key: str = "hidden_h",
    train_frac: float = 0.7,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    epochs: int = 150,
    lr: float = 1e-2,
    seed: int = 0,
) -> StateRecovery:
    """E-C4 (RQ2): probe whether the trained latent recovers the hidden state.

    Trains arm D (evolving) and arm B (memoryless) on ``dataset`` (whose
    decisions must carry the ground-truth hidden state under
    ``context[target_key]`` -- the synthetic
    :class:`~gps.synthetic.chess_players.HiddenTiltChessPlayer` records
    ``"hidden_h"``), then linearly probes each arm's latent for that state on a
    held-out temporal split. The evolving latent should recover it better,
    which is *why* it predicts behaviour better (links E-C2 to RQ2).
    """
    if not any(
        target_key in dp.context
        for t in dataset.trajectories
        for dp in t.decisions
    ):
        raise ValueError(
            f"no decision carries context[{target_key!r}]; state recovery "
            "needs ground-truth hidden state (use a synthetic player)."
        )
    splits = BoardNativeBackbone.split_indices(
        dataset.trajectories, train_frac=train_frac
    )
    out = {}
    for name, persist in (("D", True), ("B", False)):
        cfg = TrainConfig(
            epochs=epochs,
            lr=lr,
            seed=seed,
            batch_size=10_000,  # synthetic cohorts are small: one minibatch
            experiment=f"E-C4-{name}",
            extra={"timing_lambda": 0.5, "arm": name},
        )
        injector = NeuralInjector(
            kind=InjectionKind.HIDDEN,
            latent_dim=latent_dim,
            seed=seed,
            persist=persist,
        )
        inj, backbone, *_ = _train_arm(
            injector, latent_dim, hidden_dim, dataset, splits, cfg
        )
        out[name] = _probe_arm(inj, backbone, dataset, splits, target_key)
    (d_tr, d_ev, n), (b_tr, b_ev, _) = out["D"], out["B"]
    return StateRecovery(
        d_r2=d_ev,
        b_r2=b_ev,
        d_r2_train=d_tr,
        b_r2_train=b_tr,
        target=target_key,
        n_eval=n,
    )


@dataclass
class CausalIntervention:
    """Effect of clamping the latent along the hidden-state direction.

    Moves the latent +/- ``alpha`` sigma along the probe direction that encodes
    the hidden state, and measures how the *predictions* change on held-out
    decisions. A non-trivial change means the policy **uses** the latent, not
    merely that the state is decodable from it (the presence-vs-use gap).
    """

    entropy_calm: float
    entropy_tilted: float
    mu_calm: float
    mu_tilted: float
    move_kl: float
    alpha: float
    n_eval: int

    def summary(self) -> str:
        d_ent = self.entropy_tilted - self.entropy_calm
        d_mu = self.mu_tilted - self.mu_calm
        used = self.move_kl > 1e-3 or abs(d_ent) > 1e-3
        verdict = (
            "latent is USED (clamp changes predictions)"
            if used
            else "no effect (presence without use)"
        )
        return (
            f"[E-C4 causal] clamp +/-{self.alpha:g} sigma along the hidden-"
            f"state direction, n_eval={self.n_eval}\n"
            f"        move entropy: calm={self.entropy_calm:.3f} -> "
            f"tilted={self.entropy_tilted:.3f} (delta={d_ent:+.3f}; tilt "
            f"should flatten) | move KL(tilt||calm)={self.move_kl:.3f}\n"
            f"        timing mu: calm={self.mu_calm:.3f} -> "
            f"tilted={self.mu_tilted:.3f} (delta={d_mu:+.3f})\n"
            f"        verdict: {verdict}"
        )


def run_causal_intervention(
    dataset: TrajectoryDataset,
    *,
    target_key: str = "hidden_h",
    alpha: float = 2.0,
    train_frac: float = 0.7,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    epochs: int = 150,
    lr: float = 1e-2,
    seed: int = 0,
) -> CausalIntervention:
    """E-C4 (RQ2, the *use* half): clamp the latent, measure prediction change.

    Trains arm D, finds the latent direction that encodes the hidden state (the
    state-recovery probe's weight vector), then on held-out decisions perturbs
    the latent +/- ``alpha`` sigma along it and measures the change in the move
    distribution (entropy + KL) and think-time. If perturbing toward "tilted"
    flattens the move distribution and shifts timing, the policy *uses* the
    encoded state -- presence **and** use.
    """
    import torch

    splits = BoardNativeBackbone.split_indices(
        dataset.trajectories, train_frac=train_frac
    )
    injector = NeuralInjector(
        kind=InjectionKind.HIDDEN,
        latent_dim=latent_dim,
        seed=seed,
        persist=True,
    )
    cfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        seed=seed,
        batch_size=10_000,
        experiment="E-C4-causal",
        extra={"timing_lambda": 0.5, "arm": "D"},
    )
    inj, backbone, *_ = _train_arm(
        injector, latent_dim, hidden_dim, dataset, splits, cfg
    )

    device = next(backbone.parameters()).device
    backbone._build().eval()
    inj._build().eval()
    batch = backbone.encode_batch(dataset.trajectories).to(device)
    train_mask, eval_mask = backbone.train_eval_masks(batch, splits)
    with torch.no_grad():
        latent = inj.latent_trajectory(
            batch.feats, player_ids=batch.player_ids
        )

    # Probe direction u (latent -> hidden state), fit on train steps.
    lat = latent.cpu().tolist()
    tr_x, tr_y = [], []
    for b, traj in enumerate(dataset.trajectories):
        for t, dp in enumerate(traj.decisions):
            tgt = dp.context.get(target_key)
            if tgt is not None and bool(train_mask[t][b]):
                tr_x.append(lat[t][b])
                tr_y.append(float(tgt))
    fit = state_recovery_probe(tr_x, tr_y, target_name=target_key)
    w = fit.weights[:latent_dim]
    norm = sum(c * c for c in w) ** 0.5 or 1.0
    u = torch.tensor([c / norm for c in w], device=device, dtype=latent.dtype)

    # Scale the clamp by the std of the latent's projection onto u (so alpha is
    # in "hidden-state sigma" units).
    proj = (latent * u).sum(-1)[eval_mask]  # [n_eval]
    sigma = float(proj.std()) if proj.numel() > 1 else 1.0
    delta = alpha * sigma * u

    with torch.no_grad():
        logp_calm = backbone.move_logp(latent - delta, batch)
        logp_tilt = backbone.move_logp(latent + delta, batch)
        mu_calm, _ = backbone.timing_mu_sigma(latent - delta)
        mu_tilt, _ = backbone.timing_mu_sigma(latent + delta)

    def _ent(logp):
        return -(logp.exp() * logp).sum(-1)  # [T,B]

    mf = eval_mask.to(latent.dtype)
    denom = mf.sum().clamp_min(1.0)
    ent_calm = (_ent(logp_calm) * mf).sum() / denom
    ent_tilt = (_ent(logp_tilt) * mf).sum() / denom
    kl = (logp_tilt.exp() * (logp_tilt - logp_calm)).sum(-1)
    move_kl = (kl * mf).sum() / denom
    mu_c = (mu_calm * mf).sum() / denom
    mu_t = (mu_tilt * mf).sum() / denom
    return CausalIntervention(
        entropy_calm=float(ent_calm),
        entropy_tilted=float(ent_tilt),
        mu_calm=float(mu_c),
        mu_tilted=float(mu_t),
        move_kl=float(move_kl),
        alpha=alpha,
        n_eval=int(denom),
    )


# --------------------------------------------------------------------------- #
# E-C6: per-individual evolving timing vs an Elo-aggregate (Allie/B4) baseline
# --------------------------------------------------------------------------- #


@dataclass
class TimingVsAggregate:
    """Per-individual evolving timing vs an Elo+position aggregate (B4).

    The fair question is **not** "does the feature-poor latent-only model beat
    an aggregate that sees the clock?" -- it does not, the clock is too strong.
    It is "does the **evolving latent add value over** the aggregate?". So the
    headline is **B4 vs B4+z**: B4 = log-normal on aggregate features (Elo,
    move number, log time-remaining); B4+z = the same **plus the evolving
    latent**. ``d_nll`` (latent-only) is kept for context.
    """

    d_nll: float
    b4_nll: float
    b4z_nll: float
    add_ci: BootstrapCI  # (B4+z) - B4 over players (negative => latent helps)
    d_pearson: float
    b4_pearson: float
    b4z_pearson: float
    b4z_spearman: float
    n_players: int
    mode: str = "aggregate"

    def summary(self) -> str:
        verdict = (
            f"YES, evolving latent adds value over {self.mode}"
            if self.add_ci.point < 0
            else "no add"
        )
        label = {
            "aggregate": "B4(aggregate)",
            "external": "B(aggregate+released-pred)",
            "pure_external": "B(released model)",
        }.get(self.mode, "B")
        return (
            f"[E-C6 timing vs {self.mode}] n_players={self.n_players}\n"
            f"        held-out think-time NLL: {label}={self.b4_nll:.4f}"
            f" | +z(baseline+evolving latent)={self.b4z_nll:.4f}"
            f" | latent-only(D)={self.d_nll:.4f}\n"
            f"        DOES THE LATENT ADD VALUE? (B4+z)-B4 mean="
            f"{self.add_ci.point:+.4f} 95% CI "
            f"[{self.add_ci.low:+.4f}, {self.add_ci.high:+.4f}] "
            f"P(<0)={self.add_ci.p_below_zero:.2f} -> {verdict}\n"
            f"        per-player Pearson(pred,actual time): "
            f"B4={self.b4_pearson:.3f} B4+z={self.b4z_pearson:.3f} "
            f"(Spearman {self.b4z_spearman:.3f}; ChessMimic B9 r=0.41)"
        )


def _external_log_time(dp) -> float:
    """log-seconds of a released model's per-decision think-time prediction.

    The G4 add-on test (`documents/g4_plan.md`) compares against a *released*
    SOTA human-chess model instead of the hand-built aggregate. Its per-move
    predicted think-time (in **seconds**) is cached offline onto each decision
    as ``context["external_time_pred"]`` (like an ``EngineReference``); we
    consume only that scalar output, never the model's internals -- so any
    encoding / move-vocabulary mismatch is sidestepped. Raises loudly if the
    prediction is missing, rather than silently degrading to the proxy.
    """
    import math

    pred = dp.context.get("external_time_pred")
    if pred is None:
        raise ValueError(
            "external_pred/pure_external requires "
            "context['external_time_pred'] (a released model's predicted "
            "think-time in seconds) on every decision -- precompute + cache "
            "it (see documents/g4_plan.md)."
        )
    return math.log(max(float(pred), 1e-3))


def _b4_features(
    dp, position_aware: bool = False, external_pred: bool = False
):
    """Aggregate (non-individual) timing features for the Allie-style B4.

    ``position_aware`` appends the **branching factor** (number of legal moves,
    an oracle-free position-complexity proxy) -- a board-derived think-time
    driver. The fair test then asks whether the per-individual latent still
    adds value *over* an Elo + clock + complexity baseline.

    ``external_pred`` (G4) appends a **released** model's predicted
    log-think-time as one more *fitted* feature, so the baseline is at least as
    strong as that released model plus our aggregate features -- a conservative
    B for the "does the latent add value over released SOTA?" test.
    """
    import math

    ts = dp.time_signal
    elo = dp.context.get("player_elo") or 1500
    tr = ts.time_remaining or 0.0
    feats = [
        (elo - 1500.0) / 500.0,
        ts.move_number / 40.0,
        math.log(tr + 1.0) / 6.0,
        1.0,
    ]
    if position_aware:
        feats.append(len(dp.legal_actions) / 40.0)
    if external_pred:
        feats.append(_external_log_time(dp))
    return feats


def _corr(pred, actual):
    """(Pearson, Spearman) of two equal-length sequences; 0 if degenerate."""
    import numpy as np

    p = np.asarray(pred, float)
    a = np.asarray(actual, float)
    if len(p) < 3 or p.std() < 1e-9 or a.std() < 1e-9:
        return 0.0, 0.0
    pearson = float(np.corrcoef(p, a)[0, 1])
    rp = np.argsort(np.argsort(p))
    ra = np.argsort(np.argsort(a))
    spearman = (
        float(np.corrcoef(rp, ra)[0, 1])
        if rp.std() > 0 and ra.std() > 0
        else 0.0
    )
    return pearson, spearman


def run_timing_vs_aggregate(
    dataset: TrajectoryDataset,
    *,
    train_frac: float = 0.7,
    split_mode: str = "session",
    latent_dim: int = 16,
    hidden_dim: int = 64,
    epochs: int = 15,
    lr: float = 1e-2,
    seed: int = 0,
    bootstrap_n: int = 2000,
    position_aware: bool = False,
    external_pred: bool = False,
    pure_external: bool = False,
) -> TimingVsAggregate:
    """E-C6 / G4: our evolving per-individual think-time vs a baseline.

    Trains our model (arm D, log-normal timing head), and fits a baseline B
    whose ``mu`` is a least-squares function of **aggregate** features (Elo,
    move number, log time-remaining), *not* per-individual or state
    (Allie-style). ``B+z`` adds the per-step evolving latent. Both are scored
    on the same held-out steps: think-time NLL (bootstrapped over players) and
    per-player predicted-vs-actual correlation (the ChessMimic yardstick, B9).

    **G4 add-on-over-released-SOTA modes** (`documents/g4_plan.md`), both of
    which read each decision's ``context["external_time_pred"]`` (a released
    model's predicted think-time in seconds, cached offline):

    * ``external_pred=True`` -- the released prediction is one more *fitted*
      aggregate feature, so B is at least as strong as *released model + our
      aggregate features* (a conservative baseline).
    * ``pure_external=True`` -- B's ``mu`` is ``log(external pred)`` **locked**
      (weight 1, no re-fit) and ``B+z`` is that fixed offset plus the latent as
      the only free predictor: literally *released model* vs *released model +
      z*, with no chance we handicapped B by re-fitting. Implies the external
      ``position_aware``/``external_pred`` feature flags are ignored for B.
    """
    import math

    import numpy as np

    if split_mode == "session":
        splits = session_split_indices(dataset.trajectories, train_frac)
    else:
        splits = BoardNativeBackbone.split_indices(
            dataset.trajectories, train_frac
        )

    injector = NeuralInjector(
        kind=InjectionKind.HIDDEN,
        latent_dim=latent_dim,
        seed=seed,
        persist=True,
    )
    cfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        seed=seed,
        batch_size=16,
        experiment="E-C6-timing",
        extra={"timing_lambda": 0.5, "arm": "D"},
    )
    inj, backbone, *_ = _train_arm(
        injector, latent_dim, hidden_dim, dataset, splits, cfg
    )

    half_log2pi = 0.5 * math.log(2 * math.pi)

    # --- our model's per-step latent + mu on the full grid --------------
    import torch

    device = next(backbone.parameters()).device
    backbone._build().eval()
    inj._build().eval()
    batch = backbone.encode_batch(dataset.trajectories).to(device)
    with torch.no_grad():
        latent = inj.latent_trajectory(
            batch.feats, player_ids=batch.player_ids
        )
        mu_d, sig_d = backbone.timing_mu_sigma(latent)
    lat = latent.cpu().tolist()  # [T][B][L]
    mu_d = mu_d.cpu().tolist()
    sig_d = float(sig_d)

    # --- fit baseline B and B+z (baseline + evolving latent) ------------
    # B+z is the value-add test: does the latent help *beyond* the baseline?
    # Per decision each mode yields (base_feat, base_off, withz_feat, wz_off);
    # ``mu = off + w . feat`` where ``off`` is a locked (unfitted) log-time.
    #   aggregate/external: off=0; features are fitted (external pred, if on,
    #     is one more fitted feature -- baseline >= released model + feats).
    #   pure_external: off=log(released pred) LOCKED for both arms; B keeps
    #     only a single intercept (a fair global recalibration of the offset --
    #     arbitrary constant offset would otherwise inflate its NLL), and B+z
    #     adds the latent as the ONLY extra predictor -> B+z - B isolates the
    #     latent's marginal value over the released model itself.
    def _rows(dp, z):
        if pure_external:
            off = _external_log_time(dp)
            return [1.0], off, [1.0] + list(z), off
        feat = _b4_features(dp, position_aware, external_pred)
        return feat, 0.0, feat + list(z), 0.0

    base_x, base_o, wz_x, wz_o, ys = [], [], [], [], []
    for b, (traj, sp) in enumerate(zip(dataset.trajectories, splits)):
        for t in range(0, sp):
            bf, bo, wf, wo = _rows(traj.decisions[t], lat[t][b])
            base_x.append(bf)
            base_o.append(bo)
            wz_x.append(wf)
            wz_o.append(wo)
            ys.append(
                math.log(max(traj.observations[t].time_spent or 1e-3, 1e-3))
            )
    y = np.asarray(ys)

    def _fit(xs, offs):
        m = np.asarray(xs, dtype=float)
        tgt = y - np.asarray(offs, dtype=float)
        if m.shape[1] == 0:  # no free weights -> locked offset only
            return np.zeros(0), float(tgt.std()) or 1.0
        w, *_ = np.linalg.lstsq(m, tgt, rcond=None)
        sig = float((tgt - m @ w).std()) or 1.0
        return w, sig

    w4, sig4 = _fit(base_x, base_o)
    w4z, sig4z = _fit(wz_x, wz_o)

    def _nll(yt, mu, sig):
        return 0.5 * ((yt - mu) / sig) ** 2 + math.log(sig) + half_log2pi + yt

    def _mu(w, feat, off):
        return off + (float(np.dot(w, feat)) if feat else 0.0)

    # --- per-player held-out NLL + correlations -------------------------
    d_pp, b4_pp, b4z_pp = [], [], []
    b4_pr, b4z_pr, b4z_sp = [], [], []
    for b, (traj, sp) in enumerate(zip(dataset.trajectories, splits)):
        dn, bn, bzn, b4pred, bzpred, act = [], [], [], [], [], []
        for t in range(sp, len(traj.decisions)):
            yt = math.log(max(traj.observations[t].time_spent or 1e-3, 1e-3))
            bf, bo, wf, wo = _rows(traj.decisions[t], lat[t][b])
            mb = _mu(w4, bf, bo)
            mbz = _mu(w4z, wf, wo)
            dn.append(_nll(yt, mu_d[t][b], sig_d))
            bn.append(_nll(yt, mb, sig4))
            bzn.append(_nll(yt, mbz, sig4z))
            b4pred.append(math.exp(mb))
            bzpred.append(math.exp(mbz))
            act.append(traj.observations[t].time_spent or 0.0)
        if not dn:
            continue
        d_pp.append(sum(dn) / len(dn))
        b4_pp.append(sum(bn) / len(bn))
        b4z_pp.append(sum(bzn) / len(bzn))
        bp, _ = _corr(b4pred, act)
        bzp, bzs = _corr(bzpred, act)
        b4_pr.append(bp)
        b4z_pr.append(bzp)
        b4z_sp.append(bzs)

    add_diffs = [bz - b for bz, b in zip(b4z_pp, b4_pp)]
    ci = bootstrap_ci(add_diffs, n_resamples=bootstrap_n, seed=seed)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    mode = (
        "pure_external"
        if pure_external
        else "external"
        if external_pred
        else "aggregate"
    )
    return TimingVsAggregate(
        d_nll=mean(d_pp),
        b4_nll=mean(b4_pp),
        b4z_nll=mean(b4z_pp),
        add_ci=ci,
        d_pearson=0.0,
        b4_pearson=mean(b4_pr),
        b4z_pearson=mean(b4z_pr),
        b4z_spearman=mean(b4z_sp),
        n_players=len(d_pp),
        mode=mode,
    )


# --------------------------------------------------------------------------- #
# Concentration: does the latent's advantage localize to high-dynamics moments?
# --------------------------------------------------------------------------- #


@dataclass
class Concentration:
    """Held-out D-vs-B advantage bucketed by how active the hidden state is.

    If the evolving latent is really modelling *dynamics*, its edge over the
    memoryless twin should be **largest where the hidden state is most active**
    (high tilt/frustration), not spread uniformly -- the design.md section-5
    concentration check. Each bucket: ``(label, n, mean(nll_D - nll_B))``;
    more-negative = D better.
    """

    channel: str
    buckets: list[tuple[str, int, float]]

    def summary(self) -> str:
        rows = " | ".join(
            f"{lab}: dD-B={d:+.4f} (n={n})" for lab, n, d in self.buckets
        )
        lo = self.buckets[0][2]
        hi = self.buckets[-1][2]
        verdict = (
            "CONCENTRATED in high-dynamics moments"
            if hi < lo
            else "not concentrated"
        )
        return (
            f"[concentration] channel={self.channel} | {rows}\n"
            f"        low-tilt dD-B={lo:+.4f} vs high-tilt dD-B={hi:+.4f}"
            f" -> {verdict}"
        )


def run_concentration(
    dataset: TrajectoryDataset,
    *,
    channel: str = "timing",
    target_key: str = "hidden_h",
    bucket_feature: str | None = None,
    n_buckets: int = 3,
    train_frac: float = 0.7,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    epochs: int = 15,
    lr: float = 1e-2,
    seed: int = 0,
) -> Concentration:
    """Bucket the held-out D-vs-B per-decision NLL by behavioural context.

    Trains arms D and B, then on held-out decisions computes ``nll_D - nll_B``
    per decision (``channel`` = ``"timing"`` or ``"move"``) and buckets it.
    By default the bucket key is the ground-truth hidden state
    (``context[target_key]``, synthetic only). On **real** data set
    ``bucket_feature`` to an anchored dimension (``"post_loss"``,
    ``"time_pressure"``, ``"fatigue"``, ``"momentum"``) -- an *observable*
    context -- to ask **where** the latent's edge concentrates. A more-negative
    gap in the high-state bucket means the latent helps where human state
    matters most.
    """
    import torch

    from gps.latent.structured import history_features

    splits = BoardNativeBackbone.split_indices(
        dataset.trajectories, train_frac=train_frac
    )
    nets = {}
    for name, persist in (("D", True), ("B", False)):
        injector = NeuralInjector(
            kind=InjectionKind.HIDDEN,
            latent_dim=latent_dim,
            seed=seed,
            persist=persist,
        )
        cfg = TrainConfig(
            epochs=epochs,
            lr=lr,
            seed=seed,
            batch_size=10_000,
            experiment=f"concentration-{name}",
            extra={"timing_lambda": 0.5, "arm": name},
        )
        inj, backbone, *_ = _train_arm(
            injector, latent_dim, hidden_dim, dataset, splits, cfg
        )
        backbone._build().eval()
        inj._build().eval()
        nets[name] = (inj, backbone)

    device = next(nets["D"][1].parameters()).device
    batch = nets["D"][1].encode_batch(dataset.trajectories).to(device)
    _, eval_mask = nets["D"][1].train_eval_masks(batch, splits)

    def _per_step_nll(inj, backbone):
        with torch.no_grad():
            lat = inj.latent_trajectory(
                batch.feats, player_ids=batch.player_ids
            )
            if channel == "move":
                logp = backbone.move_logp(lat, batch)
                chosen = logp.gather(-1, batch.move_idx.unsqueeze(-1)).squeeze(
                    -1
                )
                return (-chosen).cpu().tolist()
            return backbone._timing_nll_steps(lat, batch).cpu().tolist()

    nll_d = _per_step_nll(*nets["D"])
    nll_b = _per_step_nll(*nets["B"])

    # Collect held-out (diff, hidden_state) pairs.
    pairs = []
    for b, traj in enumerate(dataset.trajectories):
        for t, dp in enumerate(traj.decisions):
            if not bool(eval_mask[t][b]):
                continue
            if bucket_feature is not None:
                h = history_features(dp).get(bucket_feature)
            else:
                h = dp.context.get(target_key)
            if h is None:
                continue
            pairs.append((float(h), nll_d[t][b] - nll_b[t][b]))
    if not pairs:
        key = bucket_feature or target_key
        raise ValueError(f"no held-out decision carries {key!r}")
    pairs.sort(key=lambda x: x[0])
    name = bucket_feature or "tilt"
    labels = (
        [f"low-{name}", f"mid-{name}", f"high-{name}"]
        if n_buckets == 3
        else [f"{name}-q{i}" for i in range(n_buckets)]
    )
    buckets = []
    n = len(pairs)
    for i in range(n_buckets):
        chunk = pairs[i * n // n_buckets : (i + 1) * n // n_buckets]
        diffs = [d for _, d in chunk]
        buckets.append(
            (labels[i], len(diffs), sum(diffs) / max(len(diffs), 1))
        )
    return Concentration(channel=channel, buckets=buckets)


# --------------------------------------------------------------------------- #
# E-E1 (RQ6): verbal channel (interpretable anchored dims) vs hidden vector
# --------------------------------------------------------------------------- #


@dataclass
class RQ6Result:
    """Hidden-vector vs verbal-readout injection of the *same* evolving state.

    Both arms run the identical GRU recurrence (same seed); they differ only in
    what reaches the backbone -- the full hidden vector (``hidden``) or the few
    interpretable anchored ``DIMENSIONS`` the verbal text carries (``verbal``).
    So this isolates the channel-capacity tradeoff (RQ6): hidden is richer but
    open-weight-only; verbal is portable to closed APIs but lossy.
    """

    hidden_per_player: list[float]
    verbal_per_player: list[float]
    ci: BootstrapCI  # hidden - verbal (negative => hidden channel better)
    hidden_params: int
    verbal_params: int

    def summary(self) -> str:
        h = sum(self.hidden_per_player) / len(self.hidden_per_player)
        v = sum(self.verbal_per_player) / len(self.verbal_per_player)
        if self.ci.point >= 0:
            verdict = "VERBAL ties/beats hidden (cheap channel suffices)"
        elif self.ci.high < 0:
            verdict = "HIDDEN richer (CI excludes 0)"
        else:
            verdict = "hidden better but not significant"
        return (
            f"[E-E1 verbal-vs-hidden] held-out move-NLL, "
            f"n_players={self.ci.n_units}\n"
            f"        hidden(full vector, {self.hidden_params}p)={h:.4f} | "
            f"verbal(anchored dims, {self.verbal_params}p)={v:.4f}\n"
            f"        hidden-verbal: mean={self.ci.point:+.4f} 95% CI "
            f"[{self.ci.low:+.4f}, {self.ci.high:+.4f}] "
            f"P(<0)={self.ci.p_below_zero:.2f} -> {verdict}"
        )


def run_rq6(
    dataset: TrajectoryDataset,
    *,
    train_frac: float = 0.7,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    epochs: int = 15,
    lr: float = 1e-2,
    seed: int = 0,
    batch_size: int = 16,
    bootstrap_n: int = 2000,
    split_mode: str = "session",
) -> RQ6Result:
    """RQ6: same state, full hidden vector vs the verbal anchored dims."""
    from gps.latent.structured import DIMENSIONS

    if split_mode == "session":
        splits = session_split_indices(dataset.trajectories, train_frac)
    else:
        splits = BoardNativeBackbone.split_indices(
            dataset.trajectories, train_frac
        )

    out = {}
    for name, readout, bb_dim in (
        ("hidden", False, latent_dim),
        ("verbal", True, len(DIMENSIONS)),
    ):
        injector = NeuralInjector(
            kind=InjectionKind.HIDDEN,
            latent_dim=latent_dim,
            seed=seed,
            persist=True,
            readout=readout,
        )
        cfg = TrainConfig(
            epochs=epochs,
            lr=lr,
            seed=seed,
            batch_size=batch_size,
            experiment=f"E-E1-{name}",
            extra={"timing_lambda": 0.5, "arm": name},
        )
        _, _, summ, move_pp, _ = _train_arm(
            injector, bb_dim, hidden_dim, dataset, splits, cfg
        )
        out[name] = (move_pp, summ["total_params"])

    (h_pp, h_params), (v_pp, v_params) = out["hidden"], out["verbal"]
    diffs = [h - v for h, v in zip(h_pp, v_pp)]
    ci = bootstrap_ci(diffs, n_resamples=bootstrap_n, seed=seed)
    return RQ6Result(
        hidden_per_player=h_pp,
        verbal_per_player=v_pp,
        ci=ci,
        hidden_params=h_params,
        verbal_params=v_params,
    )
