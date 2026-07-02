"""E-D1 (RQ5): the same dynamic-latent framework in a NON-game domain.

Knowledge tracing. The contribution is a game-agnostic core: if the *same*
injector + trainer, with only the encoder/oracle swapped
(:class:`~gps.policy.kt_backbone.KTBackbone`), reproduces the chess pattern --
an evolving per-individual latent beats the memoryless twin at predicting a
student's *future* responses -- then the mechanism is not a chess artifact.

The synthetic student mirrors :class:`~gps.synthetic.chess_players.\
HiddenTiltChessPlayer`: a hidden **frustration** state ``h`` (a leaky integral
of recent *errors*) lowers the probability of a correct answer, and is not
reconstructable from the windowed ``history_features`` (post-error recency,
recent accuracy, fatigue) -- so a memoryless reader has
irreducible error and an evolving latent that integrates the error stream does
not. Items vary in difficulty (the encoder's input); ``P(correct) =
sigmoid(skill - difficulty - tilt_scale*h)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from gps.eval.bootstrap import BootstrapCI, bootstrap_ci
from gps.experiments.ec import _per_player_nlls
from gps.interface import (
    DecisionPoint,
    EngineReference,
    Game,
    Outcome,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import InjectionKind, Observation
from gps.latent.neural import NeuralInjector
from gps.policy.board_native import BoardNativeBackbone  # reuse split_indices
from gps.policy.kt_backbone import KTBackbone
from gps.train.base import TrainConfig, Trajectory, TrajectoryDataset
from gps.train.sft import EvalSpec, SFTTrainer
from gps.util.rng import LCG

#: Item difficulties (the encoder's input feature). Easy (negative) -> hard.
DEFAULT_DIFFICULTIES = (-0.5, 0.0, 0.5, 1.0, 1.5)


@dataclass
class KnowledgeTracingStudent:
    """A student whose correctness rate is a hidden leaky error-integral."""

    student_id: str
    seed: int = 0
    rho: float = 0.9
    base_skill: float = 0.5
    tilt_scale: float = 3.0
    difficulties: tuple[float, ...] = field(
        default_factory=lambda: DEFAULT_DIFFICULTIES
    )

    def __post_init__(self) -> None:
        self._rng = LCG(self.seed)

    def build_trajectory(self, n_items: int) -> Trajectory:
        decisions: list[DecisionPoint] = []
        observations: list[Observation] = []
        prior: list[Outcome] = []
        h = 0.0  # hidden frustration: leaky integral of errors
        for i in range(n_items):
            difficulty = self.difficulties[i % len(self.difficulties)]
            p_correct = 1.0 / (
                1.0
                + math.exp(
                    -(self.base_skill - difficulty - self.tilt_scale * h)
                )
            )
            correct = self._rng.random() < p_correct
            # Frustration rises after an error, decays after a success.
            h = self.rho * h + (1.0 - self.rho) * (0.0 if correct else 1.0)
            # Slower (more deliberation) when frustrated; a weak timing signal.
            spent = math.exp(self._rng.uniform(0.4, 1.0) + 0.5 * h)

            stream = OutcomeStream(recent=list(prior), session_position=i)
            decisions.append(
                DecisionPoint(
                    game=Game.KNOWLEDGE_TRACING,
                    player_id=self.student_id,
                    state=(difficulty,),  # item feature(s)
                    legal_actions=("correct", "incorrect"),
                    engine_reference=EngineReference(
                        candidate_values={
                            "correct": p_correct,
                            "incorrect": 1.0 - p_correct,
                        },
                        best_move="correct"
                        if p_correct >= 0.5
                        else "incorrect",
                        unit="prob",
                    ),
                    time_signal=TimeSignal(
                        time_remaining=None,  # no clock in KT
                        time_spent=spent,
                        move_number=i,
                    ),
                    recent_outcomes=stream,
                    context={"synthetic": True, "hidden_h": h},
                )
            )
            observations.append(
                Observation(
                    move="correct" if correct else "incorrect",
                    time_spent=spent,
                )
            )
            prior.append(Outcome(won=correct))
        return Trajectory(self.student_id, decisions, observations)


def build_kt_dataset(
    n_students: int = 24,
    n_items: int = 120,
    *,
    seed: int = 0,
    rho: float = 0.9,
    base_skill: float = 0.5,
    tilt_scale: float = 3.0,
    skill_spread: float = 0.0,
) -> TrajectoryDataset:
    """A cohort of hidden-frustration students -> a ``TrajectoryDataset``.

    ``skill_spread`` > 0 gives each student a distinct ``base_skill`` drawn
    uniformly from ``base_skill +/- skill_spread`` -- *real* per-individual
    heterogeneity (the population-generation / "positive average person" test,
    Milestone F).
    """
    spread = LCG(seed)
    students = []
    for i in range(n_students):
        bs = base_skill
        if skill_spread > 0:
            bs = base_skill + spread.uniform(-skill_spread, skill_spread)
        students.append(
            KnowledgeTracingStudent(
                student_id=f"student-{i}",
                seed=seed + 1 + i,
                rho=rho,
                base_skill=bs,
                tilt_scale=tilt_scale,
            ).build_trajectory(n_items)
        )
    return TrajectoryDataset(students)


@dataclass
class KTResult:
    """D-vs-B on the per-student future split (RQ5): response + timing."""

    d_per_player: list[float]
    b_per_player: list[float]
    ci: BootstrapCI
    d_params: int
    b_params: int
    d_timing: list[float] = field(default_factory=list)
    b_timing: list[float] = field(default_factory=list)
    timing_ci: BootstrapCI | None = None

    @property
    def frac_d_wins(self) -> float:
        n = len(self.d_per_player)
        return (
            sum(
                1
                for d, b in zip(self.d_per_player, self.b_per_player)
                if d < b
            )
            / n
        )

    def summary(self) -> str:
        d = sum(self.d_per_player) / len(self.d_per_player)
        b = sum(self.b_per_player) / len(self.b_per_player)
        out = (
            f"[E-D1 knowledge-tracing] held-out NLL, "
            f"n_students={self.ci.n_units} (equal {self.d_params}p)\n"
            f"        RESPONSE: D={d:.4f} B={b:.4f} | D-B mean="
            f"{self.ci.point:+.4f} 95% CI [{self.ci.low:+.4f}, "
            f"{self.ci.high:+.4f}] | D wins {self.frac_d_wins:.0%} | "
            f"P(D-B<0)={self.ci.p_below_zero:.2f}"
        )
        if self.timing_ci is not None:
            td = sum(self.d_timing) / len(self.d_timing)
            tb = sum(self.b_timing) / len(self.b_timing)
            c = self.timing_ci
            out += (
                f"\n        TIMING: D={td:.4f} B={tb:.4f} | D-B mean="
                f"{c.point:+.4f} 95% CI [{c.low:+.4f}, {c.high:+.4f}] | "
                f"P(D-B<0)={c.p_below_zero:.2f}"
            )
        return out


def _train_kt_arm(persist, dataset, splits, latent_dim, hidden_dim, cfg):
    injector = NeuralInjector(
        kind=InjectionKind.HIDDEN,
        latent_dim=latent_dim,
        seed=cfg.seed,
        persist=persist,
    )
    backbone = KTBackbone(
        latent_dim=latent_dim,
        item_dim=len(dataset.trajectories[0].decisions[0].state),
        hidden_dim=hidden_dim,
        seed=cfg.seed,
    )
    trainer = SFTTrainer(injector, backbone, cfg)
    trainer.fit(dataset, eval_spec=EvalSpec(dataset=dataset, splits=splits))
    move_pp, timing_pp = _per_player_nlls(
        injector, backbone, dataset, splits, batch_size=cfg.batch_size
    )
    params = sum(p.numel() for p in injector.parameters()) + sum(
        p.numel() for p in backbone.parameters()
    )
    return move_pp, timing_pp, params


def run_kt(
    dataset: TrajectoryDataset,
    *,
    train_frac: float = 0.7,
    latent_dim: int = 16,
    hidden_dim: int = 32,
    epochs: int = 60,
    lr: float = 1e-2,
    seed: int = 0,
    batch_size: int = 32,
    bootstrap_n: int = 2000,
) -> KTResult:
    """RQ5: evolving latent (D) vs memoryless twin (B) on student responses."""
    splits = BoardNativeBackbone.split_indices(
        dataset.trajectories, train_frac=train_frac
    )
    out = {}
    for name, persist in (("D", True), ("B", False)):
        cfg = TrainConfig(
            epochs=epochs,
            lr=lr,
            seed=seed,
            batch_size=batch_size,
            experiment=f"E-D1-{name}",
            extra={"timing_lambda": 0.5, "arm": name},
        )
        out[name] = _train_kt_arm(
            persist, dataset, splits, latent_dim, hidden_dim, cfg
        )
    (d_pp, d_tm, d_params), (b_pp, b_tm, b_params) = out["D"], out["B"]
    ci = bootstrap_ci(
        [d - b for d, b in zip(d_pp, b_pp)], n_resamples=bootstrap_n, seed=seed
    )
    timing_ci = bootstrap_ci(
        [d - b for d, b in zip(d_tm, b_tm)], n_resamples=bootstrap_n, seed=seed
    )
    return KTResult(
        d_per_player=d_pp,
        b_per_player=b_pp,
        ci=ci,
        d_params=d_params,
        b_params=b_params,
        d_timing=d_tm,
        b_timing=b_tm,
        timing_ci=timing_ci,
    )


@dataclass
class PopulationResult:
    """Does the per-individual latent recover population heterogeneity (E-F2)?

    The field's named-but-unsolved "positive average person" problem: models
    collapse to the population mean. Here each student has a distinct skill;
    we compare how well the per-individual model vs an average-person baseline
    reproduce the *distribution* of per-student held-out accuracy.
    """

    pearson: float  # corr(model-predicted, observed) per-student accuracy
    w1_model: float  # Wasserstein-1D(model predicted dist, observed dist)
    w1_average: float  # Wasserstein-1D(average-person point mass, observed)
    model_spread: float  # std of model-predicted per-student accuracy
    observed_spread: float
    n: int
    # Generative-model metrics vs the real distribution (eval.distributional)
    js_model: float = 0.0
    js_average: float = 0.0
    recall_model: float = 0.0  # diversity/coverage of the real distribution
    recall_average: float = 0.0
    precision_model: float = 0.0  # plausibility
    precision_average: float = 0.0

    def summary(self) -> str:
        verdict = (
            "RECOVERS heterogeneity (beats average-person)"
            if self.w1_model < self.w1_average
            else "no better than average-person"
        )
        return (
            f"[E-F2 population heterogeneity] n_students={self.n}\n"
            f"        per-student accuracy: observed spread="
            f"{self.observed_spread:.3f} | model-predicted spread="
            f"{self.model_spread:.3f} (avg-person spread=0)\n"
            f"        Wasserstein-1D to observed: model={self.w1_model:.4f}"
            f" vs average-person={self.w1_average:.4f} | "
            f"corr(pred,obs)={self.pearson:.3f}\n"
            f"        JS-div: model={self.js_model:.3f} vs "
            f"average={self.js_average:.3f} | "
            f"precision/recall: model={self.precision_model:.2f}/"
            f"{self.recall_model:.2f} vs average="
            f"{self.precision_average:.2f}/{self.recall_average:.2f}\n"
            f"        VERDICT: {verdict} "
            f"(avg-person: high precision, ~0 recall = the average trap)"
        )


def run_population(
    dataset: TrajectoryDataset,
    *,
    train_frac: float = 0.7,
    latent_dim: int = 16,
    hidden_dim: int = 32,
    epochs: int = 60,
    lr: float = 1e-2,
    seed: int = 0,
) -> PopulationResult:
    """E-F2: does the per-individual latent recover skill heterogeneity?"""
    import statistics

    import numpy as np
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
    backbone = KTBackbone(
        latent_dim=latent_dim,
        item_dim=len(dataset.trajectories[0].decisions[0].state),
        hidden_dim=hidden_dim,
        seed=seed,
    )
    cfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        seed=seed,
        batch_size=10_000,
        experiment="E-F2-population",
        extra={"timing_lambda": 0.5, "arm": "D"},
    )
    SFTTrainer(injector, backbone, cfg).fit(
        dataset, eval_spec=EvalSpec(dataset=dataset, splits=splits)
    )
    backbone._build().eval()
    injector._build().eval()
    device = next(backbone.parameters()).device
    batch = backbone.encode_batch(dataset.trajectories).to(device)
    _, eval_mask = backbone.train_eval_masks(batch, splits)
    with torch.no_grad():
        latent = injector.latent_trajectory(
            batch.feats, player_ids=batch.player_ids
        )
        pcorrect = (
            torch.sigmoid(backbone._build()(batch.item, latent)).cpu().tolist()
        )

    pred, obs = [], []
    for b, traj in enumerate(dataset.trajectories):
        ps, os_ = [], []
        for t in range(len(traj.decisions)):
            if not bool(eval_mask[t][b]):
                continue
            ps.append(pcorrect[t][b])
            os_.append(1.0 if traj.observations[t].move == "correct" else 0.0)
        if ps:
            pred.append(sum(ps) / len(ps))
            obs.append(sum(os_) / len(os_))

    from gps.eval.distributional import (
        js_divergence,
        precision_recall,
        wasserstein_1d,
    )

    avg_pop = [sum(obs) / len(obs)] * len(obs)  # the average-person generation
    pearson = (
        float(np.corrcoef(pred, obs)[0, 1])
        if statistics.pstdev(pred) > 1e-9 and statistics.pstdev(obs) > 1e-9
        else 0.0
    )
    pr_m = precision_recall(obs, pred, k=3)
    pr_a = precision_recall(obs, avg_pop, k=3)
    return PopulationResult(
        pearson=pearson,
        w1_model=wasserstein_1d(pred, obs),
        w1_average=wasserstein_1d(avg_pop, obs),
        model_spread=statistics.pstdev(pred),
        observed_spread=statistics.pstdev(obs),
        n=len(obs),
        js_model=js_divergence(pred, obs),
        js_average=js_divergence(avg_pop, obs),
        recall_model=pr_m.recall,
        recall_average=pr_a.recall,
        precision_model=pr_m.precision,
        precision_average=pr_a.precision,
    )


@dataclass
class GenerationResult:
    """E-F1: GENERATE a population by sampling latents; score it vs the real.

    Invented players have no pointwise ground truth, so the generated
    population is judged *distributionally* (gps.eval.distributional) against
    the real players' behavioural stat (here: accuracy across the item pool,
    computed the same way for both, from each player's style latent).
    """

    w1: float  # Wasserstein-1D(generated dist, real dist)
    js: float
    precision: float  # share of generated players that are plausible
    recall: float  # share of real players covered by the generated population
    avg_recall: float  # the average-person generation's coverage (≈0)
    gen_spread: float
    real_spread: float
    n_real: int
    n_generated: int

    def summary(self) -> str:
        verdict = (
            "plausible AND diverse generation"
            if self.recall > self.avg_recall
            else "no coverage"
        )
        return (
            f"[E-F1 population generation] real={self.n_real} "
            f"generated={self.n_generated} (sampled latents)\n"
            f"        accuracy spread: real={self.real_spread:.3f} "
            f"generated={self.gen_spread:.3f} | W1={self.w1:.4f} "
            f"JS={self.js:.3f}\n"
            f"        generated precision/recall={self.precision:.2f}/"
            f"{self.recall:.2f} (vs average-person "
            f"recall={self.avg_recall:.2f})\n"
            f"        VERDICT: {verdict}"
        )


def run_generation(
    dataset: TrajectoryDataset,
    *,
    train_frac: float = 0.7,
    latent_dim: int = 16,
    hidden_dim: int = 32,
    epochs: int = 60,
    lr: float = 1e-2,
    seed: int = 0,
    n_generated: int = 200,
) -> GenerationResult:
    """E-F1: sample latents from a prior over the real population, generate."""
    import statistics

    import numpy as np
    import torch

    from gps.eval.distributional import (
        js_divergence,
        precision_recall,
        wasserstein_1d,
    )

    splits = BoardNativeBackbone.split_indices(
        dataset.trajectories, train_frac=train_frac
    )
    injector = NeuralInjector(
        kind=InjectionKind.HIDDEN,
        latent_dim=latent_dim,
        seed=seed,
        persist=True,
    )
    item_dim = len(dataset.trajectories[0].decisions[0].state)
    backbone = KTBackbone(
        latent_dim=latent_dim,
        item_dim=item_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    cfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        seed=seed,
        batch_size=10_000,
        experiment="E-F1-generation",
        extra={"timing_lambda": 0.5, "arm": "D"},
    )
    SFTTrainer(injector, backbone, cfg).fit(
        dataset, eval_spec=EvalSpec(dataset=dataset, splits=splits)
    )
    net = backbone._build()
    net.eval()
    injector._build().eval()
    device = next(backbone.parameters()).device

    # Per-student *style* latent = mean of their evolving latent over the run.
    batch = backbone.encode_batch(dataset.trajectories).to(device)
    with torch.no_grad():
        latent = injector.latent_trajectory(
            batch.feats, player_ids=batch.player_ids
        )  # [T,B,L]
        mask = batch.step_mask.to(latent.dtype).unsqueeze(-1)
        real_z = (latent * mask).sum(0) / mask.sum(0).clamp_min(1.0)  # [B,L]

    # Reference items = the difficulty pool (so accuracy is comparable across
    # players regardless of which items they happened to see).
    diffs = torch.tensor(
        [[d] for d in DEFAULT_DIFFICULTIES], dtype=torch.float32, device=device
    )  # [D, item_dim]

    def _acc(zs):  # zs: [N, L] -> [N] mean P(correct) over the item pool
        n = zs.shape[0]
        item = diffs.unsqueeze(0).expand(n, -1, -1)  # [N, D, item_dim]
        lat = zs.unsqueeze(1).expand(-1, diffs.shape[0], -1)  # [N, D, L]
        with torch.no_grad():
            return torch.sigmoid(net(item, lat)).mean(1).cpu().tolist()

    real_acc = _acc(real_z)

    # Generative prior: a *full-covariance* Gaussian fit to the real style
    # latents -- preserving the cross-dimension correlations that make a
    # coherent player (a diagonal prior breaks them and under-disperses).
    # Sample new, never-seen players from it.
    rz = real_z.cpu().numpy()
    mu = rz.mean(0)
    cov = np.cov(rz, rowvar=False) + 1e-6 * np.eye(latent_dim)
    rng = np.random.default_rng(seed)
    sampled = torch.tensor(
        rng.multivariate_normal(mu, cov, size=n_generated),
        dtype=torch.float32,
        device=device,
    )
    gen_acc = _acc(sampled)
    avg_acc = _acc(real_z.mean(0, keepdim=True)) * n_generated  # point mass

    pr = precision_recall(real_acc, gen_acc, k=3)
    pr_avg = precision_recall(real_acc, avg_acc, k=3)
    return GenerationResult(
        w1=wasserstein_1d(gen_acc, real_acc),
        js=js_divergence(gen_acc, real_acc),
        precision=pr.precision,
        recall=pr.recall,
        avg_recall=pr_avg.recall,
        gen_spread=statistics.pstdev(gen_acc),
        real_spread=statistics.pstdev(real_acc),
        n_real=len(real_acc),
        n_generated=len(gen_acc),
    )
