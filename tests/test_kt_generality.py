"""RQ5 generality: the dynamic-latent framework in knowledge tracing.

CPU + offline W&B. Asserts the *machinery* ports cleanly to a non-game domain
(the same injector + trainer + per-player eval, only the backbone swapped) and
is deterministic. The quantitative win (timing) is a tuned-regime result shown
in documents/results_ec.md, not asserted single-seed here (it wobbles, as in
chess).
"""

import math

import pytest

from gps.experiments.kt import (
    KnowledgeTracingStudent,
    build_kt_dataset,
    run_kt,
)
from gps.interface import Game
from gps.latent.structured import DIMENSIONS, history_features

torch = pytest.importorskip("torch")


@pytest.fixture()
def offline_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-test-key")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def test_kt_student_emits_shared_schema():
    traj = KnowledgeTracingStudent("s", seed=0).build_trajectory(20)
    assert len(traj.decisions) == 20
    d0 = traj.decisions[0]
    assert d0.game is Game.KNOWLEDGE_TRACING
    assert d0.legal_actions == ("correct", "incorrect")
    assert traj.observations[0].move in ("correct", "incorrect")
    assert d0.engine_reference is not None  # IRT-style oracle
    # The SAME history_features the chess injector uses are computable here.
    assert set(history_features(d0)) == set(DIMENSIONS)
    # Hidden frustration state recorded + actually moves over the sequence.
    hs = [d.context["hidden_h"] for d in traj.decisions]
    assert max(hs) > min(hs) + 1e-6


def test_run_kt_machinery_and_determinism(offline_wandb):
    # The whole E-C machinery runs unchanged on a non-game domain ...
    ds = build_kt_dataset(
        n_students=8, n_items=40, seed=0, base_skill=1.5, tilt_scale=1.5
    )
    kw = dict(latent_dim=8, hidden_dim=16, epochs=20, seed=0)
    r1 = run_kt(ds, bootstrap_n=200, **kw)
    r2 = run_kt(ds, bootstrap_n=200, **kw)
    assert r1.ci.n_units == 8
    assert math.isfinite(r1.ci.point)
    # Equal capacity (only the persist bit differs) + deterministic.
    assert r1.d_params == r1.b_params
    assert r1.d_per_player == r2.d_per_player
    assert r1.timing_ci is not None
    assert "E-D1" in r1.summary()


def test_population_recovers_heterogeneity(offline_wandb):
    # E-F2 (Milestone F): with real per-student skill heterogeneity, the
    # per-individual latent should reproduce the population's accuracy
    # distribution far better than a population-average baseline (point mass).
    from gps.experiments.kt import build_kt_dataset, run_population

    ds = build_kt_dataset(
        n_students=20,
        n_items=100,
        seed=0,
        base_skill=1.5,
        tilt_scale=1.5,
        rho=0.85,
        skill_spread=1.5,
    )
    res = run_population(ds, latent_dim=16, hidden_dim=32, epochs=40, seed=0)
    assert res.n == 20
    assert res.observed_spread > 0.05  # the cohort really is heterogeneous
    assert res.model_spread > 0  # the model is NOT the average person
    # The per-individual model matches the observed distribution better than
    # the average-person baseline, and tracks who is skilled.
    assert res.w1_model < res.w1_average
    assert res.pearson > 0.5
    assert "E-F2" in res.summary()


def test_distributional_metrics_average_person_signature():
    # The "positive average person" has a textbook signature: perfect precision
    # (the mean is plausible) but ~zero recall (it covers none of the spread).
    import numpy as np

    from gps.eval.distributional import (
        js_divergence,
        precision_recall,
        wasserstein_1d,
    )

    rng = np.random.default_rng(0)
    real = rng.normal(0.6, 0.15, 40).tolist()
    good = rng.normal(0.6, 0.15, 40).tolist()  # covers the spread
    avg = [0.6] * 40  # average-person point mass

    # A matching sample beats the point mass on every distance.
    assert wasserstein_1d(good, real) < wasserstein_1d(avg, real)
    assert js_divergence(good, real) < js_divergence(avg, real)
    pr_good = precision_recall(real, good, k=3)
    pr_avg = precision_recall(real, avg, k=3)
    assert pr_good.recall > 0.5  # the good model covers the diversity
    assert pr_avg.recall < 0.1  # the average person covers ~none of it
    assert pr_avg.precision > 0.8  # ... but is itself plausible


def test_population_generation_is_plausible_and_diverse(offline_wandb):
    # E-F1: sampling latents from a prior fit to the real population generates
    # novel players that are plausible AND cover the real diversity -- unlike
    # the average-person (recall 0). Distributional, not pointwise.
    from gps.experiments.kt import build_kt_dataset, run_generation

    ds = build_kt_dataset(
        n_students=24,
        n_items=120,
        seed=0,
        base_skill=1.5,
        tilt_scale=1.5,
        rho=0.85,
        skill_spread=1.5,
    )
    res = run_generation(
        ds, latent_dim=16, hidden_dim=32, epochs=50, seed=0, n_generated=200
    )
    assert res.n_generated == 200
    assert res.precision > 0.7  # generated players are plausible
    assert res.recall > 0.5  # ... and cover much of the real diversity
    assert res.recall > res.avg_recall  # beating the average-person baseline
    assert "E-F1" in res.summary()
