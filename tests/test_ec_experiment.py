"""E-C2 driver: per-player masked split, store round-trip, paired bootstrap.

CPU + offline W&B (the mandatory-tracking rule is satisfied by a present key in
offline mode, as in test_sft_training). Exercises the full real-data-shaped
path: build variable-length chess trajectories -> persist -> reload -> run_ec
-> a paired D-vs-B bootstrap over players. The synthetic targets here are
board-determined (no hidden dynamics), so this asserts the *machinery* is
correct and leak-free, not that D wins -- the win is a real-data question.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from gps.interface import (  # noqa: E402
    DecisionPoint,
    Game,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import Observation  # noqa: E402
from gps.train.base import Trajectory, TrajectoryDataset  # noqa: E402

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
POS_A = (START, ("e2e4", "d2d4", "g1f3"), "e2e4")
POS_B = (AFTER_E4, ("e7e5", "g8f6", "b8c6", "d7d5"), "e7e5")


@pytest.fixture()
def offline_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-test-key")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def _dp(fen, legal, sp):
    return DecisionPoint(
        game=Game.CHESS,
        player_id="p",
        state=fen,
        legal_actions=tuple(legal),
        engine_reference=None,
        time_signal=TimeSignal(time_remaining=40.0, move_number=1, phase="m"),
        recent_outcomes=OutcomeStream(recent=[], session_position=sp),
        context={},
    )


def _traj(player_id, n_steps):
    decisions, observations = [], []
    for i in range(n_steps):
        fen, legal, target = POS_A if i % 2 == 0 else POS_B
        decisions.append(_dp(fen, legal, sp=i))
        observations.append(Observation(move=target, time_spent=2.0 + i))
    return Trajectory(player_id, decisions, observations)


def _dataset(lengths):
    return TrajectoryDataset(
        [_traj(f"p{i}", n) for i, n in enumerate(lengths)]
    )


def test_run_ec_smoke_paired_bootstrap(offline_wandb):
    from gps.experiments.ec import run_ec

    # Variable-length trajectories (the real-data shape the global-window E-A1
    # path could not handle).
    ds = _dataset([14, 10, 12, 8])
    res = run_ec(
        ds,
        latent_dim=8,
        hidden_dim=16,
        epochs=8,
        lr=1e-2,
        seed=0,
        bootstrap_n=200,
    )
    import math

    assert math.isfinite(res.d_val_move_nll)
    assert math.isfinite(res.b_val_move_nll)
    # Equal capacity: the persist trick keeps D and B parameter-identical.
    assert res.d_params == res.b_params
    assert res.d_summary["status"] == "completed"
    assert res.b_summary["status"] == "completed"
    # One paired diff + held-out NLL per player; bootstrap over players.
    assert len(res.diff_per_player) == 4
    assert len(res.d_per_player) == len(res.b_per_player) == 4
    assert res.ci.n_units == 4
    assert res.ci.low <= res.ci.point <= res.ci.high
    assert "E-C2" in res.summary()


def test_run_ec_capacity_option_makes_b_bigger(offline_wandb):
    from gps.experiments.ec import run_ec

    res = run_ec(
        _dataset([10, 8, 12]),
        latent_dim=8,
        b_latent_dim=16,
        hidden_dim=16,
        epochs=4,
        lr=1e-2,
        seed=0,
        bootstrap_n=100,
    )
    assert res.b_params > res.d_params


def test_run_ec_from_persisted_dataset(offline_wandb, tmp_path):
    # The realistic flow: ingest persists -> experiment reloads -> run_ec.
    from gps.data.store import load_dataset, save_dataset
    from gps.experiments.ec import run_ec

    ds = _dataset([12, 10, 8])
    path = str(tmp_path / "dataset.jsonl.gz")
    save_dataset(ds, path)
    reloaded = load_dataset(path)

    res = run_ec(
        reloaded,
        latent_dim=8,
        hidden_dim=16,
        epochs=6,
        lr=1e-2,
        seed=1,
        bootstrap_n=100,
    )
    assert len(res.diff_per_player) == 3
    assert res.d_params == res.b_params


def test_b4_position_aware_adds_branching_factor():
    # The position-aware B4 baseline appends the branching factor (legal-move
    # count / 40) as a board-derived complexity proxy; default stays unchanged.
    from gps.experiments.ec import _b4_features

    dp = _dp("8/8/8/8/8/8/8/8 w - - 0 1", ["e2e4", "d2d4", "g1f3"], sp=0)
    base = _b4_features(dp)
    aware = _b4_features(dp, position_aware=True)
    assert len(aware) == len(base) + 1
    assert aware[: len(base)] == base
    assert aware[-1] == 3 / 40.0  # three legal moves


def test_b4_external_pred_appends_released_prediction():
    # G4: external_pred=True appends log(context['external_time_pred']) as one
    # more fitted feature; a missing prediction fails loudly (never silently
    # degrades to the proxy).
    from gps.experiments.ec import _b4_features

    dp = _dp("8/8/8/8/8/8/8/8 w - - 0 1", ["e2e4", "d2d4"], sp=0)
    aware = _b4_features(dp, position_aware=True)
    with pytest.raises(ValueError, match="external_time_pred"):
        _b4_features(dp, position_aware=True, external_pred=True)
    dp.context["external_time_pred"] = math.e  # log -> 1.0
    ext = _b4_features(dp, position_aware=True, external_pred=True)
    assert len(ext) == len(aware) + 1
    assert ext[: len(aware)] == aware
    assert abs(ext[-1] - 1.0) < 1e-9


def test_b4_maia_complexity_appends_released_difficulty():
    # G4: maia_complexity=True appends context['maia_entropy'] (a released
    # model's move-distribution entropy) as a difficulty feature; missing
    # entropy fails loudly.
    from gps.experiments.ec import _b4_features

    dp = _dp("8/8/8/8/8/8/8/8 w - - 0 1", ["e2e4", "d2d4"], sp=0)
    aware = _b4_features(dp, position_aware=True)
    with pytest.raises(ValueError, match="maia_entropy"):
        _b4_features(dp, position_aware=True, maia_complexity=True)
    dp.context["maia_entropy"] = 1.5
    ext = _b4_features(dp, position_aware=True, maia_complexity=True)
    assert len(ext) == len(aware) + 1
    assert ext[-1] == 1.5


def test_concentration_buckets_by_observable_feature(offline_wandb):
    # run_concentration(bucket_feature=...) buckets the held-out D-B by an
    # OBSERVABLE anchored dimension (real-data path) rather than a synthetic
    # hidden state -- it must run and label buckets by that feature.
    from gps.experiments.ec import run_concentration

    ds = _dataset([18, 16, 14, 12])
    r = run_concentration(
        ds,
        channel="timing",
        bucket_feature="time_pressure",
        n_buckets=2,
        latent_dim=8,
        hidden_dim=16,
        epochs=8,
        seed=0,
    )
    assert len(r.buckets) == 2
    assert all("time_pressure" in label for label, _, _ in r.buckets)
    assert all(math.isfinite(m) for _, _, m in r.buckets)


def test_concentration_stratified_variance_controlled(offline_wandb):
    # run_concentration_stratified (paper-readiness P0): same idea as
    # run_concentration but bootstraps over PLAYERS within each bucket and
    # also reports a variance-normalized gap, so it needs several players.
    from gps.experiments.ec import run_concentration_stratified

    ds = _dataset([20, 18, 22, 16, 24, 19])
    r = run_concentration_stratified(
        ds,
        channel="timing",
        bucket_feature="time_pressure",
        n_buckets=2,
        latent_dim=8,
        hidden_dim=16,
        epochs=8,
        seed=0,
        bootstrap_n=200,
    )
    assert len(r.buckets) == 2 == len(r.raw_ci) == len(r.normalized_ci)
    assert all("time_pressure" in label for label in r.buckets)
    assert all(math.isfinite(ci.point) for ci in r.raw_ci)
    assert all(math.isfinite(ci.point) for ci in r.normalized_ci)
    assert all(sd >= 0 for sd in r.bucket_decision_std)
    summary = r.summary()
    assert "variance-normalized ratio" in summary


def test_temporal_split_has_no_leakage(offline_wandb):
    # Belt-and-braces: the train mask the trainer uses must exclude every
    # held-out eval step (no peeking at the future), for every player.
    from gps.policy.board_native import BoardNativeBackbone

    ds = _dataset([14, 9, 11])
    bb = BoardNativeBackbone(latent_dim=8, hidden_dim=16)
    batch = bb.encode_batch(ds.trajectories)
    splits = bb.split_indices(ds.trajectories, train_frac=0.7)
    train_mask, eval_mask = bb.train_eval_masks(batch, splits)
    assert not bool((train_mask & eval_mask).any())
    # Every real decision step is in exactly one of train/eval.
    assert bool(((train_mask | eval_mask) == batch.step_mask).all())
    # Each player keeps some train AND some eval steps (non-degenerate split).
    assert bool((train_mask.sum(0) > 0).all())
    assert bool((eval_mask.sum(0) > 0).all())
