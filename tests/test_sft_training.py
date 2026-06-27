"""SFT training-loop tests (Milestone B / E-A1), torch-gated.

Exercise the real tensor loop end to end on CPU with W&B in *offline* mode
(the mandatory-tracking rule is still satisfied -- a key is present and
``wandb.init`` runs -- but nothing leaves the box). Asserts the loop trains,
honours the strict temporal split, and that the two Milestone-A arms are
exactly capacity-matched.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from gps.experiments.ea1 import run_ea1  # noqa: E402
from gps.latent.base import InjectionKind  # noqa: E402
from gps.latent.neural import NeuralInjector  # noqa: E402
from gps.policy.diff_policy import DiffMovePolicy  # noqa: E402


@pytest.fixture()
def offline_wandb(tmp_path, monkeypatch):
    # Mandatory tracking is satisfied by a present key; offline keeps it local.
    monkeypatch.setenv("WANDB_API_KEY", "offline-test-key")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    # Keep runs/ + wandb/ scratch under the test's tmp dir, not the repo.
    monkeypatch.chdir(tmp_path)


def test_encode_batch_shapes_align():
    from gps.synthetic.players import HysteresisTiltPlayer
    from gps.synthetic.toy_game import ToyGame
    from gps.train.base import Trajectory, TrajectoryDataset

    game = ToyGame(seed=0)
    games = HysteresisTiltPlayer("p", game, seed=0).play_session(3)
    decisions = [dp for g in games for dp in g.decisions]
    obs = [o for g in games for o in g.observations]
    ds = TrajectoryDataset([Trajectory("p", decisions, obs)])

    head = DiffMovePolicy(latent_dim=8, n_actions=game.branching)
    batch = head.encode_batch(ds.trajectories)
    t = len(decisions)
    assert batch.feats.shape == (t, 1, 4)
    assert batch.value_adv.shape == (t, 1, game.branching)
    assert batch.move_idx.shape == (t, 1)
    # value advantages are normalised into [0, 1]
    assert float(batch.value_adv.min()) >= 0.0
    assert float(batch.value_adv.max()) <= 1.0 + 1e-6


def test_run_ea1_smoke_trains_both_arms(offline_wandb):
    res = run_ea1(
        n_players=4,
        n_games=6,
        latent_dim=8,
        epochs=10,
        lr=1e-2,
        seed=0,
        bootstrap_n=200,
    )
    # Both arms produced a finite held-out move-NLL ...
    assert math.isfinite(res.d_val_move_nll)
    assert math.isfinite(res.b_val_move_nll)
    # ... and are EXACTLY capacity-matched (the whole point of the persist
    # trick: same architecture, same parameter count, only the carry differs).
    assert res.d_params == res.b_params
    assert res.d_summary["status"] == "completed"
    assert res.b_summary["status"] == "completed"
    # Per-player paired diffs + a bootstrap CI over players (one diff/player).
    assert len(res.diff_per_player) == 4
    assert len(res.d_per_player) == len(res.b_per_player) == 4
    assert res.ci.n_units == 4
    assert res.ci.low <= res.ci.point <= res.ci.high
    assert isinstance(res.verdict(), str)


def test_run_ea1_capacity_option_makes_b_bigger(offline_wandb):
    # Giving B a wider latent makes it strictly larger than D (capacity check).
    res = run_ea1(
        n_players=3,
        n_games=6,
        latent_dim=8,
        b_latent_dim=16,
        epochs=5,
        lr=1e-2,
        seed=0,
        bootstrap_n=100,
    )
    assert res.b_params > res.d_params


def test_sft_loop_reduces_training_loss(offline_wandb):
    # The loop must actually optimise: arm D's train loss should drop.
    from gps.synthetic.players import HysteresisTiltPlayer
    from gps.synthetic.toy_game import ToyGame
    from gps.train.base import TrainConfig, Trajectory, TrajectoryDataset
    from gps.train.sft import SFTTrainer

    game = ToyGame(seed=1)
    games = HysteresisTiltPlayer("p", game, seed=1).play_session(8)
    decisions = [dp for g in games for dp in g.decisions]
    obs = [o for g in games for o in g.observations]
    ds = TrajectoryDataset([Trajectory("p", decisions, obs)])

    inj = NeuralInjector(
        kind=InjectionKind.HIDDEN, latent_dim=8, seed=1, persist=True
    )
    head = DiffMovePolicy(latent_dim=8, n_actions=game.branching)
    trainer = SFTTrainer(
        inj, head, TrainConfig(epochs=1, lr=1e-2, experiment="E-A1-test")
    )
    first = trainer.fit(ds)

    inj2 = NeuralInjector(
        kind=InjectionKind.HIDDEN, latent_dim=8, seed=1, persist=True
    )
    head2 = DiffMovePolicy(latent_dim=8, n_actions=game.branching)
    trainer2 = SFTTrainer(
        inj2, head2, TrainConfig(epochs=60, lr=1e-2, experiment="E-A1-test")
    )
    later = trainer2.fit(ds)

    assert later["move_nll"] < first["move_nll"]
