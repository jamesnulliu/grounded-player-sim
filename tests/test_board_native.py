"""Board-native backbone: encoding, masked variable-length loss, learning.

All CPU and offline (no W&B, no GPU, no engine oracle) -- the whole point of
the minimal board-native backbone is that the real-chess move-NLL machinery is
exercisable without those. torch is required (the 'train' extra); guarded.
"""

import pytest

from gps.interface import (
    DecisionPoint,
    Game,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import InjectionKind, Observation
from gps.policy.board_native import (
    BOARD_DIM,
    BoardNativeBackbone,
    _square_index,
    board_planes,
)
from gps.train.base import Trajectory

torch = pytest.importorskip("torch")

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Two positions with distinct best replies + distinct legal-set sizes, so a
# from/to model can actually learn board -> move (used by the smoke-train).
POS_A = (START, ("e2e4", "d2d4", "g1f3"), "e2e4")  # 3 legal
POS_B = (AFTER_E4, ("e7e5", "g8f6", "b8c6", "d7d5"), "e7e5")  # 4 legal


def _dp(fen, legal, *, tr=60.0, sp=2):
    return DecisionPoint(
        game=Game.CHESS,
        player_id="p",
        state=fen,
        legal_actions=tuple(legal),
        engine_reference=None,
        time_signal=TimeSignal(time_remaining=tr, move_number=1, phase="open"),
        recent_outcomes=OutcomeStream(recent=[], session_position=sp),
        context={},
    )


def _traj(player_id, n_steps):
    """Alternate POS_A / POS_B for ``n_steps`` decisions with fixed targets."""
    decisions, observations = [], []
    for i in range(n_steps):
        fen, legal, target = POS_A if i % 2 == 0 else POS_B
        decisions.append(_dp(fen, legal, sp=i))
        observations.append(Observation(move=target, time_spent=2.0 + i))
    return Trajectory(player_id, decisions, observations)


# --------------------------------------------------------------------------- #
# Encoding primitives
# --------------------------------------------------------------------------- #


def test_square_index():
    assert _square_index("a1") == 0
    assert _square_index("e2") == 12
    assert _square_index("h8") == 63
    assert _square_index("e4") == 28


def test_board_planes_startpos():
    planes = board_planes(START)
    assert len(planes) == BOARD_DIM
    # 32 pieces on the board at the start.
    assert sum(1 for v in planes if v == 1.0) == 32 + 1  # +1 side-to-move (w)
    assert planes[BOARD_DIM - 1] == 1.0  # white to move
    # White pawn (plane 0) on e2 (square 12); black to move flips the bit.
    assert planes[0 * 64 + 12] == 1.0
    assert board_planes(AFTER_E4)[BOARD_DIM - 1] == 0.0  # black to move


# --------------------------------------------------------------------------- #
# Variable-length + masked batching
# --------------------------------------------------------------------------- #


def test_encode_batch_shapes_and_masks():
    bb = BoardNativeBackbone(latent_dim=8)
    trajs = [_traj("long", 6), _traj("short", 3)]
    batch = bb.encode_batch(trajs)
    T, B = 6, 2
    A = 4  # max legal across positions (POS_B has 4)
    assert batch.feats.shape == (T, B, 4)
    assert batch.board.shape == (T, B, BOARD_DIM)
    assert batch.legal_from.shape == (T, B, A)
    assert batch.action_mask.shape == (T, B, A)
    assert batch.step_mask.shape == (T, B)
    # step_mask reflects true lengths (col 1 padded after step 3).
    assert batch.step_mask[:, 0].sum().item() == 6
    assert batch.step_mask[:, 1].sum().item() == 3
    assert bool(batch.step_mask[3, 1]) is False
    # action_mask reflects per-position legal counts (POS_A=3, POS_B=4).
    assert batch.action_mask[0, 0].sum().item() == 3  # POS_A at step 0
    assert batch.action_mask[1, 0].sum().item() == 4  # POS_B at step 1


def test_loss_is_finite_and_differentiable():
    from gps.latent.neural import NeuralInjector

    inj = NeuralInjector(kind=InjectionKind.HIDDEN, latent_dim=8, persist=True)
    bb = BoardNativeBackbone(latent_dim=8, hidden_dim=16)
    batch = bb.encode_batch([_traj("a", 5), _traj("b", 2)])
    latent = inj.latent_trajectory(batch.feats)
    out = bb.trajectory_loss(latent, batch, lam=0.5)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["move_nll"]) and torch.isfinite(
        out["timing_nll"]
    )
    out["loss"].backward()
    # Gradients reach both the injector recurrence and the board head.
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in inj.parameters()
    )
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in bb.parameters()
    )


def test_conv_trunk_trains_and_differentiates():
    # The spatial conv trunk is an alternative backbone: it must build, give a
    # finite loss, and pass gradients to its conv weights (the latent + heads
    # are shared with the MLP path).
    from gps.latent.neural import NeuralInjector

    inj = NeuralInjector(kind=InjectionKind.HIDDEN, latent_dim=8, persist=True)
    bb = BoardNativeBackbone(latent_dim=8, hidden_dim=16, trunk="conv")
    batch = bb.encode_batch([_traj("a", 5), _traj("b", 2)])
    latent = inj.latent_trajectory(batch.feats)
    out = bb.trajectory_loss(latent, batch, lam=0.5)
    assert torch.isfinite(out["loss"])
    net = bb._build()
    assert hasattr(net, "conv")  # the conv trunk is actually in use
    out["loss"].backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in net.conv.parameters()
    )


def test_padding_does_not_change_a_players_nll():
    # The short player's per-trajectory NLL must be identical whether or not a
    # longer player pads the batch (masking must zero out the padded tail).
    from gps.latent.neural import NeuralInjector

    inj = NeuralInjector(kind=InjectionKind.HIDDEN, latent_dim=8, persist=True)
    bb = BoardNativeBackbone(latent_dim=8, hidden_dim=16)
    short = _traj("short", 3)

    solo = bb.encode_batch([short])
    padded = bb.encode_batch([_traj("long", 7), short])
    with torch.no_grad():
        nll_solo = bb.per_traj_move_nll(
            inj.latent_trajectory(solo.feats), solo
        )[0]
        nll_padded = bb.per_traj_move_nll(
            inj.latent_trajectory(padded.feats), padded
        )[1]
    assert torch.allclose(nll_solo, nll_padded, atol=1e-5)


def test_train_eval_masks_partition_steps():
    bb = BoardNativeBackbone(latent_dim=8)
    trajs = [_traj("a", 10), _traj("b", 4)]
    batch = bb.encode_batch(trajs)
    splits = bb.split_indices(trajs, train_frac=0.7)
    assert splits == [7, 3]  # round(0.7*10)=7 ; round(0.7*4)=3
    train, held = bb.train_eval_masks(batch, splits)
    # Disjoint, and together they recover exactly the real decision steps.
    assert not bool((train & held).any())
    assert bool(((train | held) == batch.step_mask).all())
    # Per player: train has `split` steps, eval has the rest.
    assert train[:, 0].sum().item() == 7 and held[:, 0].sum().item() == 3
    assert train[:, 1].sum().item() == 3 and held[:, 1].sum().item() == 1


def test_smoke_train_reduces_move_nll():
    # A genuine (tiny) learning check: the board head + evolving latent should
    # drive train move-NLL well below the uniform-over-legal baseline.
    from gps.latent.neural import NeuralInjector

    torch.manual_seed(0)
    inj = NeuralInjector(
        kind=InjectionKind.HIDDEN, latent_dim=8, persist=True, seed=0
    )
    bb = BoardNativeBackbone(latent_dim=8, hidden_dim=32)
    trajs = [_traj(f"p{i}", 12) for i in range(4)]
    batch = bb.encode_batch(trajs)
    params = list(inj.parameters()) + list(bb.parameters())
    opt = torch.optim.AdamW(params, lr=1e-2)

    def move_nll():
        with torch.no_grad():
            lat = inj.latent_trajectory(batch.feats)
            return bb.trajectory_loss(lat, batch, lam=0.0)["move_nll"].item()

    init = move_nll()
    for _ in range(200):
        opt.zero_grad()
        lat = inj.latent_trajectory(batch.feats)
        bb.trajectory_loss(lat, batch, lam=0.5)["loss"].backward()
        opt.step()
    final = move_nll()
    # Targets are deterministic per position, so a working model gets near 0;
    # require a clear drop below the ~log(3.5) uniform-legal baseline.
    assert final < init - 0.3
    assert final < 0.5


def test_zero_inflated_timing_head_handles_zeros():
    # The zero-inflated head must (a) preserve real 0s through encode_batch and
    # (b) produce a finite, different think-time NLL than the log-normal.
    from gps.latent.neural import NeuralInjector

    traj = _traj("a", 6)
    # Make the first move a 0s premove.
    traj.observations[0] = Observation(
        move=traj.observations[0].move, time_spent=0.0
    )
    inj = NeuralInjector(kind=InjectionKind.HIDDEN, latent_dim=8, persist=True)

    zi = BoardNativeBackbone(
        latent_dim=8, hidden_dim=16, timing_model="zi_lognormal", seed=0
    )
    batch = zi.encode_batch([traj])
    assert float(batch.times[0, 0]) == 0.0  # zero preserved (not 1e-3)
    lat = inj.latent_trajectory(batch.feats)
    zi_nll = zi.per_traj_timing_nll(lat, batch)
    assert torch.isfinite(zi_nll).all()

    ln = BoardNativeBackbone(
        latent_dim=8, hidden_dim=16, timing_model="lognormal", seed=0
    )
    ln_nll = ln.per_traj_timing_nll(lat, ln.encode_batch([traj]))
    assert torch.isfinite(ln_nll).all()
    assert not torch.allclose(zi_nll, ln_nll)  # different model
