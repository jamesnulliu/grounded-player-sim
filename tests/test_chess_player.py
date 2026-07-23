"""Chess-shaped hidden-tilt player + run_ec determinism (the E-C control rig).

CPU + offline W&B. These assert the *machinery* and the determinism fix (same
seed -> identical results, and arms D/B at identical capacity). They do NOT
assert D beats B: on a from-scratch board-native backbone at this data scale
the evolving latent does *not* reliably beat the memoryless twin -- a finding,
not a flake, so the test must not pretend otherwise.
"""

import pytest

from gps.latent.structured import DIMENSIONS, history_features
from gps.synthetic.chess_players import (
    HiddenTiltChessPlayer,
    build_hidden_tilt_dataset,
)

torch = pytest.importorskip("torch")


@pytest.fixture()
def offline_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "offline-test-key")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------------- #
# The synthetic player
# --------------------------------------------------------------------------- #


def test_player_emits_chess_shaped_decisions():
    traj = HiddenTiltChessPlayer(
        "p", seed=0, plies_per_game=6
    ).build_trajectory(5)
    assert len(traj.decisions) == 5 * 6
    assert len(traj.observations) == len(traj.decisions)
    d0 = traj.decisions[0]
    # Real FEN state, declared legal UCI moves, dual-use engine reference.
    assert isinstance(d0.state, str) and d0.state.split()[0].count("/") == 7
    assert all(len(m) == 4 for m in d0.legal_actions)
    assert d0.engine_reference is not None
    assert traj.observations[0].move in d0.legal_actions
    # history_features is computable (both arms' shared input).
    assert set(history_features(d0)) == set(DIMENSIONS)


def test_history_snapshot_not_aliased_across_games():
    # Each game freezes its own history snapshot; later games must not leak
    # back into earlier decisions (the aliasing trap from design.md section 5).
    traj = HiddenTiltChessPlayer(
        "p", seed=1, plies_per_game=4
    ).build_trajectory(3)
    first_game = traj.decisions[0].recent_outcomes
    last_game = traj.decisions[-1].recent_outcomes
    assert first_game is not last_game
    assert len(first_game.recent) == 0  # game 0 saw no prior games
    assert len(last_game.recent) == 2  # game 2 saw games 0 and 1


def test_hidden_state_evolves_with_losses():
    # The hidden h is a leaky loss integral -> it must actually move across the
    # session (otherwise there is nothing for the latent to track).
    traj = HiddenTiltChessPlayer(
        "p", seed=3, plies_per_game=6
    ).build_trajectory(20)
    hs = [d.context["hidden_h"] for d in traj.decisions]
    assert max(hs) > min(hs) + 1e-6


def test_dataset_population_distinct_players():
    ds = build_hidden_tilt_dataset(n_players=5, n_games=8, seed=0)
    assert len(ds) == 5
    assert ds.players() == {f"hidden-tilt-{i}" for i in range(5)}
    # Distinct seeds -> distinct realized trajectories (not all identical).
    moves = [tuple(o.move for o in t.observations) for t in ds.trajectories]
    assert len(set(moves)) > 1


# --------------------------------------------------------------------------- #
# run_ec determinism (the fix: seeded backbone init, fair D/B comparison)
# --------------------------------------------------------------------------- #


def test_run_ec_is_deterministic_and_capacity_matched(offline_wandb):
    from gps.experiments.ec import run_ec

    ds = build_hidden_tilt_dataset(n_players=6, n_games=10, seed=0)
    kw = dict(latent_dim=8, hidden_dim=24, epochs=15, lr=1e-2, seed=0)
    r1 = run_ec(ds, bootstrap_n=200, **kw)
    r2 = run_ec(ds, bootstrap_n=200, **kw)
    # Same seed -> byte-identical per-player NLLs (was false before the seeded
    # backbone init: arms drew different global-RNG initializations).
    assert r1.d_per_player == r2.d_per_player
    assert r1.b_per_player == r2.b_per_player
    # Equal capacity: D and B differ only by the persist bit.
    assert r1.d_params == r1.b_params
    assert r1.ci.n_units == 6


def _sessioned_traj(layout, per_game=3):
    """Trajectory with explicit (game_index, session_position) per game.

    Mirrors what gps.data.lichess.build_trajectory persists: game index is
    ``len(recent_outcomes.recent)`` and a session's first game has
    ``session_position == 0``.
    """
    from gps.interface import (
        DecisionPoint,
        Game,
        Outcome,
        OutcomeStream,
        TimeSignal,
    )
    from gps.latent.base import Observation
    from gps.train.base import Trajectory

    decisions, obs = [], []
    for g, sp in layout:
        stream = OutcomeStream(
            recent=[Outcome(won=True)] * g, session_position=sp
        )
        for _ in range(per_game):
            decisions.append(
                DecisionPoint(
                    game=Game.CHESS,
                    player_id="p",
                    state="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
                    legal_actions=("e2e4", "e2e3"),
                    engine_reference=None,
                    time_signal=TimeSignal(move_number=1),
                    recent_outcomes=stream,
                    context={},
                )
            )
            obs.append(Observation(move="e2e4"))
    return Trajectory("p", decisions, obs)


def test_session_split_holds_out_later_sessions():
    from gps.experiments.ec import session_split_indices

    # Session 1 = games 0,1,2 (sp 0,1,2); session 2 = games 3,4 (sp 0,1).
    # 3 decisions/game -> 15 decisions; with train_frac 0.5 over 2 sessions we
    # train on session 1 and hold out session 2 -> split at game 3 == idx 9.
    traj = _sessioned_traj([(0, 0), (1, 1), (2, 2), (3, 0), (4, 1)])
    (split,) = session_split_indices([traj], train_frac=0.5)
    assert split == 9  # first decision of the 2nd session
    # Train side is entirely session 1, eval side entirely session 2.
    assert all(
        d.recent_outcomes.session_position in (0, 1, 2)
        and len(d.recent_outcomes.recent) <= 2
        for d in traj.decisions[:split]
    )
    assert all(
        len(d.recent_outcomes.recent) >= 3 for d in traj.decisions[split:]
    )


def test_session_split_falls_back_when_single_session():
    from gps.experiments.ec import session_split_indices

    # One sitting (session_position never resets to 0 after game 0, as the
    # synthetic chess player numbers it) -> fall back to a move-fraction split.
    traj = HiddenTiltChessPlayer(
        "p", seed=0, plies_per_game=4
    ).build_trajectory(5)
    (split,) = session_split_indices([traj], train_frac=0.7)
    assert 0 < split < len(traj.decisions)


def test_board_native_seed_controls_init():
    from gps.policy.board_native import BoardNativeBackbone

    a = BoardNativeBackbone(latent_dim=4, seed=7)._build()
    b = BoardNativeBackbone(latent_dim=4, seed=7)._build()
    c = BoardNativeBackbone(latent_dim=4, seed=8)._build()
    pa = torch.cat([p.flatten() for p in a.parameters()])
    pb = torch.cat([p.flatten() for p in b.parameters()])
    pc = torch.cat([p.flatten() for p in c.parameters()])
    assert torch.equal(pa, pb)  # same seed -> identical init
    assert not torch.equal(pa, pc)  # different seed -> different init


def test_static_individual_injector_per_player_constant():
    from gps.latent.base import InjectionKind
    from gps.latent.static_individual import StaticIndividualInjector
    from gps.policy.board_native import BoardNativeBackbone

    ds = build_hidden_tilt_dataset(n_players=4, n_games=6, seed=0)
    bb = BoardNativeBackbone(latent_dim=8)
    batch = bb.encode_batch(ds.trajectories)
    inj = StaticIndividualInjector(
        [t.player_id for t in ds.trajectories],
        kind=InjectionKind.HIDDEN,
        latent_dim=8,
        seed=0,
    )
    lat = inj.latent_trajectory(batch.feats, player_ids=batch.player_ids)
    T, B, L = lat.shape
    assert (B, L) == (4, 8)
    # Constant over time (no dynamics) ...
    assert torch.allclose(lat[0], lat[-1])
    # ... but differs across players (identity is encoded).
    assert not torch.allclose(lat[0, 0], lat[0, 1])


def test_run_ec_static_control_is_ec1(offline_wandb):
    from gps.experiments.ec import run_ec

    ds = build_hidden_tilt_dataset(n_players=6, n_games=10, seed=0)
    res = run_ec(
        ds,
        latent_dim=8,
        hidden_dim=24,
        epochs=6,
        lr=1e-2,
        seed=0,
        batch_size=16,
        control="static",
        bootstrap_n=200,
    )
    assert res.label == "E-C1"
    assert len(res.diff_per_player) == 6
    # D is a GRU, B2 an embedding -> capacities differ (reported, not equal).
    assert res.d_params != res.b_params


def test_state_recovery_d_beats_b(offline_wandb):
    # RQ2: the evolving latent should recover the ground-truth hidden state
    # (hidden_h) better than the memoryless twin. Deterministic (seeded).
    from gps.experiments.ec import run_state_recovery

    # n_games=24 gives the hidden integral enough range that the evolving
    # latent's accumulation advantage is clear (fewer games -> h barely moves).
    ds = build_hidden_tilt_dataset(n_players=8, n_games=24, seed=0)
    res = run_state_recovery(
        ds,
        target_key="hidden_h",
        latent_dim=8,
        hidden_dim=32,
        epochs=150,
        seed=0,
    )
    assert res.n_eval > 0
    assert -1.0 <= res.b_r2 <= 1.0 and res.d_r2 <= 1.0
    # The accumulating latent recovers the hidden integral better than the
    # memoryless reader of instantaneous features (delta R^2 ~ +0.25 here).
    assert res.d_r2 > res.b_r2 + 0.05
    assert "E-C4" in res.summary()


def test_state_recovery_requires_ground_truth(offline_wandb):
    from gps.experiments.ec import run_state_recovery

    ds = build_hidden_tilt_dataset(n_players=4, n_games=8, seed=0)
    with pytest.raises(ValueError, match="hidden state"):
        run_state_recovery(ds, target_key="not_a_key", epochs=2)


def test_causal_intervention_latent_is_used(offline_wandb):
    # RQ2 (use, not just presence): clamping the latent toward "tilted" along
    # the hidden-state direction must flatten the move distribution (entropy
    # up) and change predictions (KL > 0) -- the policy USES the latent.
    from gps.experiments.ec import run_causal_intervention

    ds = build_hidden_tilt_dataset(n_players=10, n_games=24, seed=0)
    res = run_causal_intervention(
        ds, alpha=4.0, latent_dim=8, hidden_dim=32, epochs=120, seed=0
    )
    assert res.n_eval > 0
    # Tilt flattens moves (more blunders) -- the expected causal direction.
    assert res.entropy_tilted > res.entropy_calm
    # Predictions actually change under the clamp (not presence-without-use).
    assert res.move_kl > 1e-4
    assert "E-C4 causal" in res.summary()


def test_timing_vs_aggregate_runs(offline_wandb):
    # E-C6 B4 ablation: the aggregate (B4) and aggregate+latent (B4+z) models
    # fit and produce a bootstrap over players. (Direction is a real-data
    # question; here we only assert the machinery.)
    import math

    from gps.experiments.ec import run_timing_vs_aggregate

    ds = build_hidden_tilt_dataset(n_players=8, n_games=16, seed=0)
    res = run_timing_vs_aggregate(
        ds, split_mode="fraction", epochs=20, seed=0, bootstrap_n=200
    )
    assert res.n_players == 8
    assert res.mode == "aggregate"
    assert math.isfinite(res.b4_nll) and math.isfinite(res.b4z_nll)
    assert res.add_ci.n_units == 8
    assert "E-C6" in res.summary()


def test_g4_external_and_pure_external_modes(offline_wandb):
    # G4 (documents/g4_plan.md): the add-on test runs against a released
    # model's cached per-move think-time. We stamp a synthetic external pred
    # (a decent-but-imperfect predictor) onto every decision and assert BOTH
    # modes produce a valid bootstrap over players.
    import math

    from gps.experiments.ec import run_timing_vs_aggregate

    ds = build_hidden_tilt_dataset(n_players=8, n_games=16, seed=0)
    for traj in ds.trajectories:
        for dp, obs in zip(traj.decisions, traj.observations):
            dp.context["external_time_pred"] = max(
                0.05, (obs.time_spent or 0.5) * 0.9 + 0.2
            )

    # (a) external prediction as one more FITTED baseline feature.
    ext = run_timing_vs_aggregate(
        ds,
        split_mode="fraction",
        epochs=20,
        seed=0,
        bootstrap_n=200,
        external_pred=True,
    )
    assert ext.mode == "external"
    assert math.isfinite(ext.b4_nll) and math.isfinite(ext.b4z_nll)
    assert ext.add_ci.n_units == 8
    assert "external" in ext.summary()

    # (b) pure-external: baseline mu is log(released pred) LOCKED (+intercept);
    # B+z adds the latent as the only extra predictor.
    pure = run_timing_vs_aggregate(
        ds,
        split_mode="fraction",
        epochs=20,
        seed=0,
        bootstrap_n=200,
        pure_external=True,
    )
    assert pure.mode == "pure_external"
    assert math.isfinite(pure.b4_nll) and math.isfinite(pure.b4z_nll)
    assert pure.add_ci.n_units == 8
    assert len(pure.b4z_per_player) == len(pure.player_ids) == 8
    assert "released model" in pure.summary()

    # (c) the same locked external baseline with a static per-player latent.
    static = run_timing_vs_aggregate(
        ds,
        split_mode="fraction",
        epochs=20,
        seed=0,
        bootstrap_n=200,
        pure_external=True,
        latent_control="static",
    )
    assert static.latent_control == "static"
    assert len(static.b4z_per_player) == len(static.player_ids) == 8
    assert math.isfinite(static.b4z_nll)

    # (d) the same locked external baseline with the hand-designed
    # structured-memory control (no gradient training; lstsq readout only).
    memory = run_timing_vs_aggregate(
        ds,
        split_mode="fraction",
        epochs=20,
        seed=0,
        bootstrap_n=200,
        pure_external=True,
        latent_control="memory",
    )
    assert memory.latent_control == "memory"
    assert len(memory.b4z_per_player) == len(memory.player_ids) == 8
    assert math.isfinite(memory.b4z_nll)
    assert math.isnan(memory.d_nll)  # no trained D model in this arm
    # Same players, same held-out steps as the learned arms.
    assert memory.player_ids == pure.player_ids


def test_g4_missing_external_pred_fails_loudly(offline_wandb):
    # Without a cached external prediction the G4 modes must raise, never
    # silently fall back to the hand-built proxy.
    from gps.experiments.ec import run_timing_vs_aggregate

    ds = build_hidden_tilt_dataset(n_players=6, n_games=12, seed=0)
    with pytest.raises(ValueError, match="external_time_pred"):
        run_timing_vs_aggregate(
            ds, split_mode="fraction", epochs=5, seed=0, pure_external=True
        )


def test_concentration_in_high_dynamics_moments(offline_wandb):
    # The latent's edge should localize to high-tilt decisions (where the
    # hidden state is active), not the calm baseline -- the design.md s5 check.
    from gps.experiments.ec import run_concentration

    ds = build_hidden_tilt_dataset(n_players=10, n_games=20, seed=0)
    res = run_concentration(
        ds, channel="move", latent_dim=8, hidden_dim=32, epochs=15, seed=0
    )
    low = res.buckets[0][2]
    high = res.buckets[-1][2]
    # D helps more (more negative dD-B) at high tilt than at low tilt.
    assert high < low
    assert high < 0  # D actually beats B in the high-tilt bucket
    assert "concentration" in res.summary()


def test_rq6_verbal_vs_hidden_channel(offline_wandb):
    # RQ6: the verbal channel delivers only the anchored DIMENSIONS; the hidden
    # channel delivers the full latent vector. Same recurrence (same seed), so
    # this isolates the channel-capacity tradeoff. Machinery + readout width
    # are asserted; direction (hidden richer) is a data question.
    from gps.experiments.ec import run_rq6
    from gps.latent.structured import DIMENSIONS

    ds = build_hidden_tilt_dataset(n_players=8, n_games=20, seed=0)
    res = run_rq6(
        ds,
        split_mode="fraction",
        latent_dim=8,
        hidden_dim=16,
        epochs=15,
        seed=0,
        bootstrap_n=200,
    )
    assert res.ci.n_units == 8
    assert len(res.hidden_per_player) == len(res.verbal_per_player) == 8
    assert "E-E1" in res.summary()
    # The verbal injector emits len(DIMENSIONS)-wide latents (the readout).
    from gps.latent.neural import NeuralInjector

    inj = NeuralInjector(latent_dim=8, persist=True, readout=True)
    batch = ds.trajectories[0].decisions  # just for a feats tensor below
    assert batch is not None
    feats = torch.zeros(3, 2, len(DIMENSIONS))
    assert inj.latent_trajectory(feats).shape[-1] == len(DIMENSIONS)
