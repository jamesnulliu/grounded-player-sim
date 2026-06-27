"""Tests for the trainable recurrent latent injector (Milestone A, arm D).

These assert the three things that make ``NeuralInjector`` a usable ``f_phi``:
it exposes real (differentiable) parameters so SFT can fit it, it accumulates
state across a trajectory (unlike the memoryless control), and one learned
state renders to *both* injection channels. torch-gated so the CPU-only,
stdlib test suite still collects.
"""

import pytest

torch = pytest.importorskip("torch")

from gps.latent.base import InjectionKind  # noqa: E402
from gps.latent.neural import NeuralInjector  # noqa: E402
from gps.latent.structured import DIMENSIONS  # noqa: E402
from gps.policy.mock_backbone import MockBackbone  # noqa: E402
from gps.simulator import Simulator  # noqa: E402
from gps.synthetic.players import HysteresisTiltPlayer  # noqa: E402
from gps.synthetic.toy_game import ToyGame  # noqa: E402


def _short_trajectory(n_games=4, seed=0):
    player = HysteresisTiltPlayer(
        "p", ToyGame(seed=seed), seed=seed, base_beta=4.0
    )
    games = player.play_session(n_games=n_games)
    decisions = [dp for g in games for dp in g.decisions]
    observations = [o for g in games for o in g.observations]
    return player.player_id, decisions, observations


def test_neural_injector_has_trainable_parameters():
    # The whole point: real parameters() so SFTTrainer leaves its no-op guard.
    inj = NeuralInjector(kind=InjectionKind.HIDDEN, latent_dim=8)
    params = list(inj.parameters())
    assert params, "neural injector must expose trainable parameters"
    assert all(p.requires_grad for p in params)
    report = inj.param_report()
    assert report["n_parameters"] > 0
    assert report["latent_dim"] == 8


def test_neural_injector_unblocks_sft_guard():
    # Concretely: the SFTTrainer's parameter scan (which gates its no-op path)
    # now yields parameters, so the trainer would enter its real loop rather
    # than report "no trainable parameters".
    from gps.train.base import TrainConfig
    from gps.train.sft import SFTTrainer

    trainer = SFTTrainer(
        injector=NeuralInjector(kind=InjectionKind.HIDDEN),
        backbone=MockBackbone(),
        config=TrainConfig(),
    )
    assert list(trainer._trainable_parameters(torch))


def test_neural_injector_forward_is_differentiable():
    # f_phi must be differentiable end to end for SFT to have anything to fit.
    inj = NeuralInjector(kind=InjectionKind.HIDDEN, latent_dim=6, seed=1)
    net = inj._build()
    h = torch.zeros(1, 6)
    x = torch.rand(1, len(DIMENSIONS))
    for _ in range(3):
        h = net.step(x, h)
    loss = net.anchored(h).sum()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradients flowed to f_phi parameters"
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_neural_injector_accumulates_state():
    # Unlike the memoryless control, the probe vector must depend on history:
    # after stepping through a trajectory the state moves away from z_0.
    inj = NeuralInjector(kind=InjectionKind.HIDDEN, latent_dim=8, seed=2)
    pid, decisions, observations = _short_trajectory()
    state = inj.initial_state(pid)
    z0 = list(state.probe_vector)
    for dp, obs in zip(decisions, observations):
        state = inj.update(state, dp, obs)
    z_end = state.probe_vector
    assert len(z0) == len(z_end) == len(DIMENSIONS)
    assert z0 != z_end  # the recurrence actually carried something


def test_neural_injector_both_channels_from_one_state():
    # RQ6 / Milestone E: the same learned state renders to verbal and hidden.
    pid, decisions, _ = _short_trajectory()
    dp = decisions[10]

    hidden = NeuralInjector(kind=InjectionKind.HIDDEN, seed=3)
    verbal = NeuralInjector(kind=InjectionKind.VERBAL, seed=3)
    hs = hidden.initial_state(pid)
    vs = verbal.initial_state(pid)

    hinj = hidden.render(hs, dp)
    vinj = verbal.render(vs, dp)
    assert hinj.kind is InjectionKind.HIDDEN
    assert hinj.vector is not None and len(hinj.vector) == len(DIMENSIONS)
    assert vinj.kind is InjectionKind.VERBAL
    assert vinj.text and vinj.text.startswith("Current player state:")


def test_neural_injector_runs_through_simulator():
    # Integration: it drops into the same Simulator loop the Phase-0 arms use.
    pid, decisions, observations = _short_trajectory()
    sim = Simulator(
        NeuralInjector(kind=InjectionKind.HIDDEN, seed=4),
        MockBackbone(use_latent=True),
    )
    results = sim.run_trajectory(pid, decisions, observations)
    assert len(results) == len(decisions)
    assert all(r.prediction.moves.probs for r in results)
    assert all(len(r.latent_probe) == len(DIMENSIONS) for r in results)


def test_neural_injector_is_deterministic_given_seed():
    # LatentStateInjector contract: deterministic given parameters + inputs.
    pid, decisions, observations = _short_trajectory()

    def final_probe(seed):
        inj = NeuralInjector(kind=InjectionKind.HIDDEN, seed=seed)
        state = inj.initial_state(pid)
        for dp, obs in zip(decisions, observations):
            state = inj.update(state, dp, obs)
        return state.probe_vector

    assert final_probe(7) == final_probe(7)
