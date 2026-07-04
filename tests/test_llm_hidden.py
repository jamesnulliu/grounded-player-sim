"""Tests for the G2/G3 hidden-vs-verbal LLM SFT assembly (Milestone G).

The model forward + LoRA fit is GPU-host wiring, but the two error-prone pieces
-- per-decision example assembly and the completion-NLL step (which token
positions are scored) -- are pure/torch and must stay correct, so we test them.
torch-gated so the stdlib CPU suite still collects.
"""

import pytest

torch = pytest.importorskip("torch")

from gps.experiments.llm_hidden import (  # noqa: E402
    assemble_completion_step,
    build_examples,
)
from gps.interface import (  # noqa: E402
    DecisionPoint,
    Game,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import InjectionKind, Observation  # noqa: E402
from gps.latent.neural import NeuralInjector  # noqa: E402
from gps.policy.sglang_backbone import SGLangBackbone  # noqa: E402
from gps.train.base import Trajectory  # noqa: E402

_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _trajectory(n=3):
    decisions, observations = [], []
    for i in range(n):
        decisions.append(
            DecisionPoint(
                game=Game.CHESS,
                player_id="magnus",
                state=_FEN,
                legal_actions=("e2e4", "d2d4", "g1f3"),
                engine_reference=None,
                time_signal=TimeSignal(time_remaining=42.0 - i),
                recent_outcomes=OutcomeStream(),
            )
        )
        observations.append(Observation(move="e2e4", time_spent=1.5 + i))
    return Trajectory(
        player_id="magnus", decisions=decisions, observations=observations
    )


# --- assemble_completion_step ------------------------------------------
def test_step_no_prefix_scores_only_completion():
    embed = torch.nn.Embedding(30, 8)
    emb, attn, labels = assemble_completion_step(embed, [1, 2, 3], [4, 5])
    assert tuple(emb.shape) == (1, 5, 8)  # P + C
    assert torch.equal(attn, torch.ones(1, 5, dtype=torch.long))
    # prompt positions ignored; completion positions carry the target ids
    assert labels[0, :3].tolist() == [-100, -100, -100]
    assert labels[0, 3:].tolist() == [4, 5]


def test_step_with_prefix_ignores_prefix_and_prompt():
    embed = torch.nn.Embedding(30, 8)
    prefix = torch.zeros(2, 8)  # [n_prefix, hidden]
    emb, attn, labels = assemble_completion_step(
        embed, [1, 2, 3], [4, 5], prefix
    )
    assert tuple(emb.shape) == (1, 7, 8)  # n_prefix + P + C
    assert attn.sum().item() == 7  # every position attended
    # 2 prefix + 3 prompt ignored; only the 2 completion tokens scored
    assert labels[0, :5].tolist() == [-100] * 5
    assert labels[0, 5:].tolist() == [4, 5]


# --- build_examples ----------------------------------------------------
def test_build_examples_counts_and_completion():
    traj = _trajectory(3)
    bb = SGLangBackbone()
    inj = NeuralInjector(kind=InjectionKind.HIDDEN, latent_dim=8)
    ex = build_examples(traj, inj, bb, channel="none", target="move")
    assert len(ex) == 3
    assert all(e.completion == " e2e4" for e in ex)
    assert all(e.prompt.endswith("\nMove:") for e in ex)
    assert all(e.latent is None and e.channel == "none" for e in ex)


def test_build_examples_hidden_carries_full_latent():
    traj = _trajectory(3)
    bb = SGLangBackbone()
    inj = NeuralInjector(kind=InjectionKind.HIDDEN, latent_dim=8)
    ex = build_examples(traj, inj, bb, channel="hidden", target="move")
    assert all(e.latent is not None and len(e.latent) == 8 for e in ex)
    # the evolving latent changes across steps (it is not a constant vector)
    assert ex[0].latent != ex[-1].latent


def test_build_examples_hidden_prompt_matches_none_but_verbal_differs():
    # Channel-only contrast: hidden must NOT change the text; verbal must.
    traj = _trajectory(2)
    bb = SGLangBackbone()
    none = build_examples(
        traj, NeuralInjector(kind=InjectionKind.HIDDEN), bb, channel="none"
    )
    hidden = build_examples(
        traj, NeuralInjector(kind=InjectionKind.HIDDEN), bb, channel="hidden"
    )
    verbal = build_examples(
        traj, NeuralInjector(kind=InjectionKind.VERBAL), bb, channel="verbal"
    )
    for h, n in zip(hidden, none):
        assert h.prompt == n.prompt  # hidden rides in as embeddings, not words
    for v, n in zip(verbal, none):
        assert len(v.prompt) > len(n.prompt)  # verbal adds a state-note line


def test_build_examples_time_target_formats_completion():
    traj = _trajectory(2)
    bb = SGLangBackbone()
    inj = NeuralInjector(kind=InjectionKind.HIDDEN, latent_dim=8)
    ex = build_examples(traj, inj, bb, channel="hidden", target="time")
    assert all(e.prompt.endswith("\nThink time (s):") for e in ex)
    assert ex[0].completion == " 1.5"  # time_spent formatted to one decimal


def test_build_examples_rejects_bad_channel():
    traj = _trajectory(1)
    bb = SGLangBackbone()
    with pytest.raises(ValueError):
        build_examples(traj, NeuralInjector(), bb, channel="bogus")
