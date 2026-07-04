"""Tests for the HIDDEN soft-prompt channel into an LLM (Milestone G).

The trained ``HiddenPrefixProjector`` (latent -> soft-prompt rows) and its
wiring into ``SGLangBackbone`` are the open-weight analogue of splicing a
verbal note -- the channel RQ6 ports *into* the LLM. Everything up to the
served-model embedding hand-off is pure/torch and CPU-testable; only the final
``input_embeds`` engine call is GPU-host wiring, and it must fail *loudly*, not
silently drop the latent. torch-gated so the stdlib CPU suite still collects.
"""

import pytest

torch = pytest.importorskip("torch")

from gps.interface import (  # noqa: E402
    DecisionPoint,
    Game,
    OutcomeStream,
    TimeSignal,
)
from gps.latent.base import Injection, InjectionKind  # noqa: E402
from gps.policy.hidden_prefix import HiddenPrefixProjector  # noqa: E402
from gps.policy.sglang_backbone import SGLangBackbone  # noqa: E402


def _dp():
    return DecisionPoint(
        game=Game.CHESS,
        player_id="magnus",
        state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        legal_actions=("e2e4", "d2d4", "g1f3"),
        engine_reference=None,
        time_signal=TimeSignal(time_remaining=42.0),
        recent_outcomes=OutcomeStream(),
    )


# --- HiddenPrefixProjector ---------------------------------------------
def test_projector_shapes_single_and_batch():
    proj = HiddenPrefixProjector(latent_dim=8, hidden_size=16, n_prefix=2)
    one = proj.project([0.0] * 8)
    assert tuple(one.shape) == (2, 16)  # [n_prefix, hidden_size]
    batch = proj.project(torch.zeros(5, 8))
    assert tuple(batch.shape) == (5, 2, 16)  # [B, n_prefix, hidden_size]


def test_projector_is_deterministic_per_seed():
    z = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8]
    a = HiddenPrefixProjector(8, 16, seed=0).project(z)
    b = HiddenPrefixProjector(8, 16, seed=0).project(z)
    c = HiddenPrefixProjector(8, 16, seed=1).project(z)
    assert torch.allclose(a, b)  # same seed -> identical params
    assert not torch.allclose(a, c)  # different seed -> different params


def test_projector_has_trainable_params_and_report():
    proj = HiddenPrefixProjector(latent_dim=8, hidden_size=16, n_prefix=2)
    n = sum(p.numel() for p in proj.parameters())
    # Linear(8 -> 2*16) => 8*32 weights + 32 biases.
    assert n == 8 * 32 + 32
    assert proj.param_report()["n_parameters"] == n


def test_projector_rejects_wrong_latent_width():
    proj = HiddenPrefixProjector(latent_dim=8, hidden_size=16)
    with pytest.raises(ValueError):
        proj.project([0.0] * 4)


# --- SGLangBackbone HIDDEN wiring --------------------------------------
def test_hidden_off_by_default_accepts_verbal_only():
    assert SGLangBackbone().accepts == (InjectionKind.VERBAL,)


def test_hidden_enabled_accepts_hidden():
    bb = SGLangBackbone(enable_hidden=True, latent_dim=8, hidden_size=16)
    assert InjectionKind.HIDDEN in bb.accepts


def test_hidden_prefix_embeds_shape():
    bb = SGLangBackbone(
        enable_hidden=True, latent_dim=8, hidden_size=16, n_prefix=3
    )
    inj = Injection(kind=InjectionKind.HIDDEN, vector=[0.0] * 8)
    embeds = bb.hidden_prefix_embeds(inj)
    assert tuple(embeds.shape) == (3, 16)


def test_hidden_prefix_embeds_validation():
    bb = SGLangBackbone(enable_hidden=True, latent_dim=8, hidden_size=16)
    # wrong kind
    with pytest.raises(ValueError):
        bb.hidden_prefix_embeds(Injection(kind=InjectionKind.VERBAL, text="x"))
    # no vector
    with pytest.raises(ValueError):
        bb.hidden_prefix_embeds(Injection(kind=InjectionKind.HIDDEN))
    # hidden disabled
    off = SGLangBackbone(latent_dim=8, hidden_size=16)
    with pytest.raises(ValueError):
        off.hidden_prefix_embeds(
            Injection(kind=InjectionKind.HIDDEN, vector=[0.0] * 8)
        )


def test_projector_requires_latent_dim():
    bb = SGLangBackbone(enable_hidden=True, hidden_size=16)  # no latent_dim
    with pytest.raises(ValueError):
        bb.projector()


def test_hidden_injection_leaves_prompt_text_identical():
    # RQ6 is a channel-only contrast: hidden state must NOT change the text.
    bb = SGLangBackbone(enable_hidden=True, latent_dim=8, hidden_size=16)
    dp = _dp()
    plain = bb.build_prompt(dp, None)
    hidden = bb.build_prompt(
        dp, Injection(kind=InjectionKind.HIDDEN, vector=[0.0] * 8)
    )
    assert hidden == plain  # the vector rides in as embeddings, not words


def test_hidden_inference_fails_loudly_not_silently():
    # Without the GPU input_embeds wiring, a hidden score must raise (never
    # quietly fall back to ignoring the latent).
    bb = SGLangBackbone(enable_hidden=True, latent_dim=8, hidden_size=16)
    inj = Injection(kind=InjectionKind.HIDDEN, vector=[0.0] * 8)
    with pytest.raises((NotImplementedError, ImportError)):
        bb.move_logprobs(_dp(), inj)
