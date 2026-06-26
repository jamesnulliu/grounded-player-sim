"""Backend policy: which engine is allowed for which job, enforced in code.

Three standing rules for this project, made checkable rather than documented
and forgotten:

* **LLM training MUST use slime.** Any RL / post-training of an open-weight
  LLM backbone goes through :class:`~gps.train.slime_rl.SlimeRLTrainer`,
  which pairs a training backend with an sglang rollout engine. SFT of the
  *injector* on top of a frozen LLM also routes its LLM rollouts through
  slime+sglang.
* **LLM inference MUST use sglang.** Serving an open-weight model for
  logprob scoring / rollouts goes through
  :class:`~gps.policy.sglang_backbone.SGLangBackbone`. (A *closed* model
  behind an HTTP API uses :class:`~gps.policy.api_backbone.APIBackbone`; that
  is the RQ4 baseline, not a served open-weight model.)
* **Non-LLM training is exempt** -- the board-native CNN trunk + injector
  trains with plain torch (no LLM, so no slime/sglang). This is the cheap,
  capacity-matched control (see ``documents/milestone_a.md``).

These functions are the single gate. ``require_*`` raises with an install
hint when the engine is missing; ``assert_*`` validates that a chosen
backbone/trainer pairing obeys the rules *before* a run starts, so a
misconfiguration fails fast on a laptop instead of deep into a GPU job.

Pure stdlib at import time; the heavy engines are imported lazily inside the
``require_*`` helpers.
"""

from __future__ import annotations

import importlib.util

from gps.policy.base import PolicyBackbone


class BackendError(RuntimeError):
    """Raised when a backend rule is violated or a needed engine is absent."""


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def is_llm_backbone(backbone: PolicyBackbone) -> bool:
    """True for an open-weight, *served* LLM backbone (the sglang/slime path).

    Detected structurally (duck-typed on the sglang backbone's surface) so a
    new LLM backbone subclass is covered without editing this list, while the
    board-native CNN and the mock test double are not misclassified. The
    closed-API backbone is *not* an LLM backbone for these rules: it is a
    hosted baseline, not a model we serve or train.
    """
    from gps.policy.sglang_backbone import SGLangBackbone

    return isinstance(backbone, SGLangBackbone)


def require_sglang() -> None:
    """Ensure sglang is importable for LLM *inference*; else raise."""
    if not _installed("sglang"):
        raise BackendError(
            "LLM inference must use sglang, which is not installed. Install "
            "the 'serve' extra on a GPU host: pip install '.[serve]'"
        )


def require_slime() -> None:
    """Ensure slime is importable for LLM *training*; else raise."""
    if not _installed("slime"):
        raise BackendError(
            "LLM training must use slime, which is not installed. Install it "
            "from source on the GPU host (see documents/training.md)."
        )


def assert_inference_backend(backbone: PolicyBackbone) -> None:
    """Validate an inference/serving backbone obeys the sglang rule.

    Non-LLM backbones (board-native, mock, closed-API) are unaffected.
    """
    if is_llm_backbone(backbone):
        require_sglang()


def assert_llm_training_uses_slime(
    backbone: PolicyBackbone, *, train_backbone: bool
) -> None:
    """Validate that training touching a served LLM goes through slime+sglang.

    Called by the trainers before a run. Enforces:

    * if the backbone is a served LLM, slime (training) **and** sglang
      (rollout) must both be importable;
    * training is otherwise free to use plain torch (board-native control).

    ``train_backbone`` is accepted for symmetry / future rules (e.g. allowing
    a frozen-LLM SFT path to relax the slime requirement); today any LLM
    backbone in the loop requires the slime+sglang pair regardless, because
    even a frozen LLM is *served* by sglang for rollouts.
    """
    if not is_llm_backbone(backbone):
        return
    require_slime()
    require_sglang()
