"""Swappable policy backbones.

The policy is the agent that emits a move + timing distribution. The
backbone is *swappable* behind :class:`~gps.policy.base.PolicyBackbone` for a
deliberate reason: an LLM is a comparatively weak board-move predictor next
to a board-native model like Maia, and reviewers will benchmark next-move
NLL. Keeping the backbone a controlled variable lets us answer "does the
dynamic latent help?" on *both* an LLM backbone and a strong board-native
one -- so the latent's value is provable independent of backbone choice.

Backends
--------
* :class:`~gps.policy.sglang_backbone.SGLangBackbone` -- open-weight LLM
  (Qwen3-8B etc.) served locally via sglang. GPU.
* :class:`~gps.policy.api_backbone.APIBackbone` -- closed-source hosted LLM
  via HTTP (the persona-prompt / user-profile foil for RQ4).
* :class:`~gps.policy.board_native.BoardNativeBackbone` -- Maia/KataGo-style
  baseline backbone (no LLM).

Only :mod:`gps.policy.base` imports cleanly with no heavy deps; the concrete
backbones lazy-import torch / sglang / openai inside ``__init__`` so the
interface and the CPU-only Phase-0 path stay importable on any machine.
"""

from gps.policy.base import PolicyBackbone

__all__ = ["PolicyBackbone"]
