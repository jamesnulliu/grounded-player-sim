"""Board-native (non-LLM) policy backbone -- the controlled comparison.

Reviewers from the Maia line will benchmark next-move NLL, where an LLM is a
comparatively weak board-move predictor. To make "does the dynamic latent
help?" provable *independent of backbone*, we run the same latent-injection
experiment on a strong board-native backbone (a Maia/KataGo-style CNN with a
move head and a timing head).

It consumes the latent as a *hidden* vector only (concatenated into the head
input / FiLM-conditioned), never as text -- there is no prompt. GPU/torch;
lazily imported.
"""

from __future__ import annotations

from gps.interface import DecisionPoint
from gps.latent.base import Injection, InjectionKind
from gps.policy.base import PolicyBackbone
from gps.prediction import Prediction


class BoardNativeBackbone(PolicyBackbone):
    """Maia/KataGo-style CNN policy with latent conditioning (hidden only)."""

    accepts = (InjectionKind.HIDDEN,)

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        latent_dim: int = 4,
        condition: str = "film",  # "film" | "concat"
    ) -> None:
        self.checkpoint = checkpoint
        self.latent_dim = latent_dim
        self.condition = condition
        self._net = None

    def _network(self):
        if self._net is None:
            try:
                import torch  # noqa: F401  (lazy, GPU-only)
            except ImportError as e:  # pragma: no cover - env-dependent
                raise ImportError(
                    "torch required for BoardNativeBackbone; install the "
                    "'train' or 'serve' extra."
                ) from e
            # TODO(gpu): build/load the CNN trunk + move head + timing head,
            # with the latent vector conditioning the heads via FiLM/concat.
            raise NotImplementedError(
                "board-native network construction is wired on the GPU host."
            )
        return self._net

    def predict(
        self, dp: DecisionPoint, injection: Injection | None = None
    ) -> Prediction:
        self._network()
        raise NotImplementedError

    @property
    def name(self) -> str:
        return f"BoardNativeBackbone({self.checkpoint or 'scratch'})"
