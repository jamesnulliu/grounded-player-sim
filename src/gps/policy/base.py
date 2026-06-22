"""Policy backbone interface.

A backbone turns ``(decision point, latent injection)`` into a
:class:`~gps.prediction.Prediction`. It declares which
:class:`~gps.latent.base.InjectionKind` s it can honour so the simulator can
reject an incompatible injector/backbone pairing up front (e.g. a hidden
soft-prompt injection cannot be delivered to a closed API backbone).
"""

from __future__ import annotations

import abc

from gps.interface import DecisionPoint
from gps.latent.base import Injection, InjectionKind
from gps.prediction import Prediction


class PolicyBackbone(abc.ABC):
    """Emits a move+timing prediction, optionally conditioned on a latent.

    Subclasses must set :attr:`accepts` to the injection kinds they can
    consume. A backbone that ignores the latent entirely (an ablation, or a
    pure population baseline) advertises an empty :attr:`accepts` and is
    handed ``injection=None``.
    """

    #: Latent injection kinds this backbone can consume.
    accepts: tuple[InjectionKind, ...] = ()

    @abc.abstractmethod
    def predict(
        self,
        dp: DecisionPoint,
        injection: Injection | None = None,
    ) -> Prediction:
        """Predict the move (and timing) distribution at ``dp``.

        Implementations must normalise the move distribution over
        ``dp.legal_actions`` only. ``injection`` is ``None`` when no latent
        is in use or when this backbone advertises no accepted kinds.
        """

    def accepts_kind(self, kind: InjectionKind) -> bool:
        return kind in self.accepts

    # Optional capability hooks, overridden by LLM backbones. Defaults keep
    # board-native / mock backbones simple.
    @property
    def name(self) -> str:
        return type(self).__name__

    def close(self) -> None:  # noqa: B027  (intentional optional no-op hook)
        """Release any served process / client. No-op by default."""
