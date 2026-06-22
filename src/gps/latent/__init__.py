"""The dynamic latent-state injector -- the contribution.

``z_t`` evolves over a player's own action+timing trajectory and is
*injected* into the policy. Two interchangeable realisations share one
interface so the rest of the system never branches on which is in use:

* :class:`~gps.latent.verbal.VerbalInjector` -- the latent is natural-language
  "memory in words" spliced into the prompt (works with API backbones).
* :class:`~gps.latent.hidden.HiddenInjector` -- the latent is a vector
  realised as a soft prompt / prefix (needs an open-weight backbone).

The proposal allows the latent to be *not verbalised*; this split is exactly
that choice made swappable, so RQ2's interpretability question ("does z_t
recover time-pressure / post-loss / fatigue?") can be asked of both.
"""

from gps.latent.base import (
    Injection,
    InjectionKind,
    LatentState,
    LatentStateInjector,
)

__all__ = [
    "Injection",
    "InjectionKind",
    "LatentState",
    "LatentStateInjector",
]
