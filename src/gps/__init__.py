"""grounded-player-sim (gps).

A framework for personalized human game-play simulation built around an
LLM *policy* augmented with a trainable, per-individual **dynamic latent
state**. The central thesis: an LLM conditioned on a static verbal
persona/emotion label is not enough to reproduce how a specific person
plays *right now* -- you need a learned latent state that evolves over the
player's own action+timing trajectory and is injected into the policy.

Layout
------
``gps.interface``  Shared decision-point schema (chess and Go expose this).
``gps.games``      Game-specific encoders + engine oracles (chess concrete,
                   Go behind the same interface).
``gps.latent``     The dynamic latent-state injector -- the contribution.
                   Two interchangeable implementations: verbal (text memory)
                   and hidden (soft-prompt / prefix vectors).
``gps.policy``     Swappable policy backbones (LLM via sglang, LLM via API,
                   board-native baseline) behind one interface.
``gps.synthetic``  Phase-0 synthetic players with *known* dynamics
                   (pure-stdlib; runs anywhere).
``gps.eval``       Metrics + state-recovery probes (pure-stdlib).
``gps.train``      Trainers: SFT and slime-RL (lazy GPU imports).
"""

__version__ = "0.0.1"

from gps.interface import (
    DecisionPoint,
    EngineReference,
    Outcome,
    OutcomeStream,
    TimeSignal,
)

__all__ = [
    "DecisionPoint",
    "EngineReference",
    "Outcome",
    "OutcomeStream",
    "TimeSignal",
    "__version__",
]
