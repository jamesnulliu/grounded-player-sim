"""A CPU-only mock policy backbone.

This exists so the full pipeline -- simulator, latent injection, eval -- runs
and is unit-tested on a machine with no GPU and no LLM. It is **not** a
scientific baseline; it is a test double that behaves plausibly:

* It converts the engine reference into a softmax move distribution, so
  predictions track move quality (a sane stand-in for a real policy).
* It *consumes the latent injection* by lowering its inverse temperature
  when the latent signals degraded state (time pressure / tilt / fatigue),
  so injecting a latent measurably changes predictions -- which is what lets
  Phase-0 tests assert "the latent matters".

It advertises that it accepts *both* injection kinds (it just reads whichever
is present), so it can stand in for either an LLM or a board-native backbone
in tests.
"""

from __future__ import annotations

import math

from gps.interface import DecisionPoint
from gps.latent.base import Injection, InjectionKind
from gps.policy.base import PolicyBackbone
from gps.prediction import MoveDistribution, Prediction, TimingPrediction


class MockBackbone(PolicyBackbone):
    """Engine-value softmax policy, modulated by the latent injection."""

    accepts = (InjectionKind.VERBAL, InjectionKind.HIDDEN)

    def __init__(
        self,
        base_beta: float = 6.0,
        latent_sensitivity: float = 4.0,
        use_latent: bool = True,
    ) -> None:
        self.base_beta = base_beta
        self.latent_sensitivity = latent_sensitivity
        # When False, the backbone ignores any injection -> the no-latent
        # ablation / population baseline used in eval comparisons.
        self.use_latent = use_latent

    def predict(
        self,
        dp: DecisionPoint,
        injection: Injection | None = None,
    ) -> Prediction:
        beta = self.base_beta
        degraded = 0.0
        if self.use_latent and injection is not None:
            degraded = self._read_degradation(injection)
            beta = max(0.5, beta - self.latent_sensitivity * degraded)

        probs = self._softmax_over_moves(dp, beta)
        timing = self._timing(dp, degraded)
        return Prediction(
            moves=MoveDistribution(probs=probs),
            timing=timing,
            latent=[degraded],
            meta={"beta": beta},
        )

    # --- helpers --------------------------------------------------------
    def _read_degradation(self, injection: Injection) -> float:
        """Map an injection to a scalar 'how off-form is the player' in [0,1].

        Hidden: average of the non-momentum anchored dims. Verbal: keyword
        presence in the rendered note. Both are crude on purpose -- the mock
        only needs to react monotonically to the latent.
        """
        if injection.kind is InjectionKind.HIDDEN and injection.vector:
            # DIMENSIONS order: time_pressure, post_loss, fatigue, momentum.
            v = injection.vector
            degr = [x for x in v[:3]]  # drop momentum (signed)
            return max(0.0, min(1.0, sum(degr) / max(1, len(degr))))
        if injection.kind is InjectionKind.VERBAL and injection.text:
            text = injection.text.lower()
            # A generic "off-form" cue (emitted by the OracleInjector) maps
            # straight to strong degradation; otherwise count specific
            # mechanism keywords. This vocabulary coupling between injector
            # and backbone is intrinsic to the verbal channel -- a real
            # limitation vs. the hidden channel, surfaced here on purpose.
            if "off-form" in text:
                return 1.0
            hits = sum(
                kw in text
                for kw in ("time pressure", "tilted", "fatigued", "shaken")
            )
            return min(1.0, hits / 3.0)
        return 0.0

    def _softmax_over_moves(
        self, dp: DecisionPoint, beta: float
    ) -> dict[str, float]:
        ref = dp.engine_reference
        moves = dp.legal_actions
        if ref is None or not ref.candidate_values:
            # Uniform fallback when no oracle is available.
            p = 1.0 / len(moves)
            return {m: p for m in moves}

        vals = {m: ref.candidate_values.get(m, 0.0) for m in moves}
        vmax, vmin = max(vals.values()), min(vals.values())
        span = (vmax - vmin) or 1.0
        weights = {m: math.exp(beta * (vals[m] - vmin) / span) for m in moves}
        z = sum(weights.values())
        return {m: w / z for m, w in weights.items()}

    def _timing(self, dp: DecisionPoint, degraded: float) -> TimingPrediction:
        # Median think-time shrinks under degradation (rushing); floor sigma.
        base_mu = 1.5
        tr = dp.time_signal.time_remaining
        if tr is not None and tr < 15.0:
            base_mu -= 0.8
        base_mu -= 0.5 * degraded
        return TimingPrediction(mu=base_mu, sigma=0.5)

    @property
    def name(self) -> str:
        return f"MockBackbone(use_latent={self.use_latent})"
