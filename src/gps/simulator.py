"""The simulator: composes a latent injector with a policy backbone.

This is the runtime loop that threads ``z_t`` through a player's trajectory
(proposal section 4). It is pure-stdlib: given a mock backbone it runs on a
CPU-only laptop, which is what lets the Phase-0 synthetic experiments and the
eval harness execute end-to-end without a GPU.

    sim = Simulator(injector, backbone)
    preds = sim.run_trajectory(player_id, decision_points, observations)

When ``observations`` are supplied (teacher forcing / eval on real games),
the latent advances on ground truth. When they are absent (free rollout),
it advances on the policy's own prediction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gps.interface import DecisionPoint
from gps.latent.base import (
    InjectionKind,
    LatentStateInjector,
    Observation,
)
from gps.policy.base import PolicyBackbone
from gps.prediction import Prediction


@dataclass
class StepResult:
    """Per-decision record: the prediction and the latent that produced it."""

    decision: DecisionPoint
    prediction: Prediction
    # Numeric snapshot of z_t at the time of prediction (for RQ2 probes).
    latent_probe: list[float] | None = None


class IncompatiblePairingError(ValueError):
    """Raised when an injector's output kind no backbone can consume."""


class Simulator:
    """Drives one injector + one backbone over player trajectories."""

    def __init__(
        self,
        injector: LatentStateInjector | None,
        backbone: PolicyBackbone,
    ) -> None:
        self.injector = injector
        self.backbone = backbone
        self._check_compatibility()

    def _check_compatibility(self) -> None:
        if self.injector is None:
            # No-latent ablation / population baseline: always fine.
            return
        produced = set(self.injector.produces)
        accepted = set(self.backbone.accepts)
        if not (produced & accepted):
            raise IncompatiblePairingError(
                f"injector produces {sorted(k.value for k in produced)} "
                f"but backbone {self.backbone.name} accepts "
                f"{sorted(k.value for k in accepted)}; no overlap"
            )

    def _chosen_kind(self) -> InjectionKind | None:
        if self.injector is None:
            return None
        for kind in self.injector.produces:
            if self.backbone.accepts_kind(kind):
                return kind
        return None  # unreachable after _check_compatibility

    def run_trajectory(
        self,
        player_id: str,
        decisions: Sequence[DecisionPoint],
        observations: Sequence[Observation] | None = None,
    ) -> list[StepResult]:
        """Predict across one ordered trajectory, advancing z_t each step.

        ``observations[i]`` is the ground truth for ``decisions[i]``; pass
        ``None`` for free rollout. Length must match ``decisions`` when
        given.
        """
        if observations is not None and len(observations) != len(decisions):
            raise ValueError("observations must align 1:1 with decisions")

        results: list[StepResult] = []
        state = (
            self.injector.initial_state(player_id)
            if self.injector is not None
            else None
        )

        for i, dp in enumerate(decisions):
            injection = None
            probe = None
            if self.injector is not None and state is not None:
                injection = self.injector.render(state, dp)
                probe = state.probe_vector

            pred = self.backbone.predict(dp, injection)
            results.append(
                StepResult(decision=dp, prediction=pred, latent_probe=probe)
            )

            if self.injector is not None and state is not None:
                obs = observations[i] if observations is not None else None
                if obs is None:
                    # Free rollout: advance on the policy's own prediction.
                    obs = Observation(
                        move=pred.moves.argmax(),
                        time_spent=(
                            pred.timing.median_seconds
                            if pred.timing is not None
                            else None
                        ),
                        prediction=pred,
                    )
                state = self.injector.update(state, dp, obs)

        return results

    def close(self) -> None:
        self.backbone.close()
