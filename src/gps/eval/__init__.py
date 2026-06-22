"""Evaluation: metrics, state-recovery probes, temporal splits.

Evaluation emphasises likelihood and calibration over top-1 accuracy
(proposal section 4.2). All metrics here are pure-stdlib so the eval harness
runs on CPU against the mock backbone and the Phase-0 synthetic players.
"""

from gps.eval.metrics import (
    MoveMetrics,
    TimingMetrics,
    expected_calibration_error,
    move_metrics,
    timing_metrics,
)
from gps.eval.probes import StateRecoveryResult, state_recovery_probe
from gps.eval.splits import temporal_split

__all__ = [
    "MoveMetrics",
    "TimingMetrics",
    "expected_calibration_error",
    "move_metrics",
    "timing_metrics",
    "StateRecoveryResult",
    "state_recovery_probe",
    "temporal_split",
]
