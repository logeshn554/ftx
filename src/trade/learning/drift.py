"""Distribution and performance drift monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DriftState(str, Enum):
    NO_DRIFT = "NO_DRIFT"
    WATCH = "WATCH"
    SIGNIFICANT_DRIFT = "SIGNIFICANT_DRIFT"
    CRITICAL_DRIFT = "CRITICAL_DRIFT"


@dataclass(frozen=True)
class DriftReport:
    state: DriftState
    feature_drift: float
    expectancy_drift: float
    cost_drift: float
    reason: str


class DriftDetector:
    """Rolling evidence for distribution and performance drift."""

    def __init__(
        self,
        watch_threshold: float = 0.15,
        significant_threshold: float = 0.30,
        critical_threshold: float = 0.50,
        min_samples: int = 20,
    ):
        self.watch_threshold = watch_threshold
        self.significant_threshold = significant_threshold
        self.critical_threshold = critical_threshold
        self.min_samples = min_samples

    def assess(
        self,
        baseline_expectancy: float,
        recent_expectancy: float,
        baseline_cost: float,
        recent_cost: float,
        feature_distance: float = 0.0,
        sample_count: int = 0,
    ) -> DriftReport:
        if sample_count < self.min_samples:
            return DriftReport(DriftState.NO_DRIFT, feature_distance, 0.0, 0.0, "insufficient_samples")

        exp_drift = abs(recent_expectancy - baseline_expectancy) / max(abs(baseline_expectancy), 1e-6)
        cost_drift = abs(recent_cost - baseline_cost) / max(baseline_cost, 1e-6)
        combined = max(feature_distance, exp_drift, cost_drift)

        if combined >= self.critical_threshold:
            state = DriftState.CRITICAL_DRIFT
            reason = "critical_combined_drift"
        elif combined >= self.significant_threshold:
            state = DriftState.SIGNIFICANT_DRIFT
            reason = "significant_drift"
        elif combined >= self.watch_threshold:
            state = DriftState.WATCH
            reason = "watch_drift"
        else:
            state = DriftState.NO_DRIFT
            reason = "stable"

        return DriftReport(state, feature_distance, exp_drift, cost_drift, reason)
