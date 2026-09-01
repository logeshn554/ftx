from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Selection:
    selected_strategy: str | None
    weights: dict[str, float]
    confidence: float
    reason: str

class StrategySelector:
    """Conservative per-regime selector using expectancy with sample shrinkage."""
    def __init__(self, minimum_samples: int = 30, prior_expectancy: float = 0.0):
        self.minimum_samples, self.prior_expectancy = minimum_samples, prior_expectancy

    def select(self, regime: str, performance: dict[str, dict], uncertainty: float = 0.0) -> Selection:
        scores = {}
        for name, stats in performance.items():
            n = int(stats.get("sample_count", 0)); expectancy = float(stats.get("expectancy", self.prior_expectancy))
            drawdown = max(0., float(stats.get("drawdown", 0.)))
            scores[name] = (expectancy * n / (n + self.minimum_samples) - drawdown) if n else -float("inf")
        eligible = {k: v for k, v in scores.items() if performance[k].get("sample_count", 0) >= self.minimum_samples}
        if not eligible or uncertainty >= .5:
            return Selection(None, {}, 0., "insufficient_regime_evidence")
        best = max(eligible, key=eligible.get); spread = max(0., eligible[best])
        total = sum(max(0., v) for v in eligible.values())
        weights = {k: max(0., v) / total for k, v in eligible.items()} if total else {best: 1.}
        return Selection(best, weights, min(1., spread), f"best_shrunk_expectancy_for_{regime}")
