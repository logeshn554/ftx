"""Cross-Period Momentum & Rate-of-Change Velocity Strategy.

Rules:
- Evaluates Rate of Change (ROC-10) and multi-period return alignment (1d, 5d).
- Triggers on significant velocity exceeding momentum thresholds.
"""

from __future__ import annotations

from .base import Signal, abstain


class MomentumStrategy:
    name = "momentum"

    def __init__(self, threshold: float = 0.015):
        self.threshold = threshold

    def signal(self, i: dict) -> Signal:
        roc = float(i.get("roc_10", 0.0))
        ret_1d = float(i.get("return_1d", 0.0))
        ret_5d = float(i.get("return_5d", 0.0))
        atr_pct = float(i.get("atr_pct", i.get("atr_14", 0.01)))

        abs_roc = abs(roc)
        if abs_roc < self.threshold:
            return abstain(self.name, f"roc_below_threshold ({abs_roc:.3%} < {self.threshold:.3%})")

        # Positive Momentum -> Long
        if roc > 0 and ret_1d >= -0.005:
            alignment_boost = 0.10 if ret_5d > 0 else 0.0
            confidence = min(0.95, max(0.55, 0.55 + min(0.30, abs_roc * 5.0) + alignment_boost))
            expected_move = max(atr_pct * 2.0, abs_roc * 1.2)
            stop_dist = max(atr_pct * 1.1, abs_roc * 0.7)
            target_dist = expected_move
            reason = f"bullish_momentum_roc_{roc:+.2%}_ret1d_{ret_1d:+.2%}"
            return Signal("BUY", confidence, expected_move, stop_dist, target_dist, self.name, reason)

        # Negative Momentum -> Short
        elif roc < 0 and ret_1d <= 0.005:
            alignment_boost = 0.10 if ret_5d < 0 else 0.0
            confidence = min(0.95, max(0.55, 0.55 + min(0.30, abs_roc * 5.0) + alignment_boost))
            expected_move = max(atr_pct * 2.0, abs_roc * 1.2)
            stop_dist = max(atr_pct * 1.1, abs_roc * 0.7)
            target_dist = expected_move
            reason = f"bearish_momentum_roc_{roc:+.2%}_ret1d_{ret_1d:+.2%}"
            return Signal("SELL", confidence, expected_move, stop_dist, target_dist, self.name, reason)

        return abstain(self.name, "momentum_multi_period_conflict")
