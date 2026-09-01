"""Trend Following Specialist Strategy.

Rules:
- Uses Fast (SMA-10 / EMA-12) vs Slow (SMA-50 / EMA-26) moving averages.
- Confirms trend strength via ADX >= 20.
- Confirms direction with MACD histogram and price above/below moving average.
- Yields asymmetric R:R (targets 1.5R to 2.5R).
"""

from __future__ import annotations

from .base import Signal, abstain


class TrendStrategy:
    name = "trend"

    def __init__(self, min_adx: float = 20.0, min_separation_pct: float = 0.002):
        self.min_adx = min_adx
        self.min_separation_pct = min_separation_pct

    def signal(self, i: dict) -> Signal:
        fast = float(i.get("sma_10", i.get("ema_12", 0.0)))
        slow = float(i.get("sma_50", i.get("ema_26", 0.0)))
        adx = float(i.get("adx", 0.0))
        macd_hist = float(i.get("macd_histogram", 0.0))
        atr_pct = float(i.get("atr_pct", i.get("atr_14", 0.01)))

        if fast <= 0 or slow <= 0:
            return abstain(self.name, "missing_moving_averages")

        separation = (fast - slow) / slow
        abs_separation = abs(separation)

        if adx < self.min_adx:
            return abstain(self.name, f"adx_below_threshold ({adx:.1f} < {self.min_adx})")

        if abs_separation < self.min_separation_pct:
            return abstain(self.name, f"sma_separation_too_small ({abs_separation:.3%} < {self.min_separation_pct:.3%})")

        # Long Trend
        if separation > 0 and macd_hist >= -0.05:
            confidence = min(0.95, max(0.55, 0.50 + (adx / 100.0) * 0.35 + min(0.15, abs_separation * 10)))
            expected_move = max(atr_pct * 2.0, abs_separation * 1.5)
            stop_dist = max(atr_pct * 1.2, abs_separation * 0.8)
            target_dist = expected_move
            reason = f"bullish_trend_adx_{adx:.0f}_sep_{separation:+.2%}"
            return Signal("BUY", confidence, expected_move, stop_dist, target_dist, self.name, reason)

        # Short Trend
        elif separation < 0 and macd_hist <= 0.05:
            confidence = min(0.95, max(0.55, 0.50 + (adx / 100.0) * 0.35 + min(0.15, abs_separation * 10)))
            expected_move = max(atr_pct * 2.0, abs_separation * 1.5)
            stop_dist = max(atr_pct * 1.2, abs_separation * 0.8)
            target_dist = expected_move
            reason = f"bearish_trend_adx_{adx:.0f}_sep_{separation:+.2%}"
            return Signal("SELL", confidence, expected_move, stop_dist, target_dist, self.name, reason)

        return abstain(self.name, "trend_direction_macd_conflict")
