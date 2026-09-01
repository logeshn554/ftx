"""Volatility Squeeze & Donchian/Bollinger Breakout Strategy.

Rules:
- Triggers on Bollinger Band penetration (bb_pct >= 0.95 for BUY, bb_pct <= 0.05 for SELL).
- Requires non-zero Band Width to filter flat noise.
- Confirms volume surge if volume_ratio > 1.0.
"""

from __future__ import annotations

from .base import Signal, abstain


class BreakoutStrategy:
    name = "breakout"

    def __init__(self, upper_threshold: float = 0.95, lower_threshold: float = 0.05, min_width: float = 0.005):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.min_width = min_width

    def signal(self, i: dict) -> Signal:
        pct = float(i.get("bb_pct", 0.5))
        width = float(i.get("bb_width", 0.0))
        vol_ratio = float(i.get("volume_ratio", 1.0))
        atr_pct = float(i.get("atr_pct", i.get("atr_14", 0.01)))

        if width < self.min_width:
            return abstain(self.name, f"band_width_too_narrow ({width:.4f} < {self.min_width})")

        # Bullish Breakout
        if pct >= self.upper_threshold:
            vol_boost = min(0.15, max(0.0, (vol_ratio - 1.0) * 0.1))
            confidence = min(0.95, max(0.55, 0.55 + min(0.25, width * 5.0) + vol_boost))
            expected_move = max(atr_pct * 2.2, width * 1.5)
            stop_dist = max(atr_pct * 1.2, width * 0.8)
            target_dist = expected_move
            reason = f"bullish_breakout_bb_{pct:.2f}_w_{width:.3f}_vol_{vol_ratio:.1f}x"
            return Signal("BUY", confidence, expected_move, stop_dist, target_dist, self.name, reason)

        # Bearish Breakdown
        elif pct <= self.lower_threshold:
            vol_boost = min(0.15, max(0.0, (vol_ratio - 1.0) * 0.1))
            confidence = min(0.95, max(0.55, 0.55 + min(0.25, width * 5.0) + vol_boost))
            expected_move = max(atr_pct * 2.2, width * 1.5)
            stop_dist = max(atr_pct * 1.2, width * 0.8)
            target_dist = expected_move
            reason = f"bearish_breakdown_bb_{pct:.2f}_w_{width:.3f}_vol_{vol_ratio:.1f}x"
            return Signal("SELL", confidence, expected_move, stop_dist, target_dist, self.name, reason)

        return abstain(self.name, f"bb_within_range ({pct:.2f})")
