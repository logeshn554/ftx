"""Statistical Mean Reversion & Extreme Oscillator Snapback Strategy.

Rules:
- Triggers on extreme RSI oversold (RSI < 30) + lower Bollinger Band envelope (bb_pct < 0.20).
- Triggers on extreme RSI overbought (RSI > 70) + upper Bollinger Band envelope (bb_pct > 0.80).
- Employs quick snap-back profit targets and tight invalidation stops.
"""

from __future__ import annotations

from .base import Signal, abstain


class MeanReversionStrategy:
    name = "mean_reversion"

    def __init__(self, rsi_oversold: float = 30.0, rsi_overbought: float = 70.0):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def signal(self, i: dict) -> Signal:
        rsi = float(i.get("rsi_14", 50.0))
        pct = float(i.get("bb_pct", 0.5))
        stoch_k = float(i.get("stoch_k", 50.0))
        atr_pct = float(i.get("atr_pct", i.get("atr_14", 0.01)))

        # Oversold Reversal -> Long
        if rsi < self.rsi_oversold and pct < 0.25:
            depth = (self.rsi_oversold - rsi) / self.rsi_oversold
            stoch_confirm = 0.1 if stoch_k < 25 else 0.0
            confidence = min(0.95, max(0.55, 0.55 + depth * 0.30 + stoch_confirm))
            expected_move = max(atr_pct * 1.8, (50.0 - rsi) / 100.0 * 2.0)
            stop_dist = max(atr_pct * 1.0, 0.005)
            target_dist = expected_move
            reason = f"oversold_reversion_rsi_{rsi:.1f}_bb_{pct:.2f}"
            return Signal("BUY", confidence, expected_move, stop_dist, target_dist, self.name, reason)

        # Overbought Reversal -> Short
        elif rsi > self.rsi_overbought and pct > 0.75:
            depth = (rsi - self.rsi_overbought) / (100.0 - self.rsi_overbought)
            stoch_confirm = 0.1 if stoch_k > 75 else 0.0
            confidence = min(0.95, max(0.55, 0.55 + depth * 0.30 + stoch_confirm))
            expected_move = max(atr_pct * 1.8, (rsi - 50.0) / 100.0 * 2.0)
            stop_dist = max(atr_pct * 1.0, 0.005)
            target_dist = expected_move
            reason = f"overbought_reversion_rsi_{rsi:.1f}_bb_{pct:.2f}"
            return Signal("SELL", confidence, expected_move, stop_dist, target_dist, self.name, reason)

        return abstain(self.name, f"rsi_neutral ({rsi:.1f})")
