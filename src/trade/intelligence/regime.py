"""Market Regime Engine: detects structural market regimes and confidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np


class MarketRegime(str, Enum):
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    PANIC_CRASH = "PANIC_CRASH"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RegimeClassification:
    regime: MarketRegime
    confidence: float
    volatility_score: float
    trend_score: float
    features_used: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.regime.value


class RegimeEngine:
    """Classifies market conditions using multi-indicator structural features."""

    def __init__(
        self,
        min_trend_adx: float = 20.0,
        high_vol_atr_pct: float = 2.0,
        low_vol_atr_pct: float = 0.6,
        panic_return_threshold: float = -0.03,
    ):
        self.min_trend_adx = float(min_trend_adx)
        self.high_vol_atr_pct = float(high_vol_atr_pct)
        self.low_vol_atr_pct = float(low_vol_atr_pct)
        self.panic_return_threshold = float(panic_return_threshold)

    def classify(self, indicators: dict[str, Any]) -> RegimeClassification:
        """Classify current market state from indicators."""
        fast_ma = float(indicators.get("sma_10", indicators.get("ema_12", 0.0)))
        slow_ma = float(indicators.get("sma_50", indicators.get("ema_26", 0.0)))
        adx = float(indicators.get("adx", 15.0))
        rsi = float(indicators.get("rsi_14", 50.0))
        atr_pct = float(indicators.get("atr_pct", indicators.get("atr_14", 1.0)))
        bb_width_pct = float(indicators.get("bb_width_pct", 2.0))
        bb_pos = str(indicators.get("bb_position", "MID")).upper()
        roc_10 = float(indicators.get("roc_10", indicators.get("momentum_20", 0.0)))
        ret_1d = float(indicators.get("return_1d", roc_10))
        volume_ratio = float(indicators.get("volume_ratio", 1.0))

        features_snapshot = {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "adx": adx,
            "rsi": rsi,
            "atr_pct": atr_pct,
            "bb_width_pct": bb_width_pct,
            "bb_position": bb_pos,
            "roc_10": roc_10,
            "volume_ratio": volume_ratio,
        }

        # Calculate fundamental scores
        ma_separation = (fast_ma - slow_ma) / slow_ma if slow_ma > 0 else 0.0
        trend_score = np.clip(ma_separation * 20.0 + (adx / 50.0) * (1.0 if ma_separation >= 0 else -1.0), -1.0, 1.0)
        volatility_score = np.clip((atr_pct - 1.0) / 2.0, 0.0, 1.0)

        # 1. PANIC CRASH (Priority override: fast drop + high volume + volatility spike)
        if (ret_1d <= self.panic_return_threshold or roc_10 <= self.panic_return_threshold * 100) and (rsi <= 28.0 or atr_pct >= self.high_vol_atr_pct):
            confidence = min(0.95, 0.60 + abs(ret_1d) * 5.0)
            return RegimeClassification(
                regime=MarketRegime.PANIC_CRASH,
                confidence=confidence,
                volatility_score=1.0,
                trend_score=-1.0,
                features_used=features_snapshot,
            )

        # 2. BREAKOUT (Volatility expansion from squeeze with price breaking bands + volume)
        if bb_pos in {"ABOVE_UPPER", "BELOW_LOWER"} and volume_ratio >= 1.3 and bb_width_pct >= 1.5:
            confidence = min(0.90, 0.55 + (volume_ratio - 1.0) * 0.2)
            return RegimeClassification(
                regime=MarketRegime.BREAKOUT,
                confidence=confidence,
                volatility_score=volatility_score,
                trend_score=1.0 if bb_pos == "ABOVE_UPPER" else -1.0,
                features_used=features_snapshot,
            )

        # 3. HIGH VOLATILITY / EXTREME CHOP
        if atr_pct >= self.high_vol_atr_pct and adx < self.min_trend_adx:
            confidence = min(0.90, 0.50 + (atr_pct / self.high_vol_atr_pct) * 0.3)
            return RegimeClassification(
                regime=MarketRegime.HIGH_VOLATILITY,
                confidence=confidence,
                volatility_score=1.0,
                trend_score=0.0,
                features_used=features_snapshot,
            )

        # 4. TRENDING BULL
        if adx >= self.min_trend_adx and ma_separation > 0.003 and rsi >= 48.0:
            confidence = min(0.95, max(0.55, 0.45 + (adx / 100.0) * 0.4 + ma_separation * 10.0))
            return RegimeClassification(
                regime=MarketRegime.BULL_TREND,
                confidence=confidence,
                volatility_score=volatility_score,
                trend_score=float(trend_score),
                features_used=features_snapshot,
            )

        # 5. TRENDING BEAR
        if adx >= self.min_trend_adx and ma_separation < -0.003 and rsi <= 52.0:
            confidence = min(0.95, max(0.55, 0.45 + (adx / 100.0) * 0.4 + abs(ma_separation) * 10.0))
            return RegimeClassification(
                regime=MarketRegime.BEAR_TREND,
                confidence=confidence,
                volatility_score=volatility_score,
                trend_score=float(trend_score),
                features_used=features_snapshot,
            )

        # 6. LOW VOLATILITY / SQUEEZE
        if atr_pct <= self.low_vol_atr_pct or bb_width_pct <= 1.0:
            confidence = min(0.90, 0.60 + (self.low_vol_atr_pct - atr_pct) * 0.3 if atr_pct <= self.low_vol_atr_pct else 0.70)
            return RegimeClassification(
                regime=MarketRegime.LOW_VOLATILITY,
                confidence=confidence,
                volatility_score=0.0,
                trend_score=0.0,
                features_used=features_snapshot,
            )

        # 7. SIDEWAYS / RANGE-BOUND MEAN REVERSION
        confidence = min(0.85, max(0.50, 0.75 - (adx / 100.0)))
        return RegimeClassification(
            regime=MarketRegime.SIDEWAYS,
            confidence=confidence,
            volatility_score=volatility_score,
            trend_score=0.0,
            features_used=features_snapshot,
        )
