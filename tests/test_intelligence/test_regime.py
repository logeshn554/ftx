"""Unit tests for Market Regime Engine."""

import pytest
from trade.intelligence.regime import MarketRegime, RegimeEngine


def test_regime_bull_trend():
    engine = RegimeEngine(min_trend_adx=20.0)
    classification = engine.classify({
        "sma_10": 105.0,
        "sma_50": 100.0,
        "adx": 35.0,
        "rsi_14": 65.0,
        "atr_pct": 1.2,
        "bb_width_pct": 2.5,
        "bb_position": "MID",
        "roc_10": 0.05,
    })
    assert classification.regime == MarketRegime.BULL_TREND
    assert classification.confidence >= 0.60
    assert classification.trend_score > 0


def test_regime_bear_trend():
    engine = RegimeEngine(min_trend_adx=20.0)
    classification = engine.classify({
        "sma_10": 95.0,
        "sma_50": 100.0,
        "adx": 30.0,
        "rsi_14": 35.0,
        "atr_pct": 1.5,
        "bb_width_pct": 3.0,
        "bb_position": "MID",
        "roc_10": -0.05,
    })
    assert classification.regime == MarketRegime.BEAR_TREND
    assert classification.confidence >= 0.60
    assert classification.trend_score < 0


def test_regime_sideways():
    engine = RegimeEngine(min_trend_adx=20.0)
    classification = engine.classify({
        "sma_10": 100.1,
        "sma_50": 100.0,
        "adx": 12.0,  # Weak trend
        "rsi_14": 51.0,
        "atr_pct": 0.9,
        "bb_width_pct": 1.8,
        "bb_position": "MID",
        "roc_10": 0.001,
    })
    assert classification.regime == MarketRegime.SIDEWAYS


def test_regime_panic_crash():
    engine = RegimeEngine()
    classification = engine.classify({
        "sma_10": 90.0,
        "sma_50": 100.0,
        "adx": 40.0,
        "rsi_14": 18.0,
        "atr_pct": 3.5,
        "return_1d": -0.06,  # 6% drop
        "volume_ratio": 2.5,
    })
    assert classification.regime == MarketRegime.PANIC_CRASH
    assert classification.confidence >= 0.70


def test_regime_breakout():
    engine = RegimeEngine()
    classification = engine.classify({
        "sma_10": 105.0,
        "sma_50": 100.0,
        "adx": 22.0,
        "rsi_14": 68.0,
        "atr_pct": 1.8,
        "bb_width_pct": 2.5,
        "bb_position": "ABOVE_UPPER",
        "volume_ratio": 1.8,
    })
    assert classification.regime == MarketRegime.BREAKOUT
