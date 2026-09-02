"""Integration tests for Edge Gate effectiveness and No-Trade filtering."""

import pytest
from trade.intelligence.decision_engine import DecisionEngine


def test_edge_gate_filters_low_edge_churn():
    """Verify that Edge Gate rejects low-edge micro-move trades and audits them."""
    engine = DecisionEngine(minimum_signal_confidence=0.50, cost_safety_multiplier=1.0)

    # 1. Microscopic edge trade: strategy emits BUY, but expected move (0.05%) is lower than roundtrip friction (0.30%)
    decision_micro = engine.decide(
        indicators={
            "sma_10": 101.0,
            "sma_50": 100.0,  # separation 1%
            "adx": 28.0,
            "macd_histogram": 0.5,
            "rsi_14": 58.0,
            "atr_pct": 0.05,  # Tiny volatility: 0.05% move
            "bb_position": "MID",
            "momentum_20": 0.1,
        },
        equity=10_000.0,
        entry_price=100.0,
        regime="BULL",
        regime_confidence=0.8,
        p_win=0.51,  # Barely above coin flip
    )
    assert decision_micro.action == "HOLD"
    assert decision_micro.reason in {
        "EXPECTED_MOVE_BELOW_COST",
        "expected_move_does_not_cover_cost",
        "friction_not_covered",
        "net_edge_below_minimum",
        "no_gross_edge",
        "trade_quality_below_threshold",
    }
    assert len(engine.rejected_trades) >= 1
    rejection = engine.rejected_trades[-1]
    assert rejection["expected_cost"] > 0
    assert rejection["reason"] == decision_micro.reason

    # 2. Genuine high-edge trade where expected move > 3x friction
    decision_high = engine.decide(
        indicators={
            "sma_10": 105.0,
            "sma_50": 95.0,  # Strong bull trend
            "adx": 35.0,
            "macd_histogram": 1.5,
            "rsi_14": 65.0,
            "atr_pct": 2.5,  # Solid volatility: 2.5% move potential
            "bb_position": "ABOVE_UPPER",
            "momentum_20": 3.0,
        },
        equity=10_000.0,
        entry_price=100.0,
        regime="BULL",
        regime_confidence=0.90,
        p_win=0.75,  # Strong calibrated probability
        uncertainty=0.10,
    )
    assert decision_high.action == "TRADE"
    assert decision_high.side == "BUY"
    assert decision_high.expected_net_edge > 0
    assert decision_high.position_size > 0
    assert decision_high.trade_quality > 0
