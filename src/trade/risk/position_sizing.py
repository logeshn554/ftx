"""Deterministic conservative sizing from edge, risk, and uncertainty."""
from __future__ import annotations


def position_size(equity: float, entry_price: float, stop_distance: float, edge: float,
                  confidence: float, volatility: float, drawdown: float,
                  max_risk_per_trade: float = 0.01, max_position_pct: float = 0.20,
                  uncertainty: float = 0.0) -> float:
    if min(equity, entry_price, stop_distance) <= 0 or edge <= 0 or confidence <= 0:
        return 0.0
    risk_budget = equity * max(0.0, max_risk_per_trade) * max(0.0, 1.0 - min(drawdown, 1.0))
    risk_adjustment = min(1.0, confidence) * max(0.0, 1.0 - min(uncertainty, 1.0))
    volatility_adjustment = 1.0 / (1.0 + max(0.0, volatility))
    qty = risk_budget * risk_adjustment * volatility_adjustment / stop_distance
    return max(0.0, min(qty, equity * max_position_pct / entry_price))
