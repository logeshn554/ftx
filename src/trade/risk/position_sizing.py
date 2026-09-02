"""Deterministic conservative risk-aware position sizing."""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trade.risk.limits import RiskLimits


def position_size(
    equity: float,
    entry_price: float,
    stop_distance: float,
    edge: float,
    confidence: float,
    volatility: float,
    drawdown: float,
    max_risk_per_trade: float = 0.01,
    max_position_pct: float = 0.20,
    uncertainty: float = 0.0,
    reward_to_risk: float = 1.5,
    risk_limits: RiskLimits | None = None,
) -> float:
    """Compute risk-anchored position size.

    Args:
        equity: Current account equity ($)
        entry_price: Asset market/entry price ($)
        stop_distance: Dollar distance from entry to stop-loss ($)
        edge: Estimated net expected return of the setup (fraction > 0)
        confidence: Signal confidence (0.0 to 1.0)
        volatility: Local volatility estimate (fraction >= 0)
        drawdown: Current portfolio drawdown (0.0 to 1.0)
        max_risk_per_trade: Maximum fraction of equity to risk on 1R (e.g. 0.01 = 1%)
        max_position_pct: Maximum position notional as fraction of equity (e.g. 0.20 = 20%)
        uncertainty: Epistemic uncertainty (0.0 to 1.0, reduces size)
        reward_to_risk: Expected target to stop ratio
        risk_limits: Optional canonical RiskLimits configuration

    Returns:
        Safe quantity to trade (non-negative).
    """
    if risk_limits is not None:
        max_position_pct = risk_limits.max_position_pct / 100.0 if risk_limits.max_position_pct > 1.0 else risk_limits.max_position_pct
        max_risk_per_trade = risk_limits.max_order_pct / 100.0 if risk_limits.max_order_pct > 1.0 else risk_limits.max_order_pct

    if min(equity, entry_price, stop_distance) <= 0 or edge <= 0 or confidence <= 0:
        return 0.0

    # 1. Base Dollar Risk Budget (1R)
    drawdown_penalty = max(0.0, 1.0 - min(drawdown, 1.0))
    dollar_risk_budget = equity * max(0.0, max_risk_per_trade) * drawdown_penalty

    # 2. Confidence and Uncertainty Dampening
    confidence_scale = min(1.0, max(0.0, confidence))
    uncertainty_penalty = max(0.0, 1.0 - min(uncertainty, 1.0))
    risk_adjustment = confidence_scale * uncertainty_penalty

    # 3. Volatility Adjustment
    volatility_adjustment = 1.0 / (1.0 + max(0.0, volatility))

    # 4. Stop-Distance Anchor: Qty = Adjusted Risk Budget / Stop Distance
    effective_risk = dollar_risk_budget * risk_adjustment * volatility_adjustment
    qty_by_risk = effective_risk / stop_distance

    # 5. Edge-Scaled Fraction Cap: f* = min(1.0, edge / b)
    b = max(reward_to_risk, 0.5)
    edge_scaled_fraction = max(0.0, min(1.0, edge / b))
    edge_capped_notional = equity * edge_scaled_fraction * 0.5
    qty_by_edge = edge_capped_notional / entry_price if edge_capped_notional > 0 else qty_by_risk

    # 6. Hard Notional Cap
    hard_cap_qty = (equity * max_position_pct) / entry_price

    # Return minimum of all safety constraints
    final_qty = min(qty_by_risk, qty_by_edge, hard_cap_qty)
    return max(0.0, round(final_qty, 8))
