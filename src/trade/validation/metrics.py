"""Pure metric computation functions.

All functions operate on numpy arrays of returns or trade results.
No side effects, no state — purely functional.
"""

from __future__ import annotations

import numpy as np


def sharpe_ratio(
    returns: np.ndarray | list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio.

    Args:
        returns: Array of periodic returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of trading periods per year.

    Returns:
        Annualized Sharpe ratio.
    """
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0

    excess = r - risk_free_rate / periods_per_year
    mean_excess = np.mean(excess)
    std = np.std(excess, ddof=1)

    if std < 1e-10:
        return 0.0

    return float(mean_excess / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray | list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (penalizes only downside volatility)."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0

    excess = r - risk_free_rate / periods_per_year
    mean_excess = np.mean(excess)
    downside = r[r < 0]

    if len(downside) < 2:
        return float(mean_excess * np.sqrt(periods_per_year)) if mean_excess > 0 else 0.0

    downside_std = np.std(downside, ddof=1)
    if downside_std < 1e-10:
        return 0.0

    return float(mean_excess / downside_std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray | list[float]) -> float:
    """Maximum drawdown from peak.

    Args:
        equity_curve: Array of portfolio values over time.

    Returns:
        Maximum drawdown as a positive fraction (e.g. 0.15 = 15% drawdown).
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2:
        return 0.0

    peak = np.maximum.accumulate(eq)
    drawdowns = (peak - eq) / np.where(peak > 0, peak, 1.0)
    return float(np.max(drawdowns))


def calmar_ratio(
    returns: np.ndarray | list[float],
    equity_curve: np.ndarray | list[float],
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0

    annual_return = np.mean(r) * periods_per_year
    mdd = max_drawdown(equity_curve)

    if mdd < 1e-10:
        return 0.0

    return float(annual_return / mdd)


def win_rate(trade_pnls: np.ndarray | list[float]) -> float:
    """Fraction of profitable trades."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    if len(pnls) == 0:
        return 0.0
    return float(np.sum(pnls > 0) / len(pnls))


def profit_factor(trade_pnls: np.ndarray | list[float]) -> float:
    """Ratio of gross profits to gross losses."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = abs(np.sum(pnls[pnls < 0]))

    if gross_loss < 1e-10:
        return float("inf") if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def total_return(equity_curve: np.ndarray | list[float]) -> float:
    """Total return from start to end."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2 or eq[0] <= 0:
        return 0.0
    return float((eq[-1] - eq[0]) / eq[0])


def annualized_return(
    returns: np.ndarray | list[float],
    periods_per_year: int = 252,
) -> float:
    """Annualized compound return."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0

    cumulative = np.prod(1 + r)
    n_years = len(r) / periods_per_year

    if n_years <= 0 or cumulative <= 0:
        return 0.0

    return float(cumulative ** (1 / n_years) - 1)


def total_transaction_costs(
    trade_values: np.ndarray | list[float],
    commission_pct: float = 0.001,
) -> float:
    """Total transaction costs across all trades."""
    values = np.asarray(trade_values, dtype=np.float64)
    return float(np.sum(np.abs(values)) * commission_pct)
