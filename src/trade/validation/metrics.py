"""Pure institutional metric computation functions.

All functions operate on numpy arrays of returns, trade results, or equity curves.
No side effects, purely functional and mathematically verified.
"""

from __future__ import annotations

import numpy as np


def sharpe_ratio(
    returns: np.ndarray | list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio."""
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
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio with downside root-mean-square deviation below target."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0

    excess = r - target_return / periods_per_year
    mean_excess = np.mean(excess)
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(downside ** 2))

    if downside_dev < 1e-10:
        return float(mean_excess * np.sqrt(periods_per_year)) if mean_excess > 0 else 0.0

    return float(mean_excess / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray | list[float]) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
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
    """Calmar ratio: Compound Annual Growth Rate (CAGR) / Max Drawdown."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0

    mdd = max_drawdown(equity_curve)
    if mdd < 1e-10:
        return 0.0

    cagr = annualized_return(r, periods_per_year=periods_per_year)
    return float(cagr / mdd)


def total_return(equity_curve: np.ndarray | list[float]) -> float:
    """Total cumulative return from start to finish."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2 or eq[0] <= 0:
        return 0.0
    return float((eq[-1] - eq[0]) / eq[0])


def annualized_return(
    returns: np.ndarray | list[float],
    periods_per_year: int = 252,
) -> float:
    """Compound Annual Growth Rate (CAGR)."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0

    cumulative = float(np.prod(1.0 + r))
    n_years = len(r) / float(periods_per_year)

    if n_years <= 0 or cumulative <= 0:
        return 0.0

    return float(cumulative ** (1.0 / n_years) - 1.0)


def win_rate(trade_pnls: np.ndarray | list[float]) -> float:
    """Fraction of winning trades."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    if len(pnls) == 0:
        return 0.0
    return float(np.sum(pnls > 0) / len(pnls))


def profit_factor(trade_pnls: np.ndarray | list[float]) -> float:
    """Ratio of gross profit to gross loss."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    gp = float(np.sum(pnls[pnls > 0]))
    gl = float(abs(np.sum(pnls[pnls < 0])))

    if gl < 1e-10:
        return float("inf") if gp > 0 else 0.0
    return float(gp / gl)


def gross_profit(trade_pnls: np.ndarray | list[float]) -> float:
    """Sum of all positive trade returns/dollar amounts."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    return float(np.sum(pnls[pnls > 0])) if len(pnls) > 0 else 0.0


def gross_loss(trade_pnls: np.ndarray | list[float]) -> float:
    """Sum of all negative trade returns/dollar amounts as a positive value."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    return float(abs(np.sum(pnls[pnls < 0]))) if len(pnls) > 0 else 0.0


def net_profit(trade_pnls: np.ndarray | list[float]) -> float:
    """Net dollar/return sum of all trades."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    return float(np.sum(pnls)) if len(pnls) > 0 else 0.0


def average_win(trade_pnls: np.ndarray | list[float]) -> float:
    """Average return of winning trades."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    wins = pnls[pnls > 0]
    return float(np.mean(wins)) if len(wins) > 0 else 0.0


def average_loss(trade_pnls: np.ndarray | list[float]) -> float:
    """Average loss of losing trades as a positive value."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    losses = pnls[pnls < 0]
    return float(abs(np.mean(losses))) if len(losses) > 0 else 0.0


def payoff_ratio(trade_pnls: np.ndarray | list[float]) -> float:
    """Payoff ratio: Average Win / Average Loss."""
    avg_w = average_win(trade_pnls)
    avg_l = average_loss(trade_pnls)
    if avg_l < 1e-10:
        return float("inf") if avg_w > 0 else 0.0
    return float(avg_w / avg_l)


def expectancy(trade_pnls: np.ndarray | list[float]) -> float:
    """Mathematical expectancy per trade: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    if len(pnls) == 0:
        return 0.0
    w_rate = win_rate(pnls)
    l_rate = 1.0 - w_rate
    avg_w = average_win(pnls)
    avg_l = average_loss(pnls)
    return float(w_rate * avg_w - l_rate * avg_l)


def cvar_tail_loss(returns: np.ndarray | list[float], alpha: float = 0.05) -> float:
    """Conditional Value at Risk (Expected Shortfall) at alpha significance."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    cutoff = np.percentile(r, alpha * 100)
    tail = r[r <= cutoff]
    return float(abs(np.mean(tail))) if len(tail) > 0 else float(abs(cutoff))


def time_exposure(positions: np.ndarray | list[float]) -> float:
    """Fraction of bars with active non-zero position exposure."""
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) == 0:
        return 0.0
    return float(np.sum(pos != 0) / len(pos))


def capital_exposure(position_notionals: np.ndarray | list[float], equity_curve: np.ndarray | list[float]) -> float:
    """Mean capital allocation fraction: mean(abs(position_notional) / equity)."""
    pn = np.asarray(position_notionals, dtype=np.float64)
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(pn) == 0 or len(eq) == 0:
        return 0.0
    min_len = min(len(pn), len(eq))
    ratios = np.abs(pn[:min_len]) / np.maximum(1e-8, eq[:min_len])
    return float(np.mean(ratios))


# Alias for backward compatibility
market_exposure_pct = time_exposure


def cost_to_profit_ratio(total_costs: float, gross_profit_val: float) -> float:
    """Ratio of total transaction friction to gross profit."""
    gp = float(gross_profit_val)
    tc = float(total_costs)
    if gp <= 0:
        return float("inf") if tc > 0 else 0.0
    return float(tc / gp)


def max_consecutive_losses(trade_pnls: np.ndarray | list[float]) -> int:
    """Maximum streak of consecutive losing trades."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    max_streak = current_streak = 0
    for p in pnls:
        if p < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return int(max_streak)


def max_consecutive_wins(trade_pnls: np.ndarray | list[float]) -> int:
    """Maximum streak of consecutive winning trades."""
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    max_streak = current_streak = 0
    for p in pnls:
        if p > 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return int(max_streak)


def ulcer_index(equity_curve: np.ndarray | list[float]) -> float:
    """Peter Martin's Ulcer Index measuring depth and duration of drawdowns."""
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    drawdowns_pct = 100.0 * (peak - eq) / np.where(peak > 0, peak, 1.0)
    return float(np.sqrt(np.mean(drawdowns_pct ** 2)))


def deflated_sharpe_ratio(
    estimated_sharpe: float,
    num_trials: int,
    track_record_length: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Bailey & Lopez de Prado's Deflated Sharpe Ratio (DSR).

    Adjusts estimated Sharpe ratio for the maximum expected Sharpe under the null
    hypothesis across N independent trials with non-normal skewness/kurtosis.
    """
    from scipy.stats import norm
    if num_trials <= 1 or track_record_length <= 1:
        return float(norm.cdf(estimated_sharpe * np.sqrt(max(1, track_record_length))))

    gamma = 0.5772156649
    z = (1 - gamma) * norm.ppf(1 - 1.0 / num_trials) + gamma * norm.ppf(1 - 1.0 / (num_trials * np.e))
    expected_max_null_sharpe = float(z)

    var_sharpe = (1.0 + (0.5 * estimated_sharpe ** 2) - (skewness * estimated_sharpe) + ((kurtosis - 3) / 4.0 * estimated_sharpe ** 2)) / track_record_length
    std_sharpe = np.sqrt(max(1e-8, var_sharpe))

    dsr_stat = (estimated_sharpe - expected_max_null_sharpe) / std_sharpe
    return float(norm.cdf(dsr_stat))
