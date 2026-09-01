"""Reward functions for the trading environment.

Each function computes a scalar reward from the current portfolio state
and trading action. Selectable via config: training.reward_function.
"""

from __future__ import annotations

import numpy as np


def portfolio_reward(
    net_return: float,
    drawdown_increase: float,
    turnover: float,
    equity: float,
    drawdown_penalty: float = 0.5,
    turnover_penalty: float = 0.05,
    risk_penalty: float = 0.0,
) -> float:
    """Scale-free reward: net return minus incremental downside and turnover.

    Net return is calculated after fill prices, fees, and slippage by the
    accounting component.  The formula is ``r - a*max(0, Δdrawdown)
    - b*turnover/equity - risk_penalty`` and is clipped for stable learning.
    """
    reward = net_return - drawdown_penalty * max(0.0, drawdown_increase)
    reward -= turnover_penalty * turnover / max(equity, 1e-12)
    return float(np.clip(reward - risk_penalty, -1.0, 1.0))


def pnl_reward(
    portfolio_value: float,
    prev_portfolio_value: float,
    **kwargs,
) -> float:
    """Simple PnL change reward.

    reward = (current_value - previous_value) / previous_value
    """
    if prev_portfolio_value <= 0:
        return 0.0
    return (portfolio_value - prev_portfolio_value) / prev_portfolio_value


def sharpe_reward(
    returns_history: list[float] | np.ndarray,
    risk_free_rate: float = 0.0,
    window: int = 30,
    **kwargs,
) -> float:
    """Rolling Sharpe ratio as reward.

    Uses the trailing `window` returns to compute an annualized Sharpe ratio,
    then normalizes to a per-step reward.
    """
    if len(returns_history) < 2:
        return 0.0

    recent = np.array(returns_history[-window:])
    excess = recent - risk_free_rate / 252  # daily risk-free rate
    mean_ret = np.mean(excess)
    std_ret = np.std(excess)

    if std_ret < 1e-8:
        return 0.0

    # Annualized Sharpe, then normalized per step
    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    return float(np.clip(sharpe / 10.0, -1.0, 1.0))  # scale to [-1, 1]


def risk_adjusted_reward(
    portfolio_value: float,
    prev_portfolio_value: float,
    max_portfolio_value: float,
    returns_history: list[float] | np.ndarray,
    drawdown_penalty_weight: float = 0.5,
    volatility_penalty_weight: float = 0.3,
    **kwargs,
) -> float:
    """PnL reward penalized by drawdown and volatility.

    reward = pnl_return - α * drawdown_penalty - β * volatility_penalty
    """
    if prev_portfolio_value <= 0:
        return 0.0

    # Base PnL return
    pnl_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value

    # Drawdown penalty: how far below peak
    if max_portfolio_value > 0:
        drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value
    else:
        drawdown = 0.0

    # Volatility penalty: recent return volatility
    if len(returns_history) >= 5:
        vol = float(np.std(returns_history[-20:]))
    else:
        vol = 0.0

    reward = (
        pnl_return
        - drawdown_penalty_weight * drawdown
        - volatility_penalty_weight * vol
    )

    return float(np.clip(reward, -1.0, 1.0))


def differential_sharpe_reward(
    returns_history: list[float] | np.ndarray,
    prev_A: float = 0.0,
    prev_B: float = 0.0,
    eta: float = 0.01,
    **kwargs,
) -> tuple[float, float, float]:
    """Differential Sharpe ratio (Moody & Saffell, 1998).

    Computes an incremental approximation of the Sharpe ratio change,
    which provides a smoother gradient signal than batch Sharpe.

    Returns:
        Tuple of (reward, updated_A, updated_B) where A and B are
        the exponential moving averages of returns and squared returns.
    """
    if len(returns_history) < 1:
        return 0.0, prev_A, prev_B

    r = returns_history[-1]

    # Update exponential moving averages
    new_A = prev_A + eta * (r - prev_A)
    new_B = prev_B + eta * (r**2 - prev_B)

    # Differential Sharpe
    denominator = (new_B - new_A**2) ** 1.5
    if abs(denominator) < 1e-10:
        dsr = 0.0
    else:
        dsr = (new_B * (r - prev_A) - 0.5 * prev_A * (r**2 - prev_B)) / denominator

    reward = float(np.clip(dsr, -1.0, 1.0))
    return reward, float(new_A), float(new_B)


# Registry of reward functions
REWARD_FUNCTIONS = {
    "pnl": pnl_reward,
    "sharpe": sharpe_reward,
    "risk_adjusted": risk_adjusted_reward,
    "differential_sharpe": differential_sharpe_reward,
}
