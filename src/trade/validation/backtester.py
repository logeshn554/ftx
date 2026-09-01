"""Backtesting engine: run a model against historical data."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from trade.core.types import BacktestResult, ModelVersion
from trade.data.features import FeatureEngine
from trade.env.trading_env import TradingEnv
from trade.validation import metrics as m

logger = logging.getLogger(__name__)


class Backtester:
    """Runs a frozen model against historical data.

    Simulates the full trading loop with the same environment used
    for training, collecting detailed trade logs and equity curves
    for analysis.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        feature_window: int = 30,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.feature_window = feature_window

    def run(
        self,
        model_path: str,
        features_df: pd.DataFrame,
        feature_columns: list[str],
        model_version: ModelVersion | None = None,
    ) -> BacktestResult:
        """Run a backtest with a saved model.

        Args:
            model_path: Path to the saved SB3 model file.
            features_df: DataFrame with OHLCV + computed features.
            feature_columns: List of feature column names.
            model_version: Version descriptor for the model.

        Returns:
            BacktestResult with full metrics, equity curve, and trade log.
        """
        if model_version is None:
            model_version = ModelVersion(major=0, minor=0, patch=0)

        # Create environment
        env = TradingEnv(
            features_df=features_df,
            feature_columns=feature_columns,
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
            feature_window=self.feature_window,
            reward_function="pnl",  # Use raw PnL for backtest clarity
        )

        # Load model
        model = PPO.load(model_path, device="cpu")
        model.policy.eval()

        # Run episode
        obs, info = env.reset()
        equity_curve = [self.initial_capital]
        daily_returns: list[float] = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            portfolio_value = info["portfolio_value"]
            equity_curve.append(portfolio_value)

            if len(equity_curve) >= 2:
                prev = equity_curve[-2]
                daily_ret = (portfolio_value - prev) / prev if prev > 0 else 0.0
                daily_returns.append(daily_ret)

        # Compute metrics
        equity_arr = np.array(equity_curve)
        returns_arr = np.array(daily_returns)
        trade_log = env.trade_log

        # Extract trade PnLs
        trade_pnls = [t.get("pnl", 0.0) for t in trade_log if "pnl" in t]
        trade_values = [t.get("price", 0.0) * t.get("shares", 0.0) for t in trade_log]

        # Determine date range from features_df index
        if hasattr(features_df.index, 'date'):
            start_date = features_df.index[0].date() if hasattr(features_df.index[0], 'date') else dt.date.today()
            end_date = features_df.index[-1].date() if hasattr(features_df.index[-1], 'date') else dt.date.today()
        else:
            start_date = dt.date.today()
            end_date = dt.date.today()

        result = BacktestResult(
            model_version=model_version,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=float(equity_curve[-1]),
            total_return=m.total_return(equity_arr),
            sharpe_ratio=m.sharpe_ratio(returns_arr),
            max_drawdown=m.max_drawdown(equity_arr),
            win_rate=m.win_rate(trade_pnls) if trade_pnls else 0.0,
            profit_factor=m.profit_factor(trade_pnls) if trade_pnls else 0.0,
            total_trades=len(trade_log),
            transaction_costs=m.total_transaction_costs(trade_values, self.commission_pct),
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            trade_log=trade_log,
        )

        logger.info(
            "Backtest %s: Return=%.2f%% | Sharpe=%.2f | MDD=%.2f%% | "
            "Trades=%d | Win Rate=%.1f%%",
            model_version.tag,
            result.total_return * 100,
            result.sharpe_ratio,
            result.max_drawdown * 100,
            result.total_trades,
            result.win_rate * 100,
        )

        return result
