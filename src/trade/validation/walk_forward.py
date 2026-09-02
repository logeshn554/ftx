"""Walk-forward validation: rolling train/validation/test evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from trade.core.types import BacktestResult, ModelVersion
from trade.validation.backtester import Backtester
from trade.validation import metrics as m

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation result."""

    model_version: ModelVersion
    n_windows: int = 0
    oos_sharpe_mean: float = 0.0
    oos_sharpe_std: float = 0.0
    oos_return_mean: float = 0.0
    oos_max_drawdown_mean: float = 0.0
    oos_return_median: float = 0.0
    oos_return_std: float = 0.0
    positive_window_ratio: float = 0.0
    worst_window_return: float = 0.0
    window_results: list[dict] = field(default_factory=list)


def infer_bars_per_day(df: pd.DataFrame) -> int:
    """Infer the number of bars per day from dataframe index or timestamps."""
    if isinstance(df.index, pd.DatetimeIndex) and len(df) >= 2:
        diffs = df.index.to_series().diff().dropna()
        median_seconds = diffs.dt.total_seconds().median()
        if median_seconds and median_seconds > 0:
            bars_per_day = int(round(86400.0 / median_seconds))
            return max(1, bars_per_day)
    return 1


class WalkForwardValidator:
    """Performs rolling walk-forward validation.

    Splits data into strictly chronological, non-overlapping train,
    validation, and test windows. Supports calendar-time aware scaling for sub-daily bars.
    """

    def __init__(
        self,
        train_window_days: int = 252,
        test_window_days: int = 63,
        step_days: int = 63,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        feature_window: int = 30,
        validation_window_days: int = 42,
        timeframe: str = "1d",
    ) -> None:
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.feature_window = feature_window
        self.validation_window_days = max(0, validation_window_days)
        self.timeframe = timeframe

    def validate(
        self,
        model_path: str,
        features_df: pd.DataFrame,
        feature_columns: list[str],
        model_version: ModelVersion | None = None,
        train_callback: Callable[[pd.DataFrame, pd.DataFrame, list[str]], str] | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward validation across multiple windows."""
        if model_version is None:
            model_version = ModelVersion(major=0, minor=0, patch=0)

        if isinstance(features_df.index, pd.DatetimeIndex) and not features_df.index.is_monotonic_increasing:
            features_df = features_df.sort_index().copy()

        n_bars = len(features_df)
        bars_per_day = infer_bars_per_day(features_df)

        # Scale day-based configuration to actual bar indices
        train_bars = self.train_window_days * bars_per_day
        val_bars = self.validation_window_days * bars_per_day
        test_bars = self.test_window_days * bars_per_day
        step_bars = self.step_days * bars_per_day

        # Fallback to direct row counts if dataframe is smaller than 1 full calendar cycle
        if train_bars + val_bars + test_bars > n_bars:
            train_bars = self.train_window_days
            val_bars = self.validation_window_days
            test_bars = self.test_window_days
            step_bars = self.step_days

        total_window = train_bars + val_bars + test_bars
        window_results: list[dict] = []

        backtester = Backtester(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
            feature_window=self.feature_window,
            timeframe=self.timeframe,
        )

        start = 0
        window_id = 0

        while start + total_window <= n_bars:
            train_end = start + train_bars
            validation_end = train_end + val_bars
            test_end = validation_end + test_bars

            train_df = features_df.iloc[start:train_end].copy()
            validation_df = features_df.iloc[train_end:validation_end].copy()
            test_df = features_df.iloc[validation_end:test_end].copy()

            if len(test_df) < self.feature_window + 10:
                start += step_bars
                continue

            active_model_path = model_path
            if train_callback is not None:
                try:
                    active_model_path = train_callback(train_df, validation_df, feature_columns)
                except Exception:
                    logger.warning("Train callback failed for window %d, falling back to base model", window_id, exc_info=True)
                    active_model_path = model_path

            try:
                result = backtester.run(
                    model_path=active_model_path,
                    features_df=test_df,
                    feature_columns=feature_columns,
                    model_version=model_version,
                )

                window_results.append({
                    "window_id": window_id,
                    "train_start": start,
                    "train_end": train_end,
                    "val_start": train_end,
                    "val_end": validation_end,
                    "test_start": validation_end,
                    "test_end": test_end,
                    "total_return": result.total_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "total_trades": result.total_trades,
                    "daily_returns": result.daily_returns,
                })
            except Exception:
                logger.warning("Backtest failed for window %d", window_id, exc_info=True)

            start += step_bars
            window_id += 1

        if not window_results:
            return WalkForwardResult(model_version=model_version)

        returns = [w["total_return"] for w in window_results]
        sharpes = [w["sharpe_ratio"] for w in window_results]
        drawdowns = [w["max_drawdown"] for w in window_results]

        all_returns = []
        for w in window_results:
            all_returns.extend(w["daily_returns"])

        oos_sharpe_mean = float(np.mean(sharpes))
        oos_sharpe_std = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0

        if all_returns:
            pooled_sharpe = m.sharpe_ratio(all_returns, periods_per_year=backtester.periods_per_year)
            oos_sharpe_mean = pooled_sharpe

        return WalkForwardResult(
            model_version=model_version,
            n_windows=len(window_results),
            oos_sharpe_mean=oos_sharpe_mean,
            oos_sharpe_std=oos_sharpe_std,
            oos_return_mean=float(np.mean(returns)),
            oos_max_drawdown_mean=float(np.mean(drawdowns)),
            oos_return_median=float(np.median(returns)),
            oos_return_std=float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
            positive_window_ratio=float(np.mean([1.0 if r > 0 else 0.0 for r in returns])),
            worst_window_return=float(np.min(returns)),
            window_results=window_results,
        )
