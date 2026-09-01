"""Walk-forward validation: rolling train/test evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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
    window_results: list[dict] = field(default_factory=list)


class WalkForwardValidator:
    """Performs rolling walk-forward validation.

    Splits data into overlapping train/test windows and evaluates
    the model on each out-of-sample test window to assess
    robustness and consistency.
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
    ) -> None:
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.feature_window = feature_window

    def validate(
        self,
        model_path: str,
        features_df: pd.DataFrame,
        feature_columns: list[str],
        model_version: ModelVersion | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward validation across multiple windows.

        Args:
            model_path: Path to the saved model.
            features_df: Full feature DataFrame.
            feature_columns: Feature column names.
            model_version: Model version descriptor.

        Returns:
            WalkForwardResult with aggregated out-of-sample metrics.
        """
        if model_version is None:
            model_version = ModelVersion(major=0, minor=0, patch=0)

        n_bars = len(features_df)
        total_window = self.train_window_days + self.test_window_days
        window_results: list[dict] = []

        backtester = Backtester(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
            feature_window=self.feature_window,
        )

        # Generate windows
        start = 0
        window_id = 0

        while start + total_window <= n_bars:
            train_end = start + self.train_window_days
            test_end = train_end + self.test_window_days

            # Only evaluate on the test (OOS) window
            test_df = features_df.iloc[train_end:test_end].copy()

            if len(test_df) < self.feature_window + 10:
                start += self.step_days
                continue

            try:
                result = backtester.run(
                    model_path=model_path,
                    features_df=test_df,
                    feature_columns=feature_columns,
                    model_version=model_version,
                )

                window_results.append({
                    "window_id": window_id,
                    "train_start": start,
                    "train_end": train_end,
                    "test_start": train_end,
                    "test_end": test_end,
                    "oos_sharpe": result.sharpe_ratio,
                    "oos_return": result.total_return,
                    "oos_max_drawdown": result.max_drawdown,
                    "oos_trades": result.total_trades,
                    "oos_win_rate": result.win_rate,
                })
            except Exception:
                logger.warning("Walk-forward window %d failed", window_id, exc_info=True)

            start += self.step_days
            window_id += 1

        if not window_results:
            logger.warning("No valid walk-forward windows generated")
            return WalkForwardResult(model_version=model_version)

        # Aggregate OOS metrics
        oos_sharpes = [w["oos_sharpe"] for w in window_results]
        oos_returns = [w["oos_return"] for w in window_results]
        oos_mdds = [w["oos_max_drawdown"] for w in window_results]

        result = WalkForwardResult(
            model_version=model_version,
            n_windows=len(window_results),
            oos_sharpe_mean=float(np.mean(oos_sharpes)),
            oos_sharpe_std=float(np.std(oos_sharpes)),
            oos_return_mean=float(np.mean(oos_returns)),
            oos_max_drawdown_mean=float(np.mean(oos_mdds)),
            window_results=window_results,
        )

        logger.info(
            "Walk-forward %s: %d windows | OOS Sharpe: %.2f ± %.2f | "
            "OOS Return: %.2f%% | OOS MDD: %.2f%%",
            model_version.tag,
            result.n_windows,
            result.oos_sharpe_mean,
            result.oos_sharpe_std,
            result.oos_return_mean * 100,
            result.oos_max_drawdown_mean * 100,
        )

        return result
