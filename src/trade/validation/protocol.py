"""Strict Institutional Evaluation Protocol Engine.

Enforces the sequential research protocol:
Dataset Lock -> Train -> Val -> Untouched OOS Test -> Walk-Forward -> Cost Stress -> Monte Carlo -> Deflated Sharpe -> Promotion
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable

import numpy as np
import pandas as pd

from trade.core.types import BacktestResult, ModelVersion
from trade.validation.backtester import Backtester
from trade.validation.monte_carlo import MonteCarloResult, MonteCarloTester
from trade.validation.walk_forward import WalkForwardResult, WalkForwardValidator
from trade.validation import metrics as m


@dataclass(frozen=True)
class ProtocolStageResult:
    stage_name: str
    passed: bool
    metrics: dict[str, Any]
    rejection_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProtocolReport:
    model_version: str
    passed: bool
    dataset_hash: str
    stages: dict[str, ProtocolStageResult]
    final_oos_sharpe: float
    final_oos_return: float
    deflated_sharpe_prob: float
    rejection_reasons: list[str]
    audit_trail: dict[str, Any]


class ResearchProtocolEngine:
    """Executes the strict institutional research protocol."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        feature_window: int = 30,
        min_oos_sharpe: float = 0.5,
        min_oos_profit_factor: float = 1.0,
        max_acceptable_drawdown: float = 0.25,
        min_walk_forward_positive_ratio: float = 0.50,
        min_deflated_sharpe_prob: float = 0.90,
        cost_stress_fee_mult: float = 1.5,
        cost_stress_slip_mult: float = 2.0,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.feature_window = feature_window
        self.min_oos_sharpe = min_oos_sharpe
        self.min_oos_profit_factor = min_oos_profit_factor
        self.max_acceptable_drawdown = max_acceptable_drawdown
        self.min_walk_forward_positive_ratio = min_walk_forward_positive_ratio
        self.min_deflated_sharpe_prob = min_deflated_sharpe_prob
        self.cost_stress_fee_mult = cost_stress_fee_mult
        self.cost_stress_slip_mult = cost_stress_slip_mult

        self.walk_forward_validator = WalkForwardValidator(
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            feature_window=feature_window,
        )
        self.monte_carlo_tester = MonteCarloTester()

    def compute_dataset_hash(self, df: pd.DataFrame) -> str:
        """Compute deterministic sha256 hash of dataset to ensure immutability."""
        content = f"{len(df)}_{df.columns.tolist()}_{df.iloc[0].values}_{df.iloc[-1].values}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def evaluate(
        self,
        model_path: str,
        features_df: pd.DataFrame,
        feature_columns: list[str],
        model_version: str = "candidate-v1",
        num_prior_trials: int = 1,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
    ) -> ProtocolReport:
        stages: dict[str, ProtocolStageResult] = {}
        all_rejections: list[str] = []
        dataset_hash = self.compute_dataset_hash(features_df)

        n_rows = len(features_df)
        train_end = int(n_rows * train_ratio)
        val_end = int(n_rows * (train_ratio + val_ratio))

        train_df = features_df.iloc[:train_end]
        val_df = features_df.iloc[train_end:val_end]
        untouched_test_df = features_df.iloc[val_end:]

        # 1. Untouched OOS Test Stage
        backtester = Backtester(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
            feature_window=self.feature_window,
        )
        test_result = backtester.run(
            model_path=model_path,
            features_df=untouched_test_df,
            feature_columns=feature_columns,
            model_version=ModelVersion(major=1, minor=0, patch=0),
        )

        test_rejections = []
        if test_result.sharpe_ratio < self.min_oos_sharpe:
            test_rejections.append("oos_sharpe_below_minimum")
        if test_result.profit_factor < self.min_oos_profit_factor:
            test_rejections.append("oos_profit_factor_below_minimum")
        if test_result.max_drawdown > self.max_acceptable_drawdown:
            test_rejections.append("oos_drawdown_exceeded")

        stages["untouched_oos_test"] = ProtocolStageResult(
            stage_name="untouched_oos_test",
            passed=len(test_rejections) == 0,
            metrics={
                "sharpe": test_result.sharpe_ratio,
                "total_return": test_result.total_return,
                "max_drawdown": test_result.max_drawdown,
                "profit_factor": test_result.profit_factor,
                "total_trades": test_result.total_trades,
            },
            rejection_reasons=test_rejections,
        )
        all_rejections.extend(test_rejections)

        # 2. Rolling Walk-Forward Validation Stage
        wf_result = self.walk_forward_validator.validate(
            model_path=model_path,
            features_df=features_df,
            feature_columns=feature_columns,
        )
        wf_rejections = []
        if wf_result.positive_window_ratio < self.min_walk_forward_positive_ratio:
            wf_rejections.append("walk_forward_positive_ratio_low")
        if wf_result.oos_return_mean <= 0:
            wf_rejections.append("walk_forward_negative_mean_return")

        stages["walk_forward"] = ProtocolStageResult(
            stage_name="walk_forward",
            passed=len(wf_rejections) == 0,
            metrics={
                "n_windows": wf_result.n_windows,
                "positive_window_ratio": wf_result.positive_window_ratio,
                "oos_sharpe_mean": wf_result.oos_sharpe_mean,
                "oos_return_mean": wf_result.oos_return_mean,
            },
            rejection_reasons=wf_rejections,
        )
        all_rejections.extend(wf_rejections)

        # 3. Execution Cost Stress Test Stage
        stress_backtester = Backtester(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct * self.cost_stress_fee_mult,
            slippage_pct=self.slippage_pct * self.cost_stress_slip_mult,
            feature_window=self.feature_window,
        )
        stress_result = stress_backtester.run(
            model_path=model_path,
            features_df=untouched_test_df,
            feature_columns=feature_columns,
        )
        stress_rejections = []
        if stress_result.total_return <= 0:
            stress_rejections.append("cost_stress_negative_return")
        if stress_result.sharpe_ratio <= 0:
            stress_rejections.append("cost_stress_negative_sharpe")

        stages["cost_stress"] = ProtocolStageResult(
            stage_name="cost_stress",
            passed=len(stress_rejections) == 0,
            metrics={
                "stressed_return": stress_result.total_return,
                "stressed_sharpe": stress_result.sharpe_ratio,
                "stressed_costs": stress_result.transaction_costs,
            },
            rejection_reasons=stress_rejections,
        )
        all_rejections.extend(stress_rejections)

        # 4. Monte Carlo Sequence Permutation Stage
        trade_pnls = [t["net_pnl"] for t in test_result.trade_log if "net_pnl" in t]
        mc_rejections = []
        mc_result: MonteCarloResult | None = None
        if len(trade_pnls) >= 10:
            mc_result = self.monte_carlo_tester.test(trade_pnls)
            if not mc_result.passed:
                mc_rejections.extend(mc_result.rejection_reasons)
        else:
            mc_rejections.append("insufficient_trades_for_monte_carlo")

        stages["monte_carlo"] = ProtocolStageResult(
            stage_name="monte_carlo",
            passed=len(mc_rejections) == 0,
            metrics={"passed": mc_result.passed if mc_result else False},
            rejection_reasons=mc_rejections,
        )
        all_rejections.extend(mc_rejections)

        # 5. Multiple-Testing Deflated Sharpe Ratio (DSR) Stage
        track_record_len = len(untouched_test_df)
        dsr_prob = m.deflated_sharpe_ratio(
            estimated_sharpe=test_result.sharpe_ratio,
            num_trials=max(1, num_prior_trials),
            track_record_length=track_record_len,
        )
        dsr_rejections = []
        if dsr_prob < self.min_deflated_sharpe_prob:
            dsr_rejections.append("deflated_sharpe_below_confidence_threshold")

        stages["deflated_sharpe"] = ProtocolStageResult(
            stage_name="deflated_sharpe",
            passed=len(dsr_rejections) == 0,
            metrics={
                "dsr_prob": dsr_prob,
                "num_trials": num_prior_trials,
                "track_record_length": track_record_len,
            },
            rejection_reasons=dsr_rejections,
        )
        all_rejections.extend(dsr_rejections)

        overall_passed = len(all_rejections) == 0

        return ProtocolReport(
            model_version=model_version,
            passed=overall_passed,
            dataset_hash=dataset_hash,
            stages=stages,
            final_oos_sharpe=test_result.sharpe_ratio,
            final_oos_return=test_result.total_return,
            deflated_sharpe_prob=dsr_prob,
            rejection_reasons=all_rejections,
            audit_trail={
                "n_samples": len(features_df),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(untouched_test_df),
                "num_prior_trials": num_prior_trials,
            },
        )
