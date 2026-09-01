"""Champion vs challenger promotion with multi-criteria gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trade.evolution.evaluator import EvaluationResult


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


class ChampionSelector:
    """Promotion requires multiple robust criteria; never single-metric."""

    def __init__(
        self,
        minimum_sharpe_improvement: float = 0.1,
        minimum_expectancy_gain: float = 0.0,
        minimum_profit_factor: float = 1.0,
        minimum_positive_walkforward_ratio: float = 0.5,
        acceptable_drawdown_limit: float = 0.25,
        minimum_trade_count: int = 30,
    ):
        self.minimum_sharpe_improvement = minimum_sharpe_improvement
        self.minimum_expectancy_gain = minimum_expectancy_gain
        self.minimum_profit_factor = minimum_profit_factor
        self.minimum_positive_walkforward_ratio = minimum_positive_walkforward_ratio
        self.acceptable_drawdown_limit = acceptable_drawdown_limit
        self.minimum_trade_count = minimum_trade_count

    def decide(
        self,
        evaluation: EvaluationResult,
        champion_sharpe: float,
        challenger_sharpe: float,
        champion_expectancy: float,
        challenger_expectancy: float,
        challenger_max_drawdown: float,
        challenger_profit_factor: float,
        challenger_trade_count: int,
    ) -> PromotionDecision:
        evidence: dict[str, Any] = {
            "champion_sharpe": champion_sharpe,
            "challenger_sharpe": challenger_sharpe,
            "evaluation_passed": evaluation.passed,
        }
        if not evaluation.passed:
            return PromotionDecision(False, "evaluation_failed", {**evidence, "reasons": evaluation.rejection_reasons})

        checks = []
        if challenger_sharpe <= champion_sharpe + self.minimum_sharpe_improvement:
            checks.append("sharpe_improvement_insufficient")
        if challenger_expectancy <= champion_expectancy + self.minimum_expectancy_gain:
            checks.append("expectancy_gain_insufficient")
        if challenger_profit_factor < self.minimum_profit_factor:
            checks.append("profit_factor_below_minimum")
        if challenger_max_drawdown > self.acceptable_drawdown_limit:
            checks.append("drawdown_too_high")
        if challenger_trade_count < self.minimum_trade_count:
            checks.append("insufficient_trades")
        wf = evaluation.walk_forward
        if wf and wf.positive_window_ratio < self.minimum_positive_walkforward_ratio:
            checks.append("walk_forward_ratio_low")
        if evaluation.cost_stress_net_return <= 0:
            checks.append("cost_stress_negative")

        if checks:
            return PromotionDecision(False, ";".join(checks), evidence)
        return PromotionDecision(True, "all_promotion_criteria_met", evidence)
