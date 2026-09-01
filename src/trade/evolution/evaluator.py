"""Candidate evaluation pipeline with structured results, cost stress testing, and Monte Carlo gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trade.evolution.candidate_generator import Candidate
from trade.validation.comparator import ModelComparator
from trade.validation.monte_carlo import MonteCarloResult, MonteCarloTester
from trade.validation.walk_forward import WalkForwardResult


@dataclass
class EvaluationResult:
    candidate_id: str
    passed: bool
    rejection_reasons: list[str] = field(default_factory=list)
    walk_forward: WalkForwardResult | None = None
    cost_stress_net_return: float = 0.0
    monte_carlo: MonteCarloResult | None = None
    champion_comparison: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)


class CandidateEvaluator:
    """candidate → train → walk-forward → cost stress → monte carlo → champion comparison"""

    def __init__(
        self,
        minimum_positive_window_ratio: float = 0.5,
        minimum_profit_factor: float = 1.0,
        minimum_trade_count: int = 30,
        cost_stress_required_positive: bool = True,
        monte_carlo_tester: MonteCarloTester | None = None,
    ):
        self.minimum_positive_window_ratio = minimum_positive_window_ratio
        self.minimum_profit_factor = minimum_profit_factor
        self.minimum_trade_count = minimum_trade_count
        self.cost_stress_required_positive = cost_stress_required_positive
        self.comparator = ModelComparator()
        self.monte_carlo_tester = monte_carlo_tester or MonteCarloTester()

    def evaluate(
        self,
        candidate: Candidate,
        walk_forward: WalkForwardResult | None,
        champion_metrics: dict[str, float] | None = None,
        challenger_metrics: dict[str, float] | None = None,
        cost_stress_net_return: float = 0.0,
        candidate_trade_pnls: list[float] | None = None,
    ) -> EvaluationResult:
        reasons: list[str] = []

        # 1. Walk-Forward Stability Check
        if walk_forward is None:
            reasons.append("missing_walk_forward")
        else:
            if walk_forward.positive_window_ratio < self.minimum_positive_window_ratio:
                reasons.append("unstable_walk_forward")
            if walk_forward.oos_return_mean <= 0:
                reasons.append("negative_oos_expectancy")
            if walk_forward.n_windows < 3:
                reasons.append("insufficient_windows")

        # 2. Cost Stress Test Gate (1.5x Fee / 2.0x Slippage)
        if self.cost_stress_required_positive and cost_stress_net_return <= 0:
            reasons.append("cost_stress_failed")

        # 3. Challenger Absolute Thresholds
        if challenger_metrics:
            pf = challenger_metrics.get("profit_factor", 0.0)
            if pf < self.minimum_profit_factor:
                reasons.append("profit_factor_below_minimum")
            trades = challenger_metrics.get("total_trades", 0)
            if trades < self.minimum_trade_count:
                reasons.append("insufficient_trade_count")

        # 4. Monte Carlo Sequence & Permutation Test
        mc_result: MonteCarloResult | None = None
        if candidate_trade_pnls and len(candidate_trade_pnls) >= 10:
            mc_result = self.monte_carlo_tester.test(candidate_trade_pnls)
            if not mc_result.passed:
                reasons.extend(mc_result.rejection_reasons)

        passed = len(reasons) == 0

        # Audit dict explicitly includes candidate config so promotion can install it (Bug #7 fix)
        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            passed=passed,
            rejection_reasons=reasons,
            walk_forward=walk_forward,
            cost_stress_net_return=cost_stress_net_return,
            monte_carlo=mc_result,
            champion_comparison={"champion": champion_metrics, "challenger": challenger_metrics},
            audit={
                "hypothesis": candidate.hypothesis,
                "parent": candidate.parent_version,
                "config": candidate.config,
                "version": candidate.version,
                "candidate_id": candidate.candidate_id,
            },
        )
