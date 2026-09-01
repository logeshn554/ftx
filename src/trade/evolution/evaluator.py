"""Candidate evaluation pipeline with structured results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trade.evolution.candidate_generator import Candidate
from trade.validation.comparator import ModelComparator
from trade.validation.walk_forward import WalkForwardResult


@dataclass
class EvaluationResult:
    candidate_id: str
    passed: bool
    rejection_reasons: list[str] = field(default_factory=list)
    walk_forward: WalkForwardResult | None = None
    cost_stress_net_return: float = 0.0
    champion_comparison: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)


class CandidateEvaluator:
    """candidate → train → walk-forward → stress → champion comparison"""

    def __init__(
        self,
        minimum_positive_window_ratio: float = 0.5,
        minimum_profit_factor: float = 1.0,
        minimum_trade_count: int = 30,
        cost_stress_required_positive: bool = True,
    ):
        self.minimum_positive_window_ratio = minimum_positive_window_ratio
        self.minimum_profit_factor = minimum_profit_factor
        self.minimum_trade_count = minimum_trade_count
        self.cost_stress_required_positive = cost_stress_required_positive
        self.comparator = ModelComparator()

    def evaluate(
        self,
        candidate: Candidate,
        walk_forward: WalkForwardResult | None,
        champion_metrics: dict[str, float] | None = None,
        challenger_metrics: dict[str, float] | None = None,
        cost_stress_net_return: float = 0.0,
    ) -> EvaluationResult:
        reasons: list[str] = []
        if walk_forward is None:
            reasons.append("missing_walk_forward")
        else:
            if walk_forward.positive_window_ratio < self.minimum_positive_window_ratio:
                reasons.append("unstable_walk_forward")
            if walk_forward.oos_return_mean <= 0:
                reasons.append("negative_oos_expectancy")
            if walk_forward.n_windows < 3:
                reasons.append("insufficient_windows")

        if self.cost_stress_required_positive and cost_stress_net_return <= 0:
            reasons.append("cost_stress_failed")

        if challenger_metrics:
            pf = challenger_metrics.get("profit_factor", 0.0)
            if pf < self.minimum_profit_factor:
                reasons.append("profit_factor_below_minimum")
            trades = challenger_metrics.get("total_trades", 0)
            if trades < self.minimum_trade_count:
                reasons.append("insufficient_trade_count")

        passed = len(reasons) == 0
        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            passed=passed,
            rejection_reasons=reasons,
            walk_forward=walk_forward,
            cost_stress_net_return=cost_stress_net_return,
            champion_comparison={"champion": champion_metrics, "challenger": challenger_metrics},
            audit={"hypothesis": candidate.hypothesis, "parent": candidate.parent_version},
        )
