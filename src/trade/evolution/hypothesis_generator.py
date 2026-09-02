"""Hypothesis-Driven Candidate Generation from empirical trading experience."""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from typing import Any

from trade.evolution.experience_store import ImmutableExperienceStore


@dataclass(frozen=True)
class HypothesisCandidate:
    candidate_id: str
    parent_version: str
    hypothesis_type: str
    premise: str
    target_component: str
    proposed_mutation: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)


class HypothesisGenerator:
    """Generates structured, verifiable market hypotheses from experience logs."""

    def __init__(self, min_sample_evidence: int = 10):
        self.min_sample_evidence = min_sample_evidence

    def generate_hypotheses(
        self,
        experience_store: ImmutableExperienceStore,
        current_config: dict[str, Any],
        current_version: str = "v1.0.0",
    ) -> list[HypothesisCandidate]:
        """Analyze empirical experience and formulate targeted scientific candidate hypotheses."""
        regime_stats = experience_store.aggregate_by_strategy_regime()
        hypotheses: list[HypothesisCandidate] = []

        # 1. Hypothesis: Regime-Specific Negative Expectancy
        for strategy, regimes in regime_stats.items():
            for regime, bucket in regimes.items():
                n = int(bucket.get("sample_count", 0))
                expectancy = float(bucket.get("expectancy", 0.0))
                drawdown = float(bucket.get("drawdown", 0.0))

                if n >= self.min_sample_evidence and expectancy < 0:
                    mutation = {
                        f"strategy_thresholds_{strategy}_{regime.lower()}_min_confidence": 0.70,
                        "cost_safety_multiplier": 1.5,
                    }
                    new_cfg = dict(current_config)
                    new_cfg.update(mutation)
                    cid = f"hyp_regime_{strategy}_{regime}_{uuid.uuid4().hex[:6]}"
                    hypotheses.append(
                        HypothesisCandidate(
                            candidate_id=cid,
                            parent_version=current_version,
                            hypothesis_type="REGIME_MISMATCH",
                            premise=f"Strategy '{strategy}' exhibits negative expectancy ({expectancy:.4f}) in '{regime}' regime over {n} trades.",
                            target_component="strategy_selector",
                            proposed_mutation=mutation,
                            config=new_cfg,
                        )
                    )

        # 2. Hypothesis: Friction / Cost Drag
        trades = list(experience_store)
        if trades:
            total_fees = sum(t.fees for t in trades)
            total_gross = sum(t.gross_pnl for t in trades)
            if total_gross > 0 and (total_fees / total_gross) > 0.40:
                mutation = {
                    "minimum_edge": 0.003,
                    "cost_safety_multiplier": 1.8,
                }
                new_cfg = dict(current_config)
                new_cfg.update(mutation)
                cid = f"hyp_cost_drag_{uuid.uuid4().hex[:6]}"
                hypotheses.append(
                    HypothesisCandidate(
                        candidate_id=cid,
                        parent_version=current_version,
                        hypothesis_type="COST_DRAG",
                        premise=f"Transaction friction consumes {total_fees/total_gross:.1%} of gross profits; requires higher edge hurdle.",
                        target_component="expected_value_filter",
                        proposed_mutation=mutation,
                        config=new_cfg,
                    )
                )

        # Fallback default hypothesis if data is limited
        if not hypotheses:
            cid = f"hyp_exploration_{uuid.uuid4().hex[:6]}"
            mutation = {"reward_turnover_penalty": current_config.get("reward_turnover_penalty", 0.05) * 1.1}
            new_cfg = dict(current_config)
            new_cfg.update(mutation)
            hypotheses.append(
                HypothesisCandidate(
                    candidate_id=cid,
                    parent_version=current_version,
                    hypothesis_type="EXPLORATION",
                    premise="Insufficient failure evidence; exploring defensive turnover penalty scaling.",
                    target_component="environment_reward",
                    proposed_mutation=mutation,
                    config=new_cfg,
                )
            )

        return hypotheses
