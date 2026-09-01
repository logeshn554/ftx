"""Self-evolution orchestrator — champion trades, evidence accumulates, candidates evaluated."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from trade.evolution.candidate_generator import Candidate, CandidateGenerator
from trade.evolution.champion_selector import ChampionSelector, PromotionDecision
from trade.evolution.evaluator import CandidateEvaluator, EvaluationResult
from trade.evolution.experience_store import ImmutableExperienceStore

logger = logging.getLogger(__name__)


@dataclass
class EvolutionState:
    champion_version: str
    candidate_version: str | None = None
    evolution_state: str = "MONITORING"
    last_promotion: str | None = None
    last_rejection: str | None = None
    pending_candidates: list[Candidate] = field(default_factory=list)


class EvolutionOrchestrator:
    """Complete evolution loop without direct champion mutation from trade outcomes."""

    def __init__(
        self,
        champion_config: dict[str, Any],
        champion_version: str = "v1.0.0",
        experience_store: ImmutableExperienceStore | None = None,
        seed: int = 0,
        min_new_observations: int = 100,
    ):
        self.champion_config = champion_config
        self.champion_version = champion_version
        self.experience = experience_store or ImmutableExperienceStore()
        self.generator = CandidateGenerator(seed=seed)
        self.evaluator = CandidateEvaluator()
        self.selector = ChampionSelector()
        self.min_new_observations = min_new_observations
        self._observations_since_evolution = 0
        self.state = EvolutionState(champion_version=champion_version)
        self._rollback_config: dict[str, Any] | None = None

    def on_trade_closed(self) -> None:
        self._observations_since_evolution += 1

    def should_trigger_evolution(self, persistent_negative_expectancy: bool = False) -> bool:
        if persistent_negative_expectancy:
            return True
        return self._observations_since_evolution >= self.min_new_observations

    def generate_candidates(self, count: int = 3) -> list[Candidate]:
        stats = self.experience.aggregate_by_strategy_regime()
        hypothesis = "statistical_review"
        for strategy, regimes in stats.items():
            for regime, bucket in regimes.items():
                if bucket["sample_count"] >= 30 and bucket.get("expectancy", 0) < 0:
                    hypothesis = f"negative_expectancy_{strategy}_{regime}"
                    break
        candidates = self.generator.generate(self.champion_config, self.champion_version, count)
        self.state.pending_candidates = candidates
        self.state.evolution_state = "CANDIDATE_GENERATION"
        logger.info("Generated %d candidates; hypothesis=%s", len(candidates), hypothesis)
        return candidates

    def evaluate_candidate(
        self,
        candidate: Candidate,
        walk_forward_result: Any = None,
        champion_metrics: dict[str, float] | None = None,
        challenger_metrics: dict[str, float] | None = None,
        cost_stress_net_return: float = 0.0,
    ) -> EvaluationResult:
        self.state.evolution_state = "EVALUATING"
        return self.evaluator.evaluate(
            candidate, walk_forward_result, champion_metrics, challenger_metrics, cost_stress_net_return
        )

    def promote(self, evaluation: EvaluationResult, decision: PromotionDecision) -> bool:
        if not decision.promote:
            self.state.last_rejection = decision.reason
            self.state.evolution_state = "REJECTED"
            return False
        self._rollback_config = dict(self.champion_config)
        self.champion_config = evaluation.audit.get("config", self.champion_config)
        self.champion_version = evaluation.candidate_id
        self.state.champion_version = self.champion_version
        self.state.last_promotion = decision.reason
        self.state.evolution_state = "PROMOTED"
        self._observations_since_evolution = 0
        return True

    def rollback(self) -> bool:
        if self._rollback_config is None:
            return False
        self.champion_config = self._rollback_config
        self.state.evolution_state = "ROLLED_BACK"
        return True
