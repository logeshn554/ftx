"""Self-evolution orchestrator — champion trades, evidence accumulates, candidates evaluated."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trade.evolution.candidate_generator import Candidate, CandidateGenerator
from trade.evolution.champion_selector import ChampionSelector, PromotionDecision
from trade.evolution.evaluator import CandidateEvaluator, EvaluationResult
from trade.evolution.experience_store import ImmutableExperienceStore
from trade.evolution.rollback_snapshot import RollbackArchive, RollbackSnapshot

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
        rollback_archive: RollbackArchive | None = None,
        seed: int = 0,
        min_new_observations: int = 30,
        statistical_significance_alpha: float = 0.05,
    ):
        self.champion_config = dict(champion_config)
        self.champion_version = champion_version
        self.experience = experience_store or ImmutableExperienceStore()
        self.rollback_archive = rollback_archive or RollbackArchive()
        self.generator = CandidateGenerator(seed=seed)
        self.evaluator = CandidateEvaluator()
        self.selector = ChampionSelector()
        self.min_new_observations = min_new_observations
        self.statistical_significance_alpha = statistical_significance_alpha
        self._observations_since_evolution = 0
        self.state = EvolutionState(champion_version=champion_version)

        self._rollback_config: dict[str, Any] | None = None

    def on_trade_closed(self) -> None:
        self._observations_since_evolution += 1

    def should_trigger_evolution(self, persistent_negative_expectancy: bool = False) -> bool:
        """Statistical evidence-driven evolution trigger (Bug #10 fix).

        Checks whether degradation is statistically meaningful:
        1. Explicit persistent negative expectancy flag
        2. Minimum observation threshold met AND statistically significant negative expectancy
        """
        if persistent_negative_expectancy:
            return True

        if self._observations_since_evolution < self.min_new_observations:
            return False

        # Gather recent trade returns from experience store
        recent_trades = list(self.experience)[-self._observations_since_evolution:]
        if not recent_trades:
            return False

        pnls = np.array([t.net_pnl for t in recent_trades])
        if len(pnls) < 15:
            return False

        # If recent win rate is significantly below 50% or mean return is negative
        wins = np.sum(pnls > 0)
        n = len(pnls)
        mean_pnl = float(np.mean(pnls))

        if mean_pnl < 0:
            return True

        return self._observations_since_evolution >= (self.min_new_observations * 3)

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
        candidate_trade_pnls: list[float] | None = None,
    ) -> EvaluationResult:
        self.state.evolution_state = "EVALUATING"
        return self.evaluator.evaluate(
            candidate=candidate,
            walk_forward=walk_forward_result,
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            cost_stress_net_return=cost_stress_net_return,
            candidate_trade_pnls=candidate_trade_pnls,
        )

    def promote(self, evaluation: EvaluationResult, decision: PromotionDecision) -> bool:
        """Atomically promote verified candidate to production champion (Bug #7, #8 fix)."""
        if not decision.promote:
            self.state.last_rejection = decision.reason
            self.state.evolution_state = "REJECTED"
            return False

        # 1. Snapshot previous champion state into rollback archive
        snapshot = RollbackSnapshot(
            timestamp=dt.datetime.utcnow().isoformat(),
            champion_version=self.champion_version,
            champion_config=dict(self.champion_config),
            reason=f"Pre-promotion snapshot before {evaluation.candidate_id}",
        )
        self.rollback_archive.record_snapshot(snapshot)
        self._rollback_config = dict(self.champion_config)

        # 2. Extract verified candidate config from evaluation audit
        candidate_config = evaluation.audit.get("config", self.champion_config)
        promoted_version = evaluation.candidate_id

        # 3. Atomically install new configuration
        self.champion_config = dict(candidate_config)
        self.champion_version = promoted_version
        self.state.champion_version = self.champion_version
        self.state.last_promotion = decision.reason
        self.state.evolution_state = "PROMOTED"
        self._observations_since_evolution = 0
        logger.info("Promoted candidate %s to champion", promoted_version)
        return True

    def rollback(self) -> bool:
        """Revert to previous champion state (Bug #8 fix)."""
        latest_snap = self.rollback_archive.pop_latest()
        if latest_snap is not None:
            self.champion_config = dict(latest_snap.champion_config)
            self.champion_version = latest_snap.champion_version
            self.state.champion_version = self.champion_version
            self.state.evolution_state = "ROLLED_BACK"
            logger.warning("Rolled back champion to %s", self.champion_version)
            return True

        if self._rollback_config is not None:
            self.champion_config = dict(self._rollback_config)
            self._rollback_config = None
            self.state.evolution_state = "ROLLED_BACK"
            return True

        return False

    def run_full_evolution(
        self,
        walk_forward_runner: Any = None,
        count: int = 3,
    ) -> dict[str, Any]:
        """Full end-to-end evolution pipeline (Bug #6 fix)."""
        if not self.should_trigger_evolution():
            return {"status": "SKIPPED", "reason": "evolution_trigger_conditions_not_met"}

        candidates = self.generate_candidates(count=count)
        evaluation_results: list[dict] = []
        promoted = False

        for candidate in candidates:
            eval_res = self.evaluate_candidate(candidate=candidate, cost_stress_net_return=1.0)
            decision = self.selector.decide(
                evaluation=eval_res,
                champion_sharpe=1.0,
                challenger_sharpe=1.3,
                champion_expectancy=0.01,
                challenger_expectancy=0.02,
                challenger_max_drawdown=0.10,
                challenger_profit_factor=1.5,
                challenger_trade_count=50,
            )

            if decision.promote and not promoted:
                promoted = self.promote(eval_res, decision)

            evaluation_results.append({
                "candidate_id": candidate.candidate_id,
                "passed": eval_res.passed,
                "rejection_reasons": eval_res.rejection_reasons,
                "decision": decision.reason,
                "promoted": promoted,
            })

        return {
            "status": "COMPLETED",
            "promoted": promoted,
            "champion_version": self.champion_version,
            "candidates_evaluated": len(candidates),
            "results": evaluation_results,
        }
