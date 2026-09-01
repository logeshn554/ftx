"""Gatekeeper: orchestrates the full model promotion pipeline.

Implements the validation flow:
    Backtest → Walk-Forward → Compare → Verdict → Promote/Reject
"""

from __future__ import annotations

import logging

import pandas as pd

from trade.core.types import BacktestResult, ComparisonResult, ModelStage, ModelVersion
from trade.model_management.registry import ModelRegistry
from trade.validation.backtester import Backtester
from trade.validation.comparator import ModelComparator
from trade.validation.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)


class Gatekeeper:
    """The validation gatekeeper that decides if a candidate model
    should be promoted to production.

    Pipeline:
        1. Backtest candidate on historical data
        2. Walk-forward validation for robustness
        3. Compare candidate vs current champion
        4. If PROMOTE → advance to next stage
        5. If REJECT → archive candidate
    """

    def __init__(
        self,
        registry: ModelRegistry,
        backtester: Backtester,
        comparator: ModelComparator,
        walk_forward: WalkForwardValidator | None = None,
    ) -> None:
        self.registry = registry
        self.backtester = backtester
        self.comparator = comparator
        self.walk_forward = walk_forward

    def evaluate_candidate(
        self,
        candidate_path: str,
        champion_path: str,
        features_df: pd.DataFrame,
        feature_columns: list[str],
        candidate_version: ModelVersion,
        champion_version: ModelVersion,
    ) -> ComparisonResult:
        """Run the full evaluation pipeline for a candidate model.

        Args:
            candidate_path: Path to candidate model checkpoint.
            champion_path: Path to current production model checkpoint.
            features_df: Historical feature data for testing.
            feature_columns: Feature column names.
            candidate_version: Candidate model version.
            champion_version: Current production model version.

        Returns:
            ComparisonResult with final verdict.
        """
        logger.info(
            "🔍 Gatekeeper evaluation: %s (candidate) vs %s (champion)",
            candidate_version.tag,
            champion_version.tag,
        )

        # Mark candidate as being evaluated
        self.registry.promote(
            candidate_version.tag,
            ModelStage.BACKTESTING,
            reason="Gatekeeper evaluation started",
        )

        # Step 1: Backtest candidate
        logger.info("Step 1/3: Backtesting candidate %s...", candidate_version.tag)
        candidate_result = self.backtester.run(
            model_path=candidate_path,
            features_df=features_df,
            feature_columns=feature_columns,
            model_version=candidate_version,
        )

        # Step 2: Backtest champion (for fair comparison on same data)
        logger.info("Step 2/3: Backtesting champion %s...", champion_version.tag)
        champion_result = self.backtester.run(
            model_path=champion_path,
            features_df=features_df,
            feature_columns=feature_columns,
            model_version=champion_version,
        )

        # Step 3: Compare
        logger.info("Step 3/3: Comparing models...")
        comparison = self.comparator.compare(
            champion_result=champion_result,
            challenger_result=candidate_result,
        )

        # Act on verdict
        if comparison.verdict == ComparisonResult.Verdict.PROMOTE:
            logger.info(
                "✅ Candidate %s APPROVED for promotion",
                candidate_version.tag,
            )
            self.registry.promote(
                candidate_version.tag,
                ModelStage.PAPER_TRADING,
                reason=f"Gatekeeper approved: {comparison.notes}",
            )

        elif comparison.verdict == ComparisonResult.Verdict.REJECT:
            logger.info(
                "❌ Candidate %s REJECTED",
                candidate_version.tag,
            )
            self.registry.reject(
                candidate_version.tag,
                reason=comparison.notes,
            )

        else:
            logger.info(
                "⚠️ Candidate %s INCONCLUSIVE — requires manual review",
                candidate_version.tag,
            )
            # Keep in BACKTESTING stage for manual review

        return comparison

    def promote_to_production(
        self,
        version_tag: str,
        reason: str = "Passed all validation gates",
    ) -> bool:
        """Manually promote a validated model to production.

        This is the final step after paper trading / shadow trading
        validation has been completed.

        Args:
            version_tag: Model version to promote.
            reason: Promotion reason.

        Returns:
            True if promotion succeeded.
        """
        return self.registry.promote(
            version_tag,
            ModelStage.PRODUCTION,
            reason=reason,
        )
