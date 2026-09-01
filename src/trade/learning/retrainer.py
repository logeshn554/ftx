"""Candidate retrainer: trains new model versions when triggered.

Runs on the Learning Agent side — separate from the Production Agent.
Produces candidate models that must pass through the Validation Agent
before they can be promoted to production.
"""

from __future__ import annotations

import datetime as dt
import logging

from trade.core.config import AppConfig
from trade.core.events import RetrainingCompleted, RetrainingStarted, event_bus
from trade.core.types import ModelStage, ModelVersion, TrainingResult

logger = logging.getLogger(__name__)


class CandidateRetrainer:
    """Trains candidate models using accumulated experience.

    Flow:
        1. Triggered by PerformanceDegraded event or scheduled
        2. Pulls latest data and experience
        3. Creates new training environment
        4. Trains candidate model with PPO
        5. Registers candidate in model registry
        6. Hands off to Gatekeeper for validation
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._is_training = False
        self._last_training: dt.datetime | None = None
        self._training_count = 0

    def retrain(
        self,
        current_version: ModelVersion,
        trigger: str = "manual",
    ) -> TrainingResult | None:
        """Train a new candidate model.

        Args:
            current_version: The current production model version.
            trigger: What triggered this retraining (e.g., "performance_degraded", "scheduled").

        Returns:
            TrainingResult if successful, None if training was skipped.
        """
        if self._is_training:
            logger.warning("Retraining already in progress, skipping")
            return None

        # Check cooldown
        if self._last_training:
            hours_since = (dt.datetime.utcnow() - self._last_training).total_seconds() / 3600
            if hours_since < self.config.learning.retrain_cooldown_hours:
                logger.info(
                    "Retraining cooldown: %.1fh since last training (min: %dh)",
                    hours_since,
                    self.config.learning.retrain_cooldown_hours,
                )
                return None

        self._is_training = True
        self._training_count += 1

        # Create new version
        new_version = ModelVersion(
            major=current_version.major,
            minor=current_version.minor + 1,
            patch=0,
            stage=ModelStage.CANDIDATE,
        )

        logger.info(
            "🔄 Starting candidate retraining: %s → %s (trigger: %s)",
            current_version.tag,
            new_version.tag,
            trigger,
        )

        event_bus.publish_sync(
            RetrainingStarted(trigger=trigger, model_version=new_version.tag)
        )

        try:
            # Import here to avoid circular deps
            from trade.agent.trainer import AgentTrainer
            from trade.data.features import FeatureEngine
            from trade.data.sources.yahoo import YahooDataSource
            from trade.data.validation import DataValidator
            from trade.env.trading_env import TradingEnv

            # 1. Fetch latest data
            source = YahooDataSource(cache_dir=self.config.data.cache_dir)
            feature_engine = FeatureEngine(feature_window=self.config.data.feature_window)
            validator = DataValidator(
                max_price_jump_pct=self.config.data.max_price_jump_pct,
                min_volume=self.config.data.min_volume,
            )

            # Use first symbol for single-asset training
            symbol = self.config.data.symbols[0]
            end_date = dt.date.today()
            start_date = end_date - dt.timedelta(days=self.config.data.lookback_days)

            df = source.fetch_ohlcv(symbol, start_date, end_date, self.config.data.timeframe)
            vr = validator.validate(df, symbol)

            if not vr.passed:
                logger.error("Data validation failed for retraining, aborting")
                return None

            df = validator.clean(df)
            features_df = feature_engine.compute_features(df)
            feature_cols = feature_engine.get_feature_columns()

            # 2. Split into train/eval
            split_idx = int(len(features_df) * 0.8)
            train_df = features_df.iloc[:split_idx].copy()
            eval_df = features_df.iloc[split_idx:].copy()

            # 3. Create environments
            train_env = TradingEnv(
                features_df=train_df,
                feature_columns=feature_cols,
                initial_capital=self.config.trading.initial_capital,
                commission_pct=self.config.trading.commission_pct,
                slippage_pct=self.config.trading.slippage_pct,
                feature_window=self.config.data.feature_window,
                reward_function=self.config.training.reward_function,
            )

            eval_env = TradingEnv(
                features_df=eval_df,
                feature_columns=feature_cols,
                initial_capital=self.config.trading.initial_capital,
                commission_pct=self.config.trading.commission_pct,
                slippage_pct=self.config.trading.slippage_pct,
                feature_window=self.config.data.feature_window,
                reward_function=self.config.training.reward_function,
            )

            # 4. Train
            trainer = AgentTrainer(self.config)
            result = trainer.train(
                train_env=train_env,
                eval_env=eval_env,
                model_version=new_version,
                checkpoint_dir=f"checkpoints/{new_version.tag}",
            )

            self._last_training = dt.datetime.utcnow()

            event_bus.publish_sync(
                RetrainingCompleted(
                    model_version=new_version.tag,
                    metrics=result.final_metrics,
                )
            )

            logger.info(
                "✅ Candidate %s trained successfully in %.0fs",
                new_version.tag,
                result.training_time_seconds,
            )

            return result

        except Exception:
            logger.exception("Retraining failed for %s", new_version.tag)
            return None

        finally:
            self._is_training = False

    @property
    def is_training(self) -> bool:
        return self._is_training

    @property
    def training_count(self) -> int:
        return self._training_count
