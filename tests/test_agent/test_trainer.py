"""Tests for PPO AgentTrainer and policy features."""

import numpy as np
import pytest
from stable_baselines3 import PPO

from trade.agent.trainer import AgentTrainer, MetricsLoggerCallback
from trade.core.config import AppConfig, build_config
from trade.core.types import ModelVersion
from trade.data.features import FeaturePipeline
from trade.env.trading_env import TradingEnv


@pytest.fixture
def env_and_features(sample_ohlcv, config):
    """Fixture providing configured FeaturePipeline and TradingEnv."""
    pipeline = FeaturePipeline(config=config.data)
    feature_matrix = pipeline.compute_features(sample_ohlcv)
    env = TradingEnv(
        df=sample_ohlcv,
        features=feature_matrix,
        config=config,
    )
    return env


class TestMetricsLoggerCallback:
    """Test callback statistics calculation."""

    def test_metrics_empty(self):
        cb = MetricsLoggerCallback()
        assert cb.get_summary() == {}

    def test_metrics_with_episodes(self):
        cb = MetricsLoggerCallback()
        cb._episode_rewards = [10.0, 20.0, 30.0]
        cb._episode_lengths = [100, 100, 100]

        summary = cb.get_summary()
        assert summary["total_episodes"] == 3
        assert summary["mean_reward"] == 20.0
        assert summary["max_reward"] == 30.0
        assert summary["min_reward"] == 10.0


class TestAgentTrainer:
    """Test AgentTrainer workflow."""

    def test_create_model(self, env_and_features, config):
        trainer = AgentTrainer(config=config)
        model = trainer.create_model(env_and_features)

        assert isinstance(model, PPO)
        assert trainer.model is not None

    def test_train_short_run(self, env_and_features, config, tmp_path):
        trainer = AgentTrainer(config=config)
        version = ModelVersion(major=1, minor=0, patch=0)

        result = trainer.train(
            train_env=env_and_features,
            model_version=version,
            checkpoint_dir=str(tmp_path),
        )

        assert result.model_version.tag == "v1.0.0"
        assert result.total_timesteps == config.training.total_timesteps
        assert result.training_time_seconds >= 0.0
        assert result.checkpoint_path is not None

    def test_load_model(self, env_and_features, config, tmp_path):
        trainer = AgentTrainer(config=config)
        version = ModelVersion(major=1, minor=0, patch=0)
        result = trainer.train(
            train_env=env_and_features,
            model_version=version,
            checkpoint_dir=str(tmp_path),
        )

        # Load back
        loaded = trainer.load_model(result.checkpoint_path, env=env_and_features)
        assert isinstance(loaded, PPO)
