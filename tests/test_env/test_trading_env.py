"""Tests for the trading environment."""

import numpy as np
import pytest

from trade.data.features import FeatureEngine
from trade.env.trading_env import TradingEnv


class TestTradingEnv:
    """Test the Gymnasium trading environment."""

    def _make_env(self, sample_ohlcv):
        """Create a trading env from sample data."""
        engine = FeatureEngine(feature_window=10)
        features_df = engine.compute_features(sample_ohlcv)
        feature_cols = engine.get_feature_columns()

        return TradingEnv(
            features_df=features_df,
            feature_columns=feature_cols,
            initial_capital=100_000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
            feature_window=10,
            reward_function="pnl",
        )

    def test_reset(self, sample_ohlcv):
        """Environment resets correctly."""
        env = self._make_env(sample_ohlcv)
        obs, info = env.reset()

        assert obs.shape[0] == 10  # feature window
        assert obs.shape[1] > 0  # features + portfolio
        assert info["cash"] == 100_000.0
        assert info["position"] == 0.0
        assert info["portfolio_value"] == 100_000.0

    def test_hold_action(self, sample_ohlcv):
        """HOLD action doesn't change portfolio."""
        env = self._make_env(sample_ohlcv)
        env.reset()

        obs, reward, term, trunc, info = env.step(0)  # HOLD
        assert info["position"] == 0.0
        assert info["cash"] == 100_000.0

    def test_buy_then_sell(self, sample_ohlcv):
        """BUY followed by SELL creates a trade."""
        env = self._make_env(sample_ohlcv)
        env.reset()

        # BUY
        obs, reward, term, trunc, info = env.step(1)
        assert info["position"] > 0
        assert info["cash"] < 100_000.0

        # SELL
        obs, reward, term, trunc, info = env.step(2)
        assert info["position"] == 0.0

        # Should have some trade log entries
        assert len(env.trade_log) == 2

    def test_observation_shape_consistent(self, sample_ohlcv):
        """Observation shape remains consistent across steps."""
        env = self._make_env(sample_ohlcv)
        obs, _ = env.reset()
        expected_shape = obs.shape

        for _ in range(10):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
            assert obs.shape == expected_shape

    def test_episode_terminates(self, sample_ohlcv):
        """Episode eventually terminates."""
        env = self._make_env(sample_ohlcv)
        env.reset()

        done = False
        steps = 0
        while not done and steps < 1000:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            steps += 1

        assert done, "Episode should terminate"

    def test_no_nan_in_observations(self, sample_ohlcv):
        """Observations should never contain NaN."""
        env = self._make_env(sample_ohlcv)
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs))

        for _ in range(20):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
            assert not np.any(np.isnan(obs)), "NaN found in observation"
