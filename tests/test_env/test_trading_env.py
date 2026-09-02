"""Tests for the Gymnasium trading environment and action transitions."""

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

    def test_flat_hold_stays_flat(self, sample_ohlcv):
        """Flat + HOLD stays flat with 0 fees and 0 turnover."""
        env = self._make_env(sample_ohlcv)
        env.reset()

        obs, reward, term, trunc, info = env.step(0)  # HOLD
        assert info["position"] == 0.0
        assert info["cash"] == 100_000.0
        assert info["fees"] == 0.0
        assert info["turnover"] == 0.0

    def test_long_hold_stays_long(self, sample_ohlcv):
        """Long + HOLD maintains long position without additional fees or turnover."""
        env = self._make_env(sample_ohlcv)
        env.reset()

        # Step 1: Open Long
        _, _, _, _, info1 = env.step(1)
        initial_pos = info1["position"]
        initial_fees = info1["fees"]
        initial_turnover = info1["turnover"]
        assert initial_pos > 0

        # Step 2: HOLD
        _, _, _, _, info2 = env.step(0)
        assert info2["position"] == initial_pos
        assert info2["fees"] == initial_fees
        assert info2["turnover"] == initial_turnover

    def test_reversal_short_to_long_same_step(self, sample_ohlcv):
        """Reversal: SHORT -> BUY closes short and opens long on the exact same step."""
        env = self._make_env(sample_ohlcv)
        env.reset()

        # Step 1: SELL -> Open Short
        _, _, _, _, info_short = env.step(2)
        assert info_short["position"] < 0
        assert info_short["position_side"] == "SELL"

        # Step 2: BUY -> Reverses short to long
        _, _, _, _, info_long = env.step(1)
        assert info_long["position"] > 0
        assert info_long["position_side"] == "BUY"

        # Trade log should have entry for SELL, close of SELL, and entry for BUY
        assert len(env.trade_log) == 3

    def test_reversal_long_to_short_same_step(self, sample_ohlcv):
        """Reversal: LONG -> SELL closes long and opens short on the exact same step."""
        env = self._make_env(sample_ohlcv)
        env.reset()

        # Step 1: BUY -> Open Long
        _, _, _, _, info_long = env.step(1)
        assert info_long["position"] > 0
        assert info_long["position_side"] == "BUY"

        # Step 2: SELL -> Reverses long to short
        _, _, _, _, info_short = env.step(2)
        assert info_short["position"] < 0
        assert info_short["position_side"] == "SELL"

        assert len(env.trade_log) == 3

    def test_force_flat_outside_policy_actions(self, sample_ohlcv):
        """force_flat closes open position to cash with explicit audit reason."""
        env = self._make_env(sample_ohlcv)
        env.reset()

        env.step(1)  # BUY
        assert env._accounting.position > 0

        closed = env.force_flat(reason="RISK_KILL_SWITCH")
        assert closed is not None
        assert env._accounting.position == 0.0
        assert env.trade_log[-1]["action"] == "RISK_KILL_SWITCH"

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
