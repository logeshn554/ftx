"""Tests for Backtester execution and metric reconciliation."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from trade.core.types import ModelVersion
from trade.data.features import FeatureEngine
from trade.validation.backtester import Backtester


def test_backtester_trade_reconciliation(sample_ohlcv):
    """Verify that Backtester reconciles trade logs without cost double counting."""
    engine = FeatureEngine(feature_window=10)
    features_df = engine.compute_features(sample_ohlcv)
    feature_cols = engine.get_feature_columns()

    backtester = Backtester(
        initial_capital=100_000.0,
        commission_pct=0.001,
        slippage_pct=0.0005,
        feature_window=10,
    )

    # Mock PPO model that alternates BUY and HOLD
    mock_model = MagicMock()
    mock_model.predict.side_effect = [(1, None), (0, None), (2, None)] + [(0, None)] * 500

    with patch("trade.validation.backtester.PPO.load", return_value=mock_model):
        result = backtester.run(
            model_path="dummy_path.zip",
            features_df=features_df,
            feature_columns=feature_cols,
            model_version=ModelVersion(major=1, minor=0, patch=0),
        )

        assert result.initial_capital == 100_000.0
        assert result.total_trades >= 1
        assert len(result.equity_curve) > 1
        assert not result.daily_returns or len(result.daily_returns) == len(result.equity_curve) - 1

        # Check that friction is properly recorded from closed trades
        for trade in result.trade_log:
            if "net_pnl" in trade:
                expected_net = trade["gross_pnl"] - trade["entry_fee"] - trade["exit_fee"]
                assert abs(trade["net_pnl"] - expected_net) < 1e-6
