"""Unit tests for ResearchProtocolEngine."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from trade.data.features import FeatureEngine
from trade.validation.protocol import ResearchProtocolEngine


def test_protocol_engine_dataset_hash(sample_ohlcv):
    engine = FeatureEngine(feature_window=10)
    features_df = engine.compute_features(sample_ohlcv)

    protocol = ResearchProtocolEngine()
    hash1 = protocol.compute_dataset_hash(features_df)
    hash2 = protocol.compute_dataset_hash(features_df)
    assert hash1 == hash2
    assert len(hash1) == 16


def test_protocol_engine_sequential_evaluation(sample_ohlcv):
    engine = FeatureEngine(feature_window=10)
    features_df = engine.compute_features(sample_ohlcv)
    feature_cols = engine.get_feature_columns()

    protocol = ResearchProtocolEngine(
        min_oos_sharpe=0.1,
        min_deflated_sharpe_prob=0.1,
    )

    mock_model = MagicMock()
    mock_model.predict.return_value = (0, None)

    with patch("trade.validation.backtester.PPO.load", return_value=mock_model):
        report = protocol.evaluate(
            model_path="dummy.zip",
            features_df=features_df,
            feature_columns=feature_cols,
            num_prior_trials=5,
        )

        assert report.dataset_hash is not None
        assert "untouched_oos_test" in report.stages
        assert "walk_forward" in report.stages
        assert "cost_stress" in report.stages
        assert "deflated_sharpe" in report.stages
        assert report.audit_trail["num_prior_trials"] == 5
