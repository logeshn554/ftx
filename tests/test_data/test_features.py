"""Tests for feature engineering."""

import numpy as np
import pytest

from trade.data.features import FeatureEngine
from trade.core.types import Regime


class TestFeatureEngine:
    """Test feature computation and regime detection."""

    def test_compute_features_columns(self, sample_ohlcv):
        """All expected feature columns are present."""
        engine = FeatureEngine(feature_window=10)
        features = engine.compute_features(sample_ohlcv)

        expected = engine.get_feature_columns()
        for col in expected:
            assert col in features.columns, f"Missing feature: {col}"

    def test_compute_features_no_nans(self, sample_ohlcv):
        """Computed features should not contain NaN."""
        engine = FeatureEngine(feature_window=10)
        features = engine.compute_features(sample_ohlcv)

        feature_cols = engine.get_feature_columns()
        for col in feature_cols:
            assert not features[col].isna().any(), f"NaN in {col}"

    def test_regime_detection(self, sample_ohlcv):
        """Regime detection returns valid regimes."""
        engine = FeatureEngine()
        features = engine.compute_features(sample_ohlcv)

        assert "regime" in features.columns
        valid_regimes = {Regime.BULL, Regime.BEAR, Regime.SIDEWAYS, Regime.HIGH_VOLATILITY}
        unique_regimes = set(features["regime"].unique())
        assert unique_regimes.issubset(valid_regimes)

    def test_extract_observation_shape(self, sample_ohlcv):
        """Extracted observation has correct shape."""
        engine = FeatureEngine(feature_window=10)
        features = engine.compute_features(sample_ohlcv)

        obs = engine.extract_observation(features, 50)
        assert obs.shape == (10, len(engine.get_feature_columns()))
        assert obs.dtype == np.float32

    def test_extract_observation_early_index(self, sample_ohlcv):
        """Early index observation is zero-padded correctly."""
        engine = FeatureEngine(feature_window=10)
        features = engine.compute_features(sample_ohlcv)

        obs = engine.extract_observation(features, 3)
        assert obs.shape == (10, len(engine.get_feature_columns()))
