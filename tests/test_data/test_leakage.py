import numpy as np

from trade.data.contract import observation_columns
from trade.data.features import FeatureEngine


def test_target_columns_never_enter_observation(sample_ohlcv):
    engine = FeatureEngine(feature_window=10)
    features = engine.compute_features(sample_ohlcv)
    features["return_1_1m"] = 0.01
    features["future_target"] = 42
    before = engine.extract_observation(features, 50)
    features.loc[:, "return_1_1m"] = -999
    features.loc[:, "future_target"] = 999
    after = engine.extract_observation(features, 50)
    assert np.array_equal(before, after)
    assert observation_columns(["rsi_14", "return_1_1m", "future_target", "regime"]) == ["rsi_14"]


def test_future_rows_do_not_change_current_feature_observation(sample_ohlcv):
    engine = FeatureEngine(feature_window=10)
    baseline = engine.compute_features(sample_ohlcv)
    modified = sample_ohlcv.copy()
    modified.iloc[80:, modified.columns.get_loc("close")] *= 100
    changed = engine.compute_features(modified)
    assert np.array_equal(engine.extract_observation(baseline, 50), engine.extract_observation(changed, 50))
