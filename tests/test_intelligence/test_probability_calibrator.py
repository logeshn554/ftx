"""Unit tests for Calibrated Probability Estimator."""

import numpy as np
import pytest
from trade.intelligence.probability_calibrator import (
    CalibratedProbabilityEstimator,
    expected_calibration_error,
)


def test_ece_calculation():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.85, 0.15, 0.7, 0.3, 0.95, 0.6])
    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_calibrated_probability_estimator_fit_and_predict():
    np.random.seed(42)
    # Generate synthetic features correlated with win label
    n_samples = 100
    X = np.random.randn(n_samples, 4)
    # True win probability is sigmoid of linear combination
    logits = 1.2 * X[:, 0] - 0.8 * X[:, 1]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (np.random.rand(n_samples) < probs).astype(int)

    estimator = CalibratedProbabilityEstimator(method="sigmoid", n_splits=3)
    estimator.fit(X, y)

    assert estimator.is_fitted is True

    # Test prediction on high-signal instance
    p_win_high = estimator.predict_p_win(np.array([2.5, -2.5, 0.0, 0.0]))
    assert p_win_high is not None
    assert p_win_high > 0.65

    # Test prediction on low-signal instance
    p_win_low = estimator.predict_p_win(np.array([-2.5, 2.5, 0.0, 0.0]))
    assert p_win_low is not None
    assert p_win_low < 0.35

    # Test evaluation metrics
    eval_res = estimator.evaluate(X, y)
    assert eval_res is not None
    assert 0.0 <= eval_res.brier_score <= 0.35
    assert eval_res.expected_calibration_error >= 0.0
