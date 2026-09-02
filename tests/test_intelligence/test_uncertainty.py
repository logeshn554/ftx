"""Unit tests for Uncertainty Estimator."""

import numpy as np
import pytest
from trade.intelligence.uncertainty import UncertaintyEstimator


def test_uncertainty_estimator_in_and_out_of_distribution():
    np.random.seed(42)
    # Training distribution centered at 0 with unit variance
    ref_data = np.random.randn(100, 3)

    estimator = UncertaintyEstimator(ood_mahalanobis_threshold=3.0)
    estimator.fit(ref_data)

    # In-distribution sample (near 0)
    in_dist_sample = np.array([0.1, -0.1, 0.2])
    res_in = estimator.estimate(in_dist_sample, current_volatility_pct=1.0, baseline_volatility_pct=1.0)
    assert res_in.is_out_of_distribution is False
    assert res_in.epistemic < 0.35

    # Out-of-distribution sample (far from training domain)
    ood_sample = np.array([8.0, -10.0, 12.0])
    res_ood = estimator.estimate(ood_sample, current_volatility_pct=1.0, baseline_volatility_pct=1.0)
    assert res_ood.is_out_of_distribution is True
    assert res_ood.epistemic > 0.70
    assert res_ood.total_uncertainty > res_in.total_uncertainty


def test_uncertainty_aleatoric_volatility_scaling():
    estimator = UncertaintyEstimator()
    # Baseline normal volatility
    res_low = estimator.estimate([0.0, 0.0], current_volatility_pct=0.5, baseline_volatility_pct=1.0)
    # Extreme volatility surge
    res_high = estimator.estimate([0.0, 0.0], current_volatility_pct=3.5, baseline_volatility_pct=1.0)

    assert res_high.aleatoric > res_low.aleatoric
    assert res_high.total_uncertainty > res_low.total_uncertainty
