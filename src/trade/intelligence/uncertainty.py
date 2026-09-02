"""Epistemic and aleatoric uncertainty estimation for trading decision systems."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class UncertaintyEstimate:
    epistemic: float  # Out-of-distribution / model uncertainty [0.0, 1.0]
    aleatoric: float  # Outcome / volatility dispersion [0.0, 1.0]
    total_uncertainty: float  # Combined uncertainty [0.0, 1.0]
    is_out_of_distribution: bool


class UncertaintyEstimator:
    """Estimates epistemic (state novelty) and aleatoric (return noise) uncertainty."""

    def __init__(self, ood_mahalanobis_threshold: float = 3.0):
        self.ood_threshold = ood_mahalanobis_threshold
        self.feature_mean: np.ndarray | None = None
        self.inv_cov: np.ndarray | None = None
        self.is_fitted: bool = False

    def fit(self, reference_features: np.ndarray | list[list[float]]) -> UncertaintyEstimator:
        X = np.asarray(reference_features, dtype=np.float64)
        if len(X) < 10:
            return self

        self.feature_mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)

        # Regularized inverse covariance to avoid numerical instability
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-4
        try:
            self.inv_cov = np.linalg.inv(cov_reg)
            self.is_fitted = True
        except np.linalg.LinAlgError:
            self.inv_cov = np.eye(cov.shape[0])
            self.is_fitted = True

        return self

    def estimate(
        self,
        features: np.ndarray | list[float],
        current_volatility_pct: float,
        baseline_volatility_pct: float = 1.0,
    ) -> UncertaintyEstimate:
        """Estimate uncertainty for a new observation."""
        feat = np.asarray(features, dtype=np.float64).flatten()

        # 1. Epistemic Uncertainty (Mahalanobis Distance from Training Domain)
        if self.is_fitted and self.feature_mean is not None and self.inv_cov is not None and len(feat) == len(self.feature_mean):
            delta = feat - self.feature_mean
            dist_sq = float(np.dot(np.dot(delta, self.inv_cov), delta))
            mahalanobis_dist = np.sqrt(max(0.0, dist_sq))
            # Squeeze into [0.0, 1.0] using sigmoid-like curve
            epistemic = float(1.0 - np.exp(-mahalanobis_dist / 3.0))
            is_ood = bool(mahalanobis_dist > self.ood_threshold)
        else:
            epistemic = 0.50
            is_ood = False

        # 2. Aleatoric Uncertainty (Market Noise / Volatility Dispersion)
        vol_ratio = max(0.1, current_volatility_pct) / max(0.1, baseline_volatility_pct)
        # Squeeze volatility noise into [0.0, 1.0]
        aleatoric = float(np.clip((vol_ratio - 0.5) / 2.5, 0.0, 1.0))

        # 3. Total Combined Uncertainty
        total = float(np.clip(1.0 - (1.0 - epistemic) * (1.0 - aleatoric), 0.0, 1.0))

        return UncertaintyEstimate(
            epistemic=epistemic,
            aleatoric=aleatoric,
            total_uncertainty=total,
            is_out_of_distribution=is_ood,
        )
