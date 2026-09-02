"""Calibrated probability estimation and evaluation metrics for trading setups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class ProbabilityEvaluation:
    brier_score: float
    log_loss_value: float
    expected_calibration_error: float
    prob_true: list[float]
    prob_pred: list[float]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) across uniform confidence bins."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (n_in_bin / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


class CalibratedProbabilityEstimator:
    """Time-series safe probability calibration model for trading alpha signals."""

    def __init__(
        self,
        method: Literal["sigmoid", "isotonic"] = "sigmoid",
        n_splits: int = 3,
    ):
        self.method = method
        self.n_splits = n_splits
        self.calibrator: CalibratedClassifierCV | None = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray | list[list[float]], y: np.ndarray | list[int]) -> CalibratedProbabilityEstimator:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int32)

        if len(y_arr) < 20 or len(np.unique(y_arr)) < 2:
            # Insufficient samples for cross-validation calibration
            self.is_fitted = False
            return self

        base_estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
        cv = TimeSeriesSplit(n_splits=min(self.n_splits, max(2, len(y_arr) // 10)))

        self.calibrator = CalibratedClassifierCV(
            estimator=base_estimator,
            method=self.method,
            cv=cv,
        )
        self.calibrator.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_p_win(self, features: np.ndarray | list[float]) -> float | None:
        """Predict calibrated P(win) for a single feature vector."""
        if not self.is_fitted or self.calibrator is None:
            return None

        feat_arr = np.asarray(features, dtype=np.float64).reshape(1, -1)
        try:
            probs = self.calibrator.predict_proba(feat_arr)[0]
            # Binary classification positive class probability
            return float(probs[1]) if len(probs) > 1 else float(probs[0])
        except Exception:
            return None

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> ProbabilityEvaluation | None:
        """Evaluate calibration quality: Brier score, log-loss, and ECE."""
        if not self.is_fitted or self.calibrator is None:
            return None

        X_arr = np.asarray(X_val, dtype=np.float64)
        y_arr = np.asarray(y_val, dtype=np.int32)
        if len(y_arr) == 0:
            return None

        probs = self.calibrator.predict_proba(X_arr)[:, 1]
        brier = float(brier_score_loss(y_arr, probs))
        ll = float(log_loss(y_arr, probs))
        ece = expected_calibration_error(y_arr, probs)

        prob_true, prob_pred = calibration_curve(y_arr, probs, n_bins=5)

        return ProbabilityEvaluation(
            brier_score=brier,
            log_loss_value=ll,
            expected_calibration_error=ece,
            prob_true=prob_true.tolist(),
            prob_pred=prob_pred.tolist(),
        )
