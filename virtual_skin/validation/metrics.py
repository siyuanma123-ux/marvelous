"""
Evaluation metrics for the virtual skin system.

Four layers:
  1. Pharmacokinetic prediction accuracy
  2. Biological consistency
  3. Model credibility (calibration, uncertainty)
  4. Translational value
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


class ValidationMetrics:
    """Compute all evaluation metrics for virtual skin predictions."""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-30))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def fold_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Fold error: max(pred/true, true/pred) for each sample."""
        ratio = y_pred / (y_true + 1e-30)
        return np.maximum(ratio, 1.0 / (ratio + 1e-30))

    @staticmethod
    def within_fold(
        y_true: np.ndarray, y_pred: np.ndarray, fold: float = 2.0
    ) -> float:
        """Fraction of predictions within N-fold of observed."""
        fe = ValidationMetrics.fold_error(y_true, y_pred)
        return float(np.mean(fe <= fold))

    # ---- Calibration metrics ----

    @staticmethod
    def coverage(
        y_true: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
    ) -> float:
        """Fraction of observations falling within prediction interval."""
        inside = (y_true >= y_lower) & (y_true <= y_upper)
        return float(np.mean(inside))

    @staticmethod
    def calibration_error(
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        y_pred_std: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error across quantile bins."""
        quantiles = np.linspace(0.05, 0.95, n_bins)
        total_error = 0.0
        for q in quantiles:
            threshold = stats.norm.ppf(q) * y_pred_std + y_pred_mean
            observed_fraction = np.mean(y_true <= threshold)
            total_error += abs(observed_fraction - q)
        return float(total_error / n_bins)

    @staticmethod
    def sharpness(y_pred_std: np.ndarray) -> float:
        """Average width of prediction intervals (lower = sharper)."""
        return float(np.mean(y_pred_std))

    # ---- Concordance ----

    @staticmethod
    def concordance_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Lin's concordance correlation coefficient."""
        mean_t, mean_p = y_true.mean(), y_pred.mean()
        var_t, var_p = y_true.var(), y_pred.var()
        cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
        ccc = 2 * cov / (var_t + var_p + (mean_t - mean_p) ** 2 + 1e-30)
        return float(ccc)

    # ---- Aggregate report ----

    @classmethod
    def full_report(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_std: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        report = {
            "RMSE": cls.rmse(y_true, y_pred),
            "MAE": cls.mae(y_true, y_pred),
            "R2": cls.r2(y_true, y_pred),
            "MAPE": cls.mape(y_true, y_pred),
            "within_2fold": cls.within_fold(y_true, y_pred, 2.0),
            "within_3fold": cls.within_fold(y_true, y_pred, 3.0),
            "CCC": cls.concordance_correlation(y_true, y_pred),
        }
        if y_pred_std is not None:
            y_lo = y_pred - 1.645 * y_pred_std
            y_hi = y_pred + 1.645 * y_pred_std
            report["coverage_90"] = cls.coverage(y_true, y_lo, y_hi)
            report["calibration_error"] = cls.calibration_error(
                y_true, y_pred, y_pred_std
            )
            report["sharpness"] = cls.sharpness(y_pred_std)
        return report
