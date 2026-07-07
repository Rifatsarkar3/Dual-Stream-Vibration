"""
Isotonic Regression Post-Processor for RUL Predictions
=======================================================

Applies Pool-Adjacent-Violators (PAV) algorithm to enforce monotonicity
in bearing RUL predictions. Can be applied per-bearing or globally.

Key insight: After model training, apply isotonic regression independently
to each bearing's predictions to enforce monotonic decrease constraint.
"""

import numpy as np
from typing import Tuple, Optional

try:
    from scipy.isotonic import isotonic_regression
except ImportError:
    from sklearn.isotonic import IsotonicRegression as SklearnIsotonicRegression
    def isotonic_regression(y, y_true=None, sample_weight=None, increasing=False):
        """Wrapper around sklearn's IsotonicRegression"""
        ir = SklearnIsotonicRegression(increasing=increasing, out_of_bounds='clip')
        if sample_weight is not None:
            ir.fit(np.arange(len(y)), y, sample_weight=sample_weight)
        else:
            ir.fit(np.arange(len(y)), y)
        return ir.predict(np.arange(len(y)))


class IsotonicRegressor:
    """Wrapper around scipy's isotonic_regression with monotonicity metrics."""

    def __init__(self, increasing: bool = False):
        """
        Args:
            increasing: If False (default), enforce decreasing order (RUL degrades over time).
                       If True, enforce increasing order.
        """
        self.increasing = increasing
        self.fitted = False

    def fit_transform(self, y_pred: np.ndarray,
                     y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit isotonic regression and return transformed (monotonic) predictions.

        Args:
            y_pred: Raw model predictions, shape (n_samples,)
            y_true: Optional target values for weighted fit (default: uniform weight)

        Returns:
            y_iso: Isotonically corrected predictions, same shape as y_pred
        """
        if y_true is not None:
            # Use target values as sample weights (closer predictions get higher weight)
            weights = 1.0 / (1.0 + np.abs(y_pred - y_true))
            weights = weights / weights.sum() * len(weights)  # Normalize
        else:
            weights = None

        y_iso = isotonic_regression(y_pred, y_true=y_true, sample_weight=weights,
                                    increasing=self.increasing)
        self.fitted = True
        return y_iso


def compute_monotonicity(predictions: np.ndarray,
                        bearing_ids: Optional[np.ndarray] = None) -> float:
    """
    Compute monotonicity percentage.

    Monotonicity = fraction of consecutive pairs where pred[t+1] <= pred[t]
    (RUL should decrease or stay same over time).

    Args:
        predictions: RUL predictions, shape (n_samples,)
        bearing_ids: Optional bearing IDs for per-bearing computation.
                    If None, treats all as single sequence.

    Returns:
        mono_pct: Percentage of monotonic decreasing pairs [0, 100]
    """
    if bearing_ids is not None:
        # Compute per-bearing monotonicity, then average
        mono_list = []
        for bid in np.unique(bearing_ids):
            mask = bearing_ids == bid
            pred_bearing = predictions[mask]
            if len(pred_bearing) > 1:
                diffs = np.diff(pred_bearing)
                n_mono = np.sum(diffs <= 0)
                pct = 100.0 * n_mono / len(diffs)
                mono_list.append(pct)

        if mono_list:
            return float(np.mean(mono_list))
        else:
            return 0.0
    else:
        # Global monotonicity (all predictions treated as single sequence)
        if len(predictions) < 2:
            return 0.0

        diffs = np.diff(predictions)
        n_mono = np.sum(diffs <= 0)
        return float(100.0 * n_mono / len(diffs))


def apply_isotonic_correction(predictions: np.ndarray,
                              targets: np.ndarray,
                              bearing_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """
    Apply isotonic regression to predictions, with per-bearing support.

    KEY INSIGHT: If bearing_ids are provided, apply isotonic regression INDEPENDENTLY
    to each bearing. This preserves inter-bearing variance while enforcing intra-bearing
    monotonicity. Without bearing_ids, applies global correction (less ideal).

    Args:
        predictions: Raw model RUL predictions, shape (n_samples,)
        targets: Target RUL values, shape (n_samples,)
        bearing_ids: Optional bearing IDs for per-bearing correction.
                    If None, applies global isotonic regression (faster but less accurate).

    Returns:
        y_iso: Isotonically corrected predictions
        metrics: Dict with 'mono_raw', 'mono_corrected', 'mae_raw', 'mae_corrected'
    """
    # Compute raw metrics
    mono_raw = compute_monotonicity(predictions, bearing_ids)
    mae_raw = np.mean(np.abs(predictions - targets))

    # Apply isotonic correction
    if bearing_ids is not None and len(np.unique(bearing_ids)) > 1:
        # Per-bearing correction: treat each bearing's trajectory independently
        # This is the correct approach - each bearing has its own degradation profile
        y_iso = np.zeros_like(predictions)
        for bid in np.unique(bearing_ids):
            mask = bearing_ids == bid
            pred_bearing = predictions[mask]
            tgt_bearing = targets[mask]

            if len(pred_bearing) > 2:
                # Apply isotonic regression to enforce monotonicity WITHIN this bearing
                # WITHOUT constraining across different bearings
                regressor = IsotonicRegressor(increasing=False)
                # Use target values as weak guide (weighted fit)
                pred_iso = regressor.fit_transform(pred_bearing, tgt_bearing)
                y_iso[mask] = pred_iso
            elif len(pred_bearing) > 1:
                # Too few samples: just sort predictions
                sorted_indices = np.argsort(pred_bearing)[::-1]
                y_iso[mask] = pred_bearing[sorted_indices]
            else:
                # Single sample: keep as is
                y_iso[mask] = pred_bearing
    else:
        # Global correction: force ALL predictions to be monotonically decreasing
        # This is ONLY acceptable if all predictions represent a single bearing's trajectory
        # Otherwise it destroys inter-bearing variance
        regressor = IsotonicRegressor(increasing=False)
        y_iso = regressor.fit_transform(predictions, targets)

    # Compute corrected metrics
    mono_corrected = compute_monotonicity(y_iso, bearing_ids)
    mae_corrected = np.mean(np.abs(y_iso - targets))

    metrics = {
        'mono_raw': float(mono_raw),
        'mono_corrected': float(mono_corrected),
        'mae_raw': float(mae_raw),
        'mae_corrected': float(mae_corrected),
        'mono_improvement': float(mono_corrected - mono_raw),
    }

    return y_iso, metrics


def report_isotonic_improvement(predictions: np.ndarray,
                               targets: np.ndarray,
                               bearing_ids: Optional[np.ndarray] = None,
                               verbose: bool = True) -> dict:
    """
    Apply isotonic correction and print detailed report.

    Returns:
        metrics dict with all raw/corrected values
    """
    y_iso, metrics = apply_isotonic_correction(predictions, targets, bearing_ids)

    if verbose:
        print("\n" + "="*60)
        print("ISOTONIC REGRESSION REPORT")
        print("="*60)
        print(f"Monotonicity (raw):       {metrics['mono_raw']:.1f}%")
        print(f"Monotonicity (corrected): {metrics['mono_corrected']:.1f}%")
        print(f"  Improvement:            +{metrics['mono_improvement']:.1f}%")
        print(f"MAE (raw):                {metrics['mae_raw']:.4f}")
        print(f"MAE (corrected):          {metrics['mae_corrected']:.4f}")
        print(f"  Change:                 {metrics['mae_corrected'] - metrics['mae_raw']:+.4f}")
        print("="*60)

    return metrics
