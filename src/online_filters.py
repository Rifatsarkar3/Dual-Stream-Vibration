"""
online_filters.py — Real-Time Causal Post-Processing for RUL Predictions
==========================================================================

Implements two online (causal) monotonicity enforcement algorithms that operate
step-by-step during live inference, addressing reviewer concerns about offline
PAVA post-processing. Both filters operate on raw predictions WITHOUT seeing
future timepoints.

References:
  - Real-Time Running Minimum: Standard monotonic envelope, no parameters
  - Constrained Exponential Smoothing (CES): Smooths with bounded upward jumps
"""

import numpy as np
from typing import Tuple, Optional, Dict


class RealTimeRunningMinimum:
    """
    At each timestep t, enforces monotonic decrease by taking the minimum
    of the current raw prediction and the previous smoothed prediction.

    Formula: ŷ_t^online = min(ŷ_t^raw, ŷ_{t-1}^online)

    Properties:
      - No parameters to tune
      - Causal (only uses past and current, never future)
      - Always decreasing (or flat)
      - Aggressive: any upward jump is immediately clipped
    """

    def __init__(self):
        pass

    def process(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply real-time running minimum to a sequence of predictions.

        Args:
            predictions: (n_samples,) raw model RUL predictions

        Returns:
            filtered: (n_samples,) online minimum-filtered predictions
        """
        filtered = np.zeros_like(predictions, dtype=np.float32)
        filtered[0] = predictions[0]

        for t in range(1, len(predictions)):
            filtered[t] = min(predictions[t], filtered[t-1])

        return filtered


class ConstrainedExponentialSmoothing:
    """
    Exponential smoothing with constrained upward jumps for bearing RUL.

    Two-step process at each timestep:
      Step 1: Exponential smoothing (standard EMA)
        tilde{y}_t = α·ŷ_t^raw + (1-α)·ŷ_{t-1}^smoothed

      Step 2: Enforce monotonic constraint
        ŷ_t^smoothed = min(tilde{y}_t, ŷ_{t-1}^smoothed + ε)

    Parameters:
      - alpha (0.2): Smoothing factor. Higher → more responsive to raw predictions.
        Default 0.2 makes it conservative (75% weight on previous prediction).
      - epsilon (0.05): Max allowed upward jump per step. Larger → fewer violations.
        Default 0.05 RUL units allows gradual increase without strict monotonicity.

    Properties:
      - Causal and online
      - Balances responsiveness (alpha) and monotonicity (epsilon)
      - Tunable for different bearing types and prediction ranges
    """

    def __init__(self, alpha: float = 0.2, epsilon: float = 0.05):
        """
        Args:
            alpha: Smoothing factor in [0, 1]. Lower → more memory. Default 0.2.
            epsilon: Max upward jump allowed per timestep. Default 0.05.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")

        self.alpha = alpha
        self.epsilon = epsilon

    def process(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply constrained exponential smoothing to a sequence of predictions.

        Args:
            predictions: (n_samples,) raw model RUL predictions

        Returns:
            smoothed: (n_samples,) CES-filtered predictions
        """
        smoothed = np.zeros_like(predictions, dtype=np.float32)
        smoothed[0] = predictions[0]

        for t in range(1, len(predictions)):
            # Step 1: EMA
            tilde_yt = self.alpha * predictions[t] + (1.0 - self.alpha) * smoothed[t-1]

            # Step 2: Apply monotonic constraint with bounded jump
            smoothed[t] = min(tilde_yt, smoothed[t-1] + self.epsilon)

        return smoothed


def process_bearing_sequence(predictions: np.ndarray,
                            bearing_ids: Optional[np.ndarray] = None,
                            filter_type: str = 'rtrm',
                            alpha: float = 0.2,
                            epsilon: float = 0.05) -> np.ndarray:
    """
    Apply online filter to predictions, optionally per-bearing.

    Args:
        predictions: (n_samples,) or (n_batches, n_timesteps) predictions
        bearing_ids: (n_samples,) bearing identifier for each prediction.
                    If None, treat all as single sequence.
        filter_type: 'rtrm' (Real-Time Running Minimum) or 'ces' (Constrained Exponential Smoothing)
        alpha: CES smoothing factor (ignored if filter_type='rtrm')
        epsilon: CES constraint bound (ignored if filter_type='rtrm')

    Returns:
        filtered: Same shape as predictions, online-filtered per-bearing
    """
    if predictions.ndim != 1:
        raise ValueError(f"predictions must be 1D, got shape {predictions.shape}")

    if filter_type == 'rtrm':
        filter_fn = RealTimeRunningMinimum()
    elif filter_type == 'ces':
        filter_fn = ConstrainedExponentialSmoothing(alpha=alpha, epsilon=epsilon)
    else:
        raise ValueError(f"filter_type must be 'rtrm' or 'ces', got {filter_type}")

    if bearing_ids is None:
        return filter_fn.process(predictions)

    # Process per-bearing
    filtered = np.zeros_like(predictions, dtype=np.float32)
    unique_bearings = np.unique(bearing_ids)

    for bearing_id in unique_bearings:
        mask = bearing_ids == bearing_id
        indices = np.where(mask)[0]
        bearing_preds = predictions[indices]
        bearing_filtered = filter_fn.process(bearing_preds)
        filtered[indices] = bearing_filtered

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Utilities for batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

class OnlineFilterEvaluator:
    """Wrapper to apply filters to model predictions and compute metrics."""

    def __init__(self):
        self.rtrm_filter = RealTimeRunningMinimum()
        self.ces_filter = ConstrainedExponentialSmoothing(alpha=0.2, epsilon=0.05)

    def apply_filters(self, raw_predictions: np.ndarray,
                      bearing_ids: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Apply both online filters to raw predictions.

        Args:
            raw_predictions: (n_samples,) model predictions
            bearing_ids: (n_samples,) bearing IDs for per-bearing filtering

        Returns:
            dict with keys: 'raw', 'rtrm', 'ces'
        """
        return {
            'raw': raw_predictions,
            'rtrm': process_bearing_sequence(raw_predictions, bearing_ids, 'rtrm'),
            'ces': process_bearing_sequence(raw_predictions, bearing_ids, 'ces',
                                           alpha=0.2, epsilon=0.05)
        }


if __name__ == "__main__":
    # Quick test
    raw_preds = np.array([100.0, 95.0, 98.0, 90.0, 85.0, 88.0, 80.0], dtype=np.float32)
    print("Raw predictions:        ", raw_preds)

    rtrm = RealTimeRunningMinimum()
    rtrm_result = rtrm.process(raw_preds)
    print("RTRM filtered:          ", rtrm_result)

    ces = ConstrainedExponentialSmoothing(alpha=0.2, epsilon=0.05)
    ces_result = ces.process(raw_preds)
    print("CES filtered (α=0.2):   ", ces_result)

    # Test with different epsilon
    ces_tight = ConstrainedExponentialSmoothing(alpha=0.2, epsilon=0.02)
    ces_tight_result = ces_tight.process(raw_preds)
    print("CES filtered (ε=0.02):  ", ces_tight_result)
