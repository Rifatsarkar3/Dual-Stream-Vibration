"""
metrics.py
==========
Centralised metric computation for Paper #2.

All experiment scripts import from here — no duplicated metric code.

Functions:
    compute_regression_metrics  — MAE, RMSE, R², MBE for RUL regression
    compute_classification_metrics — Accuracy, Precision, Recall, F1, AUC-ROC
    compute_all_metrics         — both combined (primary use)
    aggregate_runs              — mean ± std over N independent runs
    print_metrics_table         — formatted console output
"""

import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)


# ════════════════════════════════════════════════════════════════════════════
# Regression
# ════════════════════════════════════════════════════════════════════════════

def compute_regression_metrics(y_true: np.ndarray,
                                y_pred: np.ndarray) -> dict:
    """
    Computes RUL regression metrics.

    Args:
        y_true: (N,) ground-truth RUL values
        y_pred: (N,) predicted RUL values

    Returns dict:
        mae   — Mean Absolute Error
        rmse  — Root Mean Squared Error
        r2    — Coefficient of Determination
        mbe   — Mean Bias Error (positive = over-prediction)
        score — Prognostic Score (asymmetric penalty from PHM literature)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae   = float(mean_absolute_error(y_true, y_pred))
    rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2    = float(r2_score(y_true, y_pred))
    mbe   = float(np.mean(y_pred - y_true))

    # Prognostic score (PHM 2008 / FEMTO convention):
    # Early predictions penalised less than late predictions
    err   = y_pred - y_true
    score = float(np.sum(
        np.where(err < 0,
                 np.exp(-err / 13) - 1,
                 np.exp(err  / 10) - 1)
    ))

    return {'mae': mae, 'rmse': rmse, 'r2': r2,
            'mbe': mbe, 'prog_score': score}


# ════════════════════════════════════════════════════════════════════════════
# Classification
# ════════════════════════════════════════════════════════════════════════════

def compute_classification_metrics(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    y_prob: np.ndarray = None) -> dict:
    """
    Computes binary fault classification metrics.

    Args:
        y_true: (N,) int  — ground-truth binary labels {0,1}
        y_pred: (N,) int  — predicted binary labels {0,1}
        y_prob: (N,) float — predicted positive-class probability (for AUC)

    Returns dict:
        accuracy, precision, recall, f1, auc_roc, confusion_matrix
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc  = float((y_true == y_pred).mean() * 100)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    cm   = confusion_matrix(y_true, y_pred).tolist()

    auc = 0.5
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            pass

    return {'accuracy':  acc, 'precision': prec,
            'recall':    rec, 'f1':        f1,
            'auc_roc':   auc, 'confusion_matrix': cm}


# ════════════════════════════════════════════════════════════════════════════
# Combined
# ════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(rul_true: np.ndarray,
                         rul_pred: np.ndarray,
                         bin_true: np.ndarray,
                         bin_pred: np.ndarray,
                         bin_prob: np.ndarray = None,
                         latency_ms: float = 0.0) -> dict:
    """
    Computes all metrics for one evaluation pass.
    Primary function called by every experiment's _eval() function.
    """
    reg = compute_regression_metrics(rul_true, rul_pred)
    clf = compute_classification_metrics(bin_true, bin_pred, bin_prob)
    return {**reg, **clf, 'latency_ms': latency_ms}


# ════════════════════════════════════════════════════════════════════════════
# Aggregation across runs
# ════════════════════════════════════════════════════════════════════════════

def aggregate_runs(run_metrics: list) -> dict:
    """
    Aggregates a list of per-run metric dicts into mean ± std.

    Args:
        run_metrics: list of dicts, each from compute_all_metrics()

    Returns:
        dict with keys like 'mae_mean', 'mae_std', 'r2_mean', 'r2_std', ...
    """
    scalar_keys = ['mae', 'rmse', 'r2', 'mbe', 'prog_score',
                   'accuracy', 'precision', 'recall', 'f1',
                   'auc_roc', 'latency_ms']
    result = {}
    for key in scalar_keys:
        vals = [r[key] for r in run_metrics if r and key in r]
        if vals:
            result[f'{key}_mean'] = float(np.mean(vals))
            result[f'{key}_std']  = float(np.std(vals))
    return result


# ════════════════════════════════════════════════════════════════════════════
# Console display
# ════════════════════════════════════════════════════════════════════════════

def print_metrics_table(results: dict, title: str = "Metrics"):
    """
    Prints a formatted metric table to console.

    Args:
        results: dict of {model_name: aggregated_metrics_dict}
        title  : table header string
    """
    sep = "─" * 78
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  {'Model':<28} {'R²':>8} {'MAE':>8} {'RMSE':>8} "
          f"{'F1':>8} {'AUC':>8} {'Lat(ms)':>9}")
    print(sep)
    for name, m in results.items():
        r2   = m.get('r2_mean',          m.get('r2',   0))
        mae  = m.get('mae_mean',         m.get('mae',  0))
        rmse = m.get('rmse_mean',        m.get('rmse', 0))
        f1   = m.get('f1_mean',          m.get('f1',   0))
        auc  = m.get('auc_roc_mean',     m.get('auc_roc', 0))
        lat  = m.get('latency_ms_mean',  m.get('latency_ms', 0))

        r2_s   = m.get('r2_std',         0)
        mae_s  = m.get('mae_std',        0)

        r2_str  = f"{r2:.4f}" + (f"±{r2_s:.4f}" if r2_s else "")
        mae_str = f"{mae:.4f}" + (f"±{mae_s:.4f}" if mae_s else "")

        print(f"  {name:<28} {r2_str:>10} {mae_str:>12} "
              f"{rmse:>8.4f} {f1:>8.4f} {auc:>8.4f} {lat:>8.2f}")
    print(sep + "\n")


# ════════════════════════════════════════════════════════════════════════════
# Monotonicity and Physical Constraints
# ════════════════════════════════════════════════════════════════════════════

def compute_monotonicity(predictions: np.ndarray) -> float:
    """
    Measure how monotonically decreasing predictions are (0-100%).
    Higher values indicate better physical realism for RUL prediction.
    """
    if len(predictions) < 2:
        return 100.0

    diffs = np.diff(predictions)
    non_increasing = np.sum(diffs <= 0)
    monotonicity = (non_increasing / len(diffs)) * 100
    return float(monotonicity)


def compute_latency(model, device, batch_size=32, x_1d_shape=(1024,), x_2d_shape=(3, 224, 224)):
    """
    Measure inference latency using CUDA events (if available).

    Args:
        model: PyTorch model
        device: torch.device
        batch_size: batch size for latency measurement
        x_1d_shape: shape of 1D input (excluding batch dimension)
        x_2d_shape: shape of 2D input (excluding batch dimension)

    Returns:
        latency_ms: per-sample latency in milliseconds
    """
    import torch
    from torch.cuda.amp import autocast

    model.eval()

    if device.type == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        x_1d = torch.randn(batch_size, 1, *x_1d_shape, device=device)
        x_2d = torch.randn(batch_size, *x_2d_shape, device=device)

        with torch.no_grad():
            with autocast():
                start.record()
                _ = model(x_1d, x_2d, use_checkpointing=False)
                end.record()

        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / batch_size
        return float(latency_ms)
    else:
        return 0.0


def apply_isotonic_monotonicity(predictions: np.ndarray) -> np.ndarray:
    """
    Enforce strict monotonic decrease on a RUL prediction sequence using
    isotonic regression (pool-adjacent violators algorithm).

    This is a valid post-processing step used in published PHM literature
    to enforce the physical constraint that bearing RUL is non-increasing.
    Unlike the soft PGMC training penalty, this guarantees 100% monotonicity
    at inference time.

    Args:
        predictions: (N,) array of raw model RUL predictions

    Returns:
        (N,) array of isotonically-constrained predictions (non-increasing)
    """
    from sklearn.isotonic import IsotonicRegression
    n = len(predictions)
    if n < 2:
        return predictions.copy()

    ir = IsotonicRegression(increasing=False)
    time_steps = np.arange(n).reshape(-1, 1)
    constrained = ir.fit_transform(time_steps.ravel(),
                                   predictions.ravel())
    return constrained.astype(np.float32)
