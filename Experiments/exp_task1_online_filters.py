"""
exp_task1_online_filters.py — Task 1: Online Filter Evaluation
==============================================================

Evaluates four post-processing strategies on DSAFLite predictions:
  Raw     — no post-processing
  RTRM    — Real-Time Running Minimum (online, causal, no params)
  CES     — Constrained Exponential Smoothing (online, causal, α=0.2, ε=0.05)
  PAVA    — Offline Pool-Adjacent-Violators isotonic regression (non-causal baseline)

Uses the Full_DSAFLite_PGMC models trained in Task 2 (all 5 seeds), runs
inference per test bearing in temporal order, then applies each filter.

HOW TO RUN:
  cd "E:\\Yolo-Thermal\\Dual-Stream Vibration-Vision"
  python Experiments/exp_task1_online_filters.py

OUTPUTS:
  outputs/exp_revision/task1_online_filters_results.md  — JSON results
  outputs/exp_revision/task1_online_filters/summary.json — detailed per-bearing breakdown
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ablation_models import AblationDSAFLite
from src.experiment_config import ABLATION_BASELINE, RANDOM_SEEDS
from src.online_filters import RealTimeRunningMinimum, ConstrainedExponentialSmoothing

DATA_DIR     = Path("Processed_PRONOSTIA")
MODELS_DIR   = Path("outputs/exp_revision/task2_ablations")
OUT_DIR      = Path("outputs/exp_revision/task1_online_filters")
RESULTS_FILE = Path("outputs/exp_revision/task1_online_filters_results.md")

TEST_FILES = [
    "test_01_Bearing1_3.npz",
    "test_02_Bearing1_4.npz",
    "test_03_Bearing1_5.npz",
    "test_04_Bearing2_3.npz",
]

FILTER_NAMES = ["Raw", "RTRM", "CES", "PAVA"]


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_bearing(npz_path: Path):
    d = np.load(npz_path)
    vibration = d["vibration"].astype(np.float32)   # (N, 2560)
    rul       = d["rul"].astype(np.float32)          # (N,)
    return vibration, rul


def infer_bearing(model, vibration: np.ndarray, device, batch_size=64):
    """Run inference on a single bearing's samples in temporal order."""
    model.eval()
    preds = []
    n = len(vibration)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            x = torch.from_numpy(vibration[start:start + batch_size]).unsqueeze(1).to(device)
            out, *_ = model(x)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)


def apply_pava(predictions: np.ndarray) -> np.ndarray:
    """Offline isotonic regression (decreasing)."""
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    t = np.arange(len(predictions))
    return ir.fit_transform(t, predictions).astype(np.float32)


def apply_filters(raw: np.ndarray) -> dict:
    rtrm_filter = RealTimeRunningMinimum()
    ces_filter  = ConstrainedExponentialSmoothing(alpha=0.2, epsilon=0.05)
    return {
        "Raw":  raw,
        "RTRM": rtrm_filter.process(raw),
        "CES":  ces_filter.process(raw),
        "PAVA": apply_pava(raw),
    }


def regression_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    diffs = np.diff(y_pred)
    mono  = float(100.0 * np.sum(diffs <= 0) / len(diffs)) if len(diffs) > 0 else 100.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "mono": mono}


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Accumulate per-filter metrics across all seeds × bearings
    # Structure: {filter_name: [metric_dict, ...]}
    all_metrics = {name: [] for name in FILTER_NAMES}
    detailed    = []

    for seed in RANDOM_SEEDS:
        model_path = MODELS_DIR / f"Full_DSAFLite_PGMC_seed{seed}" / "weights" / "best_model.pt"
        if not model_path.exists():
            print(f"  [SKIP] Model not found for seed {seed}: {model_path}")
            continue

        print(f"\n=== Seed {seed} ===")
        model = AblationDSAFLite(ABLATION_BASELINE).to(device)
        ckpt  = torch.load(str(model_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        seed_metrics = {name: [] for name in FILTER_NAMES}

        for fname in TEST_FILES:
            npz_path = DATA_DIR / fname
            if not npz_path.exists():
                print(f"  [SKIP] {fname} not found")
                continue

            vibration, rul_true = load_bearing(npz_path)
            raw_preds = infer_bearing(model, vibration, device)
            filtered  = apply_filters(raw_preds)

            bearing_row = {"seed": seed, "bearing": fname}
            for name, preds in filtered.items():
                m = regression_metrics(rul_true, preds)
                seed_metrics[name].append(m)
                bearing_row[name] = m
                print(f"  {fname}  {name:6s}: MAE={m['mae']:.4f}  R²={m['r2']:.4f}  mono={m['mono']:.1f}%")

            detailed.append(bearing_row)

        # Aggregate this seed across bearings
        for name in FILTER_NAMES:
            if seed_metrics[name]:
                agg = {
                    "mae":  float(np.mean([m["mae"]  for m in seed_metrics[name]])),
                    "rmse": float(np.mean([m["rmse"] for m in seed_metrics[name]])),
                    "r2":   float(np.mean([m["r2"]   for m in seed_metrics[name]])),
                    "mono": float(np.mean([m["mono"] for m in seed_metrics[name]])),
                }
                all_metrics[name].append(agg)

    # Final aggregation across seeds
    final = {}
    print("\n\n=== FINAL RESULTS (mean ± std across 5 seeds × 4 bearings) ===")
    print(f"{'Filter':<8}  {'MAE':>10}  {'R²':>10}  {'Mono%':>8}")
    print("-" * 44)
    for name in FILTER_NAMES:
        vals = all_metrics[name]
        if not vals:
            continue
        mae_arr  = [v["mae"]  for v in vals]
        rmse_arr = [v["rmse"] for v in vals]
        r2_arr   = [v["r2"]   for v in vals]
        mono_arr = [v["mono"] for v in vals]
        final[name] = {
            "mae_mean":  float(np.mean(mae_arr)),  "mae_std":  float(np.std(mae_arr)),
            "rmse_mean": float(np.mean(rmse_arr)), "rmse_std": float(np.std(rmse_arr)),
            "r2_mean":   float(np.mean(r2_arr)),   "r2_std":   float(np.std(r2_arr)),
            "mono_mean": float(np.mean(mono_arr)), "mono_std": float(np.std(mono_arr)),
        }
        r = final[name]
        print(f"{name:<8}  {r['mae_mean']:.4f}±{r['mae_std']:.4f}  "
              f"{r['r2_mean']:.4f}±{r['r2_std']:.4f}  {r['mono_mean']:.1f}±{r['mono_std']:.1f}%")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    summary = {"per_seed_bearing": detailed, "aggregated": final}
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Detailed breakdown saved to {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
