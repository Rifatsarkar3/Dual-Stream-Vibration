"""
select_epoch_budget.py — Epoch budget K* from LOBO validation curves.

K* = argmin over epochs k of the MEAN validation MAE at epoch k across all
fold x seed runs (curves right-censored at the shortest run). This estimator
uses validation data only and is far more stable than the median of per-run
argmins (which scatter widely on noisy cross-bearing validation).

USAGE:
  python Experiments/select_epoch_budget.py outputs/exp16_ress/lobo2_foldA/results.json ...
Prints: SELECTED_K=<int>
"""

import json
import sys

import numpy as np


def main():
    curves = []
    for path in sys.argv[1:]:
        with open(path, encoding='utf-8') as f:
            res = json.load(f)
        for seed_res in res['per_seed']:
            h = seed_res.get('val_history')
            if h:
                curves.append(h)
    if not curves:
        print('ERROR: no val histories found', file=sys.stderr)
        return 1
    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves])   # (runs, epochs)
    mean_curve = arr.mean(axis=0)
    k_star = int(np.argmin(mean_curve)) + 1          # epochs are 1-indexed
    k_star = max(k_star, 5)
    print(f'runs={len(curves)} min_len={min_len} '
          f'mean_curve_min={mean_curve.min():.4f} at epoch {k_star}')
    print(f'SELECTED_K={k_star}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
