"""
exp22_uq_calibration.py — Leak-free calibration of MC-dropout intervals.

MC-dropout intervals under-cover (PICP90 ≈ 59%). We calibrate them WITHOUT
touching test data:

  1. The LOBO fold models (sweepseq_B5_foldA/B) each have a held-out TRAINING
     bearing (Bearing1_2 / Bearing2_2) that played no role in their fitting.
  2. MC-dropout on that validation bearing yields per-window (mu, sigma, y);
     the scale c = Q_0.90(|y - mu| / (1.645 sigma)) is the smallest multiplier
     achieving 90% empirical coverage on calibration data.
  3. c* = mean over the 10 fold x seed runs is applied to the FINAL model's
     saved test-set MC outputs: sigma' = c* sigma. PICP/MPIW/NLL are recomputed.

All calibration data comes from training bearings; the test set is untouched
until the final recomputation of interval metrics.

RUN:
  .venv\\Scripts\\python.exe Experiments/exp22_uq_calibration.py ^
      --final-dir outputs/exp16_ress/final20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from Experiments.exp16_ress import (  # noqa: E402
    CONFIG, load_bearings, make_model, _enable_mc_dropout)

FOLD_DIRS = {
    'outputs/exp16_ress/sweepseq_B5_foldA': 'train_02_Bearing1_2',
    'outputs/exp16_ress/sweepseq_B5_foldB': 'train_04_Bearing2_2',
}
Z90 = 1.6449
MC_PASSES = 30


def mc_predict(model, vib, device):
    model.eval()
    _enable_mc_dropout(model)
    runs = []
    with torch.no_grad():
        for _ in range(MC_PASSES):
            ps = []
            for s in range(0, len(vib), 256):
                x = torch.from_numpy(vib[s:s + 256]).unsqueeze(1).to(device)
                p, *_ = model(x)
                ps.append(p.cpu().numpy())
            runs.append(np.concatenate(ps))
    model.eval()
    mc = np.stack(runs)
    return mc.mean(axis=0), mc.std(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final-dir', default='outputs/exp16_ress/final20')
    ap.add_argument('--out', default='outputs/exp22_uq_calibration/results.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Step 1-2: per-run calibration scales on held-out TRAINING bearings ----
    scales = []
    for fold_dir, val_pattern in FOLD_DIRS.items():
        bearing = load_bearings(CONFIG['data_dir'], [val_pattern])[0]
        for ck in sorted(Path(fold_dir).glob('model_seed*.pth')):
            model = make_model(CONFIG, device)
            model.load_state_dict(torch.load(ck, map_location=device))
            mu, sd = mc_predict(model, bearing.vib, device)
            sd = np.clip(sd, 1e-4, None)
            ratio = np.abs(bearing.rul - mu) / (Z90 * sd)
            c = float(np.quantile(ratio, 0.90))
            scales.append(c)
            print(f'{Path(fold_dir).name} {ck.stem}: c={c:.2f}')
    c_star = float(np.mean(scales))
    print(f'\ncalibration scale c* = {c_star:.3f} '
          f'(mean of {len(scales)} runs, std {np.std(scales):.3f})')

    # ---- Step 3: apply to saved final test MC outputs ----
    per_seed = []
    for npz in sorted(Path(args.final_dir).glob('preds_seed*.npz')):
        d = np.load(npz)
        names = sorted({k.split('_', 1)[1] for k in d.files if k.startswith('tgt_')})
        mu = np.concatenate([d[f'mcmean_{n}'] for n in names])
        sd = np.clip(np.concatenate([d[f'mcstd_{n}'] for n in names]), 1e-4, None)
        y = np.concatenate([d[f'tgt_{n}'] for n in names])

        def interval_metrics(scale):
            s = scale * sd
            lo, hi = mu - Z90 * s, mu + Z90 * s
            picp = float(np.mean((y >= lo) & (y <= hi)) * 100)
            mpiw = float(np.mean(hi - lo))
            nll = float(np.mean(0.5 * np.log(2 * np.pi * s ** 2)
                                + (y - mu) ** 2 / (2 * s ** 2)))
            return {'picp90': picp, 'mpiw90': mpiw, 'nll': nll}

        per_seed.append({'file': npz.name,
                         'uncalibrated': interval_metrics(1.0),
                         'calibrated': interval_metrics(c_star)})

    agg = {}
    for kind in ('uncalibrated', 'calibrated'):
        for m in ('picp90', 'mpiw90', 'nll'):
            vals = [r[kind][m] for r in per_seed]
            agg[f'{kind}_{m}_mean'] = float(np.mean(vals))
            agg[f'{kind}_{m}_std'] = float(np.std(vals))

    print(f"\nuncalibrated: PICP90={agg['uncalibrated_picp90_mean']:.1f}% "
          f"MPIW={agg['uncalibrated_mpiw90_mean']:.3f} "
          f"NLL={agg['uncalibrated_nll_mean']:.3f}")
    print(f"calibrated  : PICP90={agg['calibrated_picp90_mean']:.1f}% "
          f"MPIW={agg['calibrated_mpiw90_mean']:.3f} "
          f"NLL={agg['calibrated_nll_mean']:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'c_star': c_star, 'per_run_scales': scales,
                   'final_dir': args.final_dir,
                   'per_seed': per_seed, 'aggregated': agg}, f, indent=2)
    print(f'saved {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
