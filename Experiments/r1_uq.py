"""
r1_uq.py — Revision R1: leak-free interval calibration from the 4-fold
leave-one-bearing-out models (validation bearings only), applied to the
re-evaluated test predictions.

Two calibration targets, both fitted on held-out TRAINING bearings:
  c*  two-sided: per run, the 0.90 quantile of |y - mu| / (z s); c* = mean over runs
      (the exp22 procedure; z = 1.6449). Target metric: PICP of mu +- z c* s.
  k*  one-sided: per run, the 0.90 quantile of (mu - y) / s; k* = mean over runs.
      Target metric: LBC = P(y >= mu - k* s), the quantity the lower-bound
      maintenance trigger relies on.
Coverage is reported pooled, per bearing and per health stage (y > 0.5,
0.2 < y <= 0.5, y <= 0.2).

USAGE:
  python Experiments/r1_uq.py --budget 5 --final-preds outputs/r1_reeval/final20
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Experiments.exp16_ress import CONFIG, load_bearings, make_model  # noqa: E402
from Experiments.exp22_uq_calibration import mc_predict  # noqa: E402
from Experiments.r1_analysis import SETS  # noqa: E402

Z = 1.6449
FOLDS = {  # fold -> (run dir template, validation bearing)
    'A': ('outputs/exp16_ress/sweepseq_B{B}_foldA', 'train_02_Bearing1_2'),
    'B': ('outputs/exp16_ress/sweepseq_B{B}_foldB', 'train_04_Bearing2_2'),
    'C': ('outputs/r1_lobo/sweepseq_B{B}_foldC', 'train_01_Bearing1_1'),
    'D': ('outputs/r1_lobo/sweepseq_B{B}_foldD', 'train_03_Bearing2_1'),
}
STAGES = {'early (y>0.5)': (0.5, 1.01), 'mid (0.2<y<=0.5)': (0.2, 0.5), 'late (y<=0.2)': (-0.01, 0.2)}


def fold_scales(budget, folds, data_dir, device):
    runs = []
    for f in folds:
        run_dir, val_bearing = FOLDS[f]
        bearing = load_bearings(data_dir, [val_bearing])[0]
        for ck in sorted((ROOT / run_dir.format(B=budget)).glob('model_seed*.pth')):
            torch.manual_seed(0)
            model = make_model(CONFIG, device)
            model.load_state_dict(torch.load(ck, map_location=device))
            mu, sd = mc_predict(model, bearing.vib, device)
            sd = np.clip(sd, 1e-4, None)
            runs.append({'fold': f, 'checkpoint': ck.name,
                         'c2': float(np.quantile(np.abs(bearing.rul - mu) / (Z * sd), 0.90)),
                         'k1': float(np.quantile((mu - bearing.rul) / sd, 0.90))})
    return runs


def coverage(final_dir, bearings, c2, k1):
    per_seed = []
    for f in sorted(glob.glob(str(ROOT / final_dir / 'preds_seed*.npz'))):
        d = np.load(f)
        ys, mus, sds, names = [], [], [], []
        for b in bearings:
            ys.append(d[f'tgt_{b}'].astype(float))
            mus.append(d[f'mcmean_{b}'].astype(float))
            sds.append(np.clip(d[f'mcstd_{b}'].astype(float), 1e-4, None))
            names += [b] * len(ys[-1])
        y, mu, sd, names = np.concatenate(ys), np.concatenate(mus), np.concatenate(sds), np.array(names)
        picp_hit = (y >= mu - Z * c2 * sd) & (y <= mu + Z * c2 * sd)
        lbc_c_hit = y >= mu - Z * c2 * sd
        lbc_k_hit = y >= mu - k1 * sd
        res = {'picp_c': 100 * picp_hit.mean(), 'lbc_c': 100 * lbc_c_hit.mean(),
               'lbc_k': 100 * lbc_k_hit.mean(), 'mpiw_c': float(np.mean(2 * Z * c2 * sd)),
               'stage_lbc_k': {}, 'bearing_lbc_k': {}}
        for s, (lo, hi) in STAGES.items():
            m = (y > lo) & (y <= hi)
            res['stage_lbc_k'][s] = 100 * lbc_k_hit[m].mean()
        for b in bearings:
            res['bearing_lbc_k'][b] = 100 * lbc_k_hit[names == b].mean()
        per_seed.append(res)

    def mean(key, sub=None):
        vals = [r[key][sub] if sub else r[key] for r in per_seed]
        return round(float(np.mean(vals)), 2)
    return {'picp_c': mean('picp_c'), 'lbc_c': mean('lbc_c'), 'lbc_k': mean('lbc_k'),
            'mpiw_c': mean('mpiw_c'),
            'stage_lbc_k': {s: mean('stage_lbc_k', s) for s in STAGES},
            'bearing_lbc_k': {b: mean('bearing_lbc_k', b) for b in bearings}}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=5)
    ap.add_argument('--final-preds', default='outputs/r1_reeval/final20')
    ap.add_argument('--data-dir', default=str(ROOT / 'Processed_PRONOSTIA_v2'))
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'uq.json'))
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    runs = fold_scales(args.budget, 'ABCD', args.data_dir, device)
    out = {'budget': args.budget, 'final_preds': args.final_preds, 'runs': runs, 'calibrations': {}}
    for label, folds in (('2fold_AB', 'AB'), ('4fold', 'ABCD')):
        sel = [r for r in runs if r['fold'] in folds]
        c2 = float(np.mean([r['c2'] for r in sel]))
        k1 = float(np.mean([r['k1'] for r in sel]))
        entry = {'n_runs': len(sel), 'c_star': c2, 'c_star_sd': float(np.std([r['c2'] for r in sel])),
                 'k_star': k1, 'k_star_sd': float(np.std([r['k1'] for r in sel])), 'test': {}}
        for set_name in ('orig4', 'all10'):
            entry['test'][set_name] = coverage(args.final_preds, SETS[set_name], c2, k1)
        out['calibrations'][label] = entry
        print(f"\n{label}: c*={c2:.3f} (sd {entry['c_star_sd']:.3f}), k*={k1:.3f} "
              f"(sd {entry['k_star_sd']:.3f}), {len(sel)} runs")
        for set_name, t in entry['test'].items():
            print(f"  {set_name}: PICP(c*)={t['picp_c']:.1f}%  LBC(c*)={t['lbc_c']:.1f}%  "
                  f"LBC(k*)={t['lbc_k']:.1f}%  MPIW={t['mpiw_c']:.3f}  stages(k*)={t['stage_lbc_k']}")
    unc = coverage(args.final_preds, SETS['all10'], 1.0, Z)
    out['uncalibrated_all10'] = unc
    print(f"\nuncalibrated all10: PICP={unc['picp_c']:.1f}%  LBC={unc['lbc_c']:.1f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f'written {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
