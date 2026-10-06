"""
r1_components_smoke_check.py — Revision R1, analysis plan Amendment 10: pass criteria of the smoke
test (R, fold A, seed 42, 10 epochs, validation only):
  1. no NaN in the validation history or the final-epoch validation predictions;
  2. final-epoch validation MAE below the MAE of predicting the mean training label (pooled over
     the fold's three training bearings) on the validation bearing;
  3. standard deviation of the final-epoch validation predictions above 0.02.

USAGE:
  python Experiments/r1_components_smoke_check.py [--run outputs/r1_components_smoke/full_foldA]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='outputs/r1_components_smoke/full_foldA')
    args = ap.parse_args()
    from Experiments.exp16_ress import load_bearings
    run = ROOT / args.run
    res = json.loads((run / 'results.json').read_text(encoding='utf-8'))
    val_name = res['config']['val_bearings'][0]
    train_names = [n for n in ('train_01_Bearing1_1', 'train_02_Bearing1_2', 'train_03_Bearing2_1',
                               'train_04_Bearing2_2') if n != val_name]
    train = load_bearings(str(ROOT / res['config']['data_dir']), train_names)
    mean_label = float(np.concatenate([b.rul for b in train]).mean())

    out = {'run': args.run, 'validation_bearing': val_name, 'seeds': []}
    ok_all = True
    for r in res['per_seed']:
        seed = r['seed']
        vp = np.load(run / f'valpreds_seed{seed}.npz')
        raw, tgt = vp[f'raw_{val_name}'], vp[f'tgt_{val_name}']
        hist = np.array(r['val_history'], dtype=float)
        checks = {
            'no_nan': bool(np.isfinite(hist).all() and np.isfinite(raw).all()
                           and np.isfinite(vp[f'mcmean_{val_name}']).all()),
            'val_mae': float(hist[-1]),
            'mean_predictor_mae': float(np.mean(np.abs(mean_label - tgt))),
            'pred_sd': float(raw.std()),
        }
        checks['beats_mean_predictor'] = checks['val_mae'] < checks['mean_predictor_mae']
        checks['sd_above_0.02'] = checks['pred_sd'] > 0.02
        checks['pass'] = checks['no_nan'] and checks['beats_mean_predictor'] and checks['sd_above_0.02']
        ok_all &= checks['pass']
        out['seeds'].append({'seed': seed, **checks})
        print(f"seed {seed}: val MAE {checks['val_mae']:.4f} (mean predictor "
              f"{checks['mean_predictor_mae']:.4f}), prediction SD {checks['pred_sd']:.4f}, "
              f"no NaN {checks['no_nan']} -> {'PASS' if checks['pass'] else 'FAIL'}")
    out['pass'] = bool(ok_all)
    out['mean_training_label'] = mean_label
    (run / 'smoke_check.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('SMOKE TEST', 'PASSED' if ok_all else 'FAILED')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
