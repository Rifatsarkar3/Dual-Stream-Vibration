"""
r1_baseline_select5.py — Revision R1, analysis plan Section 5: baseline budget
selection with the same five selection seeds as our model (7, 42, 123, 777, 999)
on all four leave-one-bearing-out folds (equal tuning effort).

Seeds 42 and 123 already exist and are reused (plan Section 5): folds A/B as the mean
of their four runs ('budget_scores' in <prior>/<model>/results.json), folds C/D as
individual runs in outputs/r1_lobo/baselines/<model>.json. This script adds seeds 7,
777 and 999 on all four folds with deterministic CUDA kernels and without loading test
bearings, then selects the budget by argmin of the mean over all 20 fold-seed runs:
    score(B) = (4 * mean_AB_old(B) + sum C/D old runs(B) + sum new runs(B)) / 20
Resumable: finished runs are kept in outputs/r1_baseline_sel5/<model>.json.

USAGE:
  python Experiments/r1_baseline_select5.py [--models cnn,bilstm,cnnlstm,resnet,transformer,tsmccnn]
"""

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Experiments.exp16_ress import CONFIG  # noqa: E402
from Experiments.exp19_baselines import train_baseline  # noqa: E402
from Experiments.r1_provenance import provenance  # noqa: E402

NEW_SEEDS = [7, 777, 999]
OLD_SEEDS = [42, 123]
BUDGETS = [5, 10, 20, 40]
FOLDS = {'foldA': 'train_02_Bearing1_2', 'foldB': 'train_04_Bearing2_2',
         'foldC': 'train_01_Bearing1_1', 'foldD': 'train_03_Bearing2_1'}
PRIOR_AB = {'cnn': 'outputs/exp19_baselines', 'bilstm': 'outputs/exp19_baselines',
            'cnnlstm': 'outputs/exp19_baselines', 'resnet': 'outputs/exp19_baselines',
            'transformer': 'outputs/exp19_baselines', 'tsmccnn': 'outputs/r1_published'}
CURRENT_K = {'cnn': 40, 'bilstm': 10, 'cnnlstm': 10, 'resnet': 10, 'transformer': 5,
             'tsmccnn': 40}   # 4-fold selection with seeds 42 and 123 (R1_corrected_results.md)


def data_dir(model):
    return str(ROOT / ('Processed_PRONOSTIA_v2hv' if model == 'tsmccnn' else 'Processed_PRONOSTIA_v2'))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='cnn,bilstm,cnnlstm,resnet,transformer,tsmccnn')
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs' / 'r1_baseline_sel5'))
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in [m.strip() for m in args.models.split(',')]:
        out_f = out_dir / f'{model}.json'
        state = json.loads(out_f.read_text()) if out_f.exists() else {'model': model, 'runs': []}
        done = {(r['budget'], r['fold'], r['seed']) for r in state['runs']}
        t0 = time.time()
        for B in BUDGETS:
            for fold, val_bearing in FOLDS.items():
                for seed in NEW_SEEDS:
                    if (B, fold, seed) in done:
                        continue
                    cfg = dict(CONFIG)
                    cfg.update({'data_dir': data_dir(model), 'val_mode': 'bearing',
                                'val_bearings': [val_bearing], 'epochs': B, 'patience': 10 ** 6,
                                'mc_dropout_passes': 5, 'skip_test': True, 'test_bearings': []})
                    res, _m = train_baseline(model, seed, cfg, device, verbose=False)
                    state['runs'].append({'budget': B, 'fold': fold, 'seed': seed,
                                          'final_val_mae': res['val_history'][-1],
                                          'val_history': res['val_history']})
                    out_f.write_text(json.dumps(state, indent=1))
            print(f'{model} B={B:2d}: new runs done ({(time.time() - t0) / 60:.1f} min)', flush=True)

        prior_ab = json.loads((ROOT / PRIOR_AB[model] / model / 'results.json').read_text())
        old_cd = json.loads((ROOT / 'outputs' / 'r1_lobo' / 'baselines' / f'{model}.json').read_text())
        scores, parts = {}, {}
        for B in BUDGETS:
            mean_ab = float(prior_ab['budget_scores'][str(B)])
            cd = [r['final_val_mae'] for r in old_cd['runs']
                  if r['budget'] == B and r['seed'] in OLD_SEEDS]
            new = [r['final_val_mae'] for r in state['runs'] if r['budget'] == B]
            if len(cd) != 4 or len(new) != 12:
                raise SystemExit(f'{model} B={B}: expected 4 old C/D and 12 new runs, '
                                 f'found {len(cd)} and {len(new)}')
            scores[B] = (4 * mean_ab + sum(cd) + sum(new)) / 20
            parts[B] = {'mean_AB_old_4runs': mean_ab, 'CD_old': cd, 'new_mean': float(np.mean(new))}
        k5 = min(scores, key=scores.get)
        state.update({'scores_5seed_4fold': {str(B): s for B, s in scores.items()},
                      'parts': {str(B): p for B, p in parts.items()},
                      'K_2seed_4fold': CURRENT_K[model], 'K_5seed_4fold': k5,
                      'budget_changed': k5 != CURRENT_K[model],
                      'date': time.strftime('%Y-%m-%d %H:%M'), 'provenance': provenance()})
        out_f.write_text(json.dumps(state, indent=1))
        print(f'==> {model}: ' + '  '.join(f'B={B}:{scores[B]:.4f}' for B in BUDGETS)
              + f'  K(2 seeds)={CURRENT_K[model]}  K(5 seeds)={k5}'
              + ('  CHANGED' if k5 != CURRENT_K[model] else ''), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
