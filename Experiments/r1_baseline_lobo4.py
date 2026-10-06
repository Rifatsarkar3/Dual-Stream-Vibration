"""
r1_baseline_lobo4.py — Revision R1, comment 2: extend the baselines' budget
selection from 2-fold to 4-fold leave-one-bearing-out, so the comparison keeps
using the same selection procedure as the proposed model.

Folds A/B (validate Bearing1_2 / Bearing2_2) were run in exp19 with seeds 42 and
123; their mean final-epoch validation MAE per budget is stored in
outputs/exp19_baselines/<model>/results.json ('budget_scores'). This script runs
folds C/D (validate Bearing1_1 / Bearing2_1) with the identical settings and
combines them: 4-fold score = (score_AB + score_CD) / 2 (equal run counts).
Validation data only; test bearings are never used for selection.

USAGE:
  python Experiments/r1_baseline_lobo4.py --models cnn,bilstm,cnnlstm,resnet,transformer
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Experiments.exp16_ress import CONFIG  # noqa: E402
from Experiments.exp19_baselines import SELECT_SEEDS, train_baseline  # noqa: E402

FOLDS_CD = {'foldC': ['train_01_Bearing1_1'], 'foldD': ['train_03_Bearing2_1']}
BUDGETS = [5, 10, 20, 40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='cnn,bilstm,cnnlstm,resnet,transformer')
    ap.add_argument('--data-dir', default=str(ROOT / 'Processed_PRONOSTIA_v2'))
    ap.add_argument('--out-dir', default=str(ROOT / 'outputs' / 'r1_lobo' / 'baselines'))
    ap.add_argument('--prior-root', default=str(ROOT / 'outputs' / 'exp19_baselines'),
                    help='directory holding <model>/results.json from the 2-fold (A/B) selection')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in [m.strip() for m in args.models.split(',')]:
        out_f = out_dir / f'{model_name}.json'
        if out_f.exists():
            print(f'{model_name}: done already, skipping', flush=True)
            continue
        prior = json.loads((Path(args.prior_root) / model_name / 'results.json').read_text())
        score_ab = {int(k): v for k, v in prior['budget_scores'].items()}
        runs, score_cd = [], {}
        t0 = time.time()
        for B in BUDGETS:
            finals = []
            for fold, val_bearings in FOLDS_CD.items():
                for seed in SELECT_SEEDS:
                    cfg = dict(CONFIG)
                    cfg.update({'data_dir': args.data_dir, 'val_mode': 'bearing',
                                'val_bearings': val_bearings, 'epochs': B,
                                'patience': 10 ** 6, 'mc_dropout_passes': 5})
                    res, _m = train_baseline(model_name, seed, cfg, device, verbose=False)
                    finals.append(res['val_history'][-1])
                    runs.append({'budget': B, 'fold': fold, 'seed': seed,
                                 'final_val_mae': res['val_history'][-1],
                                 'val_history': res['val_history']})
            score_cd[B] = float(np.mean(finals))
            print(f'{model_name} B={B:2d}: CD={score_cd[B]:.4f}  AB={score_ab[B]:.4f}', flush=True)
        score_4 = {B: (score_ab[B] + score_cd[B]) / 2 for B in BUDGETS}
        result = {
            'model': model_name, 'select_seeds': SELECT_SEEDS,
            'score_2fold_AB': score_ab, 'score_2fold_CD': score_cd, 'score_4fold': score_4,
            'K_2fold': prior['epoch_budget_K'], 'K_4fold': min(score_4, key=score_4.get),
            'runs': runs, 'minutes': round((time.time() - t0) / 60, 1),
            'date': time.strftime('%Y-%m-%d %H:%M'),
        }
        out_f.write_text(json.dumps(result, indent=2))
        print(f'==> {model_name}: K_2fold={result["K_2fold"]}  K_4fold={result["K_4fold"]}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
