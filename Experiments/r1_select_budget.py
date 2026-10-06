"""
r1_select_budget.py — Revision R1, analysis plan Sections 3.1 and 7 (B2): budget selection
by the manuscript rule, argmin over B of the mean final-epoch validation MAE of the 20
fold-seed runs (4 leave-one-bearing-out folds x 5 selection seeds), reading
<root>/lam{lambda}_B{B}_fold{F}/results.json. For B2 it also applies the pre-registered
adoption rule: the variant replaces the current model only if its 4-fold mean is lower
than the current model's by more than the current model's SE.

USAGE:
  python Experiments/r1_select_budget.py --root outputs/r1_temporal [--lam 0.08]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = [5, 10, 20, 40]
FOLDS = 'ABCD'
CURRENT = {'mean': 0.1510, 'se': 0.0078, 'source': 'outputs/r1_lobo/selection_4fold.txt (B=10, lambda=0.08)'}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--lam', default='0.08')
    ap.add_argument('--no-adoption', action='store_true',
                    help='budget rule only (e.g. XJTU-SY), without the PRONOSTIA B2 adoption check')
    args = ap.parse_args()
    root = ROOT / args.root
    scores = {}
    for B in BUDGETS:
        vals = []
        for f in FOLDS:
            d = json.loads((root / f'lam{args.lam}_B{B}_fold{f}' / 'results.json').read_text(encoding='utf-8'))
            vals += [r['val_history'][-1] for r in d['per_seed']]
        if len(vals) != 20:
            raise SystemExit(f'B={B}: expected 20 fold-seed runs, found {len(vals)}')
        v = np.array(vals)
        scores[B] = {'mean': float(v.mean()), 'se': float(v.std(ddof=1) / np.sqrt(len(v))), 'n': len(v)}
    best = min(BUDGETS, key=lambda b: scores[b]['mean'])
    adopt = scores[best]['mean'] < CURRENT['mean'] - CURRENT['se']
    lines = [f"B={B:2d}: 4-fold={scores[B]['mean']:.4f}  SE={scores[B]['se']:.4f}  n={scores[B]['n']}" for B in BUDGETS]
    lines += [f'SELECTED B* = {best}']
    if not args.no_adoption:
        lines.append(f"adoption threshold (current {CURRENT['mean']:.4f} - SE {CURRENT['se']:.4f}) = "
                     f"{CURRENT['mean'] - CURRENT['se']:.4f}; variant at B*: {scores[best]['mean']:.4f} -> "
                     + ('better by more than one SE (candidate for adoption after the lambda rule)' if adopt
                        else 'NOT better by more than one SE: not adopted'))
    text = '\n'.join(lines)
    print(text)
    (root / 'budget_selection.txt').write_text(text + '\n', encoding='utf-8')
    (root / 'budget_selection.json').write_text(json.dumps(
        {'scores': {str(b): s for b, s in scores.items()}, 'selected': best,
         **({} if args.no_adoption else {'current_model': CURRENT,
                                         'better_by_more_than_one_se': bool(adopt)})},
        indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
