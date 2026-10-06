"""
r1_select_lambda.py — Revision R1, analysis plan Sections 3.2 and 3.4.

Scores each lambda_PGMC by the mean final-epoch validation MAE over its 20 fold-seed
runs (4 leave-one-bearing-out folds x 5 selection seeds), SE = sd / sqrt(n), and applies
the pre-registered one-standard-error rule:
  keep lambda = 0.08 if m(0.08) <= min m + SE(argmin);
  otherwise take the smallest lambda with m(lambda) <= min m + SE(argmin).
Section 3.4: the lambda = 0.08 runs repeat the existing budget-sweep runs at the same
budget; the largest absolute difference in final-epoch validation MAE is reported.

USAGE:
  python Experiments/r1_select_lambda.py [--budget 10] [--root outputs/r1_lambda]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LAMBDAS = [0.08, 0.2, 0.5, 1.0, 2.0, 5.0]
DEFAULT = 0.08
FOLDS = 'ABCD'
PRIOR = {'A': 'outputs/exp16_ress/sweepseq_B{B}_foldA', 'B': 'outputs/exp16_ress/sweepseq_B{B}_foldB',
         'C': 'outputs/r1_lobo/sweepseq_B{B}_foldC', 'D': 'outputs/r1_lobo/sweepseq_B{B}_foldD'}


def final_val(run_dir: Path) -> dict:
    d = json.loads((run_dir / 'results.json').read_text(encoding='utf-8'))
    return {r['seed']: r['val_history'][-1] for r in d['per_seed']}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=10)
    ap.add_argument('--root', default='outputs/r1_lambda')
    ap.add_argument('--no-repro-check', action='store_true',
                    help='skip the Section 3.4 comparison, which only applies to the submitted model '
                         '(use for other variants, e.g. B2 under outputs/r1_temporal)')
    args = ap.parse_args()
    root = ROOT / args.root

    scores, per_fold = {}, {}
    for lam in LAMBDAS:
        vals, per_fold[lam] = [], {}
        for f in FOLDS:
            v = final_val(root / f'lam{lam}_B{args.budget}_fold{f}')
            per_fold[lam][f] = float(np.mean(list(v.values())))
            vals += list(v.values())
        if len(vals) != 20:
            raise SystemExit(f'lambda {lam}: expected 20 fold-seed runs, found {len(vals)}')
        vals = np.array(vals)
        scores[lam] = {'mean': float(vals.mean()),
                       'se': float(vals.std(ddof=1) / np.sqrt(len(vals))), 'n': len(vals)}

    best = min(LAMBDAS, key=lambda lam: scores[lam]['mean'])
    threshold = scores[best]['mean'] + scores[best]['se']
    if scores[DEFAULT]['mean'] <= threshold:
        selected, reason = DEFAULT, 'default within one SE of the minimum: kept'
    else:
        selected = min(lam for lam in LAMBDAS if scores[lam]['mean'] <= threshold)
        reason = 'default beaten by more than one SE: smallest lambda within one SE of the minimum'

    lines = [f"lambda={lam:<4}  " + '  '.join(f"{f}={per_fold[lam][f]:.4f}" for f in FOLDS)
             + f"  4-fold={scores[lam]['mean']:.4f}  SE={scores[lam]['se']:.4f}  n={scores[lam]['n']}"
             for lam in LAMBDAS]
    lines += [f'argmin lambda = {best}; threshold (min + SE) = {threshold:.4f}',
              f'SELECTED lambda (pre-registered one-SE rule) = {selected}  [{reason}]']

    diffs = []
    for f in ('' if args.no_repro_check else FOLDS):
        new = final_val(root / f'lam{DEFAULT}_B{args.budget}_fold{f}')
        old_dir = ROOT / PRIOR[f].format(B=args.budget)
        if (old_dir / 'results.json').exists():
            old = final_val(old_dir)
            diffs += [abs(new[s] - old[s]) for s in new if s in old]
    if diffs:
        lines.append(f'reproducibility (Section 3.4): {len(diffs)} repeated runs, '
                     f'max |delta val MAE| = {max(diffs):.5f}, mean = {np.mean(diffs):.5f}')

    text = '\n'.join(lines)
    print(text)
    (root / 'selection.txt').write_text(text + '\n', encoding='utf-8')
    (root / 'selection.json').write_text(json.dumps(
        {'budget': args.budget, 'scores': {str(k): v for k, v in scores.items()},
         'per_fold': {str(k): v for k, v in per_fold.items()}, 'argmin': best,
         'threshold': threshold, 'selected': selected, 'reason': reason,
         'reproducibility_abs_diffs': diffs}, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
