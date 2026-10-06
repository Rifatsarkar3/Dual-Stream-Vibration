"""
r1_ensembles.py — Revision R1, analysis plan Section 7 (B1): seed ensembles for every model,
post hoc on the saved test predictions. Nothing is trained or selected here.

For each model, bearing-averaged raw and CES MAE on all10, orig4, new6 and zs3:
  single model: mean over the 20 final seeds;
  5-member: the 20 final seeds in four disjoint groups of five, in seed-list order; member
    raw predictions are averaged and CES is applied to the averaged prediction; metrics are
    averaged over the four groups;
  20-member: all 20 seeds.
Single models stay the primary comparison (Section 8). As a secondary check, the comparison
of our model with each baseline is repeated at the bearing level for 20-member ensembles
(exact two-sided sign-flip test, Holm across the baselines), to show whether the conclusion
depends on seed averaging at the model level.

USAGE:
  python Experiments/r1_ensembles.py [--provisional]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
from r1_analysis import load_variant, variants_of  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import REF, SETS, baseline_paths, holm, n_preds, signflip_p  # noqa: E402

FINAL_SEEDS = [42, 123, 777, 999, 7, 8, 21, 34, 55, 89, 144, 233, 377, 610, 987,
               1597, 2584, 4181, 6765, 10946]
GROUPS = [FINAL_SEEDS[i:i + 5] for i in range(0, 20, 5)]


def bearing_mae(raw, y):
    v = variants_of(raw)
    return {m: float(np.mean(np.abs(v[m] - y))) for m in ('raw', 'ces')}


def per_bearing(variant, bearings, members):
    """Per-bearing MAE of the ensemble of `members` (one seed = single model)."""
    return {b: bearing_mae(np.mean([variant[s][b]['raw'] for s in members], axis=0),
                           variant[members[0]][b]['y']) for b in bearings}


def summarize(variant, bearings):
    single = [per_bearing(variant, bearings, [s]) for s in FINAL_SEEDS]
    five = [per_bearing(variant, bearings, g) for g in GROUPS]
    twenty = per_bearing(variant, bearings, FINAL_SEEDS)
    out = {}
    for m in ('raw', 'ces'):
        out[m] = {'single': float(np.mean([np.mean([r[b][m] for b in bearings]) for r in single])),
                  '5-member': float(np.mean([np.mean([r[b][m] for b in bearings]) for r in five])),
                  '20-member': float(np.mean([twenty[b][m] for b in bearings]))}
    return out, twenty


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'ensembles.json'))
    ap.add_argument('--provisional', action='store_true',
                    help='allow an incomplete baseline family (labelled provisional)')
    ap.add_argument('--ref', default=REF, help='our model predictions (20 seeds)')
    args = ap.parse_args()

    paths = baseline_paths()
    ready = {k: v for k, v in paths.items() if n_preds(v) == 20}
    missing = sorted(set(paths) - set(ready))
    if missing and not args.provisional:
        raise SystemExit(f'baseline family incomplete (missing: {missing}); use --provisional')
    models = {'DSAFLite+PGMC (ours)': args.ref, **ready}
    loaded = {k: load_variant(ROOT / v) for k, v in models.items()}
    for k, v in loaded.items():
        if sorted(v) != sorted(FINAL_SEEDS):
            raise SystemExit(f'{k}: seeds differ from the final seed list')

    report = {'status': 'complete' if not missing else 'PROVISIONAL, missing ' + ', '.join(missing),
              'models': models, 'groups': GROUPS, 'sets': {}}
    lines = [f"# Seed ensembles (plan B1), status: {report['status']}", '',
             'Bearing-averaged MAE. 5-member: mean over four disjoint groups of five final seeds. '
             'CES is applied to the ensemble-mean prediction.']
    for set_name in ('all10', 'orig4', 'new6', 'zs3'):
        bearings = SETS[set_name]
        summ, twenty = {}, {}
        for k, v in loaded.items():
            summ[k], twenty[k] = summarize(v, bearings)
        ref = 'DSAFLite+PGMC (ours)'
        tests = {}
        for k in loaded:
            if k == ref:
                continue
            d = np.array([twenty[k][b]['ces'] - twenty[ref][b]['ces'] for b in bearings])
            tests[k] = {'k': int((d > 0).sum()), 'n': len(d), 'p_flip': signflip_p(d),
                        'rel': 100 * float(d.mean() / np.mean([twenty[k][b]['ces'] for b in bearings]))}
        adj = holm(np.array([t['p_flip'] for t in tests.values()])) if tests else []
        for (k, t), pa in zip(tests.items(), adj):
            t['p_holm'] = float(pa)
        report['sets'][set_name] = {'mae': summ, 'twenty_member_vs_ours_ces': tests}
        lines += ['', f'## {set_name} ({len(bearings)} bearings)', '',
                  '| Model | raw single | raw 5-member | raw 20-member | CES single | CES 5-member | '
                  'CES 20-member | 20-member: ours lower by (CES) | ours better on | flip p | Holm p |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for k in loaded:
            s = summ[k]
            t = tests.get(k)
            cmp_cols = (f"{t['rel']:+.1f} % | {t['k']}/{t['n']} | {t['p_flip']:.4f} | {t['p_holm']:.4f}"
                        if t else '– | – | – | –')
            lines.append(f"| {k} | {s['raw']['single']:.4f} | {s['raw']['5-member']:.4f} | "
                         f"{s['raw']['20-member']:.4f} | {s['ces']['single']:.4f} | {s['ces']['5-member']:.4f} | "
                         f"{s['ces']['20-member']:.4f} | {cmp_cols} |")
    report['provenance'] = provenance(('Experiments/r1_ensembles.py', 'Experiments/r1_strict_stats.py',
                                       'Experiments/r1_analysis.py'))
    text = '\n'.join(lines) + '\n'
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding='utf-8')
    Path(args.out).with_suffix('.md').write_text(text, encoding='utf-8')
    print(f'written {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
