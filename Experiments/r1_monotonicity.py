"""
r1_monotonicity.py — Revision R1, exploratory analysis (not in the analysis plan; registered as such):
does the monotonicity regularization (PGMC) change the monotonicity of the raw predictions?

The pre-registered ablations compare errors only. This script compares the shape of the raw
(unsmoothed) test predictions of the full model with the PGMC ablations, for both versions of
our model, with the bearing as the unit:
  non-increasing steps: share of consecutive raw predictions of a bearing with diff <= 0 (%)
  mean upward step:     mean of max(diff, 0) over the consecutive predictions of a bearing (RUL units)
Each quantity is averaged over the 20 seeds per bearing, then compared bearing by bearing with the
exact two-sided sign-flip test (r1_strict_stats.signflip_p). p-values are unadjusted; the analysis
is descriptive and informs wording only.

USAGE:
  python Experiments/r1_monotonicity.py [--set all10] [--out outputs/r1_analysis/monotonicity.md]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
from r1_analysis import load_variant  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import SETS, signflip_p  # noqa: E402

MODELS = {'DSAFLite+PGMC with temporal context (adopted)': 'outputs/r1_final_temporal_B5',
          'DSAFLite+PGMC, submitted architecture': 'outputs/r1_final_B10'}
VARIANTS = {'full model': 'final20', 'without PGMC': 'abl20_no_pgmc',
            'without PGMC and DANN': 'abl20_neither', 'PGMC batch ordering': 'abl20_pgmc_soft'}


def shape_per_bearing(variant, bearings):
    """seed-mean per bearing of the share of non-increasing steps and the mean upward step"""
    out = {'nonincreasing_pct': [], 'mean_up_step': []}
    for b in bearings:
        diffs = [np.diff(variant[s][b]['raw']) for s in sorted(variant)]
        out['nonincreasing_pct'].append(float(np.mean([100.0 * np.mean(d <= 0) for d in diffs])))
        out['mean_up_step'].append(float(np.mean([np.mean(np.clip(d, 0, None)) for d in diffs])))
    return {k: np.array(v) for k, v in out.items()}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='all10', choices=sorted(SETS))
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'monotonicity.md'))
    args = ap.parse_args()
    bearings = SETS[args.set]

    report = {'set': args.set, 'bearings': bearings, 'status': 'exploratory (not in the analysis plan)',
              'models': {}}
    lines = [f'# Shape of the raw predictions with and without the monotonicity regularization '
             f'({args.set}, {len(bearings)} bearings; exploratory)', '',
             'Per bearing: seed mean (20 seeds) of the share of non-increasing consecutive raw predictions '
             'and of the mean upward step. Δ: variant minus full model, averaged over bearings. '
             'p: exact two-sided sign-flip test over bearings, unadjusted.', '',
             '| Model | Variant | non-increasing steps (%) | Δ (pp) | full model higher on | p | '
             'mean upward step | Δ (%) | full model smaller on | p |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for model, root in MODELS.items():
        shapes = {v: shape_per_bearing(load_variant(ROOT / root / d), bearings) for v, d in VARIANTS.items()}
        full = shapes['full model']
        report['models'][model] = {'root': root, 'variants': {}}
        for v, s in shapes.items():
            entry = {k: {'per_bearing': s[k].tolist(), 'mean': float(s[k].mean())} for k in s}
            cells = ['–', '–', '–', '–', '–', '–']
            if v != 'full model':
                d_pct = s['nonincreasing_pct'] - full['nonincreasing_pct']
                d_up = s['mean_up_step'] - full['mean_up_step']
                entry['vs_full'] = {
                    'nonincreasing_pp': float(d_pct.mean()),
                    'full_higher_on': int((d_pct < 0).sum()), 'p_nonincreasing': signflip_p(d_pct),
                    'up_step_rel_pct': float(100 * d_up.mean() / s['mean_up_step'].mean()),
                    'full_smaller_on': int((d_up > 0).sum()), 'p_up_step': signflip_p(d_up)}
                e = entry['vs_full']
                cells = [f"{e['nonincreasing_pp']:+.2f}", f"{e['full_higher_on']}/{len(bearings)}",
                         f"{e['p_nonincreasing']:.3f}", f"{e['up_step_rel_pct']:+.1f}",
                         f"{e['full_smaller_on']}/{len(bearings)}", f"{e['p_up_step']:.3f}"]
            report['models'][model]['variants'][v] = entry
            lines.append(f"| {model} | {v} | {s['nonincreasing_pct'].mean():.2f} | {cells[0]} | {cells[1]} | "
                         f"{cells[2]} | {s['mean_up_step'].mean():.5f} | {cells[3]} | {cells[4]} | {cells[5]} |")
    lines += ['', 'Δ (%) for the mean upward step is relative to the variant: (variant − full) / variant.']
    report['provenance'] = provenance(('Experiments/r1_monotonicity.py', 'Experiments/r1_strict_stats.py',
                                       'Experiments/r1_analysis.py'))
    text = '\n'.join(lines) + '\n'
    print(text)
    Path(args.out).write_text(text, encoding='utf-8')
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'written {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
