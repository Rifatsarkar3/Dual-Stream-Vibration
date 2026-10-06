"""
r1_pooled_framework.py — Revision R1, analysis plan Amendment 5.

  A5-P1 (primary): pooled framework effect. Standard practice (in-bearing early-stopping checkpoints,
      raw trigger) against the framework (final runs, cost-matched trigger), seeds 42, 123, 777, 999, 7,
      tau = 0.20, all10, exactly as Amendment 2, P3. Per bearing b and model m the symmetric relative
      difference d = (L_std - L_fw) / (L_std + L_fw) (positive: framework cheaper); D_b = mean of d over
      the eight architectures (submitted architecture and the seven baselines); exact sign-flip test of D
      over the ten bearings at r = 5 and 10, Holm over the two ratios.
      Secondary: nine models; selection and decision layer separately; absolute differences; per-bearing
      values. Check: per-model bearing-averaged losses reproduce Amendment 2, P3.
  A5-D (descriptive): raw (c = 1) against calibrated (c*) two-sided 90 % MC-dropout test intervals for
      the eight models with dropout, equal-terms predictions of Amendment 4.

USAGE:
  python Experiments/r1_pooled_framework.py
  python Experiments/r1_pooled_framework.py --dry-run     # synthetic data, scratch output, no real files read
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import r1_decision_models as dm  # noqa: E402  (Amendment 2; used unchanged)
import r1_decision_models_mc30 as dm30  # noqa: E402  (Amendment 4 paths)
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import SETS, holm, signflip_p  # noqa: E402

PRIMARY = [dm.SUBMITTED] + list(dm.BASELINES)        # eight architectures, the paper's model once
NINE = list(dm.VERSIONS) + list(dm.BASELINES)


def srd(a, b):
    """symmetric relative difference (a - b) / (a + b) per bearing; 0 where both are 0"""
    s = a + b
    return np.where(s > 0, (a - b) / np.where(s > 0, s, 1.0), 0.0)


def pooled(per, names, r, a_key, b_key, bearings, absolute=False):
    f = (lambda x, y: x - y) if absolute else srd
    d = np.mean([f(per[n][r][a_key], per[n][r][b_key]) for n in names], axis=0)
    return {'per_bearing': {b: float(v) for b, v in zip(bearings, d)}, 'mean': float(d.mean()),
            'k_positive': int((d > 0).sum()), 'n': int(len(d)), 'p_flip': float(signflip_p(d))}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'pooled_framework.md'))
    ap.add_argument('--dry-run', action='store_true', help='synthetic data; writes to a temporary folder')
    args = ap.parse_args()
    if args.dry_run:
        models = dm.synthetic_models(np.random.default_rng(0))
        out_path = Path(tempfile.mkdtemp(prefix='pooled_framework_dry_')) / 'pooled_framework.md'
    else:
        dm.baseline_paths = dm30.mc30_paths          # equal-terms predictions (Amendment 4)
        models = dm.real_models()
        out_path = Path(args.out)
    a10 = SETS['all10']
    names = NINE
    cal = {n: dm.calibrate(models[n]) for n in names}

    # ---- per-model, per-bearing losses on the five shared seeds (as Amendment 2, P3) ----------
    per = {}
    for n in names:
        t_es = dm.timing(models[n]['es'], a10, cal[n], dm.ES_SEEDS)[dm.TAU]
        t_fin = dm.timing(models[n]['final'], a10, cal[n], dm.ES_SEEDS)[dm.TAU]
        per[n] = {r: {'std': dm.losses(t_es, 'raw', r, a10), 'fw': dm.losses(t_fin, f'cmq{r}', r, a10),
                      'fin_raw': dm.losses(t_fin, 'raw', r, a10)} for r in dm.R_PRIMARY}

    check = None
    if not args.dry_run:
        p3 = json.loads((ROOT / 'outputs' / 'r1_analysis' / 'decision_models.json').read_text(encoding='utf-8'))['P3']
        diffs = [max(abs(per[n][r]['std'].mean() - p3[str(r)]['framework_vs_standard'][n]['other']),
                     abs(per[n][r]['fw'].mean() - p3[str(r)]['framework_vs_standard'][n]['ref']))
                 for n in names for r in dm.R_PRIMARY]
        check = {'max_abs_diff_vs_amendment2_P3': float(max(diffs)), 'pass': bool(max(diffs) == 0.0)}
        if not check['pass']:
            raise SystemExit(f'check failed: per-model losses differ from Amendment 2, P3 by {max(diffs):.3g}')

    # ---- A5-P1 primary and secondary -----------------------------------------------------------
    primary = {str(r): pooled(per, PRIMARY, r, 'std', 'fw', a10) for r in dm.R_PRIMARY}
    adj = holm(np.array([primary[str(r)]['p_flip'] for r in dm.R_PRIMARY]))
    for r, a in zip(dm.R_PRIMARY, adj):
        primary[str(r)]['p_holm'] = float(a)
    secondary = {
        'nine models (framework vs standard)': {str(r): pooled(per, NINE, r, 'std', 'fw', a10) for r in dm.R_PRIMARY},
        'selection only (final raw vs early-stopping raw)': {str(r): pooled(per, PRIMARY, r, 'std', 'fin_raw', a10)
                                                             for r in dm.R_PRIMARY},
        'decision layer only (cost-matched vs raw, final runs)': {str(r): pooled(per, PRIMARY, r, 'fin_raw', 'fw', a10)
                                                                  for r in dm.R_PRIMARY},
        'absolute difference, windows (framework vs standard)': {str(r): pooled(per, PRIMARY, r, 'std', 'fw', a10,
                                                                                absolute=True)
                                                                 for r in dm.R_PRIMARY},
    }

    # ---- A5-D: raw against calibrated intervals -------------------------------------------------
    drop = [n for n in names if models[n]['dropout']]
    a5d = {}
    for n in drop:
        c = cal[n]['c_star']
        seeds = sorted(models[n]['final'])
        a5d[n] = {'c_star': c,
                  'raw': dm.intervals(models[n]['final'], a10, seeds, lambda d: (d['mu'] - dm.Z * d['s'], d['mu'] + dm.Z * d['s'])),
                  'calibrated': dm.intervals(models[n]['final'], a10, seeds,
                                             lambda d, c=c: (d['mu'] - dm.Z * c * d['s'], d['mu'] + dm.Z * c * d['s']))}

    def summ(e):
        return (e['pooled_coverage'], e['late_coverage'], float(np.mean([e['per_bearing'][b]['width'] for b in a10])),
                float(np.mean([e['per_bearing'][b]['interval_score'] for b in a10])))

    lines = ['# Pooled framework effect and raw against calibrated intervals (revision R1, Amendment 5)', '',
             ('**DRY RUN on synthetic data: numbers are meaningless.**' if args.dry_run else
              'Generated by `Experiments/r1_pooled_framework.py`; definitions in analysis plan Section 11, Amendment 5.'),
             '', ('Check: per-model losses reproduce Amendment 2, P3 exactly.' if check else 'Check: not run (dry run).'),
             '', 'Ten test bearings (all10), seeds 42, 123, 777, 999 and 7, τ = 0.20. Per bearing, the symmetric relative '
             'difference (L_std − L_fw) / (L_std + L_fw), averaged over the models; positive: the framework is cheaper. '
             'p: exact sign-flip test over bearings (smallest attainable 0.002).', '',
             '## A5-P1 (primary): eight architectures, framework against standard practice', '',
             '| r | mean D | bearings with D > 0 | p | Holm over two |', '|---|---|---|---|---|']
    for r in dm.R_PRIMARY:
        e = primary[str(r)]
        lines.append(f"| {r} | {e['mean']:+.3f} | {e['k_positive']}/{e['n']} | {e['p_flip']:.3f} | {e['p_holm']:.3f} |")
    lines += ['', '| Bearing | ' + ' | '.join(f'D, r = {r}' for r in dm.R_PRIMARY) + ' |', '|---|' + '---|' * len(dm.R_PRIMARY)]
    for b in a10:
        lines.append(f'| {b} | ' + ' | '.join(f"{primary[str(r)]['per_bearing'][b]:+.3f}" for r in dm.R_PRIMARY) + ' |')
    lines += ['', '## Secondary', '', '| Family | r | mean | bearings > 0 | p |', '|---|---|---|---|---|']
    for fam, block in secondary.items():
        for r in dm.R_PRIMARY:
            e = block[str(r)]
            fmt = f"{e['mean']:+.1f}" if fam.startswith('absolute') else f"{e['mean']:+.3f}"
            lines.append(f"| {fam} | {r} | {fmt} | {e['k_positive']}/{e['n']} | {e['p_flip']:.3f} |")
    lines += ['', '## A5-D (descriptive): raw against calibrated two-sided 90 % MC-dropout test intervals', '',
              '| Model | c* | raw: coverage | raw: late life | raw: width | raw: interval score | calibrated: coverage | '
              'calibrated: late life | calibrated: width | calibrated: interval score |', '|---|---|---|---|---|---|---|---|---|---|']
    for n in drop:
        rc, rl, rw, ri = summ(a5d[n]['raw'])
        cc, cl, cw, ci = summ(a5d[n]['calibrated'])
        lines.append(f"| {n} | {a5d[n]['c_star']:.2f} | {rc:.1f} % | {rl:.1f} % | {rw:.4f} | {ri:.4f} | {cc:.1f} % | "
                     f"{cl:.1f} % | {cw:.4f} | {ci:.4f} |")

    report = {'dry_run': args.dry_run, 'check': check, 'primary_family': PRIMARY, 'A5_P1': primary,
              'secondary': secondary, 'A5_D': a5d,
              'per_model_per_bearing': {n: {str(r): {k: [float(x) for x in v] for k, v in per[n][r].items()}
                                            for r in dm.R_PRIMARY} for n in names},
              'provenance': provenance(('Experiments/r1_pooled_framework.py', 'Experiments/r1_decision_models.py',
                                        'Experiments/r1_decision_models_mc30.py', 'Experiments/r1_strict_stats.py'))}
    text = '\n'.join(lines) + '\n'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')
    out_path.with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(text)
    print(f'written {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
