"""
r1_xjtu_calibration.py — Revision R1, analysis plan Amendment 1, E5: interval calibration on XJTU-SY
with validation data only, for both versions of our model.

Validation source: final-epoch MC-dropout predictions of the four-fold XJTU-SY selection runs at the
selected setting (B 5, lambda 0.08, five seeds; outputs/r1_xjtu_calib/<version>/lam0.08_B5_fold*,
written by r1_xjtu_calib_runs.sh). Factors as in plan Section 4 (r1_calibration functions): c* = mean
over the 20 runs of the 90 % quantile of |y - mu| / (z s); k* = mean of the 90 % quantile of
(mu - y) / s. Applied once to the same version's final test predictions on the six XJTU-SY test
bearings (the zero-shot prediction files store no MC-dropout arrays). Reported: two-sided and
one-sided coverage, uncalibrated and calibrated, pooled, per bearing and per true-RUL band.
Check for the adopted variant: the repeated selection runs must reproduce the final-epoch validation
MAE of its original selection runs exactly (deterministic kernels).

USAGE:
  python Experiments/r1_xjtu_calibration.py [--out outputs/r1_analysis/xjtu_calibration.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import r1_xjtu_stats as xj  # noqa: E402
from r1_calibration import FOLDS, Z, c2, k1, load_test, load_val, test_coverage  # noqa: E402
from r1_provenance import provenance  # noqa: E402

VERSIONS = {'adopted': ('outputs/r1_xjtu_calib/temporal', xj.MODELS[xj.ADOPTED], 'outputs/r1_temporal_xjtu'),
            'submitted': ('outputs/r1_xjtu_calib/submitted', xj.MODELS[xj.SUBMITTED], 'outputs/r1_xjtu_lobo')}


def final_val(path):
    d = json.loads((ROOT / path / 'results.json').read_text(encoding='utf-8'))
    return {r['seed']: r['val_history'][-1] for r in d['per_seed']}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'xjtu_calibration.json'))
    args = ap.parse_args()
    bearings = xj.SETS['xjtu_test']
    out, lines = {'bearings': bearings, 'versions': {}}, [
        '# XJTU-SY interval calibration, validation data only (revision R1, Amendment 1, E5)', '',
        'Coverage of the 90 % intervals on the six XJTU-SY test bearings (pooled over 20 seeds and all windows). '
        'Two-sided: mu ± z c s; one-sided lower bound: mu − k s (k = k*) or mu − z c* s.', '',
        '| Version | c* | k* | uncalibrated two-sided | calibrated two-sided | late life [0, 0.2) | lower bound k* | '
        'lower bound c* | per bearing, calibrated two-sided |', '|---|---|---|---|---|---|---|---|---|']
    for version, (val_root, test_dir, original) in VERSIONS.items():
        val = load_val(val_root, '0.08', 5)
        test = load_test(test_dir)
        if len(test) != 20:
            raise SystemExit(f'{version}: expected 20 test seeds, found {len(test)}')
        # replay check against the original selection runs (exact for deterministic originals)
        diffs = []
        for f in FOLDS:
            name = f'lam0.08_B5_fold{f}' if version == 'adopted' else f'sweepseq_B5_fold{f}'
            new, old = final_val(f'{val_root}/lam0.08_B5_fold{f}'), final_val(f'{original}/{name}')
            diffs += [abs(new[s] - old[s]) for s in new]
        c_star = float(np.mean([c2(r['y'], r['mu'], r['s']) for r in val]))
        k_star = float(np.mean([k1(r['y'], r['mu'], r['s']) for r in val]))
        res = {'c_star': c_star, 'k_star': k_star,
               'c2_per_run_sd': float(np.std([c2(r['y'], r['mu'], r['s']) for r in val])),
               'replay_max_abs_diff_final_val_mae': float(max(diffs)),
               'replay_reference': original,
               'uncalibrated_two_sided': test_coverage(test, bearings, lambda d: (d['mu'] - Z * d['s'], d['mu'] + Z * d['s'])),
               'two_sided_c_star': test_coverage(test, bearings, lambda d: (d['mu'] - Z * c_star * d['s'],
                                                                            d['mu'] + Z * c_star * d['s'])),
               'lower_bound_k_star': test_coverage(test, bearings, lambda d: (d['mu'] - k_star * d['s'], None)),
               'lower_bound_c_star': test_coverage(test, bearings, lambda d: (d['mu'] - Z * c_star * d['s'], None))}
        out['versions'][version] = res
        t = res['two_sided_c_star']
        lines.append(f"| {version} | {c_star:.3f} | {k_star:.3f} | {res['uncalibrated_two_sided']['pooled']:.1f} % | "
                     f"{t['pooled']:.1f} % | {t['late [0, 0.2)']:.1f} % | {res['lower_bound_k_star']['pooled']:.1f} % | "
                     f"{res['lower_bound_c_star']['pooled']:.1f} % | "
                     + ', '.join(f"{b.split('_', 2)[-1]} {v:.0f}" for b, v in t['per_bearing'].items()) + ' |')
        lines.append('')
    lines += ['Replay check (final-epoch validation MAE of the repeated selection runs against the originals): '
              + '; '.join(f"{v}: max |difference| {r['replay_max_abs_diff_final_val_mae']:.2e} "
                          f"({'deterministic original, must be 0' if v == 'adopted' else 'original run with non-deterministic kernels'})"
                          for v, r in out['versions'].items()) + '.']
    out['provenance'] = provenance(('Experiments/r1_xjtu_calibration.py', 'Experiments/r1_calibration.py'))
    Path(args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    Path(args.out).with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'written {args.out}')
    if out['versions']['adopted']['replay_max_abs_diff_final_val_mae'] > 1e-9:
        raise SystemExit('adopted variant: repeated selection runs do not reproduce the originals')
    return 0


if __name__ == '__main__':
    sys.exit(main())
