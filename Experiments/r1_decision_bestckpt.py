"""
r1_decision_bestckpt.py — Revision R1, analysis plan Amendment 7, part B: calibrating on the checkpoint that
was selected on the calibration data. Eight PRONOSTIA architectures; the deployed models are fixed (each
architecture's 20 final runs, equal-terms test predictions of Amendment 4), so only the calibration data differ.

  BV  best-epoch predictions of the four-fold selection runs on their own held-out bearing (baselines: B7,
      r1_bestckpt_valpreds.py; our model: T6c of Amendment 6).
  FE  final-epoch predictions of the same runs (Amendment 2, V; our model: Section 4), as in the protocol.
  A7-P4  pooled: per bearing, mean over the eight architectures of |cov_BV - 90| - |cov_FE - 90| for the
         two-sided 90 % residual-quantile interval (positive: final-epoch calibration closer); sign-flip over
         the ten test bearings.
  A7-P5  per architecture, the same; Holm over the eight.
  Secondary: best epoch statistics; ratio of the calibration-data 0.9-quantiles of |error|; MC-dropout
         interval (seven architectures with dropout); late alarms and L(r) (descriptive).

USAGE:
  python Experiments/r1_decision_bestckpt.py
  python Experiments/r1_decision_bestckpt.py --dry-run     # synthetic data, scratch output, no real files read
"""

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import r1_decision_inbearing as di  # noqa: E402  (Amendment 6; used unchanged)
import r1_decision_models as dm  # noqa: E402  (Amendment 2; used unchanged)
import r1_decision_models_mc30 as dm30  # noqa: E402  (Amendment 4 paths)
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import SETS  # noqa: E402

ARCH = di.ARCH
BV_DIR = ROOT / 'outputs' / 'r1_bestckpt_valpreds'
T6C = ROOT / 'outputs' / 'r1_inbearing' / 't6c' / 'ours'


def ours_best_epochs():
    eps = []
    for f in 'ABCD':
        res = json.loads((ROOT / 'outputs' / 'r1_lambda' / f'lam0.08_B10_fold{f}' / 'results.json').read_text(encoding='utf-8'))
        eps += [(int(r['best_epoch']), int(r.get('epochs_trained') or len(r['val_history']))) for r in res['per_seed']
                if int(r['seed']) in dm.ES_SEEDS]
    return {'share_best_is_final': float(np.mean([b == e for b, e in eps])), 'mean_best_epoch': float(np.mean([b for b, _ in eps])),
            'K': 10, 'n_runs': len(eps)}


def real_data():
    dm.baseline_paths = dm30.mc30_paths
    models = dm.real_models()
    fe, bv, dep, drop, best_stats = {}, {}, {}, {}, {}
    for label, short in ARCH.items():
        if short == 'ours':
            m = json.loads((T6C / 'meta.json').read_text(encoding='utf-8'))
            if m['n_runs'] != 20 or m['max_abs_diff'] > 1e-4:
                raise SystemExit('T6c incomplete or failed its check')
            files = sorted(glob.glob(str(T6C / 'fold*_seed*.npz')))
            best_stats[label] = ours_best_epochs()
        else:
            m = json.loads((BV_DIR / short / 'meta.json').read_text(encoding='utf-8'))
            if m['n_runs'] != 20 or m['max_final_raw_diff_vs_V'] > 1e-6 or m['max_best_mae_vs_history_min'] > 1e-5:
                raise SystemExit(f'{short}: B7 incomplete or failed its checks')
            files = sorted(glob.glob(str(BV_DIR / short / 'fold*_seed*_best.npz')))
            best_stats[label] = {k: m[k] for k in ('share_best_is_final', 'mean_best_epoch', 'K', 'n_runs')}
        if len(files) != 20:
            raise SystemExit(f'{label}: expected 20 best-epoch validation runs, found {len(files)}')
        bv[label] = dm.load_val(files)
        fe[label] = models[label]['val']
        dep[label] = models[label]['final']
        drop[label] = label not in dm.NO_DROPOUT
    return fe, bv, dep, drop, best_stats


def synthetic_data(rng):
    syn = dm.synthetic_models(rng)
    fe, bv, dep, drop, best_stats = {}, {}, {}, {}, {}
    for label in ARCH:
        m = syn[label]
        fe[label], dep[label], drop[label] = m['val'], m['final'], m['dropout']
        bv[label] = [{'y': r['y'], 'raw': r['y'] + 0.8 * (r['raw'] - r['y']), 'mu': r['y'] + 0.8 * (r['mu'] - r['y']), 's': r['s']}
                     for r in m['val']]
        best_stats[label] = {'share_best_is_final': 0.5, 'mean_best_epoch': 5.0, 'K': 10, 'n_runs': 20}
    return fe, bv, dep, drop, best_stats


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'decision_bestckpt.md'))
    ap.add_argument('--dry-run', action='store_true', help='synthetic data; writes to a temporary folder')
    args = ap.parse_args()
    if args.dry_run:
        fe, bv, dep, drop, best_stats = synthetic_data(np.random.default_rng(0))
        out_path = Path(tempfile.mkdtemp(prefix='decision_bestckpt_dry_')) / 'decision_bestckpt.md'
    else:
        fe, bv, dep, drop, best_stats = real_data()
        out_path = Path(args.out)
    a10, names = SETS['all10'], list(ARCH)
    with_drop = [n for n in names if drop[n]]
    cal_fe = {n: dm.calibrate({'val': fe[n], 'dropout': drop[n]}) for n in names}
    cal_bv = {n: dm.calibrate({'val': bv[n], 'dropout': drop[n]}) for n in names}
    seeds = {n: sorted(dep[n]) for n in names}

    riv_bv = {n: di.res_iv(dep[n], cal_bv[n], a10, seeds[n]) for n in names}
    riv_fe = {n: di.res_iv(dep[n], cal_fe[n], a10, seeds[n]) for n in names}
    primary = di.test_rows({n: di.cov_err(riv_bv[n], a10) - di.cov_err(riv_fe[n], a10) for n in names})

    ratio = {n: {'BV_q90': di.q90_abs(bv[n]), 'FE_q90': di.q90_abs(fe[n])} for n in names}
    for n in names:
        ratio[n]['BV_over_FE'] = ratio[n]['BV_q90'] / ratio[n]['FE_q90']
    is_res = di.test_rows({n: di.per_b(riv_bv[n], 'interval_score', a10) - di.per_b(riv_fe[n], 'interval_score', a10) for n in names})
    miv_bv = {n: di.mc_iv(dep[n], cal_bv[n], a10, seeds[n]) for n in with_drop}
    miv_fe = {n: di.mc_iv(dep[n], cal_fe[n], a10, seeds[n]) for n in with_drop}
    mc_cov = di.test_rows({n: di.cov_err(miv_bv[n], a10) - di.cov_err(miv_fe[n], a10) for n in with_drop})
    mc_is = di.test_rows({n: di.per_b(miv_bv[n], 'interval_score', a10) - di.per_b(miv_fe[n], 'interval_score', a10) for n in with_drop})
    dec = {}
    for n in names:
        t_bv = dm.timing(dep[n], a10, cal_bv[n], seeds[n])[dm.TAU]
        t_fe = dm.timing(dep[n], a10, cal_fe[n], seeds[n])[dm.TAU]
        dec[n] = {str(r): {'late_BV': float(np.mean([t_bv['late'][f'cmq{r}'][b] for b in a10])),
                           'late_FE': float(np.mean([t_fe['late'][f'cmq{r}'][b] for b in a10])),
                           'L_BV': float(dm.losses(t_bv, f'cmq{r}', r, a10).mean()),
                           'L_FE': float(dm.losses(t_fe, f'cmq{r}', r, a10).mean())} for r in dm.R_PRIMARY}

    def row(e):
        return f"{e['pooled_coverage']:.1f} % | {e['late_coverage']:.1f} % | {di.per_b(e, 'interval_score', a10).mean():.3f}"

    def cell(r, holm_=True):
        return f"{r['mean']:+.2f} ({r['k_positive']}/{r['n']}, {r['p_flip']:.3f}" + (f", {r['p_holm']:.3f})" if holm_ else ')')

    lines = ['# Calibrating on the selected checkpoint (revision R1, Amendment 7, part B)', '',
             ('**DRY RUN on synthetic data: numbers are meaningless.**' if args.dry_run else
              'Generated by `Experiments/r1_decision_bestckpt.py`; definitions in analysis plan Section 11, Amendment 7.'),
             '', 'Deployed models fixed: each architecture\'s 20 final runs (equal-terms test predictions). Calibration data: the '
             'four-fold selection runs on their held-out bearing, from the best-epoch state (BV, the checkpoint exp19\'s harness '
             'restores) or the final-epoch state (FE, the protocol). Ten test bearings; 90 % two-sided intervals. Differences are '
             'BV minus FE (positive: final-epoch calibration better). p: exact sign-flip test over bearings.', '',
             '## A7-P4 and A7-P5 (primary): coverage error |coverage − 90| of the residual-quantile interval', '',
             '| Architecture | best epoch = final epoch (mean best epoch / K) | BV: coverage, late life, interval score | '
             'FE: coverage, late life, interval score | coverage error BV − FE, points (better on, p, Holm over eight) |',
             '|---|---|---|---|---|']
    for n in names:
        s = best_stats[n]
        lines.append(f"| {n} | {100 * s['share_best_is_final']:.0f} % ({s['mean_best_epoch']:.1f} / {s['K']}) | {row(riv_bv[n])} | "
                     f"{row(riv_fe[n])} | {cell(primary['per_model'][n])} |")
    p, o = primary['pooled'], primary['outcome']
    lines += ['', f"A7-P4, pooled over the eight: mean {p['mean']:+.2f} points, final-epoch calibration closer on {p['k_positive']}/{p['n']} "
                  f"bearings, p {p['p_flip']:.3f} (registered prediction: d ≥ 0).",
              f"A7-P5 outcome: final-epoch calibration significantly closer for {len(o['protocol_significantly_better'])} of 8 "
              f"({', '.join(o['protocol_significantly_better']) or 'none'}); significantly farther for "
              f"{len(o['protocol_significantly_worse'])} ({', '.join(o['protocol_significantly_worse']) or 'none'}).", '',
              '## Secondary', '', '### Calibration data: 0.9-quantile of |error|, residual-quantile interval score', '',
              '| Architecture | BV | FE | BV / FE | interval score BV − FE (better on, p, Holm) |', '|---|---|---|---|---|']
    for n in names:
        a = ratio[n]
        lines.append(f"| {n} | {a['BV_q90']:.4f} | {a['FE_q90']:.4f} | {a['BV_over_FE']:.2f} | {cell(is_res['per_model'][n])} |")
    lines += [f"| pooled | | | | {cell(is_res['pooled'], False)} |", '', '### MC-dropout interval μ ± z c* s (seven architectures with dropout)', '',
              '| Architecture | c* BV / FE | BV: coverage, late life, interval score | FE: coverage, late life, interval score | '
              'coverage error BV − FE (better on, p, Holm) | interval score BV − FE (better on, p, Holm) |', '|---|---|---|---|---|---|']
    for n in with_drop:
        lines.append(f"| {n} | {cal_bv[n]['c_star']:.2f} / {cal_fe[n]['c_star']:.2f} | {row(miv_bv[n])} | {row(miv_fe[n])} | "
                     f"{cell(mc_cov['per_model'][n])} | {cell(mc_is['per_model'][n])} |")
    lines += [f"| pooled | | | | {cell(mc_cov['pooled'], False)} | {cell(mc_is['pooled'], False)} |", '',
              f'### Cost-matched trigger, τ = {dm.TAU:.2f} (descriptive)', '',
              '| Architecture | ' + ' | '.join(f'late alarms BV / FE, r = {r} | L(r = {r}) BV / FE' for r in dm.R_PRIMARY) + ' |',
              '|---|' + '---|---|' * len(dm.R_PRIMARY)]
    for n in names:
        lines.append(f'| {n} | ' + ' | '.join(
            f"{100 * dec[n][str(r)]['late_BV']:.0f} / {100 * dec[n][str(r)]['late_FE']:.0f} % | "
            f"{dec[n][str(r)]['L_BV']:.1f} / {dec[n][str(r)]['L_FE']:.1f}" for r in dm.R_PRIMARY) + ' |')

    report = {'dry_run': args.dry_run, 'calibration_BV': cal_bv, 'calibration_FE': cal_fe, 'best_epochs': best_stats,
              'A7_P4_P5': primary, 'residual_intervals': {'BV': riv_bv, 'FE': riv_fe},
              'secondary': {'calibration_q90': ratio, 'interval_score': is_res, 'mc_coverage': mc_cov, 'mc_interval_score': mc_is,
                            'mc_intervals': {'BV': miv_bv, 'FE': miv_fe}, 'decisions': dec},
              'provenance': provenance(('Experiments/r1_decision_bestckpt.py', 'Experiments/r1_decision_inbearing.py',
                                        'Experiments/r1_decision_models.py', 'Experiments/r1_decision_models_mc30.py',
                                        'Experiments/r1_bestckpt_valpreds.py', 'Experiments/r1_strict_stats.py'))}
    text = '\n'.join(lines) + '\n'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')
    out_path.with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(text)
    print(f'written {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
