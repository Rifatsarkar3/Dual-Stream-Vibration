"""
r1_decision_inbearing.py — Revision R1, analysis plan Amendment 6: what standard-practice calibration
(on the in-bearing validation windows that also select the model) does to the error distribution on
unseen bearings, against the protocol's held-out-bearing calibration. Functions of
r1_decision_models.py are used unchanged; equal-terms predictions (30 MC-dropout passes everywhere).

  Standard practice (SP): in-bearing early-stopping checkpoints (seeds 42, 123, 777, 999, 7), calibrated on
      their own in-bearing validation windows (T6a); test predictions: ours r1_reeval/full_inbearing_es,
      dropout baselines T6b, TSMC-CNN stored.
  Protocol (PR): final runs on the same five seeds, calibrated on held-out bearings (Amendment 2).
  A6-P1  pooled: per bearing, mean over the eight architectures of |cov_SP - 90| - |cov_PR - 90| for the
         two-sided 90 % residual-quantile interval (positive: protocol closer to target); sign-flip over bearings.
  A6-P2  per architecture: the same, Holm over the eight.
  Secondary: (a) understatement ratio; (b) interval score; (c) MC-dropout interval (seven with dropout);
         (d) cost-matched trigger and late alarms; (e) our model, deployed model fixed, calibrated from
         best-validation checkpoints (T6c) against final-epoch predictions.

USAGE:
  python Experiments/r1_decision_inbearing.py
  python Experiments/r1_decision_inbearing.py --dry-run     # synthetic data, scratch output, no real files read
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

ARCH = {dm.SUBMITTED: 'ours', '1D CNN': 'cnn', 'BiLSTM': 'bilstm', 'CNN-LSTM': 'cnnlstm', '1D ResNet': 'resnet',
        'Transformer': 'transformer', 'TSMC-CNN [21]': 'tsmccnn', 'MSCNN [7]': 'mscnn'}
SEEDS = dm.ES_SEEDS
TARGET = 90.0
INB = ROOT / 'outputs' / 'r1_inbearing'
SP_TEST = {'ours': 'outputs/r1_reeval/full_inbearing_es', 'tsmccnn': 'outputs/r1_protocol_baselines/tsmccnn',
           **{m: f'outputs/r1_inbearing/t6b/{m}' for m in ('cnn', 'bilstm', 'cnnlstm', 'resnet', 'transformer', 'mscnn')}}


# ---------------------------------------------------------------- data --------------------------
def concat_runs(files):
    runs = []
    for f in files:
        d = np.load(f)
        names = sorted(k[4:] for k in d.files if k.startswith('tgt_'))

        def cat(p):
            return np.concatenate([d[p + n].astype(float) for n in names])
        runs.append({'y': cat('tgt_'), 'raw': cat('raw_'), 'mu': cat('mcmean_'), 's': np.clip(cat('mcstd_'), 1e-4, None)})
    return runs


def meta_ok(path, key, limit, count_key, count):
    m = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
    if m[count_key] != count or m[key] > limit:
        raise SystemExit(f'{path}: incomplete or failed its check ({m[count_key]} runs, {key} {m[key]})')
    return m


def real_data():
    dm.baseline_paths = dm30.mc30_paths
    models = dm.real_models()
    sp, pr = {}, {}
    for label, short in ARCH.items():
        meta_ok(INB / 't6a' / short, 'max_abs_diff', 1e-4, 'n_runs', 5)
        if short not in ('ours', 'tsmccnn'):
            meta_ok(INB / 't6b' / short, 'raw_check_max_abs_diff', 1e-5, 'n_checkpoints', 5)
        test = dm.load_preds(SP_TEST[short])
        if sorted(test) != sorted(SEEDS):
            raise SystemExit(f'{label}: standard-practice test seeds {sorted(test)}')
        dropout = label not in dm.NO_DROPOUT
        sp[label] = {'val': concat_runs([INB / 't6a' / short / f'inval_seed{s}.npz' for s in SEEDS]),
                     'test': test, 'dropout': dropout}
        pr[label] = {'val': models[label]['val'], 'test': {s: models[label]['final'][s] for s in SEEDS},
                     'test_all': models[label]['final'], 'dropout': dropout}
    meta_ok(INB / 't6c' / 'ours', 'max_abs_diff', 1e-4, 'n_runs', 20)
    best_val = dm.load_val(sorted(str(p) for p in (INB / 't6c' / 'ours').glob('fold*_seed*.npz')))
    return sp, pr, best_val


def synthetic_data(rng):
    """dry run: protocol models from r1_decision_models' generator; standard practice with understated
    validation errors (in-bearing windows) and slightly different test predictions"""
    syn = dm.synthetic_models(rng)
    sp, pr = {}, {}
    for label in ARCH:
        m = syn[label]
        pr[label] = {'val': m['val'], 'test': {s: m['final'][s] for s in SEEDS}, 'test_all': m['final'],
                     'dropout': m['dropout']}
        runs = []
        for r in m['val'][:5]:
            runs.append({'y': r['y'], 'raw': r['y'] + 0.3 * (r['raw'] - r['y']), 'mu': r['y'] + 0.3 * (r['mu'] - r['y']),
                         's': r['s']})
        sp[label] = {'val': runs, 'test': m['es'], 'dropout': m['dropout']}
    best_val = [{'y': r['y'], 'raw': r['y'] + 0.8 * (r['raw'] - r['y']), 'mu': r['y'] + 0.8 * (r['mu'] - r['y']),
                 's': r['s']} for r in syn[dm.SUBMITTED]['val']]
    return sp, pr, best_val


# ---------------------------------------------------------------- helpers -----------------------
def res_iv(data, cal, bearings, seeds):
    q = cal['q']
    return dm.intervals(data, bearings, seeds, lambda d: (d['raw'] - q[0.95], d['raw'] - q[0.05]))


def mc_iv(data, cal, bearings, seeds):
    c = cal['c_star']
    return dm.intervals(data, bearings, seeds, lambda d: (d['mu'] - dm.Z * c * d['s'], d['mu'] + dm.Z * c * d['s']))


def per_b(e, key, bearings):
    return np.array([e['per_bearing'][b][key] for b in bearings])


def cov_err(e, bearings):
    return np.abs(per_b(e, 'coverage', bearings) - TARGET)


def test_rows(d):
    """d: {label: per-bearing array, positive = protocol better}; per-model sign-flip with Holm, and pooled"""
    rows = {n: {'mean': float(v.mean()), 'k_positive': int((v > 0).sum()), 'n': int(len(v)), 'p_flip': float(signflip_p(v))}
            for n, v in d.items()}
    adj = holm(np.array([r['p_flip'] for r in rows.values()]))
    for r, a in zip(rows.values(), adj):
        r['p_holm'] = float(a)
    pooled_v = np.mean(list(d.values()), axis=0)
    pooled = {'per_bearing': [float(x) for x in pooled_v], 'mean': float(pooled_v.mean()),
              'k_positive': int((pooled_v > 0).sum()), 'n': int(len(pooled_v)), 'p_flip': float(signflip_p(pooled_v))}
    outcome = {'protocol_significantly_better': [n for n, r in rows.items() if r['p_holm'] < 0.05 and r['mean'] > 0],
               'protocol_significantly_worse': [n for n, r in rows.items() if r['p_holm'] < 0.05 and r['mean'] < 0]}
    return {'per_model': rows, 'pooled': pooled, 'outcome': outcome}


def q90_abs(runs):
    return float(np.mean([np.quantile(np.abs(r['raw'] - r['y']), 0.9) for r in runs]))


def q90_abs_test(test, bearings, seeds):
    return float(np.mean([np.quantile(np.abs(np.concatenate([test[s][b]['raw'] - test[s][b]['y'] for b in bearings])), 0.9)
                          for s in seeds]))


def srd(a, b):
    s = a + b
    return np.where(s > 0, (a - b) / np.where(s > 0, s, 1.0), 0.0)


# ---------------------------------------------------------------- main --------------------------
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'decision_inbearing.md'))
    ap.add_argument('--dry-run', action='store_true', help='synthetic data; writes to a temporary folder')
    args = ap.parse_args()
    if args.dry_run:
        sp, pr, best_val = synthetic_data(np.random.default_rng(0))
        out_path = Path(tempfile.mkdtemp(prefix='decision_inbearing_dry_')) / 'decision_inbearing.md'
    else:
        sp, pr, best_val = real_data()
        out_path = Path(args.out)
    a10, names = SETS['all10'], list(ARCH)
    drop = [n for n in names if pr[n]['dropout']]
    cal_sp = {n: dm.calibrate({'val': sp[n]['val'], 'dropout': sp[n]['dropout']}) for n in names}
    cal_pr = {n: dm.calibrate({'val': pr[n]['val'], 'dropout': pr[n]['dropout']}) for n in names}

    # ---- A6-P1 / A6-P2: residual-quantile interval coverage error ---------------------------------
    riv_sp = {n: res_iv(sp[n]['test'], cal_sp[n], a10, SEEDS) for n in names}
    riv_pr = {n: res_iv(pr[n]['test'], cal_pr[n], a10, SEEDS) for n in names}
    primary = test_rows({n: cov_err(riv_sp[n], a10) - cov_err(riv_pr[n], a10) for n in names})
    # ---- secondary ---------------------------------------------------------------------------------
    ratio = {n: {'SP_calibration_q90': q90_abs(sp[n]['val']), 'SP_test_q90': q90_abs_test(sp[n]['test'], a10, SEEDS),
                 'PR_calibration_q90': q90_abs(pr[n]['val']), 'PR_test_q90': q90_abs_test(pr[n]['test'], a10, SEEDS)}
             for n in names}
    for n in names:
        ratio[n]['SP_ratio'] = ratio[n]['SP_calibration_q90'] / ratio[n]['SP_test_q90']
        ratio[n]['PR_ratio'] = ratio[n]['PR_calibration_q90'] / ratio[n]['PR_test_q90']
    is_res = test_rows({n: per_b(riv_sp[n], 'interval_score', a10) - per_b(riv_pr[n], 'interval_score', a10) for n in names})
    miv_sp = {n: mc_iv(sp[n]['test'], cal_sp[n], a10, SEEDS) for n in drop}
    miv_pr = {n: mc_iv(pr[n]['test'], cal_pr[n], a10, SEEDS) for n in drop}
    mc_cov = test_rows({n: cov_err(miv_sp[n], a10) - cov_err(miv_pr[n], a10) for n in drop})
    mc_is = test_rows({n: per_b(miv_sp[n], 'interval_score', a10) - per_b(miv_pr[n], 'interval_score', a10) for n in drop})
    dec = {}
    for r in dm.R_PRIMARY:
        L, late = {}, {}
        for n in names:
            t_sp = dm.timing(sp[n]['test'], a10, cal_sp[n], SEEDS)[dm.TAU]
            t_pr = dm.timing(pr[n]['test'], a10, cal_pr[n], SEEDS)[dm.TAU]
            L[n] = (dm.losses(t_sp, f'cmq{r}', r, a10), dm.losses(t_pr, f'cmq{r}', r, a10))
            late[n] = {'SP': float(np.mean([t_sp['late'][f'cmq{r}'][b] for b in a10])),
                       'PR': float(np.mean([t_pr['late'][f'cmq{r}'][b] for b in a10]))}
        d = np.mean([srd(L[n][0], L[n][1]) for n in names], axis=0)
        dec[str(r)] = {'L_mean': {n: {'SP': float(L[n][0].mean()), 'PR': float(L[n][1].mean())} for n in names},
                       'late_share': late,
                       'pooled_srd': {'per_bearing': [float(x) for x in d], 'mean': float(d.mean()),
                                      'k_positive': int((d > 0).sum()), 'n': int(len(d)), 'p_flip': float(signflip_p(d))}}
    # (e) our model, deployed model fixed
    ours = dm.SUBMITTED
    cal_best = dm.calibrate({'val': best_val, 'dropout': True})
    seeds_all = sorted(pr[ours]['test_all'])
    fixed = {}
    for tag, cal in (('final-epoch predictions (protocol)', cal_pr[ours]), ('best-validation checkpoints', cal_best)):
        rv, mv = res_iv(pr[ours]['test_all'], cal, a10, seeds_all), mc_iv(pr[ours]['test_all'], cal, a10, seeds_all)
        t = dm.timing(pr[ours]['test_all'], a10, cal, seeds_all)[dm.TAU]
        fixed[tag] = {'c_star': cal['c_star'], 'q90': cal['q'][0.9], 'q95': cal['q'][0.95],
                      'residual_coverage': rv['pooled_coverage'], 'residual_late': rv['late_coverage'],
                      'residual_is': float(per_b(rv, 'interval_score', a10).mean()),
                      'mc_coverage': mv['pooled_coverage'], 'mc_late': mv['late_coverage'],
                      'mc_is': float(per_b(mv, 'interval_score', a10).mean()),
                      'L': {str(r): float(dm.losses(t, f'cmq{r}', r, a10).mean()) for r in dm.R_PRIMARY}}

    # ---- report ------------------------------------------------------------------------------------
    def row(e):
        return f"{e['pooled_coverage']:.1f} % | {e['late_coverage']:.1f} % | {per_b(e, 'interval_score', a10).mean():.3f}"

    def cell(r, holm_=True):
        return f"{r['mean']:+.2f} ({r['k_positive']}/{r['n']}, {r['p_flip']:.3f}" + (f", {r['p_holm']:.3f})" if holm_ else ')')

    lines = ['# Standard-practice calibration against the protocol (revision R1, Amendment 6)', '',
             ('**DRY RUN on synthetic data: numbers are meaningless.**' if args.dry_run else
              'Generated by `Experiments/r1_decision_inbearing.py`; definitions in analysis plan Section 11, Amendment 6.'),
             '', 'Standard practice (SP): in-bearing early-stopping checkpoints calibrated on their own in-bearing validation '
             'windows. Protocol (PR): final runs calibrated on held-out bearings. Seeds 42, 123, 777, 999, 7; ten test '
             'bearings; 90 % two-sided intervals. Differences are SP minus PR (positive: the protocol is better). '
             'p: exact sign-flip test over bearings.', '',
             '## A6-P1 and A6-P2 (primary): coverage error |coverage − 90| of the residual-quantile interval', '',
             '| Architecture | SP: coverage, late life, interval score | PR: coverage, late life, interval score | '
             'coverage error SP − PR, points (better on, p, Holm over eight) |', '|---|---|---|---|']
    for n in names:
        lines.append(f'| {n} | {row(riv_sp[n])} | {row(riv_pr[n])} | {cell(primary["per_model"][n])} |')
    p = primary['pooled']
    o = primary['outcome']
    lines += ['', f"A6-P1, pooled over the eight: mean {p['mean']:+.2f} points, protocol closer on {p['k_positive']}/{p['n']} "
                  f"bearings, p {p['p_flip']:.3f}.",
              f"A6-P2 outcome: protocol significantly closer to the target for {len(o['protocol_significantly_better'])} of 8 "
              f"({', '.join(o['protocol_significantly_better']) or 'none'}); significantly farther for "
              f"{len(o['protocol_significantly_worse'])} ({', '.join(o['protocol_significantly_worse']) or 'none'}).",
              '', '## Secondary', '', '### (a) Understatement: 0.9-quantile of |error| on the calibration data against the test bearings', '',
              '| Architecture | SP calibration | SP test | SP ratio | PR calibration | PR test | PR ratio |', '|---|---|---|---|---|---|---|']
    for n in names:
        a = ratio[n]
        lines.append(f"| {n} | {a['SP_calibration_q90']:.4f} | {a['SP_test_q90']:.4f} | {a['SP_ratio']:.2f} | "
                     f"{a['PR_calibration_q90']:.4f} | {a['PR_test_q90']:.4f} | {a['PR_ratio']:.2f} |")
    lines += ['', '### (b) Interval score of the residual-quantile interval, SP − PR', '',
              '| Architecture | SP − PR (better on, p, Holm) |', '|---|---|']
    lines += [f'| {n} | {cell(is_res["per_model"][n])} |' for n in names]
    lines += [f"| pooled | {cell(is_res['pooled'], False)} |", '',
              '### (c) MC-dropout interval μ ± z c* s (seven architectures with dropout)', '',
              '| Architecture | c* SP / PR | SP: coverage, late life, interval score | PR: coverage, late life, interval score | '
              'coverage error SP − PR (better on, p, Holm) | interval score SP − PR (better on, p, Holm) |',
              '|---|---|---|---|---|---|']
    for n in drop:
        lines.append(f"| {n} | {cal_sp[n]['c_star']:.2f} / {cal_pr[n]['c_star']:.2f} | {row(miv_sp[n])} | {row(miv_pr[n])} | "
                     f"{cell(mc_cov['per_model'][n])} | {cell(mc_is['per_model'][n])} |")
    lines += [f"| pooled | | | | {cell(mc_cov['pooled'], False)} | {cell(mc_is['pooled'], False)} |", '',
              f'### (d) Cost-matched trigger with each pipeline\'s own quantiles, τ = {dm.TAU:.2f}', '',
              '| r | pooled (L_SP − L_PR) / (L_SP + L_PR): mean (bearings > 0, p) | late alarms SP / PR, mean over architectures |',
              '|---|---|---|']
    for r in dm.R_PRIMARY:
        e = dec[str(r)]
        ls = np.mean([e['late_share'][n]['SP'] for n in names]), np.mean([e['late_share'][n]['PR'] for n in names])
        lines.append(f"| {r} | {e['pooled_srd']['mean']:+.3f} ({e['pooled_srd']['k_positive']}/{e['pooled_srd']['n']}, "
                     f"{e['pooled_srd']['p_flip']:.3f}) | {100 * ls[0]:.0f} % / {100 * ls[1]:.0f} % |")
    lines += ['', '| Architecture | ' + ' | '.join(f'L(r = {r}) SP / PR' for r in dm.R_PRIMARY) + ' |',
              '|---|' + '---|' * len(dm.R_PRIMARY)]
    for n in names:
        lines.append(f'| {n} | ' + ' | '.join(f"{dec[str(r)]['L_mean'][n]['SP']:.1f} / {dec[str(r)]['L_mean'][n]['PR']:.1f}"
                                             for r in dm.R_PRIMARY) + ' |')
    lines += ['', '### (e) Our model, deployed model fixed (final runs, 20 seeds): calibration source', '',
              '| Calibration source | c* | q(0.9) | residual: coverage, late life, interval score | MC: coverage, late life, '
              'interval score | ' + ' | '.join(f'L(r = {r})' for r in dm.R_PRIMARY) + ' |', '|---|---|---|---|---|' + '---|' * len(dm.R_PRIMARY)]
    for tag, f in fixed.items():
        lines.append(f"| {tag} | {f['c_star']:.3f} | {f['q90']:.4f} | {f['residual_coverage']:.1f} %, {f['residual_late']:.1f} %, "
                     f"{f['residual_is']:.3f} | {f['mc_coverage']:.1f} %, {f['mc_late']:.1f} %, {f['mc_is']:.3f} | "
                     + ' | '.join(f"{f['L'][str(r)]:.1f}" for r in dm.R_PRIMARY) + ' |')

    report = {'dry_run': args.dry_run, 'calibration_SP': cal_sp, 'calibration_PR': cal_pr,
              'A6_P1_P2': primary, 'residual_intervals': {'SP': riv_sp, 'PR': riv_pr},
              'secondary': {'understatement': ratio, 'interval_score': is_res, 'mc_coverage': mc_cov, 'mc_interval_score': mc_is,
                            'mc_intervals': {'SP': miv_sp, 'PR': miv_pr}, 'decisions': dec, 'ours_fixed_model': fixed},
              'provenance': provenance(('Experiments/r1_decision_inbearing.py', 'Experiments/r1_decision_models.py',
                                        'Experiments/r1_decision_models_mc30.py', 'Experiments/r1_inbearing_preds.py',
                                        'Experiments/r1_strict_stats.py'))}
    text = '\n'.join(lines) + '\n'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')
    out_path.with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(text)
    print(f'written {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
