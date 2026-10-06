"""
r1_decision_xjtu_inbearing.py — Revision R1, analysis plan Amendment 7, part A: in-bearing calibration (SP)
against held-out calibration (PR) on XJTU-SY, for the submitted architecture and the three XJTU-SY baselines.
Functions of r1_decision_models.py and r1_decision_inbearing.py are used unchanged; equal-terms predictions
(30 MC-dropout passes everywhere).

  SP  in-bearing early-stopping checkpoints (seeds 42, 123, 777, 999, 7), calibrated on their own in-bearing
      validation windows (T7a); test predictions T7b.
  PR  final runs on the same five seeds (T7c), calibrated on held-out bearings (ours: E5's 20 validation runs;
      baselines: V7).
  Bearings: the six XJTU-SY test bearings and the five Condition-3 bearings (eleven).
  A7-P1  pooled: per bearing, mean over the four architectures of |cov_SP - 90| - |cov_PR - 90| for the two-sided
         90 % residual-quantile interval (positive: held-out calibration closer); sign-flip over the eleven bearings.
  A7-P2  per architecture, the same; Holm over the four.
  A7-P3  late alarms of the cost-matched trigger (tau 0.20, r = 5 and 10): per bearing, SP share minus PR share,
         mean over the four architectures; sign-flip over the eleven bearings; Holm over the two ratios.
  Secondary (a) to (f) as in the plan.

USAGE:
  python Experiments/r1_decision_xjtu_inbearing.py
  python Experiments/r1_decision_xjtu_inbearing.py --dry-run     # synthetic data, scratch output, no real files read
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
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import holm, signflip_p  # noqa: E402
from r1_xjtu_stats import SETS as XSETS  # noqa: E402

ARCH = {dm.SUBMITTED: 'ours', '1D CNN': 'cnn', 'BiLSTM': 'bilstm', 'CNN-LSTM': 'cnnlstm'}
SEEDS = [42, 123, 777, 999, 7]
BEARINGS = XSETS['xjtu_test'] + XSETS['xjtu_zs']
SUBSETS = {'test6': XSETS['xjtu_test'], 'zs5': XSETS['xjtu_zs']}
XI = ROOT / 'outputs' / 'r1_xjtu_inbearing'
OURS_PR_VAL = 'outputs/r1_xjtu_calib/submitted/lam0.08_B5_fold*/valpreds_seed*.npz'
A6_JSON = ROOT / 'outputs' / 'r1_analysis' / 'decision_inbearing.json'


# ---------------------------------------------------------------- data --------------------------
def meta_ok(path, key, limit, count_key, count):
    m = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
    if m[count_key] != count or (key is not None and m[key] > limit):
        raise SystemExit(f'{path}: incomplete or failed its check ({m[count_key]} runs, {key} {m.get(key)})')
    return m


def real_data():
    sp, pr = {}, {}
    for label, short in ARCH.items():
        meta_ok(XI / 't7a' / short, 'max_abs_diff', 1e-4, 'n_runs', 5)
        meta_ok(XI / 't7b' / short, None, None, 'n_checkpoints', 5)
        meta_ok(XI / 't7c' / short, 'raw_check_max_abs_diff', 1e-5, 'n_checkpoints', 5)
        if short == 'ours':
            val_files = sorted(glob.glob(str(ROOT / OURS_PR_VAL)))
        else:
            m = meta_ok(XI / 'v7' / short, 'replay_max_abs_diff', 1e-9, 'n_runs', 20)
            if not m['all_histories_identical']:
                raise SystemExit(f'{short}: V7 validation histories do not reproduce E6')
            val_files = sorted(str(p) for p in (XI / 'v7' / short).glob('fold*_seed*.npz'))
        if len(val_files) != 20:
            raise SystemExit(f'{label}: expected 20 held-out validation runs, found {len(val_files)}')
        sp_test = dm.load_preds(str((XI / 't7b' / short).relative_to(ROOT)))
        pr_test = dm.load_preds(str((XI / 't7c' / short).relative_to(ROOT)))
        for name, t in (('SP', sp_test), ('PR', pr_test)):
            if sorted(t) != sorted(SEEDS) or any(sorted(t[s]) != sorted(BEARINGS) for s in t):
                raise SystemExit(f'{label} {name}: expected seeds {SEEDS} with the eleven bearings')
        sp[label] = {'val': di.concat_runs([XI / 't7a' / short / f'inval_seed{s}.npz' for s in SEEDS]), 'test': sp_test,
                     'dropout': True}
        pr[label] = {'val': dm.load_val(val_files), 'test': pr_test, 'dropout': True}
    return sp, pr


def synthetic_data(rng):
    """dry run: random trajectories; in-bearing validation errors understated, early-stopping predictions biased"""
    lengths = {b: int(rng.integers(150, 600)) for b in BEARINGS}

    def run(n, bias, noise):
        y = np.linspace(1.0, 0.0, n)
        raw = np.clip(y + bias + noise * rng.standard_normal(n), -0.2, 1.2)
        return {'y': y, 'raw': raw, 'mu': raw + 0.01 * rng.standard_normal(n), 's': np.full(n, max(noise / 4, 1e-4))}

    sp, pr = {}, {}
    for i, label in enumerate(ARCH):
        bias, noise = 0.02 * (i % 3 - 1), 0.08 + 0.01 * i
        sp[label] = {'val': [run(300, 0.0, 0.3 * noise) for _ in range(5)],
                     'test': {s: {b: run(lengths[b], bias + 0.03, noise) for b in BEARINGS} for s in SEEDS}, 'dropout': True}
        pr[label] = {'val': [run(300, bias, noise) for _ in range(20)],
                     'test': {s: {b: run(lengths[b], bias, noise) for b in BEARINGS} for s in SEEDS}, 'dropout': True}
    return sp, pr


# ---------------------------------------------------------------- main --------------------------
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'decision_xjtu_inbearing.md'))
    ap.add_argument('--dry-run', action='store_true', help='synthetic data; writes to a temporary folder')
    args = ap.parse_args()
    if args.dry_run:
        sp, pr = synthetic_data(np.random.default_rng(0))
        out_path = Path(tempfile.mkdtemp(prefix='decision_xjtu_inbearing_dry_')) / 'decision_xjtu_inbearing.md'
    else:
        sp, pr = real_data()
        out_path = Path(args.out)
    names, B = list(ARCH), BEARINGS
    cal_sp = {n: dm.calibrate({'val': sp[n]['val'], 'dropout': True}) for n in names}
    cal_pr = {n: dm.calibrate({'val': pr[n]['val'], 'dropout': True}) for n in names}

    # ---- A7-P1 / A7-P2: residual-quantile interval coverage error ---------------------------------
    riv_sp = {n: di.res_iv(sp[n]['test'], cal_sp[n], B, SEEDS) for n in names}
    riv_pr = {n: di.res_iv(pr[n]['test'], cal_pr[n], B, SEEDS) for n in names}
    d_cov = {n: di.cov_err(riv_sp[n], B) - di.cov_err(riv_pr[n], B) for n in names}
    primary = di.test_rows(d_cov)

    # ---- A7-P3: late alarms of the cost-matched trigger --------------------------------------------
    tim_sp = {n: dm.timing(sp[n]['test'], B, cal_sp[n], SEEDS)[dm.TAU] for n in names}
    tim_pr = {n: dm.timing(pr[n]['test'], B, cal_pr[n], SEEDS)[dm.TAU] for n in names}
    p3 = {}
    for r in dm.R_PRIMARY:
        rule = f'cmq{r}'
        late = {n: {'SP': np.array([tim_sp[n]['late'][rule][b] for b in B]),
                    'PR': np.array([tim_pr[n]['late'][rule][b] for b in B])} for n in names}
        d = np.mean([late[n]['SP'] - late[n]['PR'] for n in names], axis=0)
        p3[str(r)] = {'per_bearing': [float(x) for x in d], 'mean': float(d.mean()), 'k_positive': int((d > 0).sum()),
                      'k_negative': int((d < 0).sum()), 'n': int(len(d)), 'p_flip': float(signflip_p(d)),
                      'late_share': {n: {'SP': float(late[n]['SP'].mean()), 'PR': float(late[n]['PR'].mean())} for n in names}}
    for r, a in zip(dm.R_PRIMARY, holm(np.array([p3[str(r)]['p_flip'] for r in dm.R_PRIMARY]))):
        p3[str(r)]['p_holm'] = float(a)

    # ---- secondary ---------------------------------------------------------------------------------
    ratio = {n: {'SP_calibration_q90': di.q90_abs(sp[n]['val']), 'SP_test_q90': di.q90_abs_test(sp[n]['test'], B, SEEDS),
                 'PR_calibration_q90': di.q90_abs(pr[n]['val']), 'PR_test_q90': di.q90_abs_test(pr[n]['test'], B, SEEDS)}
             for n in names}
    for n in names:
        ratio[n]['SP_ratio'] = ratio[n]['SP_calibration_q90'] / ratio[n]['SP_test_q90']
        ratio[n]['PR_ratio'] = ratio[n]['PR_calibration_q90'] / ratio[n]['PR_test_q90']
    is_res = di.test_rows({n: di.per_b(riv_sp[n], 'interval_score', B) - di.per_b(riv_pr[n], 'interval_score', B) for n in names})
    miv_sp = {n: di.mc_iv(sp[n]['test'], cal_sp[n], B, SEEDS) for n in names}
    miv_pr = {n: di.mc_iv(pr[n]['test'], cal_pr[n], B, SEEDS) for n in names}
    mc_cov = di.test_rows({n: di.cov_err(miv_sp[n], B) - di.cov_err(miv_pr[n], B) for n in names})
    mc_is = di.test_rows({n: di.per_b(miv_sp[n], 'interval_score', B) - di.per_b(miv_pr[n], 'interval_score', B) for n in names})
    dec = {}
    for r in dm.R_PRIMARY:
        rule = f'cmq{r}'
        L = {n: (dm.losses(tim_sp[n], rule, r, B), dm.losses(tim_pr[n], rule, r, B)) for n in names}
        dt = {n: {'SP': float(np.mean([tim_sp[n]['A'][rule][b] - tim_sp[n]['B'][rule][b] for b in B])),
                  'PR': float(np.mean([tim_pr[n]['A'][rule][b] - tim_pr[n]['B'][rule][b] for b in B]))} for n in names}
        d = np.mean([di.srd(L[n][0], L[n][1]) for n in names], axis=0)
        dec[str(r)] = {'L_mean': {n: {'SP': float(L[n][0].mean()), 'PR': float(L[n][1].mean())} for n in names},
                       'mean_alarm_time_dt': dt,
                       'pooled_srd': {'per_bearing': [float(x) for x in d], 'mean': float(d.mean()),
                                      'k_positive': int((d > 0).sum()), 'n': int(len(d)), 'p_flip': float(signflip_p(d))}}
    subsets = {}
    for s, bs in SUBSETS.items():
        idx = [B.index(b) for b in bs]
        cov = np.array(primary['pooled']['per_bearing'])[idx]
        subsets[s] = {'A7_P1': {'mean': float(cov.mean()), 'k_positive': int((cov > 0).sum()), 'n': len(idx)},
                      'A7_P3': {str(r): {'mean': float(np.array(p3[str(r)]['per_bearing'])[idx].mean()),
                                         'k_positive': int((np.array(p3[str(r)]['per_bearing'])[idx] > 0).sum()),
                                         'k_negative': int((np.array(p3[str(r)]['per_bearing'])[idx] < 0).sum()),
                                         'n': len(idx)} for r in dm.R_PRIMARY}}
    side = None
    if not args.dry_run and A6_JSON.exists():
        a6 = json.loads(A6_JSON.read_text(encoding='utf-8'))
        side = {n: {'PRONOSTIA_SP_coverage': a6['residual_intervals']['SP'][n]['pooled_coverage'],
                    'PRONOSTIA_PR_coverage': a6['residual_intervals']['PR'][n]['pooled_coverage'],
                    'PRONOSTIA_SP_ratio': a6['secondary']['understatement'][n]['SP_ratio'],
                    'PRONOSTIA_PR_ratio': a6['secondary']['understatement'][n]['PR_ratio']} for n in names}
        side['_pooled'] = {'PRONOSTIA_A6_P1_mean': a6['A6_P1_P2']['pooled']['mean']}

    # ---- report ------------------------------------------------------------------------------------
    def row(e):
        return f"{e['pooled_coverage']:.1f} % | {e['late_coverage']:.1f} % | {di.per_b(e, 'interval_score', B).mean():.3f}"

    def cell(r, holm_=True):
        return f"{r['mean']:+.2f} ({r['k_positive']}/{r['n']}, {r['p_flip']:.3f}" + (f", {r['p_holm']:.3f})" if holm_ else ')')

    lines = ['# In-bearing against held-out calibration on XJTU-SY (revision R1, Amendment 7, part A)', '',
             ('**DRY RUN on synthetic data: numbers are meaningless.**' if args.dry_run else
              'Generated by `Experiments/r1_decision_xjtu_inbearing.py`; definitions in analysis plan Section 11, Amendment 7.'),
             '', 'In-bearing (SP): early-stopping checkpoints calibrated on their own in-bearing validation windows. Held-out (PR): '
             'final runs calibrated on held-out bearings. Seeds 42, 123, 777, 999, 7; six test and five Condition-3 bearings; '
             '90 % two-sided intervals. Differences are SP minus PR (positive: held-out calibration better, or for late alarms: '
             'more late alarms under SP). p: exact sign-flip test over the eleven bearings (smallest attainable 0.001).', '',
             '## A7-P1 and A7-P2 (primary): coverage error |coverage − 90| of the residual-quantile interval', '',
             '| Architecture | SP: coverage, late life, interval score | PR: coverage, late life, interval score | '
             'coverage error SP − PR, points (better on, p, Holm over four) |', '|---|---|---|---|']
    for n in names:
        lines.append(f'| {n} | {row(riv_sp[n])} | {row(riv_pr[n])} | {cell(primary["per_model"][n])} |')
    p, o = primary['pooled'], primary['outcome']
    lines += ['', f"A7-P1, pooled over the four: mean {p['mean']:+.2f} points, held-out calibration closer on {p['k_positive']}/{p['n']} "
                  f"bearings, p {p['p_flip']:.4f} (registered prediction: positive, p < 0.05).",
              f"A7-P2 outcome: held-out calibration significantly closer to the target for {len(o['protocol_significantly_better'])} of 4 "
              f"({', '.join(o['protocol_significantly_better']) or 'none'}); significantly farther for "
              f"{len(o['protocol_significantly_worse'])} ({', '.join(o['protocol_significantly_worse']) or 'none'}) "
              '(registered prediction: closer for every architecture).', '',
              '## A7-P3 (primary): late alarms of the cost-matched trigger, τ = 0.20', '',
              '| r | SP − PR share of late alarms, mean over architectures (bearings SP more / fewer, p, Holm over two) | '
              'late alarms SP / PR per architecture |', '|---|---|---|']
    for r in dm.R_PRIMARY:
        e = p3[str(r)]
        lines.append(f"| {r} | {e['mean']:+.3f} ({e['k_positive']} / {e['k_negative']} of {e['n']}, {e['p_flip']:.4f}, {e['p_holm']:.4f}) | "
                     + '; '.join(f"{n} {100 * e['late_share'][n]['SP']:.0f} / {100 * e['late_share'][n]['PR']:.0f} %" for n in names) + ' |')
    lines += ['', 'Registered prediction: positive (more late alarms under in-bearing calibration).', '', '## Secondary', '',
              '### (a) Understatement: 0.9-quantile of |error| on the calibration data against the eleven bearings', '',
              '| Architecture | SP calibration | SP test | SP ratio | PR calibration | PR test | PR ratio |', '|---|---|---|---|---|---|---|']
    for n in names:
        a = ratio[n]
        lines.append(f"| {n} | {a['SP_calibration_q90']:.4f} | {a['SP_test_q90']:.4f} | {a['SP_ratio']:.2f} | "
                     f"{a['PR_calibration_q90']:.4f} | {a['PR_test_q90']:.4f} | {a['PR_ratio']:.2f} |")
    lines += ['', '### (b) Interval score of the residual-quantile interval, SP − PR', '',
              '| Architecture | SP − PR (better on, p, Holm) |', '|---|---|']
    lines += [f'| {n} | {cell(is_res["per_model"][n])} |' for n in names]
    lines += [f"| pooled | {cell(is_res['pooled'], False)} |", '',
              '### (c) MC-dropout interval μ ± z c* s', '',
              '| Architecture | c* SP / PR | SP: coverage, late life, interval score | PR: coverage, late life, interval score | '
              'coverage error SP − PR (better on, p, Holm) | interval score SP − PR (better on, p, Holm) |', '|---|---|---|---|---|---|']
    for n in names:
        lines.append(f"| {n} | {cal_sp[n]['c_star']:.2f} / {cal_pr[n]['c_star']:.2f} | {row(miv_sp[n])} | {row(miv_pr[n])} | "
                     f"{cell(mc_cov['per_model'][n])} | {cell(mc_is['per_model'][n])} |")
    lines += [f"| pooled | | | | {cell(mc_cov['pooled'], False)} | {cell(mc_is['pooled'], False)} |", '',
              f'### (d) Cost-matched trigger with each pipeline\'s own quantiles, τ = {dm.TAU:.2f}', '',
              '| r | pooled (L_SP − L_PR) / (L_SP + L_PR): mean (bearings > 0, p) |', '|---|---|']
    for r in dm.R_PRIMARY:
        e = dec[str(r)]['pooled_srd']
        lines.append(f"| {r} | {e['mean']:+.3f} ({e['k_positive']}/{e['n']}, {e['p_flip']:.3f}) |")
    lines += ['', '| Architecture | ' + ' | '.join(f'L(r = {r}) SP / PR | mean Δt (r = {r}) SP / PR' for r in dm.R_PRIMARY) + ' |',
              '|---|' + '---|---|' * len(dm.R_PRIMARY)]
    for n in names:
        lines.append(f'| {n} | ' + ' | '.join(
            f"{dec[str(r)]['L_mean'][n]['SP']:.1f} / {dec[str(r)]['L_mean'][n]['PR']:.1f} | "
            f"{dec[str(r)]['mean_alarm_time_dt'][n]['SP']:+.1f} / {dec[str(r)]['mean_alarm_time_dt'][n]['PR']:+.1f}"
            for r in dm.R_PRIMARY) + ' |')
    lines += ['', '### (e) Bearing sets separately (descriptive)', '',
              '| Set | A7-P1 mean (bearings > 0) | ' + ' | '.join(f'A7-P3 r = {r} mean (more / fewer)' for r in dm.R_PRIMARY) + ' |',
              '|---|---|' + '---|' * len(dm.R_PRIMARY)]
    for s, e in subsets.items():
        lines.append(f"| {s} | {e['A7_P1']['mean']:+.2f} ({e['A7_P1']['k_positive']}/{e['A7_P1']['n']}) | " + ' | '.join(
            f"{e['A7_P3'][str(r)]['mean']:+.3f} ({e['A7_P3'][str(r)]['k_positive']} / {e['A7_P3'][str(r)]['k_negative']} of "
            f"{e['A7_P3'][str(r)]['n']})" for r in dm.R_PRIMARY) + ' |')
    if side:
        lines += ['', '### (f) Beside Amendment 6 (PRONOSTIA, ten test bearings; descriptive, no combined test)', '',
                  '| Architecture | PRONOSTIA SP / PR coverage | XJTU-SY SP / PR coverage | PRONOSTIA SP / PR ratio | XJTU-SY SP / PR ratio |',
                  '|---|---|---|---|---|']
        for n in names:
            s_ = side[n]
            lines.append(f"| {n} | {s_['PRONOSTIA_SP_coverage']:.1f} / {s_['PRONOSTIA_PR_coverage']:.1f} % | "
                         f"{riv_sp[n]['pooled_coverage']:.1f} / {riv_pr[n]['pooled_coverage']:.1f} % | "
                         f"{s_['PRONOSTIA_SP_ratio']:.2f} / {s_['PRONOSTIA_PR_ratio']:.2f} | "
                         f"{ratio[n]['SP_ratio']:.2f} / {ratio[n]['PR_ratio']:.2f} |")
        lines += ['', f"Pooled coverage-error gain: PRONOSTIA (eight architectures) {side['_pooled']['PRONOSTIA_A6_P1_mean']:+.2f} points; "
                      f"XJTU-SY (four) {p['mean']:+.2f} points."]

    report = {'dry_run': args.dry_run, 'bearings': B, 'seeds': SEEDS, 'calibration_SP': cal_sp, 'calibration_PR': cal_pr,
              'A7_P1_P2': primary, 'A7_P3': p3, 'residual_intervals': {'SP': riv_sp, 'PR': riv_pr},
              'secondary': {'understatement': ratio, 'interval_score': is_res, 'mc_coverage': mc_cov, 'mc_interval_score': mc_is,
                            'mc_intervals': {'SP': miv_sp, 'PR': miv_pr}, 'decisions': dec, 'subsets': subsets,
                            'beside_amendment6': side},
              'provenance': provenance(('Experiments/r1_decision_xjtu_inbearing.py', 'Experiments/r1_decision_inbearing.py',
                                        'Experiments/r1_decision_models.py', 'Experiments/r1_xjtu_inbearing.py',
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
