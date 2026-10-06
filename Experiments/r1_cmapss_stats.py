"""
r1_cmapss_stats.py - Revision R1, Amendment 11 (plan Section 11): analysis of the C-MAPSS runs
of Experiments/r1_cmapss.py. Run once on the real runs; --dry-run checks it on synthetic data.

Per cell (dataset, fleet size n, repeat r), each pipeline (SP: in-unit calibration; PR: held-out
units) with its own deployed model and calibration data:
  e = y_hat - y (cycles); q(p) the p-quantile of e on the calibration data (numpy linear);
  coverage of [y_hat - q(0.95), y_hat - q(0.05)] over all test windows (inclusive bounds);
  rho = 0.9-quantile of |e| on the calibration data / that on the test windows.
Primary family per dataset (FD001 primary, FD003 replication), Holm over six, size-matched as in
Amendment 11, correction 1 (the dry run showed that |coverage - 90| and the 0.9-quantile depend on
the number of calibration errors, and PR's calibration sets are about ten times larger than SP's):
  A11-P1(n), n = 4, 10, 25, 80: d = |cov_SP - 90| - mean over 1,000 subsamples of |cov_PR,k - 90|,
    subsample k = as many PR calibration errors as SP has validation windows (without replacement);
    exact sign-flip over the 20 repeats;
  A11-P2: log rho~_SP at n = 80, exact sign-flip; rho~ = q_cal / median over 1,000 subsamples of the
    0.9-quantile of |e| over as many test windows as there are calibration errors (median: correction 2);
  A11-P3: sum over repeats of Kendall's S between n and rho~_SP, exact permutation test.
The tests as first written (full calibration sets; rho) are reported as secondary.

Usage:
  python Experiments/r1_cmapss_stats.py --dry-run
  python Experiments/r1_cmapss_stats.py
Outputs: outputs/r1_analysis/cmapss.md and cmapss.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Experiments'))
from r1_mechanism import exact_test, monte_carlo_p  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import holm, signflip_p  # noqa: E402

RUNS = ROOT / 'outputs' / 'r1_cmapss'
OUT = ROOT / 'outputs' / 'r1_analysis'
PLAN_SHA_RUNS = {   # the runs' specification is the same under both (correction 1 concerns the analysis only)
    'bc8bbb9bbe3df61eaea6c96b1b5dacd89acd6196196bda9443d248e15999bc2a': 'Amendment 11',
    '51a48275ebf60a07e89074ad904d5e60d4ce3b14d876a0f8c716c460f45578c2': 'Amendment 11, correction 1',
    '0d4d7f6d6e30d18bb2d07ec7055a0083ff64e3534d3f6ba8ccdc7edf73ac134c': 'Amendment 11, correction 2',
}
DATASETS = {'FD001': 'primary', 'FD003': 'replication'}
DS_CODE = {'FD001': 1, 'FD003': 3}
N_SUB = 1000
FLEETS = [4, 10, 25, 80]
REPEATS = list(range(20))
TARGET = 90.0
DECISION_R = [5, 10]
TESTS = ['P1_n4', 'P1_n10', 'P1_n25', 'P1_n80', 'P2', 'P3']
CODE = ['Experiments/r1_cmapss.py', 'Experiments/r1_cmapss_stats.py', 'Experiments/r1_mechanism.py',
        'Experiments/r1_strict_stats.py']


# ---------------------------------------------------------------- per cell ----------------------
def coverage(cal_e, test_sorted):
    """coverage (%) of [y_hat - q(0.95), y_hat - q(0.05)]: share of test errors in [q(0.05), q(0.95)]"""
    lo, hi = np.quantile(cal_e, [0.05, 0.95])
    inside = np.searchsorted(test_sorted, hi, 'right') - np.searchsorted(test_sorted, lo, 'left')
    return 100.0 * inside / len(test_sorted)


def pipeline_metrics(cal_e, test_e, last):
    q = {p: float(np.quantile(cal_e, p)) for p in (0.05, 0.95)}
    inside = (test_e >= q[0.05]) & (test_e <= q[0.95])      # y in [y_hat - q95, y_hat - q05]
    q_cal, q_test = float(np.quantile(np.abs(cal_e), 0.9)), float(np.quantile(np.abs(test_e), 0.9))
    out = {'coverage': 100.0 * float(inside.mean()), 'width': q[0.95] - q[0.05],
           'q_cal': q_cal, 'q_test': q_test, 'rho': q_cal / q_test,
           'rmse_last': float(np.sqrt(np.mean(test_e[last] ** 2))), 'n_cal': int(len(cal_e))}
    for r in DECISION_R:
        p = r / (1 + r)
        out[f'decision_cov_r{r}'] = 100.0 * float((test_e <= np.quantile(cal_e, p)).mean())
    return out


def matched(sp_cal, pr_cal, sp_test, pr_test, rng, n_sub=N_SUB):
    """correction 1: PR's coverage error with calibration sets of SP's size (mean over subsamples),
    then the size-matched ratio rho~_SP; subsamples drawn in this order from rng"""
    m = len(sp_cal)
    pr_sorted = np.sort(pr_test)
    cov_err = np.array([abs(coverage(pr_cal[rng.choice(len(pr_cal), m, replace=False)], pr_sorted) - TARGET)
                        for _ in range(n_sub)])
    abs_test = np.abs(sp_test)
    q_sub = np.array([np.quantile(abs_test[rng.choice(len(abs_test), m, replace=False)], 0.9)
                      for _ in range(n_sub)])
    q_cal = float(np.quantile(np.abs(sp_cal), 0.9))
    q_ref = float(np.median(q_sub))                      # correction 2: median, not mean
    return {'pr_cov_err_matched': float(cov_err.mean()), 'q_test_matched': q_ref,
            'rho_matched': q_cal / q_ref, 'm': m}


def cell_metrics(z, meta, rng):
    y = z['test_y']
    last = z['test_last'].astype(bool)
    sp_cal, sp_test = z['sp_val_pred'] - z['sp_val_y'], z['sp_test_pred'] - y
    pr_cal, pr_test = z['pr_cal_e'], z['pr_test_pred'] - y
    sp, pr = pipeline_metrics(sp_cal, sp_test, last), pipeline_metrics(pr_cal, pr_test, last)
    mt = matched(sp_cal, pr_cal, sp_test, pr_test, rng)
    sp['kept_step'] = int(meta['sp']['best_step'])
    pr['s_star'] = int(meta['pr']['s_star'])
    sp['rho_matched'] = mt['rho_matched']
    sp['q_test_matched'] = mt['q_test_matched']
    pr['cov_err_matched'] = mt['pr_cov_err_matched']
    return {'SP': sp, 'PR': pr,
            'd': abs(sp['coverage'] - TARGET) - abs(pr['coverage'] - TARGET),
            'd_matched': abs(sp['coverage'] - TARGET) - mt['pr_cov_err_matched'],
            'n_test_windows': int(meta['n_test_windows']), 'n_test_short': int(meta['n_test_engines_short'])}


def load_dataset(ds):
    cells, plans = {}, {}
    for n in FLEETS:
        for r in REPEATS:
            stem = RUNS / ds / f'n{n:02d}_r{r:02d}'
            meta = json.loads(stem.with_suffix('.json').read_text(encoding='utf-8'))
            plan = meta['provenance']['plan_sha256']
            if plan not in PLAN_SHA_RUNS:
                raise SystemExit(f'{stem}: run made under plan {plan[:8]}')
            plans[PLAN_SHA_RUNS[plan]] = plans.get(PLAN_SHA_RUNS[plan], 0) + 1
            if (meta['dataset'], meta['n'], meta['repeat']) != (ds, n, r):
                raise SystemExit(f'{stem}: metadata mismatch')
            rng = np.random.default_rng([DS_CODE[ds], n, r])
            with np.load(stem.with_suffix('.npz')) as z:
                cells[(n, r)] = cell_metrics(z, meta, rng)
    return cells, plans


# ---------------------------------------------------------------- tests -------------------------
def family(cells, d_key, rho_key):
    tests = {}
    for n in FLEETS:
        d = np.array([cells[(n, r)][d_key] for r in REPEATS])
        tests[f'P1_n{n}'] = {'stat': 'mean d (points)', 'mean': float(d.mean()), 'p': signflip_p(d),
                             'n_positive': int((d > 0).sum()), 'n': len(d), 'predicted_sign': 1}
    lr = np.log(np.array([cells[(80, r)]['SP'][rho_key] for r in REPEATS]))
    tests['P2'] = {'stat': f'mean log {rho_key} (SP, n = 80)', 'mean': float(lr.mean()), 'p': signflip_p(lr),
                   'geometric_mean_rho': float(np.exp(lr.mean())), 'n_negative': int((lr < 0).sum()),
                   'n': len(lr), 'predicted_sign': -1}
    groups = [(FLEETS, [cells[(n, r)]['SP'][rho_key] for n in FLEETS]) for r in REPEATS]
    et = exact_test(groups)
    tests['P3'] = {'stat': f'T (sum of Kendall S, n against {rho_key} of SP)', 'mean': float(et['T']),
                   'p': et['p_two_sided'], 'T_range': et['T_min_max'], 'mean_tau_b': et['mean_tau_b'],
                   'n_tau_negative': et['n_tau_negative'], 'n': len(groups), 'predicted_sign': 0}
    adj = holm(np.array([tests[k]['p'] for k in TESTS]))
    for k, a in zip(TESTS, adj):
        t = tests[k]
        t['p_holm'] = float(a)
        sign = int(np.sign(t['mean']))
        t['direction'] = sign
        t['significant'] = bool(a < 0.05 and ((t['predicted_sign'] == 0 and sign != 0) or sign == t['predicted_sign']))
    return tests


def secondary(cells):
    out = {}
    for n in FLEETS:
        rows = [cells[(n, r)] for r in REPEATS]
        s = {}
        for pipe in ('SP', 'PR'):
            keys = ['coverage', 'width', 'q_cal', 'q_test', 'rho', 'rmse_last', 'decision_cov_r5', 'decision_cov_r10',
                    'n_cal'] + (['rho_matched', 'q_test_matched'] if pipe == 'SP' else ['cov_err_matched'])
            for key in keys:
                v = np.array([c[pipe][key] for c in rows], dtype=float)
                s[f'{pipe}_{key}'] = {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max())}
        for key in ('d', 'd_matched'):
            v = np.array([c[key] for c in rows])
            s[key] = {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max())}
        steps = np.array([c['SP']['kept_step'] for c in rows])
        rho = np.array([c['SP']['rho_matched'] for c in rows])
        s['SP_kept_step'] = {'median': float(np.median(steps)), 'min': int(steps.min()), 'max': int(steps.max())}
        s['PR_s_star_counts'] = {str(b): int(sum(c['PR']['s_star'] == b for c in rows))
                                 for b in (500, 1000, 2000, 4000, 8000)}
        tau = kendalltau(steps, rho)
        s['tau_kept_step_rho_matched_SP'] = float(tau.statistic) if np.isfinite(tau.statistic) else None
        out[n] = s
    return out


def analyse(cells_by_ds):
    res = {}
    for ds, cells in cells_by_ds.items():
        any_cell = next(iter(cells.values()))
        res[ds] = {'tests': family(cells, 'd_matched', 'rho_matched'),
                   'tests_as_first_written': family(cells, 'd', 'rho'),
                   'secondary': secondary(cells),
                   'n_test_windows': any_cell['n_test_windows'], 'n_test_short': any_cell['n_test_short']}
    res['replicated'] = {k: bool(res['FD001']['tests'][k]['significant'] and res['FD003']['tests'][k]['significant']
                                 and res['FD001']['tests'][k]['direction'] == res['FD003']['tests'][k]['direction'])
                         for k in TESTS}
    return res


# ---------------------------------------------------------------- report ------------------------
def fmt_p(p):
    return f'{p:.2g}' if p < 0.001 else f'{p:.3f}'


def test_label(k):
    return f'A11-P1(n = {k[4:]})' if k.startswith('P1') else f'A11-{k}'


def test_rows(tests):
    L = ['| Test | Statistic | Value | Repeats in predicted direction | p | Holm p | Claim allowed |',
         '|---|---|---|---|---|---|---|']
    for k in TESTS:
        t = tests[k]
        if k.startswith('P1'):
            cnt = f"{t['n_positive']}/{t['n']}"
        elif k == 'P2':
            cnt = f"{t['n_negative']}/{t['n']} (geometric mean {t['geometric_mean_rho']:.2f})"
        else:
            cnt = f"mean tau_b {t['mean_tau_b']:.2f}; negative in {t['n_tau_negative']}/{t['n']}"
        L.append(f"| {test_label(k)} | {t['stat']} | {t['mean']:.3f} | {cnt} | {fmt_p(t['p'])} | "
                 f"{fmt_p(t['p_holm'])} | {'yes' if t['significant'] else 'no'} |")
    return L


def report(res):
    L = ['# Amendment 11: in-unit against held-out-unit calibration on C-MAPSS', '',
         'Generated by `Experiments/r1_cmapss_stats.py` (plan Section 11, Amendment 11 and its correction 1). '
         'Unit: the repeat (random fleet and seed), 20 repeats; the 100 test engines of each subset are fixed. '
         'SP: calibration on every 10th window of the fleet engines after in-unit early stopping; PR: calibration '
         'on held-out engines (four folds), deployed model trained for the budget selected by those folds. Coverage '
         'of the two-sided 90 % residual-quantile interval over all test windows. Primary tests size-matched '
         '(correction 1): PR coverage error and the test quantile at the size of SP\'s calibration set.', '']
    for ds, role in DATASETS.items():
        r = res[ds]
        L += [f'## {ds} ({role} family; Holm over six)', ''] + test_rows(r['tests'])
        L += ['', f'As first written (secondary; full calibration sets, plain rho):', ''] + test_rows(r['tests_as_first_written'])
        L += ['', '| n | calibration errors SP / PR | coverage SP | coverage PR | d matched (d as written) | '
                  'rho~ SP (rho SP) | rho PR | q_cal / q_test SP | q_cal / q_test PR | '
                  'decision coverage r = 5, SP / PR (target 83.3) | r = 10 (target 90.9) | last-window RMSE SP / PR | '
                  'SP kept step, median (range) | PR S* counts (500 to 8,000) |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for n in FLEETS:
            s = r['secondary'][n]

            def m(key, f='.1f'):
                v = s[key]
                return f"{v['mean']:{f}} ({v['min']:{f}} to {v['max']:{f}})"
            L.append(f"| {n} | {s['SP_n_cal']['mean']:.0f} / {s['PR_n_cal']['mean']:.0f} | {m('SP_coverage')} | "
                     f"{m('PR_coverage')} | {s['d_matched']['mean']:.1f} ({s['d']['mean']:.1f}) | "
                     f"{m('SP_rho_matched', '.2f')} ({s['SP_rho']['mean']:.2f}) | {m('PR_rho', '.2f')} | "
                     f"{s['SP_q_cal']['mean']:.1f} / {s['SP_q_test']['mean']:.1f} | "
                     f"{s['PR_q_cal']['mean']:.1f} / {s['PR_q_test']['mean']:.1f} | "
                     f"{s['SP_decision_cov_r5']['mean']:.1f} / {s['PR_decision_cov_r5']['mean']:.1f} | "
                     f"{s['SP_decision_cov_r10']['mean']:.1f} / {s['PR_decision_cov_r10']['mean']:.1f} | "
                     f"{s['SP_rmse_last']['mean']:.1f} / {s['PR_rmse_last']['mean']:.1f} | "
                     f"{s['SP_kept_step']['median']:.0f} ({s['SP_kept_step']['min']} to {s['SP_kept_step']['max']}) | "
                     + ', '.join(str(c) for c in s['PR_s_star_counts'].values()) + ' |')
        L += ['', 'Within each n, Kendall tau between SP kept step and rho~ SP: '
              + '; '.join(f"n = {n}: {r['secondary'][n]['tau_kept_step_rho_matched_SP']:.2f}"
                          if r['secondary'][n]['tau_kept_step_rho_matched_SP'] is not None else f'n = {n}: undefined'
                          for n in FLEETS) + '.', '',
              f"Test windows: {r['n_test_windows']}; test engines with fewer than 30 cycles: {r['n_test_short']}. "
              f"Runs by plan version: {r.get('plans', {})}.", '']
    L += ['## Replication (primary tests)', '']
    for k in TESTS:
        a, b = res['FD001']['tests'][k], res['FD003']['tests'][k]
        L.append(f"- {test_label(k)}: FD001 {'significant' if a['significant'] else 'not significant'}, "
                 f"FD003 {'significant' if b['significant'] else 'not significant'}"
                 f"{'; replicated' if res['replicated'][k] else ''}.")
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- dry run -----------------------
SIZES = {4: 70, 10: 180, 25: 450, 80: 1400}       # SP calibration sizes like those of the runs


def synthetic_cell(rng, m, shrink, n_test=4000):
    """Laplace errors; SP calibration errors scaled by shrink (1: no effect); PR calibration 10 m"""
    last = np.zeros(n_test, bool)
    last[::40] = True
    scale = 10.0 + rng.normal(0, 0.5)
    z = {'test_y': rng.uniform(0, 125, n_test), 'test_last': last}
    z['sp_test_pred'] = z['test_y'] + rng.laplace(0, scale, n_test)
    z['pr_test_pred'] = z['test_y'] + rng.laplace(0, scale, n_test)
    z['sp_val_y'] = rng.uniform(0, 125, m)
    z['sp_val_pred'] = z['sp_val_y'] + rng.laplace(0, scale * shrink, m)
    z['pr_cal_e'] = rng.laplace(0, scale, 10 * m)
    meta = {'sp': {'best_step': int(rng.integers(50, 8000))}, 'pr': {'s_star': 1000},
            'n_test_windows': n_test, 'n_test_engines_short': 0}
    return z, meta


SIGNS20 = None


def signflip_p_fast(d):
    """vectorized equivalent of signflip_p (all 2^n sign patterns); used only in the dry run's
    rejection-rate check, after checking that it equals signflip_p"""
    global SIGNS20
    n = len(d)
    if SIGNS20 is None or SIGNS20.shape[1] != n:
        bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
        SIGNS20 = (1 - 2 * bits).astype(np.float64)
    obs = abs(d.mean())
    return float(np.mean(np.abs(SIGNS20 @ d / n) >= obs - 1e-15))


def dry_run():
    """dry run 3 (Amendment 11, correction 2); every check is printed, failures are listed at the end"""
    t0 = time.time()
    rng = np.random.default_rng(0)
    failures = []
    # 1. exact tests against Monte Carlo, and the fast sign-flip against signflip_p
    d = rng.normal(0.3, 1.0, 20)
    mc = np.mean([abs((d * rng.choice([-1, 1], 20)).mean()) >= abs(d.mean()) - 1e-15 for _ in range(200000)])
    p = signflip_p(d)
    print(f'sign-flip exact {p:.4f} against Monte Carlo {mc:.4f}')
    if abs(p - mc) >= 0.01:
        failures.append('sign-flip against Monte Carlo')
    for _ in range(3):
        v = rng.normal(0.2, 1.0, 20)
        if abs(signflip_p(v) - signflip_p_fast(v)) > 1e-12:
            failures.append('fast sign-flip differs')
    groups = [(FLEETS, list(rng.normal(size=4) + 0.3 * np.arange(4))) for _ in REPEATS]
    et, mcp = exact_test(groups)['p_two_sided'], monte_carlo_p(groups, 20000, rng)
    print(f'Kendall exact {et:.4f} against Monte Carlo {mcp:.4f}')
    if abs(et - mcp) >= 0.01:
        failures.append('Kendall against Monte Carlo')
    # 2. no effect: bias at each calibration size, 2,000 cells each
    for n, m in SIZES.items():
        d, dm, lr, rm = [], [], [], []
        for _ in range(2000):
            z, meta = synthetic_cell(rng, m, 1.0)
            c = cell_metrics(z, meta, rng)
            d.append(c['d'])
            dm.append(c['d_matched'])
            lr.append(np.log(c['SP']['rho']))
            rm.append(c['SP']['rho_matched'])
        rm = np.array(rm)
        print(f'no effect, m = {m}: mean d as written {np.mean(d):+.3f}, size-matched {np.mean(dm):+.3f} points; '
              f'mean log rho as written {np.mean(lr):+.4f}; rho~ median {np.median(rm):.4f}, '
              f'mean log {np.mean(np.log(rm)):+.4f}, share below 1 {np.mean(rm < 1):.3f}')
        if not (abs(np.mean(dm)) < 0.2 and abs(np.median(rm) - 1) < 0.01 and abs(np.mean(np.log(rm))) < 0.01):
            failures.append(f'bias check at m = {m}')
    # 3. no effect: rejection rates over 200 families of 80 cells
    rej = {k: 0 for k in TESTS}
    for _ in range(200):
        cells = {(n, r): cell_metrics(*synthetic_cell(rng, SIZES[n], 1.0, n_test=2000), rng)
                 for n in FLEETS for r in REPEATS}
        for n in FLEETS:
            rej[f'P1_n{n}'] += signflip_p_fast(np.array([cells[(n, r)]['d_matched'] for r in REPEATS])) < 0.05
        rej['P2'] += signflip_p_fast(np.log(np.array([cells[(80, r)]['SP']['rho_matched'] for r in REPEATS]))) < 0.05
        rej['P3'] += exact_test([(FLEETS, [cells[(n, r)]['SP']['rho_matched'] for n in FLEETS])
                                 for r in REPEATS])['p_two_sided'] < 0.05
    rates = {k: v / 200 for k, v in rej.items()}
    print('no effect, share of 200 families with p < 0.05 (unadjusted):', {k: round(v, 3) for k, v in rates.items()})
    for k in TESTS:
        if k != 'P2' and rates[k] > 0.08:
            failures.append(f'rejection rate {k}')
    # 4. planted effect (understatement shrinking with n) still detected
    cells = {}
    for i, n in enumerate(FLEETS):
        for r in REPEATS:
            z, meta = synthetic_cell(rng, SIZES[n], 0.5 + 0.1 * i)
            cells[(n, r)] = cell_metrics(z, meta, rng)
    tests = family(cells, 'd_matched', 'rho_matched')
    print('planted', {k: (round(tests[k]['mean'], 3), fmt_p(tests[k]['p_holm']), tests[k]['significant'])
                      for k in TESTS})
    if not (all(tests[k]['significant'] for k in TESTS) and tests['P3']['direction'] == 1):
        failures.append('planted effect missed')
    print(f"dry run {'FAILED: ' + '; '.join(failures) if failures else 'passed'} ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return
    loaded = {ds: load_dataset(ds) for ds in DATASETS}
    res = analyse({ds: cells for ds, (cells, _) in loaded.items()})
    for ds, (_, plans) in loaded.items():
        res[ds]['plans'] = plans
    res['provenance'] = provenance(CODE)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cmapss.json').write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
    (OUT / 'cmapss.md').write_text(report(res), encoding='utf-8')
    print(report(res))


if __name__ == '__main__':
    main()
