"""
r1_cmapss_blocked_stats.py - Revision R1, Amendment 12 (plan Section 11): analysis of the blocked
in-unit scheme (SB, Experiments/r1_cmapss_blocked.py) against the protocol (PR) and standard practice
(SP) of Amendment 11, whose outputs are reused, not rerun. Run once on the real runs; --dry-run checks
it on synthetic data.

Per cell (dataset, fleet size n, repeat r), each pipeline with its own deployed model and calibration
errors, measured as in Amendment 11 (r1_cmapss_stats.pipeline_metrics): coverage of
[y_hat - q(0.95), y_hat - q(0.05)] over all test windows; rho = 0.9-quantile of |e| on the calibration
data / that on the test windows. SB and PR have one calibration error per fleet window each, so no size
matching. Primary family per dataset (FD001 primary, FD003 replication), Holm over six:
  A12-P1(n), n = 4, 10, 25: c = cov_PR - cov_SB (points); exact sign-flip over 20 repeats; predicted > 0.
  A12-P2(n): L = log rho_SB - log rho_PR; exact sign-flip; predicted < 0.
Secondary: SB beside SP (size-matched rho~ exactly as in Amendment 11) and PR; the share of SP's
understatement removed by blocking g = (log rho_SB - log rho~_SP) / (log rho_PR - log rho~_SP), median,
reported where the denominator is positive in at least 15 of 20 repeats (correction 1).

Usage:
  python Experiments/r1_cmapss_blocked_stats.py --dry-run
  python Experiments/r1_cmapss_blocked_stats.py
Outputs: outputs/r1_analysis/cmapss_blocked.md and cmapss_blocked.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Experiments'))
import r1_cmapss_stats as S11  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import holm, signflip_p  # noqa: E402

RUNS = ROOT / 'outputs' / 'r1_cmapss_blocked'
RUNS11 = ROOT / 'outputs' / 'r1_cmapss'
OUT = ROOT / 'outputs' / 'r1_analysis'
PLAN_SHA_RUNS = {
    '7b779917a6c3d4198bb778bf74c287721d93415a4c6d744ee37551895eb1d598': 'Amendment 12',
    'f13f98792b12d5828592f2e913306d29e516a6c1d1e63176ff558069a4a3bf13': 'Amendment 12, correction 1',
}
DATASETS = {'FD001': 'primary', 'FD003': 'replication'}
FLEETS = [4, 10, 25]
REPEATS = list(range(20))
TESTS = ['P1_n4', 'P1_n10', 'P1_n25', 'P2_n4', 'P2_n10', 'P2_n25']
CODE = ['Experiments/r1_cmapss_blocked.py', 'Experiments/r1_cmapss_blocked_stats.py',
        'Experiments/r1_cmapss_stats.py', 'Experiments/r1_strict_stats.py']


# ---------------------------------------------------------------- per cell ----------------------
def primary(sb_cal, sb_test, pr_cal, pr_test, last):
    sb, pr = S11.pipeline_metrics(sb_cal, sb_test, last), S11.pipeline_metrics(pr_cal, pr_test, last)
    return sb, pr, pr['coverage'] - sb['coverage'], float(np.log(sb['rho']) - np.log(pr['rho']))


def load_cell(ds, n, r):
    stem, stem11 = RUNS / ds / f'n{n:02d}_r{r:02d}', RUNS11 / ds / f'n{n:02d}_r{r:02d}'
    meta = json.loads(stem.with_suffix('.json').read_text(encoding='utf-8'))
    meta11 = json.loads(stem11.with_suffix('.json').read_text(encoding='utf-8'))
    plan = meta['provenance']['plan_sha256']
    if plan not in PLAN_SHA_RUNS:
        raise SystemExit(f'{stem}: run made under plan {plan[:8]}')
    if (meta['dataset'], meta['n'], meta['repeat']) != (ds, n, r) or meta['fleet'] != meta11['fleet'] \
            or meta['seed'] != meta11['seed']:
        raise SystemExit(f'{stem}: does not match the Amendment 11 cell')
    with np.load(stem.with_suffix('.npz')) as zs, np.load(stem11.with_suffix('.npz')) as z11:
        zs, z11 = dict(zs), dict(z11)
    for key in ('test_y', 'test_unit', 'test_index', 'test_last'):
        if not np.array_equal(zs[key], z11[key]):
            raise SystemExit(f'{stem}: {key} differs from Amendment 11')
    if len(zs['sb_cal_e']) != len(z11['pr_cal_e']):
        raise SystemExit(f'{stem}: calibration sizes differ ({len(zs["sb_cal_e"])} against {len(z11["pr_cal_e"])})')
    y, last = z11['test_y'], z11['test_last'].astype(bool)
    sb, pr, c, L = primary(zs['sb_cal_e'], zs['sb_test_pred'] - y, z11['pr_cal_e'], z11['pr_test_pred'] - y, last)
    c11 = S11.cell_metrics(z11, meta11, np.random.default_rng([S11.DS_CODE[ds], n, r]))   # as in Amendment 11
    if abs(c11['PR']['coverage'] - pr['coverage']) > 1e-9:
        raise SystemExit(f'{stem}: PR coverage differs from Amendment 11')
    sp = c11['SP']
    sb['s_star'], pr['s_star'] = int(meta['sb']['s_star']), int(meta11['pr']['s_star'])
    b = [int(v) for v in zs['sb_budgets']].index(sb['s_star'])
    sb['block_mae'] = [float(np.mean(np.abs(zs[f'sb_fold{k}_pred'][b] - zs[f'sb_fold{k}_y']))) for k in range(4)]
    return {'SB': sb, 'PR': pr, 'SP': sp, 'c': c, 'L': L,
            'g_num': float(np.log(sb['rho']) - np.log(sp['rho_matched'])),
            'g_den': float(np.log(pr['rho']) - np.log(sp['rho_matched'])),
            'same_s_star': sb['s_star'] == pr['s_star'],
            'identical_test_pred': bool(np.array_equal(zs['sb_test_pred'], z11['pr_test_pred'])),
            'plan': PLAN_SHA_RUNS[plan], 'n_test_windows': int(meta['n_test_windows'])}


def load_dataset(ds):
    return {(n, r): load_cell(ds, n, r) for n in FLEETS for r in REPEATS}


# ---------------------------------------------------------------- tests -------------------------
def family(cells, pvalue=signflip_p):
    tests = {}
    for n in FLEETS:
        c = np.array([cells[(n, r)]['c'] for r in REPEATS])
        tests[f'P1_n{n}'] = {'stat': 'mean c = cov_PR - cov_SB (points)', 'mean': float(c.mean()), 'p': pvalue(c),
                             'n_predicted': int((c > 0).sum()), 'n': len(c), 'predicted_sign': 1}
        L = np.array([cells[(n, r)]['L'] for r in REPEATS])
        tests[f'P2_n{n}'] = {'stat': 'mean L = log rho_SB - log rho_PR', 'mean': float(L.mean()), 'p': pvalue(L),
                             'n_predicted': int((L < 0).sum()), 'n': len(L), 'predicted_sign': -1,
                             'geometric_mean_ratio': float(np.exp(L.mean()))}
    adj = holm(np.array([tests[k]['p'] for k in TESTS]))
    for k, a in zip(TESTS, adj):
        t = tests[k]
        t['p_holm'] = float(a)
        sign = int(np.sign(t['mean']))
        t['direction'] = sign
        t['significant'] = bool(a < 0.05 and sign == t['predicted_sign'])
        t['significant_opposite'] = bool(a < 0.05 and sign == -t['predicted_sign'])
    return tests


def secondary(cells):
    out = {}
    for n in FLEETS:
        rows = [cells[(n, r)] for r in REPEATS]
        s = {}
        for pipe, keys in (('SB', ['coverage', 'width', 'q_cal', 'q_test', 'rho', 'rmse_last', 'decision_cov_r5',
                                   'decision_cov_r10', 'n_cal']),
                           ('PR', ['coverage', 'width', 'q_cal', 'q_test', 'rho', 'rmse_last', 'decision_cov_r5',
                                   'decision_cov_r10', 'n_cal']),
                           ('SP', ['coverage', 'width', 'q_cal', 'q_test', 'rho', 'rho_matched', 'decision_cov_r5',
                                   'decision_cov_r10', 'n_cal'])):
            for key in keys:
                v = np.array([c[pipe][key] for c in rows], dtype=float)
                s[f'{pipe}_{key}'] = {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max())}
        for key in ('c', 'L'):
            v = np.array([c[key] for c in rows])
            s[key] = {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max())}
        den = np.array([c['g_den'] for c in rows])
        g = np.array([c['g_num'] for c in rows]) / den
        s['g'] = {'n_den_positive': int((den > 0).sum()),
                  'median': float(np.median(g)) if (den > 0).sum() >= 15 else None}
        s['SB_s_star_counts'] = {str(b): int(sum(c['SB']['s_star'] == b for c in rows)) for b in (500, 1000, 2000, 4000, 8000)}
        s['PR_s_star_counts'] = {str(b): int(sum(c['PR']['s_star'] == b for c in rows)) for b in (500, 1000, 2000, 4000, 8000)}
        s['same_s_star'] = int(sum(c['same_s_star'] for c in rows))
        s['identical_test_pred'] = int(sum(c['identical_test_pred'] for c in rows))
        s['SB_block_mae'] = [float(np.mean([c['SB']['block_mae'][k] for c in rows])) for k in range(4)]
        out[n] = s
    return out


def analyse(cells_by_ds):
    res = {}
    for ds, cells in cells_by_ds.items():
        plans = {}
        for c in cells.values():
            plans[c['plan']] = plans.get(c['plan'], 0) + 1
        res[ds] = {'tests': family(cells), 'secondary': secondary(cells), 'plans': plans,
                   'n_test_windows': next(iter(cells.values()))['n_test_windows']}
    res['replicated'] = {k: bool(res['FD001']['tests'][k]['significant'] and res['FD003']['tests'][k]['significant'])
                         for k in TESTS}
    res['replicated_opposite'] = {k: bool(res['FD001']['tests'][k]['significant_opposite']
                                          and res['FD003']['tests'][k]['significant_opposite']) for k in TESTS}
    return res


# ---------------------------------------------------------------- report ------------------------
def fmt_p(p):
    return f'{p:.2g}' if p < 0.001 else f'{p:.3f}'


def label(k):
    return f"A12-{k[:2]}(n = {k[4:]})"


def report(res):
    L = ['# Amendment 12: blocked in-unit calibration against held-out engines on C-MAPSS', '',
         'Generated by `Experiments/r1_cmapss_blocked_stats.py` (plan Section 11, Amendment 12 and its correction 1). '
         'Unit: the repeat (random fleet and seed), 20 repeats; the 100 test engines of each subset are fixed. '
         'SB: four folds of contiguous time blocks inside every fleet engine, 29-window buffers (no shared cycle), '
         'budget and calibration from the block errors, deployed model trained on the whole fleet; PR and SP: '
         'Amendment 11 (reused). Coverage of the two-sided 90 % residual-quantile interval over all test windows.', '']
    for ds, role in DATASETS.items():
        r = res[ds]
        L += [f'## {ds} ({role} family; Holm over six)', '',
              '| Test | Statistic | Value | Repeats in predicted direction | p | Holm p | Claim |', '|---|---|---|---|---|---|---|']
        for k in TESTS:
            t = r['tests'][k]
            claim = 'predicted direction' if t['significant'] else ('opposite direction' if t['significant_opposite']
                                                                    else 'not detected')
            extra = f" (geometric mean ratio {t['geometric_mean_ratio']:.2f})" if 'geometric_mean_ratio' in t else ''
            L.append(f"| {label(k)} | {t['stat']} | {t['mean']:.3f}{extra} | {t['n_predicted']}/{t['n']} | "
                     f"{fmt_p(t['p'])} | {fmt_p(t['p_holm'])} | {claim} |")
        L += ['', '| n | coverage SP / SB / PR | rho~ SP / rho SB / rho PR | q_cal / q_test SB | width SB / PR | '
                  'decision coverage r = 5, SB / PR | r = 10, SB / PR | last-window RMSE SB / PR | g (median; den > 0) | '
                  'S* SB counts (500 to 8,000) | S* PR counts | same S* / identical test predictions | block MAE at S* (folds 0 to 3) |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for n in FLEETS:
            s = r['secondary'][n]

            def m(key, f='.1f'):
                return f"{s[key]['mean']:{f}}"
            g = s['g']
            L.append(f"| {n} | {m('SP_coverage')} / {m('SB_coverage')} ({s['SB_coverage']['min']:.1f} to "
                     f"{s['SB_coverage']['max']:.1f}) / {m('PR_coverage')} | {m('SP_rho_matched', '.2f')} / {m('SB_rho', '.2f')} / "
                     f"{m('PR_rho', '.2f')} | {m('SB_q_cal')} / {m('SB_q_test')} | {m('SB_width')} / {m('PR_width')} | "
                     f"{m('SB_decision_cov_r5')} / {m('PR_decision_cov_r5')} | {m('SB_decision_cov_r10')} / "
                     f"{m('PR_decision_cov_r10')} | {m('SB_rmse_last')} / {m('PR_rmse_last')} | "
                     f"{'n/a' if g['median'] is None else format(g['median'], '.2f')} ({g['n_den_positive']}/20) | "
                     + ', '.join(str(v) for v in s['SB_s_star_counts'].values()) + ' | '
                     + ', '.join(str(v) for v in s['PR_s_star_counts'].values())
                     + f" | {s['same_s_star']} / {s['identical_test_pred']} | "
                     + ', '.join(f'{v:.1f}' for v in s['SB_block_mae']) + ' |')
        L += ['', f"Test windows: {r['n_test_windows']}. Runs by plan version: {r['plans']}.", '']
    L += ['## Replication', '']
    for k in TESTS:
        a, b = res['FD001']['tests'][k], res['FD003']['tests'][k]
        state = 'replicated' if res['replicated'][k] else ('replicated, opposite direction'
                                                           if res['replicated_opposite'][k] else 'not replicated')
        L.append(f"- {label(k)}: FD001 Holm p {fmt_p(a['p_holm'])}, FD003 Holm p {fmt_p(b['p_holm'])}; {state}.")
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------- dry run -----------------------
SIZES = {4: 700, 10: 1800, 25: 4400}      # calibration errors per pipeline, like one per fleet window


def synthetic_primary(rng, m, shrink, n_test=4000):
    """Laplace errors; SB calibration errors scaled by shrink (1: no effect); PR and SB calibration of
    size m; their test errors independent, or shared in half of the cells (as when S*_SB = S*_PR)"""
    last = np.zeros(n_test, bool)
    last[::40] = True
    scale = 10.0 + rng.normal(0, 0.5)
    pr_test = rng.laplace(0, scale, n_test)
    sb_test = pr_test if rng.random() < 0.5 else rng.laplace(0, scale, n_test)
    pr_cal, sb_cal = rng.laplace(0, scale, m), rng.laplace(0, scale * shrink, m)
    _, _, c, L = primary(sb_cal, sb_test, pr_cal, pr_test, last)
    return {'c': c, 'L': L}


def dry_run():
    t0 = time.time()
    rng = np.random.default_rng(12)
    failures = []
    d = rng.normal(0.3, 1.0, 20)
    mc = np.mean([abs((d * rng.choice([-1, 1], 20)).mean()) >= abs(d.mean()) - 1e-15 for _ in range(200000)])
    p = signflip_p(d)
    print(f'sign-flip exact {p:.4f} against Monte Carlo {mc:.4f}')
    if abs(p - mc) >= 0.01:
        failures.append('sign-flip against Monte Carlo')
    for _ in range(3):
        v = rng.normal(0.2, 1.0, 20)
        if abs(signflip_p(v) - S11.signflip_p_fast(v)) > 1e-12:
            failures.append('fast sign-flip differs')
    for n, m in SIZES.items():
        cells = [synthetic_primary(rng, m, 1.0) for _ in range(2000)]
        mc_, ml = np.mean([c['c'] for c in cells]), np.mean([c['L'] for c in cells])
        print(f'no effect, m = {m}: mean c {mc_:+.3f} points, mean L {ml:+.4f}')
        if not (abs(mc_) < 0.2 and abs(ml) < 0.01):
            failures.append(f'bias check at m = {m}')
    rej = {k: 0 for k in TESTS}
    for _ in range(200):
        cells = {(n, r): synthetic_primary(rng, SIZES[n], 1.0, n_test=2000) for n in FLEETS for r in REPEATS}
        tests = family(cells, pvalue=S11.signflip_p_fast)
        for k in TESTS:
            rej[k] += tests[k]['p'] < 0.05
    rates = {k: v / 200 for k, v in rej.items()}
    print('no effect, share of 200 families with p < 0.05 (unadjusted):', {k: round(v, 3) for k, v in rates.items()})
    for k in TESTS:
        if rates[k] > 0.08:
            failures.append(f'rejection rate {k}')
    cells = {(n, r): synthetic_primary(rng, SIZES[n], shrink) for n, shrink in zip(FLEETS, (0.6, 0.7, 0.8))
             for r in REPEATS}
    tests = family(cells)
    print('planted', {k: (round(tests[k]['mean'], 3), fmt_p(tests[k]['p_holm']), tests[k]['significant']) for k in TESTS})
    if not all(tests[k]['significant'] for k in TESTS):
        failures.append('planted effect missed')
    print(f"dry run {'FAILED: ' + '; '.join(failures) if failures else 'passed'} ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return
    res = analyse({ds: load_dataset(ds) for ds in DATASETS})
    res['provenance'] = provenance(CODE)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cmapss_blocked.json').write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
    (OUT / 'cmapss_blocked.md').write_text(report(res), encoding='utf-8')
    print(report(res))


if __name__ == '__main__':
    main()
