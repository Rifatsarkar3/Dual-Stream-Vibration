"""
r1_decision_ablations.py — Revision R1, analysis plan Amendment 3: the calibration and
maintenance-decision analysis of Amendment 2 (P1, P2) for the submitted architecture and its
seven ablations. The functions of r1_decision_models.py are used unchanged.

Inputs: full model, stored final20 predictions (30 MC passes) and the validation runs of
Section 3.2 (outputs/r1_lambda/lam0.08_B10_fold*); ablations, the 30-pass re-evaluation of their
final checkpoints (T3, outputs/r1_ablation_mc30/) and their four-fold validation runs (V3,
outputs/r1_ablation_val/). Each variant is calibrated on its own 20 validation runs only.

  A3-P2  primary: bearing-averaged interval score of the calibrated two-sided 90 % MC-dropout
         intervals, all10, 20 seeds; full model against each ablation, exact sign-flip test over
         bearings, Holm over the seven.
  A3-P1  primary: L(r) of the cost-matched quantile trigger, tau = 0.20, r = 5 and 10, all10;
         full model against each ablation, Holm over the seven at each ratio.
  Secondary: c*, coverage, width; residual-quantile interval; every deployable rule; Kendall tau
         (CES MAE ranking vs L(r) ranking); new6, orig4, zs3 (descriptive); sensitivity of T3
         (the full model re-evaluated by r1_ablation_mc30.py, A3-P2 repeated).

USAGE:
  python Experiments/r1_decision_ablations.py --check-replay   # V3 replay check only
  python Experiments/r1_decision_ablations.py                  # the analysis
  python Experiments/r1_decision_ablations.py --dry-run        # synthetic data, scratch output, no real files read
"""

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import r1_decision_models as dm  # noqa: E402  (Amendment 2; used unchanged)
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import SETS  # noqa: E402

FULL = 'full model'
ABLATIONS = {'static fusion': 'abl20_static', 'temporal stream only': 'abl20_temporal',
             'spectral stream only': 'abl20_spectral', 'without PGMC': 'abl20_no_pgmc',
             'without DANN': 'abl20_no_dann', 'without PGMC and DANN': 'abl20_neither',
             'PGMC batch ordering (soft)': 'abl20_pgmc_soft'}
FULL_FINAL = 'outputs/r1_final_B10/final20'
FULL_VAL = 'outputs/r1_lambda/lam0.08_B10_fold{f}'
SENS_FINAL = 'outputs/r1_ablation_mc30/final20'
REPLAY_NEW, REPLAY_OLD = 'outputs/r1_ablation_val_replay/full_foldA', 'outputs/r1_lambda/lam0.08_B10_foldA'
TOL = 1e-5
ALPHA = 0.05


# ---------------------------------------------------------------- data --------------------------
def val_files(pattern):
    return sum((sorted(glob.glob(str(ROOT / pattern.format(f=f) / 'valpreds_seed*.npz'))) for f in 'ABCD'), [])


def check_replay():
    """V3 replay check: the full model's fold-A runs repeated by r1_ablation_valpreds.sh must reproduce
    the stored validation predictions and validation histories exactly."""
    new_dir, old_dir = ROOT / REPLAY_NEW, ROOT / REPLAY_OLD
    old_files = sorted(old_dir.glob('valpreds_seed*.npz'))
    if len(old_files) != 5:
        raise SystemExit(f'replay: expected 5 stored validation files in {old_dir}, found {len(old_files)}')
    max_diff, n_arrays = 0.0, 0
    for f in old_files:
        g = new_dir / f.name
        if not g.exists():
            raise SystemExit(f'replay: {g} missing')
        a, b = np.load(f), np.load(g)
        if sorted(a.files) != sorted(b.files):
            raise SystemExit(f'replay: {f.name} has different arrays')
        for k in a.files:
            n_arrays += 1
            if not np.array_equal(a[k], b[k]):
                max_diff = max(max_diff, float(np.max(np.abs(a[k].astype(float) - b[k].astype(float)))))
    hist = {}
    for d, tag in ((old_dir, 'old'), (new_dir, 'new')):
        res = json.loads((d / 'results.json').read_text(encoding='utf-8'))
        hist[tag] = {r['seed']: r['val_history'] for r in res['per_seed']}
    hist_equal = hist['old'] == hist['new']
    exact = max_diff == 0.0 and hist_equal
    print(f"replay check: {len(old_files)} seeds, {n_arrays} arrays, max |difference| {max_diff:.3g}, "
          f"validation histories {'identical' if hist_equal else 'DIFFERENT'}: {'EXACT' if exact else 'FAILED'}")
    return {'exact': exact, 'max_abs_diff': max_diff, 'histories_identical': hist_equal, 'n_arrays': n_arrays}


def real_models():
    models = {FULL: {'final': dm.load_preds(FULL_FINAL), 'val': dm.load_val(val_files(FULL_VAL)), 'dropout': True,
                     'paths': {'final': FULL_FINAL, 'val': FULL_VAL}}}
    for label, short in ABLATIONS.items():
        fin, val = f'outputs/r1_ablation_mc30/{short}', f'outputs/r1_ablation_val/{short}_fold{{f}}'
        meta = json.loads((ROOT / fin / 'meta.json').read_text(encoding='utf-8'))
        if meta['mc_dropout_passes'] != 30 or meta['raw_check_max_abs_diff'] > TOL or meta['n_checkpoints'] != 20:
            raise SystemExit(f'{label}: T3 output {fin} incomplete or failed its check ({meta})')
        models[label] = {'final': dm.load_preds(fin), 'val': dm.load_val(val_files(val)), 'dropout': True,
                         'paths': {'final': fin, 'val': val, 't3_raw_check_max_abs_diff': meta['raw_check_max_abs_diff']}}
    for name, m in models.items():
        if len(m['val']) != 20 or sorted(m['final']) != sorted(dm.FINAL_SEEDS):
            raise SystemExit(f"{name}: expected 20 validation runs and the 20 final seeds; "
                             f"found {len(m['val'])}, {sorted(m['final'])}")
    meta = json.loads((ROOT / SENS_FINAL / 'meta.json').read_text(encoding='utf-8'))
    if meta['mc_dropout_passes'] != 30 or meta['raw_check_max_abs_diff'] > TOL or meta['n_checkpoints'] != 20:
        raise SystemExit(f'sensitivity: T3 output {SENS_FINAL} incomplete or failed its check ({meta})')
    return models, dm.load_preds(SENS_FINAL)


def synthetic_models(rng):
    """dry run: random degradation trajectories with variant-specific bias and noise"""
    bearings = sorted(set(sum((SETS[s] for s in dm.SET_NAMES), [])))
    lengths = {b: int(rng.integers(150, 600)) for b in bearings}

    def traj(bias, noise, seeds, spread):
        out = {}
        for s in seeds:
            out[s] = {}
            for b in bearings:
                y = np.linspace(1.0, 0.0, lengths[b])
                raw = np.clip(y + bias + noise * rng.standard_normal(len(y)), -0.2, 1.2)
                out[s][b] = {'y': y, 'raw': raw, 'mu': raw + 0.01 * rng.standard_normal(len(y)),
                             's': np.full(len(y), spread)}
        return out

    models = {}
    for i, name in enumerate([FULL] + list(ABLATIONS)):
        bias, noise, spread = 0.02 * (i % 3 - 1), 0.05 + 0.01 * i, 0.01 + 0.004 * i
        val = [traj(bias, noise, [0], spread)[0][bearings[0]] for _ in range(20)]
        models[name] = {'final': traj(bias, noise, dm.FINAL_SEEDS, spread), 'val': val, 'dropout': True,
                        'paths': {'final': 'synthetic', 'val': 'synthetic'}}
    return models, traj(0.0, 0.05, dm.FINAL_SEEDS, 0.01)


# ---------------------------------------------------------------- helpers -----------------------
def mc_intervals(data, bearings, c):
    return dm.intervals(data, bearings, sorted(data), lambda d: (d['mu'] - dm.Z * c * d['s'], d['mu'] + dm.Z * c * d['s']))


def per_bearing_is(e, bearings):
    return np.array([e['per_bearing'][b]['interval_score'] for b in bearings])


def outcome(rows):
    """ablations significantly worse / better than the full model (Holm-adjusted p < 0.05)"""
    worse = [n for n, r in rows.items() if r['p_holm'] < ALPHA and r['other'] > r['ref']]
    better = [n for n, r in rows.items() if r['p_holm'] < ALPHA and r['other'] < r['ref']]
    return {'significantly_worse': worse, 'significantly_better': better}


def cell(c, holm=True):
    return f"{c['rel']:+.1f} % ({c['k']}/{c['n']}, {c['p_flip']:.3f}" + (f", {c['p_holm']:.3f})" if holm else ')')


# ---------------------------------------------------------------- main --------------------------
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'decision_ablations.md'))
    ap.add_argument('--dry-run', action='store_true', help='synthetic data; writes to a temporary folder')
    ap.add_argument('--check-replay', action='store_true', help='V3 replay check only')
    args = ap.parse_args()
    if args.check_replay:
        return 0 if check_replay()['exact'] else 1
    if args.dry_run:
        models, sens = synthetic_models(np.random.default_rng(0))
        replay = {'exact': None, 'note': 'dry run'}
        out_path = Path(tempfile.mkdtemp(prefix='decision_ablations_dry_')) / 'decision_ablations.md'
    else:
        replay = check_replay()
        if not replay['exact']:
            raise SystemExit('V3 replay check failed; analysis not run (Amendment 3)')
        models, sens = real_models()
        out_path = Path(args.out)

    names, abl, a10 = [FULL] + list(ABLATIONS), list(ABLATIONS), SETS['all10']
    cal = {n: dm.calibrate(models[n]) for n in names}
    tim = {s: {n: dm.timing(models[n]['final'], SETS[s], cal[n], sorted(models[n]['final'])) for n in names}
           for s in dm.SET_NAMES}
    mae = {n: dm.ces_mae_bavg(models[n]['final'], a10) for n in names}
    report = {'dry_run': args.dry_run, 'replay_check': replay, 'paths': {n: models[n]['paths'] for n in names},
              'calibration': cal, 'ces_mae_bavg_all10': mae}

    # ---- A3-P2: calibrated MC-dropout intervals ----------------------------------------------------
    mc = {n: mc_intervals(models[n]['final'], a10, cal[n]['c_star']) for n in names}
    ref = per_bearing_is(mc[FULL], a10)
    p2 = dm.add_holm({n: dm.compare(ref, per_bearing_is(mc[n], a10)) for n in abl})
    report['A3_P2'] = {'intervals': mc, 'tests': p2, 'outcome': outcome(p2)}

    # ---- A3-P1: cost-matched quantile trigger ------------------------------------------------------
    p1, lv = {}, {}
    for r in dm.R_PRIMARY:
        lv[str(r)] = {n: dm.losses(tim['all10'][n][dm.TAU], f'cmq{r}', r, a10) for n in names}
        p1[str(r)] = dm.add_holm({n: dm.compare(lv[str(r)][FULL], lv[str(r)][n]) for n in abl})
    report['A3_P1'] = {str(r): {'L_mean': {n: float(lv[str(r)][n].mean()) for n in names}, 'tests': p1[str(r)],
                                'outcome': outcome(p1[str(r)])} for r in dm.R_PRIMARY}

    lines = ['# Components of the submitted architecture: calibration and maintenance decisions (revision R1, Amendment 3)',
             '', ('**DRY RUN on synthetic data: numbers are meaningless.**' if args.dry_run else
                  'Generated by `Experiments/r1_decision_ablations.py`; definitions in analysis plan Section 11, Amendment 3.'),
             '', ('V3 replay check: ' + ('exact (the full model\'s fold-A validation runs reproduced bit for bit).'
                                         if replay.get('exact') else 'not run (dry run).')),
             '', 'Ten test bearings (all10), 20 seeds per variant. Δ = (ablation − full) / ablation; positive: the full '
             'model is better. p: exact sign-flip test over bearings; Holm over the seven ablations.', '',
             '## Calibration (validation runs only) and accuracy', '',
             '| Variant | c* | q(0.9) | q(5/6) | q(10/11) | CES MAE, bearing avg |', '|---|---|---|---|---|---|']
    for n in names:
        q = cal[n]['q']
        lines.append(f"| {n} | {cal[n]['c_star']:.3f} | {q[0.9]:.4f} | {q[5 / 6]:.4f} | {q[10 / 11]:.4f} | {mae[n]:.4f} |")

    lines += ['', '## A3-P2 (primary): calibrated two-sided 90 % MC-dropout intervals', '',
              '| Variant | coverage, pooled | coverage, late life | mean width | interval score, bearing avg | '
              'full better by (better on, p, Holm) |', '|---|---|---|---|---|---|']
    for n in names:
        e = mc[n]
        wid = np.mean([e['per_bearing'][b]['width'] for b in a10])
        lines.append(f"| {n} | {e['pooled_coverage']:.1f} % | {e['late_coverage']:.1f} % | {wid:.4f} | "
                     f"{per_bearing_is(e, a10).mean():.4f} | {'reference' if n == FULL else cell(p2[n])} |")
    o = report['A3_P2']['outcome']
    lines += ['', f"Outcome: significantly worse than the full model {len(o['significantly_worse'])} of 7 "
                  f"({', '.join(o['significantly_worse']) or 'none'}); significantly better {len(o['significantly_better'])} "
                  f"of 7 ({', '.join(o['significantly_better']) or 'none'})."]

    lines += ['', f'## A3-P1 (primary): cost-matched quantile trigger, τ = {dm.TAU:.2f}', '',
              '| Variant | ' + ' | '.join(f'L(r = {r})' for r in dm.R_PRIMARY) + ' | '
              + ' | '.join(f'full lower by, r = {r} (better on, p, Holm)' for r in dm.R_PRIMARY) + ' |',
              '|---|' + '---|' * (2 * len(dm.R_PRIMARY))]
    for n in names:
        lines.append(f'| {n} | ' + ' | '.join(f"{lv[str(r)][n].mean():.1f}" for r in dm.R_PRIMARY) + ' | '
                     + ' | '.join('reference' if n == FULL else cell(p1[str(r)][n]) for r in dm.R_PRIMARY) + ' |')
    for r in dm.R_PRIMARY:
        o = report['A3_P1'][str(r)]['outcome']
        lines.append(f"\nOutcome at r = {r}: significantly worse than the full model {len(o['significantly_worse'])} of 7 "
                     f"({', '.join(o['significantly_worse']) or 'none'}); significantly better "
                     f"{len(o['significantly_better'])} of 7 ({', '.join(o['significantly_better']) or 'none'}).")

    # ---- secondary ----------------------------------------------------------------------------------
    res = {n: dm.intervals(models[n]['final'], a10, sorted(models[n]['final']),
                           lambda d, q=cal[n]['q']: (d['raw'] - q[0.95], d['raw'] - q[0.05])) for n in names}
    res_tests = {n: dm.compare(per_bearing_is(res[FULL], a10), per_bearing_is(res[n], a10)) for n in abl}
    report['secondary'] = {'residual_interval': res, 'residual_interval_tests': res_tests}
    lines += ['', '## Secondary', '', '### Residual-quantile interval [ŷ − q(0.95), ŷ − q(0.05)]', '',
              '| Variant | coverage, pooled | coverage, late life | mean width | interval score, bearing avg | '
              'full better by (better on, p) |', '|---|---|---|---|---|---|']
    for n in names:
        e = res[n]
        wid = np.mean([e['per_bearing'][b]['width'] for b in a10])
        lines.append(f"| {n} | {e['pooled_coverage']:.1f} % | {e['late_coverage']:.1f} % | {wid:.4f} | "
                     f"{per_bearing_is(e, a10).mean():.4f} | {'reference' if n == FULL else cell(res_tests[n], holm=False)} |")

    rules = {}
    lines += ['', f'### Every deployable rule, τ = {dm.TAU:.2f} (all10)', '',
              '| Variant | rule | ' + ' | '.join(f'L(r = {r})' for r in dm.R_SECONDARY) + ' |',
              '|---|---|' + '---|' * len(dm.R_SECONDARY)]
    for n in names:
        t = tim['all10'][n][dm.TAU]
        rules[n] = {rule: {str(r): float(dm.losses(t, rule, r, a10).mean()) for r in dm.R_SECONDARY}
                    for rule in dm.DEPLOYABLE if rule in t['A']}
        rules[n]['cmq (r = own)'] = {str(r): float(dm.losses(t, f'cmq{r}', r, a10).mean()) for r in dm.R_SECONDARY}
        for rule, vals in rules[n].items():
            lines.append(f'| {n} | {rule} | ' + ' | '.join(f'{vals[str(r)]:.1f}' for r in dm.R_SECONDARY) + ' |')
    report['secondary']['rules_all10'] = rules

    kt = {}
    lines += ['', '### Ranking: CES MAE against L(r) of the cost-matched trigger (all10)', '']
    for r in dm.R_PRIMARY:
        lmean = report['A3_P1'][str(r)]['L_mean']
        tau_k, p_k = stats.kendalltau([mae[n] for n in names], [lmean[n] for n in names])
        kt[str(r)] = {'tau': float(tau_k), 'p': float(p_k), 'ranking_ces_mae': sorted(names, key=mae.get),
                      'ranking_loss': sorted(names, key=lmean.get)}
        lines.append(f"- r = {r}: Kendall tau {tau_k:.2f} (p {p_k:.3f}). CES MAE: " + ', '.join(kt[str(r)]['ranking_ces_mae'])
                     + '. Loss: ' + ', '.join(kt[str(r)]['ranking_loss']) + '.')
    report['secondary']['kendall'] = kt

    other = {}
    lines += ['', '### Other bearing sets (descriptive; unadjusted p)', '',
              '| Set | Variant | interval score (MC) | full better by (better on, p) | '
              + ' | '.join(f'L(r = {r})' for r in dm.R_PRIMARY) + ' | '
              + ' | '.join(f'full lower by, r = {r} (better on, p)' for r in dm.R_PRIMARY) + ' |',
              '|---|---|---|---|' + '---|' * (2 * len(dm.R_PRIMARY))]
    for s in ('new6', 'orig4', 'zs3'):
        bs = SETS[s]
        e = {n: mc_intervals(models[n]['final'], bs, cal[n]['c_star']) for n in names}
        ls = {r: {n: dm.losses(tim[s][n][dm.TAU], f'cmq{r}', r, bs) for n in names} for r in dm.R_PRIMARY}
        other[s] = {}
        for n in names:
            c_is = None if n == FULL else dm.compare(per_bearing_is(e[FULL], bs), per_bearing_is(e[n], bs))
            c_l = {r: None if n == FULL else dm.compare(ls[r][FULL], ls[r][n]) for r in dm.R_PRIMARY}
            other[s][n] = {'interval_score': float(per_bearing_is(e[n], bs).mean()), 'is_vs_full': c_is,
                           'L': {str(r): float(ls[r][n].mean()) for r in dm.R_PRIMARY},
                           'L_vs_full': {str(r): c_l[r] for r in dm.R_PRIMARY}}
            lines.append(f"| {s} | {n} | {per_bearing_is(e[n], bs).mean():.4f} | "
                         f"{'reference' if c_is is None else cell(c_is, holm=False)} | "
                         + ' | '.join(f'{ls[r][n].mean():.1f}' for r in dm.R_PRIMARY) + ' | '
                         + ' | '.join('reference' if c_l[r] is None else cell(c_l[r], holm=False) for r in dm.R_PRIMARY)
                         + ' |')
    report['secondary']['other_sets'] = other

    # sensitivity of T3: the full model re-evaluated by r1_ablation_mc30.py (same calibration factor)
    e_sens = mc_intervals(sens, a10, cal[FULL]['c_star'])
    ref_s = per_bearing_is(e_sens, a10)
    p2s = dm.add_holm({n: dm.compare(ref_s, per_bearing_is(mc[n], a10)) for n in abl})
    report['secondary']['sensitivity_T3'] = {'full_interval_score_stored': float(ref.mean()),
                                             'full_interval_score_reevaluated': float(ref_s.mean()),
                                             'full_coverage_reevaluated': e_sens['pooled_coverage'],
                                             'tests': p2s, 'outcome': outcome(p2s)}
    o = outcome(p2s)
    lines += ['', '### Sensitivity of T3: A3-P2 with the full model re-evaluated by `r1_ablation_mc30.py`', '',
              f'Full model interval score: stored predictions {ref.mean():.4f}, re-evaluated {ref_s.mean():.4f} '
              f"(coverage {e_sens['pooled_coverage']:.1f} %).", '',
              '| Ablation | full better by (better on, p, Holm) |', '|---|---|']
    lines += [f'| {n} | {cell(p2s[n])} |' for n in abl]
    lines += ['', f"Outcome: significantly worse {len(o['significantly_worse'])} of 7 "
                  f"({', '.join(o['significantly_worse']) or 'none'}); significantly better "
                  f"{len(o['significantly_better'])} of 7 ({', '.join(o['significantly_better']) or 'none'})."]

    report['provenance'] = provenance(('Experiments/r1_decision_ablations.py', 'Experiments/r1_decision_models.py',
                                       'Experiments/exp29_maintenance_timing.py', 'Experiments/r1_strict_stats.py',
                                       'Experiments/r1_ablation_mc30.py', 'Experiments/r1_ablation_valpreds.sh'))
    text = '\n'.join(lines) + '\n'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')
    out_path.with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(text)
    print(f'written {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
