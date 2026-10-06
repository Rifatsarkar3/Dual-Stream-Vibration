"""
r1_calibration.py — Revision R1, analysis plan Sections 4, 7 (B1, B3, B4) and 9.

Fitting data: final-epoch MC-dropout predictions (30 passes) on the held-out training
bearing of each leave-one-bearing-out fold run at the selected setting,
<val-root>/lam{lambda}_B{budget}_fold{A..D}/valpreds_seed*.npz (4 folds x 5 seeds).
Test data: the 20 final runs' predictions (--test-preds). Nothing here is trained.

Section 4 (global factors, the exp22 procedure on the corrected source): per run,
  c2 = q90(|y - mu| / (z s))  (two-sided),  k1 = q90((mu - y) / s)  (one-sided);
  c* and k* are the means over the 20 runs.
B3 (stage-aware): two-sided factors per predicted-RUL band [0, 0.3), [0.3, 0.6), [0.6, 1],
  fitted the same way within each band; leave-one-fold-out on the four folds first, then
  fitted on all four folds and applied once to the test predictions.
B1 (ensembles): mean of member MC means; variance = mean member variance + variance of
  member means; 5-member validation ensembles (one per fold) give the ensemble factor;
  test: four disjoint 5-member groups of the final seeds (seed-list order) and one
  20-member ensemble.
B4 (smoothing strength): CES MAE of the deterministic validation predictions for
  alpha in {0.05, 0.1, 0.2, 0.3, 0.5}, epsilon = 0.05; keep 0.2 unless another value is
  lower by more than one SE, then take the argmin.
Coverage is reported on all10, orig4, new6 and zs3, pooled, per bearing and per true-RUL
band [0, 0.2), [0.2, 0.5), [0.5, 1].

USAGE:
  python Experiments/r1_calibration.py --val-root outputs/r1_lambda --lam 0.08 --budget 10 \
      --test-preds outputs/r1_final_B10/final20
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
from r1_analysis import SETS  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from src.online_filters import ConstrainedExponentialSmoothing  # noqa: E402

Z = 1.6449
SUPERSEDED = {'exp22 (submitted, best-validation checkpoints)': 2.414251375198364,
              'r1_uq (first R1 estimate, best-validation checkpoints)': 3.18}
FOLDS = 'ABCD'
FINAL_SEEDS = [42, 123, 777, 999, 7, 8, 21, 34, 55, 89, 144, 233, 377, 610, 987,
               1597, 2584, 4181, 6765, 10946]
PRED_BANDS = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.01)]
TRUE_BANDS = {'late [0, 0.2)': (-0.01, 0.2), 'mid [0.2, 0.5)': (0.2, 0.5), 'early [0.5, 1]': (0.5, 1.01)}
ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5]
SETS = dict(SETS)
SETS['new6'] = [b for b in SETS['all10'] if b not in SETS['orig4']]


def load_val(val_root, lam, budget):
    runs = []
    for f in FOLDS:
        for p in sorted(glob.glob(str(ROOT / val_root / f'lam{lam}_B{budget}_fold{f}' / 'valpreds_seed*.npz'))):
            d = np.load(p)
            b = next(k[4:] for k in d.files if k.startswith('tgt_'))
            runs.append({'fold': f, 'seed': int(os.path.basename(p)[13:-4]), 'y': d[f'tgt_{b}'].astype(float),
                         'raw': d[f'raw_{b}'].astype(float), 'mu': d[f'mcmean_{b}'].astype(float),
                         's': np.clip(d[f'mcstd_{b}'].astype(float), 1e-4, None)})
    if len(runs) != 20:
        raise SystemExit(f'expected 20 validation runs, found {len(runs)}')
    return runs


def load_test(test_dir):
    out = {}
    for p in sorted(glob.glob(str(ROOT / test_dir / 'preds_seed*.npz'))):
        d = np.load(p)
        seed = int(os.path.basename(p)[10:-4])
        out[seed] = {k[4:]: {'y': d[k].astype(float), 'raw': d['raw_' + k[4:]].astype(float),
                             'mu': d['mcmean_' + k[4:]].astype(float),
                             's': np.clip(d['mcstd_' + k[4:]].astype(float), 1e-4, None)}
                     for k in d.files if k.startswith('tgt_')}
    return out


def c2(y, mu, s):
    return float(np.quantile(np.abs(y - mu) / (Z * s), 0.90))


def k1(y, mu, s):
    return float(np.quantile((mu - y) / s, 0.90))


def band_factors(runs, min_windows=20):
    out = []
    for lo, hi in PRED_BANDS:
        vals = [c2(r['y'][m], r['mu'][m], r['s'][m]) for r in runs
                for m in [(r['mu'] >= lo) & (r['mu'] < hi)] if m.sum() >= min_windows]
        out.append(float(np.mean(vals)) if vals else float('nan'))
    return out


def scale_for(mu, factors, global_c):
    sc = np.full_like(mu, global_c)
    for (lo, hi), f in zip(PRED_BANDS, factors):
        if np.isfinite(f):
            sc[(mu >= lo) & (mu < hi)] = f
    return sc


def coverage(y, lo, hi=None):
    inside = (y >= lo) if hi is None else ((y >= lo) & (y <= hi))
    res = {'pooled': 100.0 * float(inside.mean())}
    for name, (a, b) in TRUE_BANDS.items():
        m = (y >= a) & (y < b)
        res[name] = 100.0 * float(inside[m].mean()) if m.any() else float('nan')
    return res


def test_coverage(test, bearings, interval):
    """interval(d) -> (lo, hi or None); pooled over seeds and bearings, plus per bearing."""
    ys, ins = [], []
    per_b = {}
    for b in bearings:
        yb, ib = [], []
        for seed in sorted(test):
            d = test[seed][b]
            lo, hi = interval(d)
            yb.append(d['y'])
            ib.append((d['y'] >= lo) if hi is None else ((d['y'] >= lo) & (d['y'] <= hi)))
        per_b[b] = 100.0 * float(np.concatenate(ib).mean())
        ys += yb
        ins += ib
    y, inside = np.concatenate(ys), np.concatenate(ins)
    res = {'pooled': 100.0 * float(inside.mean()), 'per_bearing': per_b}
    for name, (a, b) in TRUE_BANDS.items():
        m = (y >= a) & (y < b)
        res[name] = 100.0 * float(inside[m].mean()) if m.any() else float('nan')
    return res


def ensemble(members):
    mu = np.mean([m['mu'] for m in members], axis=0)
    var = np.mean([m['s'] ** 2 for m in members], axis=0) + np.var([m['mu'] for m in members], axis=0)
    return {'y': members[0]['y'], 'raw': np.mean([m['raw'] for m in members], axis=0),
            'mu': mu, 's': np.sqrt(np.clip(var, 1e-8, None))}


def ces(x, alpha):
    return np.asarray(ConstrainedExponentialSmoothing(alpha=alpha, epsilon=0.05).process(x), float)


def val_only(val, args):
    c_runs = [c2(r['y'], r['mu'], r['s']) for r in val]
    print(f"validation only: c* = {np.mean(c_runs):.3f} (sd {np.std(c_runs):.3f}), "
          f"k* = {np.mean([k1(r['y'], r['mu'], r['s']) for r in val]):.3f}")
    for f in FOLDS:
        fit = [r for r in val if r['fold'] != f]
        held = [r for r in val if r['fold'] == f]
        g = float(np.mean([c2(r['y'], r['mu'], r['s']) for r in fit]))
        bf = band_factors(fit)
        y, mu, s = (np.concatenate([r[k] for r in held]) for k in ('y', 'mu', 's'))
        sc = scale_for(mu, bf, g)
        cg, cs = coverage(y, mu - Z * g * s, mu + Z * g * s), coverage(y, mu - Z * sc * s, mu + Z * sc * s)
        print(f"  B3 LOFO fold {f}: global {cg['pooled']:.1f}% (late {cg['late [0, 0.2)']:.1f}%)  "
              f"stage-aware {cs['pooled']:.1f}% (late {cs['late [0, 0.2)']:.1f}%)  bands {[round(x, 2) for x in bf]}")
    for a in ALPHAS:
        v = np.array([np.mean(np.abs(ces(r['raw'], a) - r['y'])) for r in val])
        print(f"  B4 alpha={a}: CES val MAE {v.mean():.4f} (SE {v.std(ddof=1) / np.sqrt(len(v)):.4f})")
    raw = np.array([np.mean(np.abs(r['raw'] - r['y'])) for r in val])
    ens = [ensemble([r for r in val if r['fold'] == f]) for f in FOLDS]
    print(f"  B1 validation raw MAE: single {raw.mean():.4f}; 5-member ensembles "
          f"{np.mean([np.mean(np.abs(e['raw'] - e['y'])) for e in ens]):.4f}")
    return 0


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-root', default='outputs/r1_lambda')
    ap.add_argument('--lam', default='0.08')
    ap.add_argument('--budget', type=int, default=10)
    ap.add_argument('--test-preds', default='outputs/r1_final_B10/final20')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'calibration_v2.json'))
    ap.add_argument('--val-only', action='store_true',
                    help='validation-side quantities only (factors, B3 leave-one-fold-out, B4); '
                         'test predictions are not read')
    args = ap.parse_args()
    val = load_val(args.val_root, args.lam, args.budget)
    if args.val_only:
        return val_only(val, args)
    test = load_test(args.test_preds)
    out = {'val_root': args.val_root, 'lambda': args.lam, 'budget': args.budget,
           'test_preds': args.test_preds, 'n_val_runs': len(val), 'n_test_seeds': len(test)}

    # ---- Section 4: global factors on the corrected source ------------------------------
    c_star = float(np.mean([c2(r['y'], r['mu'], r['s']) for r in val]))
    k_star = float(np.mean([k1(r['y'], r['mu'], r['s']) for r in val]))
    out['global'] = {'c_star': c_star, 'k_star': k_star,
                     'c2_per_run_sd': float(np.std([c2(r['y'], r['mu'], r['s']) for r in val]))}
    out['test'] = {}
    for set_name in ('all10', 'orig4', 'new6', 'zs3'):
        bs = SETS[set_name]
        out['test'][set_name] = {
            'uncalibrated_two_sided': test_coverage(test, bs, lambda d: (d['mu'] - Z * d['s'], d['mu'] + Z * d['s'])),
            'two_sided_c_star': test_coverage(test, bs, lambda d: (d['mu'] - Z * c_star * d['s'],
                                                                   d['mu'] + Z * c_star * d['s'])),
            'lower_bound_k_star': test_coverage(test, bs, lambda d: (d['mu'] - k_star * d['s'], None)),
            'lower_bound_c_star': test_coverage(test, bs, lambda d: (d['mu'] - Z * c_star * d['s'], None)),
            'superseded_two_sided': {name: test_coverage(test, bs, lambda d, c=c: (d['mu'] - Z * c * d['s'],
                                                                                   d['mu'] + Z * c * d['s']))
                                     for name, c in SUPERSEDED.items()},
        }

    # ---- B3: stage-aware factors -------------------------------------------------------
    lofo = {}
    for f in FOLDS:
        fit = [r for r in val if r['fold'] != f]
        held = [r for r in val if r['fold'] == f]
        g = float(np.mean([c2(r['y'], r['mu'], r['s']) for r in fit]))
        bf = band_factors(fit)
        y = np.concatenate([r['y'] for r in held])
        mu = np.concatenate([r['mu'] for r in held])
        s = np.concatenate([r['s'] for r in held])
        sc = scale_for(mu, bf, g)
        lofo[f] = {'global': coverage(y, mu - Z * g * s, mu + Z * g * s),
                   'stage_aware': coverage(y, mu - Z * sc * s, mu + Z * sc * s),
                   'band_factors': bf, 'global_factor': g}
    bands_all = band_factors(val)
    out['B3'] = {'band_factors': bands_all, 'leave_one_fold_out': lofo, 'test': {}}
    for set_name in ('all10', 'orig4', 'new6', 'zs3'):
        out['B3']['test'][set_name] = test_coverage(
            test, SETS[set_name],
            lambda d: (d['mu'] - Z * scale_for(d['mu'], bands_all, c_star) * d['s'],
                       d['mu'] + Z * scale_for(d['mu'], bands_all, c_star) * d['s']))

    # ---- B1: ensembles -----------------------------------------------------------------
    val_ens = [ensemble([r for r in val if r['fold'] == f]) for f in FOLDS]
    c_ens = float(np.mean([c2(e['y'], e['mu'], e['s']) for e in val_ens]))
    seeds_present = [s for s in FINAL_SEEDS if s in test]
    groups = [seeds_present[i:i + 5] for i in range(0, 20, 5)]
    out['B1'] = {'c_star_ensemble': c_ens, 'groups': groups, 'test': {}}

    def ens_entry(grp, bs):
        ens = {0: {b: ensemble([test[s][b] for s in grp]) for b in bs}}
        y = np.concatenate([ens[0][b]['y'] for b in bs])
        raw = np.concatenate([ens[0][b]['raw'] for b in bs])
        ces_p = np.concatenate([ces(ens[0][b]['raw'], 0.2) for b in bs])
        cov = test_coverage(ens, bs, lambda d: (d['mu'] - Z * c_ens * d['s'], d['mu'] + Z * c_ens * d['s']))
        return {'raw_mae': float(np.mean(np.abs(raw - y))), 'ces_mae': float(np.mean(np.abs(ces_p - y))),
                'raw_mae_bearing_avg': float(np.mean([np.mean(np.abs(ens[0][b]['raw'] - ens[0][b]['y']))
                                                      for b in bs])),
                'ces_mae_bearing_avg': float(np.mean([np.mean(np.abs(ces(ens[0][b]['raw'], 0.2) - ens[0][b]['y']))
                                                      for b in bs])),
                'two_sided_pooled': cov['pooled'], 'two_sided_late': cov['late [0, 0.2)']}

    for set_name in ('all10', 'orig4', 'new6', 'zs3'):
        bs = SETS[set_name]
        five = [ens_entry(g, bs) for g in groups]
        single_raw = [np.mean(np.abs(np.concatenate([test[s][b]['raw'] for b in bs])
                                     - np.concatenate([test[s][b]['y'] for b in bs]))) for s in seeds_present]
        single_b = {m: float(np.mean([np.mean([np.mean(np.abs(f(test[s][b]['raw']) - test[s][b]['y'])) for b in bs])
                                      for s in seeds_present]))
                    for m, f in (('raw', lambda x: x), ('ces', lambda x: ces(x, 0.2)))}
        out['B1']['test'][set_name] = {
            'single_model_raw_mae_mean_over_seeds': float(np.mean(single_raw)),
            'single_model_raw_mae_bearing_avg': single_b['raw'],
            'single_model_ces_mae_bearing_avg': single_b['ces'],
            '5-member_mean_of_4_groups': {k: float(np.mean([e[k] for e in five])) for k in five[0]},
            '20-member': ens_entry(seeds_present, bs)}

    # ---- B4: smoothing strength --------------------------------------------------------
    b4 = {}
    for a in ALPHAS:
        v = np.array([np.mean(np.abs(ces(r['raw'], a) - r['y'])) for r in val])
        b4[a] = {'mean': float(v.mean()), 'se': float(v.std(ddof=1) / np.sqrt(len(v)))}
    best = min(ALPHAS, key=lambda a: b4[a]['mean'])
    chosen = 0.2 if b4[0.2]['mean'] <= b4[best]['mean'] + b4[best]['se'] else best
    out['B4'] = {'scores': {str(a): s for a, s in b4.items()}, 'argmin': best, 'selected_alpha': chosen}
    out['provenance'] = provenance(('Experiments/r1_calibration.py', 'Experiments/r1_analysis.py'))

    Path(args.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    a10 = out['test']['all10']
    print(f"Section 4: c* = {c_star:.3f} (sd over runs {out['global']['c2_per_run_sd']:.3f}), k* = {k_star:.3f}")
    print(f"  all10 two-sided: uncalibrated {a10['uncalibrated_two_sided']['pooled']:.1f}% -> "
          f"{a10['two_sided_c_star']['pooled']:.1f}%  (late {a10['two_sided_c_star']['late [0, 0.2)']:.1f}%)")
    print(f"  all10 lower bound: k* {a10['lower_bound_k_star']['pooled']:.1f}% (late "
          f"{a10['lower_bound_k_star']['late [0, 0.2)']:.1f}%), c* {a10['lower_bound_c_star']['pooled']:.1f}%")
    print('B3 band factors', [round(x, 3) for x in bands_all], '| LOFO pooled global vs stage-aware: ' +
          ', '.join(f"{f}: {lofo[f]['global']['pooled']:.1f}/{lofo[f]['stage_aware']['pooled']:.1f}" for f in FOLDS))
    t3 = out['B3']['test']['all10']
    print(f"  B3 test all10 two-sided {t3['pooled']:.1f}% (late {t3['late [0, 0.2)']:.1f}%)")
    e = out['B1']['test']['all10']
    print(f"B1 c*_ens = {c_ens:.3f}; all10 raw MAE single {e['single_model_raw_mae_mean_over_seeds']:.4f}, "
          f"5-member {e['5-member_mean_of_4_groups']['raw_mae']:.4f}, 20-member {e['20-member']['raw_mae']:.4f}; "
          f"two-sided {e['5-member_mean_of_4_groups']['two_sided_pooled']:.1f}% / "
          f"{e['20-member']['two_sided_pooled']:.1f}%")
    print(f"  all10 bearing-averaged CES MAE: single {e['single_model_ces_mae_bearing_avg']:.4f}, "
          f"5-member {e['5-member_mean_of_4_groups']['ces_mae_bearing_avg']:.4f}, "
          f"20-member {e['20-member']['ces_mae_bearing_avg']:.4f}")
    for name, cov in a10['superseded_two_sided'].items():
        print(f"  superseded factor, {name}: all10 two-sided {cov['pooled']:.1f}% (late {cov['late [0, 0.2)']:.1f}%)")
    print('B4 CES alpha: ' + ', '.join(f"{a}:{b4[a]['mean']:.4f}" for a in ALPHAS) + f' -> selected {chosen}')
    print(f'written {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
