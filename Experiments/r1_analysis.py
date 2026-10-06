"""
r1_analysis.py — Revision R1: corrected PRONOSTIA result tables from the
re-evaluated predictions in outputs/r1_reeval/<variant>/preds_seed*.npz.

Bearing sets
  orig4 : the four test bearings of the submitted manuscript
  all10 : all Condition 1-2 bearings of the official test set (none used for selection)
  zs3   : the three Condition-3 bearings (operating condition unseen in training)

Reported per variant: deployable (Raw, CES, RTRM) and offline (PAVA) MAE/RMSE/R2,
monotonicity; paired seed-level tests against the full model; bearing-level
statistics (per-bearing means, sign counts, bootstrap over bearings); two-sided
interval coverage (PICP) and one-sided lower-bound coverage (LBC) at a given c*.

USAGE:
  python Experiments/r1_analysis.py [--c-star 2.414] [--sets orig4,all10,zs3]
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.isotonic_eval import IsotonicRegressor  # noqa: E402
from src.online_filters import (ConstrainedExponentialSmoothing,  # noqa: E402
                                RealTimeRunningMinimum)

SETS = {
    'orig4': ['test_01_Bearing1_3', 'test_02_Bearing1_4', 'test_03_Bearing1_5',
              'test_04_Bearing2_3'],
    'all10': ['test_01_Bearing1_3', 'test_02_Bearing1_4', 'test_03_Bearing1_5',
              'test_04_Bearing2_3', 'test_05_Bearing1_6', 'test_06_Bearing1_7',
              'test_07_Bearing2_4', 'test_08_Bearing2_5', 'test_09_Bearing2_6',
              'test_10_Bearing2_7'],
    'zs3': ['zs_01_Bearing3_1', 'zs_02_Bearing3_2', 'zs_03_Bearing3_3'],
}
METHODS = ('raw', 'ces', 'rtrm', 'pava')
Z = 1.6449
DEFAULT_ROOTS = f"={ROOT / 'outputs' / 'r1_reeval'}"


def load_variant(path):
    out = {}
    for f in sorted(glob.glob(str(Path(path) / 'preds_seed*.npz'))):
        d = np.load(f)
        seed = int(os.path.basename(f)[10:-4])
        bearings = {k.split('_', 1)[1] for k in d.files if k.startswith('tgt_')}
        out[seed] = {b: {'raw': d[f'raw_{b}'].astype(np.float64),
                         'y': d[f'tgt_{b}'].astype(np.float64),
                         'mu': d[f'mcmean_{b}'].astype(np.float64),
                         's': np.clip(d[f'mcstd_{b}'].astype(np.float64), 1e-4, None)}
                     for b in bearings}
    return out


def variants_of(raw):
    return {'raw': raw,
            'ces': np.asarray(ConstrainedExponentialSmoothing(alpha=0.2, epsilon=0.05).process(raw), float),
            'rtrm': np.asarray(RealTimeRunningMinimum().process(raw), float),
            'pava': np.asarray(IsotonicRegressor(increasing=False).fit_transform(raw), float)}


def seed_metrics(per_bearing, bearings, c_star):
    preds = {m: [] for m in METHODS}
    ys, mono_dec, mono_tot = [], 0, 0
    per_b = {}
    cov2, cov1, n_all = 0, 0, 0
    for b in bearings:
        d = per_bearing[b]
        v = variants_of(d['raw'])
        for m in METHODS:
            preds[m].append(v[m])
        ys.append(d['y'])
        diff = np.diff(d['raw'])
        mono_dec += int((diff <= 0).sum())
        mono_tot += len(diff)
        per_b[b] = {m: float(np.mean(np.abs(v[m] - d['y']))) for m in METHODS}
        lo, hi = d['mu'] - Z * c_star * d['s'], d['mu'] + Z * c_star * d['s']
        cov2 += int(np.sum((d['y'] >= lo) & (d['y'] <= hi)))
        cov1 += int(np.sum(d['y'] >= lo))
        n_all += len(d['y'])
    y = np.concatenate(ys)
    res = {'per_bearing': per_b, 'mono_raw': 100.0 * mono_dec / max(mono_tot, 1),
           'picp': 100.0 * cov2 / n_all, 'lbc': 100.0 * cov1 / n_all}
    for m in METHODS:
        p = np.concatenate(preds[m])
        res[f'{m}_mae'] = float(np.mean(np.abs(p - y)))
        res[f'{m}_rmse'] = float(np.sqrt(np.mean((p - y) ** 2)))
        res[f'{m}_r2'] = float(1 - np.sum((y - p) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))
    return res


def summarize(per_seed, key):
    v = np.array([r[key] for r in per_seed.values()])
    return float(v.mean()), float(v.std())


def paired(full, other, key):
    seeds = sorted(set(full) & set(other))
    a = np.array([full[s][key] for s in seeds])
    b = np.array([other[s][key] for s in seeds])
    diff = b - a
    return {'n': len(seeds), 'rel_pct': float(100 * (b.mean() - a.mean()) / b.mean()),
            'p_t': float(stats.ttest_rel(b, a)[1]),
            'p_wilcoxon': float(stats.wilcoxon(b, a)[1]) if len(seeds) > 5 else None,
            'd_z': float(diff.mean() / (diff.std(ddof=1) + 1e-12))}


def bearing_level(full, other, bearings, method, n_boot=10000, rng_seed=0):
    """Per-bearing seed-mean MAE; differences resampled over bearings, not seeds."""
    seeds = sorted(set(full) & set(other))
    fa = np.array([np.mean([full[s]['per_bearing'][b][method] for s in seeds]) for b in bearings])
    ob = np.array([np.mean([other[s]['per_bearing'][b][method] for s in seeds]) for b in bearings])
    diff = ob - fa                                   # > 0: full model better on that bearing
    rng = np.random.default_rng(rng_seed)
    boot = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return {'full_better_on': int((diff > 0).sum()), 'n_bearings': len(bearings),
            'mean_diff': float(diff.mean()),
            'ci95': [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            'sign_test_p': float(stats.binomtest(int((diff > 0).sum()), len(diff)).pvalue)}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--c-star', type=float, default=2.414251375198364,
                    help='two-sided interval scale from validation folds (exp22)')
    ap.add_argument('--sets', default='orig4,all10,zs3')
    ap.add_argument('--roots', default=DEFAULT_ROOTS,
                    help='comma-separated label=dir pairs; variants are "label:subdir" '
                         '(or "subdir" for an empty label)')
    ap.add_argument('--ref', default='final20', help='reference variant for paired tests')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'results.json'))
    args = ap.parse_args()

    paths = {}
    for item in args.roots.split(','):
        label, root = item.split('=', 1)
        for p in sorted(Path(root).iterdir()):
            if p.is_dir() and any(p.glob('preds_seed*.npz')):
                paths[f'{label}:{p.name}' if label else p.name] = p
    names = list(paths)
    data = {n: load_variant(paths[n]) for n in names}
    report = {'c_star': args.c_star, 'roots': args.roots, 'variants': names, 'sets': {}}

    for set_name in args.sets.split(','):
        bearings = SETS[set_name]
        per = {n: {s: seed_metrics(d, bearings, args.c_star) for s, d in data[n].items()}
               for n in names}
        ref = args.ref
        block = {'reference': ref, 'summary': {}, 'paired_vs_ref': {}, 'bearing_level_vs_ref': {},
                 'per_bearing_ref': {}}
        print(f'\n### {set_name} ({len(bearings)} bearings)\n')
        print('| Variant | n | Raw MAE | CES MAE | RTRM MAE | PAVA MAE | PAVA R2 | Mono% raw | PICP | LBC |')
        print('|---|---|---|---|---|---|---|---|---|---|')
        for n in names:
            s = {k: summarize(per[n], k) for k in
                 ('raw_mae', 'ces_mae', 'rtrm_mae', 'pava_mae', 'raw_r2', 'ces_r2', 'pava_r2',
                  'mono_raw', 'picp', 'lbc')}
            block['summary'][n] = s
            print(f"| {n} | {len(per[n])} | {s['raw_mae'][0]:.4f} ± {s['raw_mae'][1]:.4f} | "
                  f"{s['ces_mae'][0]:.4f} ± {s['ces_mae'][1]:.4f} | {s['rtrm_mae'][0]:.4f} | "
                  f"{s['pava_mae'][0]:.4f} ± {s['pava_mae'][1]:.4f} | {s['pava_r2'][0]:.3f} | "
                  f"{s['mono_raw'][0]:.1f} | {s['picp'][0]:.1f} | {s['lbc'][0]:.1f} |")
        if ref in per:
            print(f'\n| vs {ref} | Raw Δ% (p, d) | CES Δ% (p, d) | PAVA Δ% (p, d) | '
                  'bearings where reference better (raw / CES / PAVA) | bootstrap CI over bearings, CES |')
            print('|---|---|---|---|---|---|')
            for n in names:
                if n == ref:
                    continue
                pr = {m: paired(per[ref], per[n], f'{m}_mae') for m in ('raw', 'ces', 'pava')}
                bl = {m: bearing_level(per[ref], per[n], bearings, m) for m in ('raw', 'ces', 'pava')}
                block['paired_vs_ref'][n] = pr
                block['bearing_level_vs_ref'][n] = bl
                cells = ' | '.join(f"{pr[m]['rel_pct']:+.1f} ({pr[m]['p_t']:.1e}, {pr[m]['d_z']:.2f})"
                                   for m in ('raw', 'ces', 'pava'))
                signs = ' / '.join(f"{bl[m]['full_better_on']}/{bl[m]['n_bearings']}"
                                   for m in ('raw', 'ces', 'pava'))
                ci = bl['ces']['ci95']
                print(f'| {n} | {cells} | {signs} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |')
            for b in bearings:
                block['per_bearing_ref'][b] = {
                    m: float(np.mean([r['per_bearing'][b][m] for r in per[ref].values()]))
                    for m in METHODS}
        report['sets'][set_name] = block

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f'\nwritten {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
