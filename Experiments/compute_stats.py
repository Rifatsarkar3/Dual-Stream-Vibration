"""
compute_stats.py — Paired t-tests + Cohen's d for the manuscript.

Pairs per-seed PAVA MAE of the full model (final4) against each ablation and
each baseline (exp19). Seeds are matched, so a paired test is appropriate.

RUN: .venv\\Scripts\\python.exe Experiments/compute_stats.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
E16 = ROOT / 'outputs' / 'exp16_ress'
E19 = ROOT / 'outputs' / 'exp19_baselines'


def seed_maes(results_path: Path, metric='pava'):
    with open(results_path, encoding='utf-8') as f:
        res = json.load(f)
    out = {}
    for s in res['per_seed']:
        out[s['seed']] = s[metric]['mae']
    return out


def compare(name, full, other):
    seeds = sorted(set(full) & set(other))
    a = np.array([full[s] for s in seeds])
    b = np.array([other[s] for s in seeds])
    diff = b - a
    t, p = stats.ttest_rel(b, a)
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    rel = (b.mean() - a.mean()) / b.mean() * 100
    print(f"{name:32s} other={b.mean():.4f} full={a.mean():.4f} "
          f"Δ={rel:+.1f}% (full better if +) t={t:.2f} p={p:.4f} d={d:.2f}")
    return {'other_mean': float(b.mean()), 'full_mean': float(a.mean()),
            'rel_improvement_pct': float(rel), 't': float(t), 'p': float(p),
            'cohens_d': float(d), 'n_seeds': len(seeds)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', default='final4', help='full-model variant dir name')
    ap.add_argument('--abl-prefix', default='abl4', help='ablation dir prefix')
    ap.add_argument('--baseline-root', default=str(E19))
    ap.add_argument('--out', default='outputs/stats_tests.json')
    args = ap.parse_args()

    full = seed_maes(E16 / args.final / 'results.json')
    out = {}

    print(f'=== Ablations (PAVA MAE, paired over {len(full)} seeds) ===')
    for label, suffix in [('w/o PGMC', 'no_pgmc'),
                          ('w/o DANN', 'no_dann'),
                          ('w/o adaptive gate', 'static'),
                          ('temporal-only', 'temporal'),
                          ('spectral-only', 'spectral'),
                          ('PGMC batch-ordering', 'pgmc_soft')]:
        p = E16 / f'{args.abl_prefix}_{suffix}' / 'results.json'
        if p.exists():
            out[label] = compare(label, full, seed_maes(p))

    print('\n=== Baselines (PAVA MAE, paired) ===')
    broot = Path(args.baseline_root)
    if broot.exists():
        for mdir in sorted(broot.iterdir()):
            p = mdir / 'results.json'
            if p.exists():
                out[f'baseline {mdir.name}'] = compare(
                    f'baseline {mdir.name}', full, seed_maes(p))

    with open(ROOT / args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
