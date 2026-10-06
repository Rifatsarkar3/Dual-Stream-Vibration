"""
r1_power.py — Revision R1, analysis plan Amendment 7, part C (descriptive): what a bearing-level comparison
with ten or eleven bearings can detect. For each paired bearing-level endpoint of Amendments 5, 6 and 7, from
the per-bearing differences already computed: n, mean, standard deviation, and the minimum detectable mean
difference at 80 % power and two-sided alpha 0.05,
    MDE = (t(0.975, n - 1) + t(0.80, n - 1)) * SD / sqrt(n)    (paired t approximation);
for the cost endpoints also as a cost ratio (1 + MDE) / (1 - MDE), and the number of bearings needed to detect
cost differences of 10 % and 25 % (symmetric relative difference 0.048 and 0.111); for the coverage endpoints
the number needed for 5 and 10 points (normal approximation, same SD). No hypothesis test. Written after the
cost results of Amendments 5 and 6 were known.

USAGE:
  python Experiments/r1_power.py
  python Experiments/r1_power.py --dry-run     # synthetic per-bearing values, scratch output
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
from r1_provenance import provenance  # noqa: E402

AN = ROOT / 'outputs' / 'r1_analysis'
COST_TARGETS = {'10 %': 0.1 / 2.1, '25 %': 0.25 / 2.25}
COVERAGE_TARGETS = {'5 points': 5.0, '10 points': 10.0}


def values(x):
    return np.array(list(x.values()) if isinstance(x, dict) else x, dtype=float)


def endpoints_real():
    """(name, kind, per-bearing differences); kind: cost, coverage or late"""
    out, missing = [], []
    a5 = json.loads((AN / 'pooled_framework.json').read_text(encoding='utf-8'))
    for r in ('5', '10'):
        out.append((f'A5-P1 framework against standard practice, PRONOSTIA, r = {r}', 'cost', values(a5['A5_P1'][r]['per_bearing'])))
    a6 = json.loads((AN / 'decision_inbearing.json').read_text(encoding='utf-8'))
    out.append(('A6-P1 coverage error, in-bearing against held-out calibration, PRONOSTIA', 'coverage',
                values(a6['A6_P1_P2']['pooled']['per_bearing'])))
    for r in ('5', '10'):
        out.append((f'A6 (d) cost, in-bearing against held-out calibration, PRONOSTIA, r = {r}', 'cost',
                    values(a6['secondary']['decisions'][r]['pooled_srd']['per_bearing'])))
    f7a = AN / 'decision_xjtu_inbearing.json'
    if f7a.exists():
        a7 = json.loads(f7a.read_text(encoding='utf-8'))
        out.append(('A7-P1 coverage error, in-bearing against held-out calibration, XJTU-SY', 'coverage',
                    values(a7['A7_P1_P2']['pooled']['per_bearing'])))
        for r in ('5', '10'):
            out.append((f'A7-P3 late-alarm share, XJTU-SY, r = {r}', 'late', values(a7['A7_P3'][r]['per_bearing'])))
            out.append((f'A7-A (d) cost, XJTU-SY, r = {r}', 'cost', values(a7['secondary']['decisions'][r]['pooled_srd']['per_bearing'])))
    else:
        missing.append(str(f7a.relative_to(ROOT)))
    f7b = AN / 'decision_bestckpt.json'
    if f7b.exists():
        b7 = json.loads(f7b.read_text(encoding='utf-8'))
        out.append(('A7-P4 coverage error, best-epoch against final-epoch calibration, PRONOSTIA', 'coverage',
                    values(b7['A7_P4_P5']['pooled']['per_bearing'])))
    else:
        missing.append(str(f7b.relative_to(ROOT)))
    return out, missing


def endpoints_synthetic(rng):
    return [('synthetic cost, n = 10', 'cost', rng.normal(0.01, 0.27, 10)),
            ('synthetic coverage, n = 10', 'coverage', rng.normal(39, 11, 10)),
            ('synthetic late share, n = 11', 'late', rng.normal(0.1, 0.15, 11))], []


def describe(name, kind, d):
    n, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    mde = float((stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * sd / np.sqrt(n))
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    row = {'endpoint': name, 'kind': kind, 'n': n, 'mean': mean, 'sd': sd, 'mde': mde}
    if kind == 'cost':
        row['mde_as_cost_ratio'] = (1 + mde) / (1 - mde) if mde < 1 else float('inf')
        row['bearings_needed'] = {k: int(np.ceil((z * sd / v) ** 2)) for k, v in COST_TARGETS.items()}
    elif kind == 'coverage':
        row['bearings_needed'] = {k: int(np.ceil((z * sd / v) ** 2)) for k, v in COVERAGE_TARGETS.items()}
    return row


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(AN / 'power.md'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        eps, missing = endpoints_synthetic(np.random.default_rng(0))
        out_path = Path(tempfile.mkdtemp(prefix='power_dry_')) / 'power.md'
    else:
        eps, missing = endpoints_real()
        out_path = Path(args.out)
    rows = [describe(*e) for e in eps]
    lines = ['# Power of bearing-level comparisons (revision R1, Amendment 7, part C; descriptive)', '',
             ('**DRY RUN on synthetic data: numbers are meaningless.**' if args.dry_run else
              'Generated by `Experiments/r1_power.py`; definitions in analysis plan Section 11, Amendment 7. Written after the '
              'cost results of Amendments 5 and 6 were known; no hypothesis test.'), '',
             'MDE: minimum detectable mean difference at 80 % power, two-sided α = 0.05, paired t approximation. Cost endpoints: '
             'per-bearing symmetric relative difference (L_a − L_b) / (L_a + L_b); as a cost ratio (1 + MDE) / (1 − MDE). '
             'Bearings needed: normal approximation with the same SD.', '',
             '| Endpoint | n | mean | SD | MDE | MDE as cost ratio | bearings needed |', '|---|---|---|---|---|---|---|']
    for r in rows:
        ratio = f"{r['mde_as_cost_ratio']:.2f}" if 'mde_as_cost_ratio' in r else ''
        need = '; '.join(f'{k}: {v}' for k, v in r.get('bearings_needed', {}).items())
        lines.append(f"| {r['endpoint']} | {r['n']} | {r['mean']:+.3f} | {r['sd']:.3f} | {r['mde']:.3f} | {ratio} | {need} |")
    if missing:
        lines += ['', 'Not yet available: ' + ', '.join(missing) + '.']
    report = {'dry_run': args.dry_run, 'rows': rows, 'missing': missing,
              'provenance': provenance(('Experiments/r1_power.py',))}
    text = '\n'.join(lines) + '\n'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')
    out_path.with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(text)
    print(f'written {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
