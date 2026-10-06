"""
r1_decision.py — Revision R1: maintenance-decision analysis on corrected predictions.

1. Interval-triggered rule (exp23 logic): per-window false-alarm / missed-detection
   rates at tau = 0.20 for point, uncalibrated and calibrated lower-bound triggers,
   macro-averaged over bearings per seed (Tables 8-9).
2. Alarm timing (exp29 logic, imported unchanged): first-crossing error dt for
   Raw, PAVA, RTRM, CES, the calibrated lower bound on the raw mean, and the
   calibrated lower bound on the CES-smoothed mean (the filtered-mean trigger that
   the submitted manuscript lists as not evaluated, Sec. 6.5.4).
3. Cost-ratio analysis (reviewer comment 5): with r = c_late / c_early, each
   variant's mean decision loss is r * A + B (A = mean lateness, B = mean
   earliness), so the decision-optimal rule over all r is the lower envelope of
   straight lines and rankings change only at r* = (B_j - B_i) / (A_i - A_j).

USAGE:
  python Experiments/r1_decision.py --preds-dir outputs/r1_final_B10/final20 --set all10 \
      --uq-json outputs/r1_analysis/uq.json
"""

import argparse
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import exp29_maintenance_timing as e29  # noqa: E402
from r1_analysis import SETS  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from src.online_filters import ConstrainedExponentialSmoothing  # noqa: E402

SETS = dict(SETS)
SETS['new6'] = [b for b in SETS['all10'] if b not in SETS['orig4']]   # never used before R1
Z = 1.6449
TAUS = (0.30, 0.20, 0.10)
TAU_RATES = 0.20
TIMING_VARIANTS = ('raw', 'pava', 'rtrm', 'ces', 'lb_raw', 'lb_ces')
SECONDS_PER_WINDOW = 10.0


def load(preds_dir, bearings):
    seeds = []
    for f in sorted(glob.glob(str(Path(preds_dir) / 'preds_seed*.npz'))):
        d = np.load(f)
        seeds.append({b: dict(y=d[f'tgt_{b}'].astype(float), raw=d[f'raw_{b}'].astype(float),
                              mu=d[f'mcmean_{b}'].astype(float),
                              s=np.clip(d[f'mcstd_{b}'].astype(float), 1e-4, None))
                      for b in bearings})
    return seeds


def window_rates(b, c_star, tau):
    y, mu, s = b['y'], b['mu'], b['s']
    should = y <= tau
    n_pos, n_neg = max(int(should.sum()), 1), max(int((~should).sum()), 1)
    triggers = {'point': mu <= tau, 'uncalibrated': (mu - Z * s) <= tau,
                'calibrated': (mu - Z * c_star * s) <= tau}
    return {k: {'false_alarm': 100.0 * float((t & ~should).sum()) / n_neg,
                'missed_detection': 100.0 * float((~t & should).sum()) / n_pos}
            for k, t in triggers.items()}


def timing_sequences(b, c_star):
    seq = e29.build_variants({'raw': b['raw'], 'mu': b['mu'], 's': b['s']}, c_star)
    mu_ces = np.asarray(ConstrainedExponentialSmoothing(alpha=0.2, epsilon=0.05).process(b['mu']), float)
    return {'raw': seq['raw'], 'pava': seq['pava'], 'rtrm': seq['rtrm'], 'ces': seq['ces'],
            'lb_raw': seq['lower_bound'], 'lb_ces': mu_ces - e29.Z90 * c_star * b['s']}


def envelope(stats_tau, variants, r_min=0.5, r_max=100.0):
    grid = np.logspace(np.log10(r_min), np.log10(r_max), 600)
    best = [min(variants, key=lambda v: r * stats_tau[v]['A'] + stats_tau[v]['B']) for r in grid]
    segs, start = [], grid[0]
    for i in range(1, len(grid)):
        if best[i] != best[i - 1]:
            segs.append({'variant': best[i - 1], 'r_from': float(start), 'r_to': float(grid[i])})
            start = grid[i]
    segs.append({'variant': best[-1], 'r_from': float(start), 'r_to': float(grid[-1])})
    breakeven = []
    for i, j in itertools.combinations(variants, 2):
        dA = stats_tau[i]['A'] - stats_tau[j]['A']
        if abs(dA) > 1e-12:
            r = (stats_tau[j]['B'] - stats_tau[i]['B']) / dA
            if r_min <= r <= r_max:
                breakeven.append({'pair': [i, j], 'r_star': float(r)})
    return segs, breakeven


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds-dir', required=True)
    ap.add_argument('--set', default='all10', choices=list(SETS))
    ap.add_argument('--uq-json', default=None, help='r1_uq.py output; uses its 4-fold c*')
    ap.add_argument('--c-star', type=float, default=None)
    ap.add_argument('--calibration-json', default=None,
                    help='r1_calibration.py output; uses its Section 4 global c*')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.c_star is not None:
        c_star = args.c_star
    elif args.calibration_json:
        c_star = json.loads(Path(args.calibration_json).read_text())['global']['c_star']
    elif args.uq_json:
        c_star = json.loads(Path(args.uq_json).read_text())['calibrations']['4fold']['c_star']
    else:
        c_star = 2.414251375198364
    bearings = SETS[args.set]
    seeds = load(args.preds_dir, bearings)
    out = {'preds_dir': args.preds_dir, 'set': args.set, 'c_star': c_star, 'n_seeds': len(seeds),
           'c_star_source': (args.calibration_json or args.uq_json
                             or ('--c-star' if args.c_star is not None else 'exp22 default'))}

    # ---- 1. window-level alarm rates at tau = 0.20 ------------------------------------
    pols = ('point', 'uncalibrated', 'calibrated')
    macro = {p: {'false_alarm': [], 'missed_detection': []} for p in pols}
    per_b = {b: {p: {'false_alarm': [], 'missed_detection': []} for p in pols} for b in bearings}
    for sd in seeds:
        rates = {b: window_rates(sd[b], c_star, TAU_RATES) for b in bearings}
        for p in pols:
            for m in ('false_alarm', 'missed_detection'):
                macro[p][m].append(np.mean([rates[b][p][m] for b in bearings]))
                for b in bearings:
                    per_b[b][p][m].append(rates[b][p][m])
    out['rates_tau020'] = {p: {m: [float(np.mean(v)), float(np.std(v))] for m, v in macro[p].items()}
                           for p in pols}
    out['rates_tau020_per_bearing'] = {b: {p: {m: [float(np.mean(v)), float(np.std(v))]
                                               for m, v in per_b[b][p].items()} for p in pols}
                                       for b in bearings}
    print(f'\n## Alarm rates, tau = {TAU_RATES}, {args.set}, c* = {c_star:.3f} '
          f'({len(seeds)} seeds, macro over bearings)\n')
    print('| Policy | False alarm % | Missed detection % |\n|---|---|---|')
    for p in pols:
        fa, md = out['rates_tau020'][p]['false_alarm'], out['rates_tau020'][p]['missed_detection']
        print(f'| {p} | {fa[0]:.1f} ± {fa[1]:.1f} | {md[0]:.1f} ± {md[1]:.1f} |')

    # ---- 2. alarm timing + 3. break-even ------------------------------------------------
    out['timing'] = {}
    for tau in TAUS:
        acc = {v: {k: [] for k in ('dt', 'late', 'severe', 'early', 'A', 'B')} for v in TIMING_VARIANTS}
        late_by_b = {v: {b: [] for b in bearings} for v in TIMING_VARIANTS}
        for sd in seeds:
            per_seed = {v: {k: [] for k in acc[v]} for v in TIMING_VARIANTS}
            for b in bearings:
                seqs = timing_sequences(sd[b], c_star)
                for v in TIMING_VARIANTS:
                    dt, severe = e29.bearing_deltas(sd[b]['y'], np.asarray(seqs[v], float), tau)
                    late_by_b[v][b].append(bool(dt > 0))
                    for k, val in (('dt', dt), ('late', dt > 0), ('severe', severe), ('early', dt < 0),
                                   ('A', max(0, dt)), ('B', max(0, -dt))):
                        per_seed[v][k].append(float(val))
            for v in TIMING_VARIANTS:
                for k in acc[v]:
                    acc[v][k].append(np.mean(per_seed[v][k]))
        stats_tau = {v: {k: float(np.mean(x)) for k, x in acc[v].items()} for v in TIMING_VARIANTS}
        segs, be = envelope(stats_tau, TIMING_VARIANTS)
        # bearings on which a rule fires late for the majority of seeds (plan Section 9)
        late_bearings = {v: [b for b in bearings if np.mean(late_by_b[v][b]) > 0.5] for v in TIMING_VARIANTS}
        out['timing'][f'{tau:.2f}'] = {
            'stats': stats_tau, 'optimal_by_ratio': segs, 'breakeven': be,
            'late_bearings_majority': late_bearings,
            'late_fraction_per_bearing': {v: {b: float(np.mean(late_by_b[v][b])) for b in bearings}
                                          for v in TIMING_VARIANTS}}
        print(f'\n## Alarm timing, tau = {tau:.2f} (dt in windows of {SECONDS_PER_WINDOW:.0f} s)\n')
        print('| Variant | mean dt | late % | severe-late % | early % | loss 5:1 | late bearings |\n'
              '|---|---|---|---|---|---|---|')
        for v in TIMING_VARIANTS:
            s = stats_tau[v]
            print(f"| {v} | {s['dt']:+.1f} | {100 * s['late']:.1f} | {100 * s['severe']:.1f} | "
                  f"{100 * s['early']:.1f} | {5 * s['A'] + s['B']:.1f} | "
                  f"{len(late_bearings[v])}/{len(bearings)} |")
        print('decision-optimal by r = c_late/c_early: ' +
              '; '.join(f"{g['variant']} [{g['r_from']:.2f}, {g['r_to']:.2f}]" for g in segs))

    out_path = Path(args.out or ROOT / 'outputs' / 'r1_analysis' /
                    f'decision_{Path(args.preds_dir).name}_{args.set}.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out['provenance'] = provenance(('Experiments/r1_decision.py', 'Experiments/exp29_maintenance_timing.py',
                                    'Experiments/r1_analysis.py'))
    out_path.write_text(json.dumps(out, indent=1))
    print(f'\nwritten {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
