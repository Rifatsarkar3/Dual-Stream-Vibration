"""
r1_eval_choices.py — Revision R1, analysis plan Amendment 1, E1 and E2 (existing predictions only).

E1, unit of inference. For every comparison of plan Section 8 (each version of our model against
the seven baselines, against its seven ablations, the protocol demonstrations, the two versions
against each other; all10) and for both versions against the three XJTU-SY baselines (xjtu_test,
xjtu_zs), three tests on CES MAE (raw co-reported), all on the seeds both runs share:
  (a) as in the submission: paired two-sided t-test over seeds on window-pooled MAE
  (b) paired two-sided t-test over seeds on bearing-averaged MAE
  (c) plan Section 8: exact two-sided sign-flip test over bearings on bearing-averaged seed-mean MAE
Counts of p < 0.05, unadjusted and Holm-adjusted within each family, and the disagreements.

E2, aggregation. For the same comparisons and for every set (all10, new6, orig4, zs3, xjtu_test,
xjtu_zs): which model is better under window-pooled and under bearing-averaged MAE, and Kendall's
tau between the rankings of the main models of a set (both versions and the baselines; each
model's own seeds) under the two aggregations.

USAGE:
  python Experiments/r1_eval_choices.py [--out outputs/r1_analysis/eval_choices.md]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import r1_xjtu_stats as xj  # noqa: E402
from r1_analysis import load_variant  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import ABLATIONS, PROTOCOL, SETS, baseline_paths, holm, signflip_p  # noqa: E402
from src.online_filters import ConstrainedExponentialSmoothing  # noqa: E402

REFS = {'adopted': {'path': 'outputs/r1_final_temporal_B5/final20', 'ablation_root': 'outputs/r1_final_temporal_B5',
                    'protocol': {'in-bearing early stopping': 'outputs/r1_protocol_temporal/full_inbearing_es'}},
        'submitted': {'path': 'outputs/r1_final_B10/final20', 'ablation_root': 'outputs/r1_final_B10',
                      'protocol': PROTOCOL}}
P_SETS = ('all10', 'new6', 'orig4', 'zs3')
X_SETS = ('xjtu_test', 'xjtu_zs')


class Errors:
    """absolute errors per (seed, bearing, method), computed once per loaded variant"""

    def __init__(self, variant):
        self.v = variant
        self.cache = {}

    def err(self, seed, b, method):
        key = (seed, b, method)
        if key not in self.cache:
            d = self.v[seed][b]
            pred = (d['raw'] if method == 'raw' else
                    np.asarray(ConstrainedExponentialSmoothing(alpha=0.2, epsilon=0.05).process(d['raw']), float))
            self.cache[key] = np.abs(pred - d['y'])
        return self.cache[key]

    def seeds(self):
        return sorted(self.v)

    def pooled(self, seed, bearings, method):
        return float(np.mean(np.concatenate([self.err(seed, b, method) for b in bearings])))

    def bearing_avg(self, seed, bearings, method):
        return float(np.mean([np.mean(self.err(seed, b, method)) for b in bearings]))


def compare(ref, other, bearings, method):
    seeds = sorted(set(ref.seeds()) & set(other.seeds()))
    pa = np.array([ref.pooled(s, bearings, method) for s in seeds])
    pb = np.array([other.pooled(s, bearings, method) for s in seeds])
    ba = np.array([ref.bearing_avg(s, bearings, method) for s in seeds])
    bb = np.array([other.bearing_avg(s, bearings, method) for s in seeds])
    per_b = np.array([np.mean([np.mean(other.err(s, b, method)) for s in seeds])
                      - np.mean([np.mean(ref.err(s, b, method)) for s in seeds]) for b in bearings])
    return {'n_seeds': len(seeds), 'n_bearings': len(bearings),
            'pooled_diff': float((pb - pa).mean()), 'bavg_diff': float((bb - ba).mean()),
            'p_a_seed_pooled': float(stats.ttest_rel(pb, pa).pvalue),
            'p_b_seed_bavg': float(stats.ttest_rel(bb, ba).pvalue),
            'p_c_bearing_flip': float(signflip_p(per_b)),
            'bearings_ref_better': int((per_b > 0).sum())}


def family(ref, members, bearings, method):
    res = {k: compare(ref, v, bearings, method) for k, v in members.items()}
    for test in ('p_a_seed_pooled', 'p_b_seed_bavg', 'p_c_bearing_flip'):
        adj = holm(np.array([r[test] for r in res.values()])) if len(res) > 1 else [r[test] for r in res.values()]
        for r, pa in zip(res.values(), adj):
            r[test + '_holm'] = float(pa)
    return res


def kendall(models, bearings, method):
    pooled = [np.mean([m.pooled(s, bearings, method) for s in m.seeds()]) for m in models.values()]
    bavg = [np.mean([m.bearing_avg(s, bearings, method) for s in m.seeds()]) for m in models.values()]
    tau = stats.kendalltau(pooled, bavg)
    order = lambda v: [k for _, k in sorted(zip(v, models))]  # noqa: E731
    return {'tau': float(tau.statistic), 'p': float(tau.pvalue), 'rank_pooled': order(pooled),
            'rank_bearing_avg': order(bavg),
            'pooled': dict(zip(models, map(float, pooled))), 'bearing_avg': dict(zip(models, map(float, bavg)))}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'eval_choices.md'))
    args = ap.parse_args()

    base = {k: Errors(load_variant(ROOT / v)) for k, v in baseline_paths().items()}
    refs = {k: Errors(load_variant(ROOT / v['path'])) for k, v in REFS.items()}
    abl = {k: {a: Errors(load_variant(ROOT / v['ablation_root'] / d)) for a, d in ABLATIONS.items()}
           for k, v in REFS.items()}
    prot = {k: {p: Errors(load_variant(ROOT / d)) for p, d in v['protocol'].items()} for k, v in REFS.items()}
    xmodels = {k: Errors(xj.load(v)) for k, v in xj.MODELS.items()}
    xref = {'adopted': xmodels[xj.ADOPTED], 'submitted': xmodels[xj.SUBMITTED]}
    xbase = {b: xmodels[b] for b in xj.BASELINES}

    report = {'E1': {}, 'E2': {'flips': {}, 'kendall': {}}}
    for method in ('ces', 'raw'):
        fams = {}
        for r in REFS:
            fams[f'{r}: baselines (all10)'] = family(refs[r], base, SETS['all10'], method)
            fams[f'{r}: ablations (all10)'] = family(refs[r], abl[r], SETS['all10'], method)
            fams[f'{r}: protocol (all10)'] = family(refs[r], prot[r], SETS['all10'], method)
            for s in X_SETS:
                fams[f'{r}: XJTU-SY baselines ({s})'] = family(xref[r], xbase, xj.SETS[s], method)
        fams['adopted vs submitted (all10)'] = family(refs['adopted'], {'submitted architecture': refs['submitted']},
                                                      SETS['all10'], method)
        report['E1'][method] = fams

        # E2: direction under the two aggregations, every set
        flips = {}
        for set_name in P_SETS + X_SETS:
            bearings = SETS[set_name] if set_name in P_SETS else xj.SETS[set_name]
            pairs = ([(f'{r} vs {b}', xref[r], xbase[b]) for r in xref for b in xbase] if set_name in X_SETS else
                     [(f'{r} vs {b}', refs[r], m) for r in refs for b, m in base.items()]
                     + [(f'{r} vs {a}', refs[r], m) for r in refs for a, m in abl[r].items()]
                     + [(f'{r} vs {p}', refs[r], m) for r in refs for p, m in prot[r].items()]
                     + [('adopted vs submitted', refs['adopted'], refs['submitted'])])
            rows = []
            for name, a, b in pairs:
                c = compare(a, b, bearings, method)
                rows.append({'comparison': name, 'pooled_diff': c['pooled_diff'], 'bavg_diff': c['bavg_diff'],
                             'flip': bool(np.sign(c['pooled_diff']) != np.sign(c['bavg_diff']))})
            flips[set_name] = rows
            main_models = ({'adopted': xref['adopted'], 'submitted': xref['submitted'], **xbase} if set_name in X_SETS
                           else {'adopted': refs['adopted'], 'submitted': refs['submitted'], **base})
            report['E2']['kendall'].setdefault(method, {})[set_name] = kendall(main_models, bearings, method)
        report['E2']['flips'][method] = flips

    # ---- report ----
    lines = ['# Evaluation choices: unit of inference (E1) and aggregation (E2)',
             '(revision R1, analysis plan Amendment 1; generated by Experiments/r1_eval_choices.py)', '',
             'Tests on the seeds both runs share. (a) paired t-test over seeds on window-pooled MAE (the submission\'s '
             'test); (b) paired t-test over seeds on bearing-averaged MAE; (c) exact sign-flip test over bearings on '
             'bearing-averaged seed-mean MAE (plan Section 8). Holm within each family.', '']
    for method in ('ces', 'raw'):
        lines += [f'## E1, {method.upper()} MAE: comparisons with p < 0.05 (unadjusted / Holm)', '',
                  '| Family | comparisons | (a) seeds, pooled | (b) seeds, bearing-averaged | (c) bearings |',
                  '|---|---|---|---|---|']
        tot = {t: [0, 0] for t in ('a', 'b', 'c')}
        n_tot = 0
        for fam, res in report['E1'][method].items():
            cells = []
            for t, key in (('a', 'p_a_seed_pooled'), ('b', 'p_b_seed_bavg'), ('c', 'p_c_bearing_flip')):
                u = sum(r[key] < 0.05 for r in res.values())
                h = sum(r[key + '_holm'] < 0.05 for r in res.values())
                tot[t][0] += u
                tot[t][1] += h
                cells.append(f'{u} / {h}')
            n_tot += len(res)
            lines.append(f'| {fam} | {len(res)} | ' + ' | '.join(cells) + ' |')
        lines.append(f"| **all** | {n_tot} | " + ' | '.join(f'**{tot[t][0]} / {tot[t][1]}**' for t in 'abc') + ' |')
        lines += ['', f'Comparisons where the seed-level tests and the bearing-level test disagree ({method.upper()}, '
                      'Holm-adjusted, 0.05):', '',
                  '| Family | Comparison | Δ pooled | Δ bearing-avg | (a) Holm | (b) Holm | (c) Holm | bearings ref. better |',
                  '|---|---|---|---|---|---|---|---|']
        for fam, res in report['E1'][method].items():
            for name, r in res.items():
                sig = [r[k + '_holm'] < 0.05 for k in ('p_a_seed_pooled', 'p_b_seed_bavg', 'p_c_bearing_flip')]
                if len(set(sig)) > 1:
                    lines.append(f"| {fam} | {name} | {r['pooled_diff']:+.4f} | {r['bavg_diff']:+.4f} | "
                                 f"{r['p_a_seed_pooled_holm']:.2g} | {r['p_b_seed_bavg_holm']:.2g} | "
                                 f"{r['p_c_bearing_flip_holm']:.3f} | {r['bearings_ref_better']}/{r['n_bearings']} |")
        lines += ['', f'## E2, {method.upper()} MAE: window-pooled vs bearing-averaged', '',
                  '| Set | comparisons | direction differs | Kendall tau (main models) | ranking, pooled | ranking, bearing-averaged |',
                  '|---|---|---|---|---|---|']
        for set_name, rows in report['E2']['flips'][method].items():
            k = report['E2']['kendall'][method][set_name]
            lines.append(f"| {set_name} | {len(rows)} | {sum(r['flip'] for r in rows)} | {k['tau']:.2f} | "
                         f"{', '.join(k['rank_pooled'])} | {', '.join(k['rank_bearing_avg'])} |")
        lines += ['', 'Comparisons whose direction differs:', '']
        for set_name, rows in report['E2']['flips'][method].items():
            fl = [f"{r['comparison']} (pooled {r['pooled_diff']:+.4f}, bearing-avg {r['bavg_diff']:+.4f})"
                  for r in rows if r['flip']]
            if fl:
                lines.append(f"- {set_name}: " + '; '.join(fl))
        lines.append('')
    report['provenance'] = provenance(('Experiments/r1_eval_choices.py', 'Experiments/r1_strict_stats.py',
                                       'Experiments/r1_xjtu_stats.py', 'Experiments/r1_analysis.py'))
    text = '\n'.join(lines) + '\n'
    print(text)
    Path(args.out).write_text(text, encoding='utf-8')
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'written {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
