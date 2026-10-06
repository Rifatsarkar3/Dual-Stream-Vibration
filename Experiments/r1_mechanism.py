"""
r1_mechanism.py — Revision R1, analysis plan Amendment 9: does the in-bearing understatement of the error tail
grow with training progress? Computation only, on the saved runs of Amendments 6 and 7 (no training, no inference).

  Runs   the 60 in-bearing early-stopping runs (seeds 42, 123, 777, 999, 7) in twelve combinations: PRONOSTIA, our
         network and the seven baselines; XJTU-SY, our network and the three baselines.
  e      epoch of the checkpoint selected by in-bearing early stopping, as recorded (best_epoch). Our PRONOSTIA runs
         (exp16, 10 June 2026) record only the epochs trained: e = epochs trained - 30 (patience 30); a run that
         trained 120 epochs without stopping is ranked above every other run of its combination and tied with any
         other such run. Check: e = epochs trained - 30 for every recorded run that stopped before 120 epochs.
  rho    q_cal / q_test: 0.9-quantile of |raw - y| over the run's in-bearing validation windows (T6a, T7a) over the
         0.9-quantile over the run's windows of the ten PRONOSTIA test bearings (Amendment 6) or the eleven XJTU-SY
         bearings (Amendment 7), from the same test files. Check: the means over the five runs reproduce the values
         stored by Amendments 6 and 7 within 1e-12, otherwise the analysis stops.
  A9-P1  T = sum over the combinations of Kendall's S between e and rho across the five runs; exact two-sided p
         from the convolution of the within-combination permutation distributions. Prediction: T < 0.
  A9-S1  A9-P1 without our XJTU-SY network.
  A9-S2  the A9-P1 statistic with q_cal and with q_test in place of rho.
  A9-S3  descriptive: between combinations, Kendall tau_b of the median e and the architecture-level ratio.
  Figure rho against e (logarithmic axis), the 60 runs.

USAGE:
  python Experiments/r1_mechanism.py --dry-run     # synthetic data and a Monte Carlo check of the exact test
  python Experiments/r1_mechanism.py
"""

import argparse
import json
import sys
import tempfile
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import r1_decision_inbearing as di  # noqa: E402  (Amendment 6; used unchanged)
import r1_decision_models as dm  # noqa: E402  (Amendment 2; used unchanged)
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import SETS  # noqa: E402
from r1_xjtu_stats import SETS as XSETS  # noqa: E402

SEEDS = [42, 123, 777, 999, 7]
PATIENCE, MAX_EPOCHS = 30, 120
CENSORED = 10 ** 6  # rank of a run that never stopped: above every other run, tied with any other such run
TOL = 1e-12
OUT = ROOT / 'outputs'
XJTU_ARCH = {dm.SUBMITTED: 'ours', '1D CNN': 'cnn', 'BiLSTM': 'bilstm', 'CNN-LSTM': 'cnnlstm'}
EXCLUDED_S1 = ('XJTU-SY', dm.SUBMITTED)


# ---------------------------------------------------------------- statistics --------------------
def kendall_s(x, y):
    """concordant minus discordant pairs; pairs tied in x or y count zero"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            s += int(np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]))
    return s


def perm_counts(x, y):
    """{S: number of orderings of y} over all len(y)! orderings, x fixed"""
    counts = {}
    for p in permutations(range(len(y))):
        s = kendall_s(x, [y[k] for k in p])
        counts[s] = counts.get(s, 0) + 1
    return counts


def exact_test(groups):
    """groups: list of (x, y). T = sum of Kendall's S; exact two-sided p (integer arithmetic)"""
    t_obs = sum(kendall_s(x, y) for x, y in groups)
    dist, total = {0: 1}, 1
    for x, y in groups:
        c = perm_counts(x, y)
        n = sum(c.values())
        new = {}
        for a, ca in dist.items():
            for b, cb in c.items():
                new[a + b] = new.get(a + b, 0) + ca * cb
        dist, total = new, total * n
    tail = sum(v for t, v in dist.items() if abs(t) >= abs(t_obs))
    taus = [kendalltau(x, y).statistic for x, y in groups]
    taus = [float(t) for t in taus if np.isfinite(t)]
    return {'T': int(t_obs), 'p_two_sided': tail / total, 'n_groups': len(groups),
            'T_min_max': [min(dist), max(dist)], 'mean_tau_b': float(np.mean(taus)) if taus else None,
            'n_tau_negative': int(sum(t < 0 for t in taus)), 'n_tau_defined': len(taus)}


def monte_carlo_p(groups, n_perm, rng):
    t_obs = abs(sum(kendall_s(x, y) for x, y in groups))
    hits = 0
    for _ in range(n_perm):
        t = sum(kendall_s(x, rng.permutation(np.asarray(y))) for x, y in groups)
        hits += abs(t) >= t_obs
    return hits / n_perm


# ---------------------------------------------------------------- data --------------------------
def check_meta(path, count_key, count, key=None, limit=None):
    m = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
    if m[count_key] != count or (key is not None and m[key] > limit):
        raise SystemExit(f'{path}: incomplete or failed its check ({m[count_key]}, {key} {m.get(key)})')


def epochs_of(results_file, harness):
    """{seed: (e, censored)} and the rule check {seed: ok or None}"""
    d = json.loads(Path(results_file).read_text(encoding='utf-8'))
    out, check = {}, {}
    for r in d['per_seed']:
        s, trained = int(r['seed']), int(r['epochs_trained'])
        if 'best_epoch' in r:
            out[s] = (int(r['best_epoch']), False)
            check[s] = (int(r['best_epoch']) == trained - PATIENCE) if trained < MAX_EPOCHS else None
        elif trained < MAX_EPOCHS:
            out[s] = (trained - PATIENCE, False)
        else:
            out[s] = (CENSORED, True)
    if sorted(out) != sorted(SEEDS):
        raise SystemExit(f'{results_file}: seeds {sorted(out)}')
    return out, {'harness': harness, 'check': check}


def per_run_quantiles(val_files, test, bearings):
    runs = di.concat_runs(val_files)
    q_cal = [float(np.quantile(np.abs(r['raw'] - r['y']), 0.9)) for r in runs]
    q_test = [float(np.quantile(np.abs(np.concatenate([test[s][b]['raw'] - test[s][b]['y'] for b in bearings])), 0.9))
              for s in SEEDS]
    return q_cal, q_test


def real_data():
    a6 = json.loads((OUT / 'r1_analysis' / 'decision_inbearing.json').read_text(encoding='utf-8'))
    a7 = json.loads((OUT / 'r1_analysis' / 'decision_xjtu_inbearing.json').read_text(encoding='utf-8'))
    combos, rule = [], []
    a10 = SETS['all10']
    for label, short in di.ARCH.items():
        check_meta(OUT / 'r1_inbearing' / 't6a' / short, 'n_runs', 5, 'max_abs_diff', 1e-4)
        if short not in ('ours', 'tsmccnn'):
            check_meta(OUT / 'r1_inbearing' / 't6b' / short, 'n_checkpoints', 5, 'raw_check_max_abs_diff', 1e-5)
        test = dm.load_preds(di.SP_TEST[short])
        q_cal, q_test = per_run_quantiles([OUT / 'r1_inbearing' / 't6a' / short / f'inval_seed{s}.npz' for s in SEEDS],
                                          test, a10)
        rec = (OUT / 'exp16_ress' / 'full' / 'results.json' if short == 'ours'
               else OUT / 'r1_protocol_baselines' / short / 'results.json')
        ep, chk = epochs_of(rec, 'exp16 (10 June 2026)' if short == 'ours' else 'r1_protocol_baselines')
        rule.append({'dataset': 'PRONOSTIA', 'model': label, **chk})
        combos.append({'dataset': 'PRONOSTIA', 'model': label, 'stored': a6['secondary']['understatement'][label],
                       'q_cal': q_cal, 'q_test': q_test, 'e': [ep[s][0] for s in SEEDS],
                       'censored': [ep[s][1] for s in SEEDS]})
    xb = XSETS['xjtu_test'] + XSETS['xjtu_zs']
    xi = OUT / 'r1_xjtu_inbearing'
    for label, short in XJTU_ARCH.items():
        check_meta(xi / 't7a' / short, 'n_runs', 5, 'max_abs_diff', 1e-4)
        check_meta(xi / 't7b' / short, 'n_checkpoints', 5)
        test = dm.load_preds(str((xi / 't7b' / short).relative_to(ROOT)))
        q_cal, q_test = per_run_quantiles([xi / 't7a' / short / f'inval_seed{s}.npz' for s in SEEDS], test, xb)
        ep, chk = epochs_of(xi / 'es' / short / 'results.json',
                            'exp16 (XJTU-SY, Amendment 7)' if short == 'ours' else 'r1_xjtu_inbearing')
        rule.append({'dataset': 'XJTU-SY', 'model': label, **chk})
        combos.append({'dataset': 'XJTU-SY', 'model': label, 'stored': a7['secondary']['understatement'][label],
                       'q_cal': q_cal, 'q_test': q_test, 'e': [ep[s][0] for s in SEEDS],
                       'censored': [ep[s][1] for s in SEEDS]})
    return combos, rule


def synthetic_data(rng, slope):
    """dry run: twelve combinations, rho falling with log(e) by `slope` (0: null), one combination with tied epochs"""
    combos = []
    for i in range(12):
        e = rng.integers(2, 90, 5) if i else np.array([3, 40, 3, 40, 3])
        q_test = 0.3 + 0.05 * rng.random(5)
        rho = np.clip(0.6 - slope * np.log(e) + 0.05 * rng.standard_normal(5), 0.05, 1.5)
        q_cal = rho * q_test
        combos.append({'dataset': 'PRONOSTIA' if i < 8 else 'XJTU-SY', 'model': f'model {i % 8}',
                       'stored': {'SP_calibration_q90': float(np.mean(q_cal)), 'SP_test_q90': float(np.mean(q_test)),
                                  'SP_ratio': float(np.mean(q_cal) / np.mean(q_test))},
                       'q_cal': list(map(float, q_cal)), 'q_test': list(map(float, q_test)),
                       'e': list(map(int, e)), 'censored': [False] * 5})
    rule = [{'dataset': c['dataset'], 'model': c['model'], 'harness': 'synthetic', 'check': {}} for c in combos]
    return combos, rule


# ---------------------------------------------------------------- figure ------------------------
def figure(combos, path_stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9})
    models = list(dict.fromkeys(c['model'] for c in combos))
    colors = {m: plt.get_cmap('tab10')(i % 10) for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, ds in zip(axes, ('PRONOSTIA', 'XJTU-SY')):
        for c in (c for c in combos if c['dataset'] == ds):
            x = np.array([MAX_EPOCHS if cen else e for e, cen in zip(c['e'], c['censored'])], float)
            rho = np.array(c['q_cal']) / np.array(c['q_test'])
            order = np.argsort(x)
            ax.plot(x[order], rho[order], '-', color=colors[c['model']], lw=0.8, alpha=0.6)
            ax.scatter(x, rho, s=18, color=colors[c['model']], label=c['model'], zorder=3)
        ax.set_xscale('log')
        ax.set_title(ds, fontsize=9)
        ax.set_xlabel('Epoch of the selected checkpoint')
        ax.axhline(1.0, color='0.5', lw=0.6, ls=':')
    axes[0].set_ylabel('Calibration / test 0.9-quantile')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path_stem}.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- main --------------------------
def analyse(combos, rule):
    exp16_fail = any(v is False for r in rule if r['harness'].startswith('exp16') for v in r['check'].values())
    for c in combos:
        c['rho'] = [a / b for a, b in zip(c['q_cal'], c['q_test'])]
        c['repro_diff'] = [abs(float(np.mean(c['q_cal'])) - c['stored']['SP_calibration_q90']),
                           abs(float(np.mean(c['q_test'])) - c['stored']['SP_test_q90'])]
    worst = max(max(c['repro_diff']) for c in combos)
    if worst > TOL:
        raise SystemExit(f'reproduction check failed: largest difference {worst:.3e}; analysis stopped')
    p1_combos = [c for c in combos if not (exp16_fail and c['dataset'] == 'PRONOSTIA' and c['model'] == dm.SUBMITTED)]
    s1_combos = [c for c in p1_combos if (c['dataset'], c['model']) != EXCLUDED_S1]
    res = {'A9_P1': exact_test([(c['e'], c['rho']) for c in p1_combos]),
           'A9_S1': exact_test([(c['e'], c['rho']) for c in s1_combos]),
           'A9_S2_q_cal': exact_test([(c['e'], c['q_cal']) for c in p1_combos]),
           'A9_S2_q_test': exact_test([(c['e'], c['q_test']) for c in p1_combos])}
    med_e = [float(np.median(c['e'])) for c in combos]
    arch_ratio = [c['stored']['SP_ratio'] for c in combos]
    res['A9_S3_between'] = {'tau_b': float(kendalltau(med_e, arch_ratio).statistic), 'n': len(combos)}
    res['exclusion_exp16'] = exp16_fail
    res['reproduction_max_abs_diff'] = worst
    p1 = res['A9_P1']
    res['A9_P1']['outcome'] = 'confirmed' if (p1['T'] < 0 and p1['p_two_sided'] < 0.05) else 'not confirmed'
    return res


def report(combos, rule, res, dry):
    def fmt(r):
        tau = 'n/a' if r['mean_tau_b'] is None else f"{r['mean_tau_b']:+.3f}"
        return (f"T = {r['T']:+d} (range {r['T_min_max'][0]} to {r['T_min_max'][1]}), exact two-sided p {r['p_two_sided']:.4f}; "
                f"mean within-combination tau_b {tau}, negative in {r['n_tau_negative']} of "
                f"{r['n_tau_defined']} combinations")
    lines = ['# In-bearing understatement and training progress (revision R1, Amendment 9)', '',
             ('**DRY RUN on synthetic data: numbers are meaningless.**' if dry else
              'Generated by `Experiments/r1_mechanism.py`; definitions in analysis plan Section 11, Amendment 9.'), '',
             'e: epoch of the checkpoint selected by in-bearing early stopping. rho = q_cal / q_test (0.9-quantiles of '
             '|raw - y| on the run\'s in-bearing validation windows and on its test windows). Kendall S per combination '
             'over the five runs; T is their sum; exact permutation test within combinations.', '',
             f"Reproduction of the Amendment 6 and 7 means: largest absolute difference {res['reproduction_max_abs_diff']:.1e}.",
             '', '## Rule e = epochs trained - 30 (runs that stopped before 120 epochs)', '',
             '| Dataset | Model | Harness | Runs checked | Agree |', '|---|---|---|---|---|']
    for r in rule:
        vals = [v for v in r['check'].values() if v is not None]
        lines.append(f"| {r['dataset']} | {r['model']} | {r['harness']} | {len(vals)} | {sum(vals)} |")
    lines += ['', f"exp16-harness rule failure, our PRONOSTIA runs excluded: {res['exclusion_exp16']}", '',
              '## Per run', '', '| Dataset | Model | Seed | e | q_cal | q_test | rho |', '|---|---|---|---|---|---|---|']
    for c in combos:
        for s, e, cen, a, b, r in zip(SEEDS, c['e'], c['censored'], c['q_cal'], c['q_test'], c['rho']):
            lines.append(f"| {c['dataset']} | {c['model']} | {s} | {'>90 (120 trained)' if cen else e} | "
                         f"{a:.4f} | {b:.4f} | {r:.3f} |")
    lines += ['', '## Per combination', '', '| Dataset | Model | Kendall S | tau_b | ratio (A6/A7) |', '|---|---|---|---|---|']
    for c in combos:
        t = kendalltau(c['e'], c['rho']).statistic
        lines.append(f"| {c['dataset']} | {c['model']} | {kendall_s(c['e'], c['rho']):+d} | "
                     f"{'n/a' if not np.isfinite(t) else f'{t:+.2f}'} | {c['stored']['SP_ratio']:.2f} |")
    lines += ['', '## Tests', '',
              f"- A9-P1 (primary, prediction T < 0): {fmt(res['A9_P1'])}. Outcome: **{res['A9_P1']['outcome']}**.",
              f"- A9-S1 (without our XJTU-SY network): {fmt(res['A9_S1'])}.",
              f"- A9-S2, q_cal (expectation T < 0): {fmt(res['A9_S2_q_cal'])}.",
              f"- A9-S2, q_test (no expectation): {fmt(res['A9_S2_q_test'])}.",
              f"- A9-S3 (descriptive), between the {res['A9_S3_between']['n']} combinations, median e against the "
              f"architecture-level ratio: tau_b {res['A9_S3_between']['tau_b']:+.3f}."]
    return '\n'.join(lines) + '\n'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT / 'r1_analysis' / 'mechanism.md'))
    ap.add_argument('--dry-run', action='store_true', help='synthetic data; writes to a temporary folder')
    args = ap.parse_args()
    if args.dry_run:
        rng = np.random.default_rng(0)
        out_path = Path(tempfile.mkdtemp(prefix='mechanism_dry_')) / 'mechanism.md'
        checks = []
        for slope in (0.12, 0.0):
            combos, rule = synthetic_data(rng, slope)
            res = analyse(combos, rule)
            groups = [(c['e'], c['rho']) for c in combos]
            mc = monte_carlo_p(groups, 20000, np.random.default_rng(1))
            checks.append(f"slope {slope}: exact p {res['A9_P1']['p_two_sided']:.4f}, Monte Carlo p {mc:.4f}, "
                          f"T {res['A9_P1']['T']:+d}, outcome {res['A9_P1']['outcome']}")
            if abs(mc - res['A9_P1']['p_two_sided']) > 0.015:
                raise SystemExit('exact and Monte Carlo p disagree: ' + checks[-1])
        print('\n'.join(checks))
    else:
        combos, rule = real_data()
        res = analyse(combos, rule)
        out_path = Path(args.out)
    text = report(combos, rule, res, args.dry_run)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')
    figure(combos, str(out_path.with_suffix('')))
    payload = {'dry_run': args.dry_run, 'seeds': SEEDS, 'combinations': combos, 'rule_check': rule, 'results': res,
               'provenance': provenance(('Experiments/r1_mechanism.py', 'Experiments/r1_decision_inbearing.py',
                                         'Experiments/r1_decision_models.py', 'Experiments/r1_strict_stats.py',
                                         'Experiments/r1_xjtu_stats.py'))}
    out_path.with_suffix('.json').write_text(json.dumps(payload, indent=1), encoding='utf-8')
    print(text)
    print(f'written {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
