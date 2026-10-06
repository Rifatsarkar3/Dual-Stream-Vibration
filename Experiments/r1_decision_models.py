"""
r1_decision_models.py — Revision R1, analysis plan Amendment 2, P1 to P3: the same
validation-only calibration and the same maintenance-decision analysis for all nine PRONOSTIA
models (our two versions and the seven baselines).

Calibration, fitted on each model's 20 validation runs (final-epoch predictions on the
held-out training bearing of each four-fold selection run) and applied once to its test
predictions:
  residual quantile (every model): q(p) = mean over runs of the p-quantile of (yhat - y),
      yhat the deterministic prediction; lower bound at level alpha: yhat - q(1 - alpha)
  MC dropout (models with dropout): c* = mean over runs of q90(|y - mu| / (z s)) (Section 4);
      lower bound mu - z c* s (Section 9); two-sided interval mu +- z c* s
Rules: raw, CES, PAVA (offline), 90 % residual-quantile bound (lb90res), MC-dropout bound
(lbmc), cost-matched quantile trigger cmq_r = residual-quantile bound at alpha = 1/(1 + r).
Alarm timing as r1_decision.py / exp29 (first crossing of tau; A lateness, B earliness);
per bearing A_b and B_b are seed means and L_b(r) = r A_b + B_b.

  P1  primary: L(r) of cmq_r, r = 5 and 10, tau = 0.20, all10, 20 final seeds; each version
      of our model against each baseline, exact sign-flip test over bearings, Holm over the
      seven baselines. Secondary: all rules, tau in {0.30, 0.20, 0.10}, r in {1, 2, 5, 10,
      1000}, break-even ratios, Kendall tau (CES MAE ranking vs L(r) ranking), other sets.
  P1a cmq_r against the best other deployable rule, r in {2, 5, 10, 20, 100}.
  P2  two-sided 90 % MC-dropout intervals (models with dropout): coverage, width, interval
      score; primary bearing-averaged interval score, Holm over the six baselines with dropout.
      Secondary: residual-quantile interval [yhat - q(0.95), yhat - q(0.05)] for all nine.
  P3  standard practice (in-bearing early-stopping checkpoints, raw trigger) against the
      framework (final runs on the same five seeds, cmq_r), every model, Holm over nine.

USAGE:
  python Experiments/r1_decision_models.py [--out outputs/r1_analysis/decision_models.md]
  python Experiments/r1_decision_models.py --dry-run     # synthetic data, scratch output, no real files read
"""

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import exp29_maintenance_timing as e29  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import SETS, baseline_paths, holm, signflip_p  # noqa: E402
from src.isotonic_eval import IsotonicRegressor  # noqa: E402
from src.online_filters import ConstrainedExponentialSmoothing  # noqa: E402

Z = 1.6449
ADOPTED, SUBMITTED = 'adopted variant', 'submitted architecture'
VERSIONS = (ADOPTED, SUBMITTED)
OURS = {ADOPTED: {'final': 'outputs/r1_final_temporal_B5/final20', 'val': 'outputs/r1_temporal/lam0.08_B5_fold{f}',
                  'es': 'outputs/r1_protocol_temporal/full_inbearing_es'},
        SUBMITTED: {'final': 'outputs/r1_final_B10/final20', 'val': 'outputs/r1_lambda/lam0.08_B10_fold{f}',
                    'es': 'outputs/r1_reeval/full_inbearing_es'}}
BASELINES = {'1D CNN': 'cnn', 'BiLSTM': 'bilstm', 'CNN-LSTM': 'cnnlstm', '1D ResNet': 'resnet',
             'Transformer': 'transformer', 'TSMC-CNN [21]': 'tsmccnn', 'MSCNN [7]': 'mscnn'}
NO_DROPOUT = {'TSMC-CNN [21]'}
ES_SEEDS = [42, 123, 777, 999, 7]
FINAL_SEEDS = [42, 123, 777, 999, 7, 8, 21, 34, 55, 89, 144, 233, 377, 610, 987,
               1597, 2584, 4181, 6765, 10946]
TAUS = (0.30, 0.20, 0.10)
TAU = 0.20
R_PRIMARY = (5, 10)
R_SECONDARY = (1, 2, 5, 10, 1000)
R_THEORY = (2, 5, 10, 20, 100)
R_ALL = sorted(set(R_PRIMARY) | set(R_SECONDARY) | set(R_THEORY))
FIXED_RULES = ('raw', 'ces', 'pava', 'lb90res', 'lbmc')
DEPLOYABLE = ('raw', 'ces', 'lb90res', 'lbmc')
SET_NAMES = ('all10', 'new6', 'orig4', 'zs3')
LATE = 0.2


# ---------------------------------------------------------------- data --------------------------
def load_preds(path):
    """{seed: {bearing: {y, raw, mu, s}}} from <path>/preds_seed*.npz"""
    out = {}
    for f in sorted(glob.glob(str(ROOT / path / 'preds_seed*.npz'))):
        d = np.load(f)
        seed = int(os.path.basename(f)[10:-4])
        out[seed] = {k[4:]: {'y': d[k].astype(float), 'raw': d['raw_' + k[4:]].astype(float),
                             'mu': d['mcmean_' + k[4:]].astype(float),
                             's': np.clip(d['mcstd_' + k[4:]].astype(float), 1e-4, None)}
                     for k in d.files if k.startswith('tgt_')}
    return out


def load_val(files):
    runs = []
    for f in files:
        d = np.load(f)
        b = next(k[4:] for k in d.files if k.startswith('tgt_'))
        runs.append({'y': d[f'tgt_{b}'].astype(float), 'raw': d[f'raw_{b}'].astype(float),
                     'mu': d[f'mcmean_{b}'].astype(float), 's': np.clip(d[f'mcstd_{b}'].astype(float), 1e-4, None)})
    return runs


def real_models():
    finals = baseline_paths()
    models = {}
    for v in VERSIONS:
        files = sum((sorted(glob.glob(str(ROOT / OURS[v]['val'].format(f=f) / 'valpreds_seed*.npz'))) for f in 'ABCD'), [])
        models[v] = {'final': load_preds(OURS[v]['final']), 'val': load_val(files), 'es': load_preds(OURS[v]['es']),
                     'dropout': True, 'paths': {'final': OURS[v]['final'], 'val': OURS[v]['val'], 'es': OURS[v]['es']}}
    for label, short in BASELINES.items():
        files = sorted(glob.glob(str(ROOT / 'outputs' / 'r1_baseline_valpreds' / short / 'fold*_seed*.npz')))
        es_dir = f'outputs/r1_protocol_baselines/{short}'
        models[label] = {'final': load_preds(finals[label]), 'val': load_val(files), 'es': load_preds(es_dir),
                         'dropout': label not in NO_DROPOUT,
                         'paths': {'final': finals[label], 'val': f'outputs/r1_baseline_valpreds/{short}', 'es': es_dir}}
    for name, m in models.items():
        if len(m['val']) != 20 or sorted(m['final']) != sorted(FINAL_SEEDS) or sorted(m['es']) != sorted(ES_SEEDS):
            raise SystemExit(f"{name}: expected 20 validation runs, the 20 final seeds and early-stopping seeds {ES_SEEDS}; "
                             f"found {len(m['val'])}, {sorted(m['final'])}, {sorted(m['es'])}")
    return models


def synthetic_models(rng):
    """dry run: random degradation trajectories with model-specific bias and noise"""
    bearings = sorted(set(sum((SETS[s] for s in SET_NAMES), [])))
    lengths = {b: int(rng.integers(150, 600)) for b in bearings}

    def traj(bias, noise, seeds):
        out = {}
        for s in seeds:
            out[s] = {}
            for b in bearings:
                y = np.linspace(1.0, 0.0, lengths[b])
                raw = np.clip(y + bias + noise * rng.standard_normal(len(y)), -0.2, 1.2)
                out[s][b] = {'y': y, 'raw': raw, 'mu': raw + 0.01 * rng.standard_normal(len(y)),
                             's': np.full(len(y), max(noise / 4, 1e-4))}
        return out

    models = {}
    for i, name in enumerate(list(VERSIONS) + list(BASELINES)):
        bias, noise = 0.02 * (i % 3 - 1), 0.05 + 0.01 * i
        val = [traj(bias, noise, [0])[0][bearings[0]] for _ in range(20)]
        models[name] = {'final': traj(bias, noise, FINAL_SEEDS), 'val': val,
                        'es': traj(bias + 0.03, noise, ES_SEEDS), 'dropout': name not in NO_DROPOUT,
                        'paths': {'final': 'synthetic', 'val': 'synthetic', 'es': 'synthetic'}}
        if name in NO_DROPOUT:
            for d in (models[name]['final'], models[name]['es']):
                for s in d:
                    for b in d[s]:
                        d[s][b]['s'] = np.full_like(d[s][b]['s'], 1e-4)
    return models


# ---------------------------------------------------------- calibration -------------------------
def calibrate(m):
    ps = sorted({0.9, 0.95, 0.05} | {r / (1 + r) for r in R_ALL})
    q = {p: float(np.mean([np.quantile(r['raw'] - r['y'], p) for r in m['val']])) for p in ps}
    c = (float(np.mean([np.quantile(np.abs(r['y'] - r['mu']) / (Z * r['s']), 0.90) for r in m['val']]))
         if m['dropout'] else None)
    return {'q': q, 'c_star': c}


def ces(x):
    return np.asarray(ConstrainedExponentialSmoothing(alpha=0.2, epsilon=0.05).process(x), float)


def sequences(d, cal):
    raw = d['raw']
    seq = {'raw': raw, 'ces': ces(raw), 'pava': np.asarray(IsotonicRegressor(increasing=False).fit_transform(raw), float),
           'lb90res': raw - cal['q'][0.9]}
    for r in R_ALL:
        seq[f'cmq{r}'] = raw - cal['q'][r / (1 + r)]
    if cal['c_star'] is not None:
        seq['lbmc'] = d['mu'] - Z * cal['c_star'] * d['s']
    return seq


def timing(data, bearings, cal, seeds):
    """{tau: {'A': {rule: {bearing: seed mean}}, 'B': ..., 'late': ...}}"""
    acc = {tau: {} for tau in TAUS}
    for b in bearings:
        for s in seeds:
            seq = sequences(data[s][b], cal)
            for v, x in seq.items():
                for tau in TAUS:
                    dt, _severe = e29.bearing_deltas(data[s][b]['y'], np.asarray(x, float), tau)
                    acc[tau].setdefault(v, {}).setdefault(b, []).append(float(dt))
    out = {}
    for tau in TAUS:
        out[tau] = {k: {v: {b: float(np.mean([f(d) for d in dts])) for b, dts in per_b.items()}
                        for v, per_b in acc[tau].items()}
                    for k, f in (('A', lambda d: max(0.0, d)), ('B', lambda d: max(0.0, -d)),
                                 ('late', lambda d: float(d > 0)))}
    return out


def losses(t, rule, r, bearings):
    return np.array([r * t['A'][rule][b] + t['B'][rule][b] for b in bearings])


def compare(ref, other):
    """ref, other: per-bearing arrays (lower is better); Δ = (other - ref) / other, positive: ref better"""
    d = other - ref
    return {'ref': float(ref.mean()), 'other': float(other.mean()),
            'rel': float(100 * (other.mean() - ref.mean()) / other.mean()) if other.mean() else float('nan'),
            'k': int((d > 0).sum()), 'n': int(len(d)), 'p_flip': float(signflip_p(d))}


def add_holm(rows, key='p_flip'):
    adj = holm(np.array([r[key] for r in rows.values()]))
    for r, a in zip(rows.values(), adj):
        r['p_holm'] = float(a)
    return rows


def envelope(t, bearings, rules, r_min=0.5, r_max=1000.0):
    stats_ = {v: {'A': float(np.mean([t['A'][v][b] for b in bearings])),
                  'B': float(np.mean([t['B'][v][b] for b in bearings]))} for v in rules}
    grid = np.logspace(np.log10(r_min), np.log10(r_max), 800)
    best = [min(rules, key=lambda v: r * stats_[v]['A'] + stats_[v]['B']) for r in grid]
    segs, start = [], grid[0]
    for i in range(1, len(grid)):
        if best[i] != best[i - 1]:
            segs.append({'rule': best[i - 1], 'r_from': float(start), 'r_to': float(grid[i])})
            start = grid[i]
    segs.append({'rule': best[-1], 'r_from': float(start), 'r_to': float(grid[-1])})
    return segs


def intervals(data, bearings, seeds, lo_hi):
    per_b, ys, ins = {}, [], []
    for b in bearings:
        cov, wid, isc = [], [], []
        for s in seeds:
            d = data[s][b]
            lo, hi = lo_hi(d)
            y = d['y']
            inside = (y >= lo) & (y <= hi)
            cov.append(inside)
            wid.append(hi - lo)
            isc.append((hi - lo) + (2 / 0.1) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None)))
            ys.append(y)
            ins.append(inside)
        per_b[b] = {'coverage': 100.0 * float(np.concatenate(cov).mean()), 'width': float(np.concatenate(wid).mean()),
                    'interval_score': float(np.concatenate(isc).mean())}
    y, inside = np.concatenate(ys), np.concatenate(ins)
    return {'per_bearing': per_b, 'pooled_coverage': 100.0 * float(inside.mean()),
            'late_coverage': 100.0 * float(inside[y < LATE].mean()) if (y < LATE).any() else float('nan')}


def ces_mae_bavg(data, bearings):
    return float(np.mean([np.mean([np.mean(np.abs(ces(data[s][b]['raw']) - data[s][b]['y'])) for s in sorted(data)])
                          for b in bearings]))


# ---------------------------------------------------------------- main --------------------------
def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'decision_models.md'))
    ap.add_argument('--dry-run', action='store_true', help='synthetic data; writes to a temporary folder')
    args = ap.parse_args()
    if args.dry_run:
        models = synthetic_models(np.random.default_rng(0))
        out_path = Path(tempfile.mkdtemp(prefix='decision_models_dry_')) / 'decision_models.md'
    else:
        models = real_models()
        out_path = Path(args.out)
    names = list(models)
    base_names = list(BASELINES)
    drop_names = [n for n in names if models[n]['dropout']]
    cal = {n: calibrate(models[n]) for n in names}
    report = {'dry_run': args.dry_run, 'paths': {n: models[n]['paths'] for n in names},
              'calibration': cal, 'sets': {}}

    # timing on the final runs (all seeds) per set; P3 needs the five early-stopping seeds as well
    tim = {s: {n: timing(models[n]['final'], SETS[s], cal[n], sorted(models[n]['final'])) for n in names}
           for s in SET_NAMES}
    lines = ['# Calibration and maintenance decisions for all nine models (revision R1, Amendment 2)',
             '', ('**DRY RUN on synthetic data: numbers are meaningless.**' if args.dry_run else
                  'Generated by `Experiments/r1_decision_models.py`; definitions in analysis plan Section 11, Amendment 2.'),
             '', 'Loss L(r) = r A + B per bearing (A mean lateness, B mean earliness, in windows of 10 s; seed means), '
             'bearing-averaged. Δ = (comparator − reference) / comparator; positive: reference better. p: exact sign-flip '
             'test over bearings; Holm within each family.', '',
             '## Calibration factors (validation runs only)', '',
             '| Model | q(0.9) | q(5/6) | q(10/11) | c* (MC dropout) |', '|---|---|---|---|---|']
    for n in names:
        q, c = cal[n]['q'], cal[n]['c_star']
        c_text = '–' if c is None else f'{c:.3f}'
        lines.append(f'| {n} | {q[0.9]:.4f} | {q[5 / 6]:.4f} | {q[10 / 11]:.4f} | {c_text} |')

    for s in SET_NAMES:
        bearings = SETS[s]
        block = {'P1': {}, 'losses_cmq': {}, 'secondary': {}}
        for r in R_PRIMARY:
            block['losses_cmq'][str(r)] = {n: float(losses(tim[s][n][TAU], f'cmq{r}', r, bearings).mean()) for n in names}
            for v in VERSIONS:
                ref = losses(tim[s][v][TAU], f'cmq{r}', r, bearings)
                rows = {b: compare(ref, losses(tim[s][b][TAU], f'cmq{r}', r, bearings)) for b in base_names}
                block['P1'][f'{v} | r={r}'] = add_holm(rows)
        # secondary: every rule, tau and r
        for tau in TAUS:
            block['secondary'][f'{tau:.2f}'] = {
                n: {'L': {rule: {str(r): float(losses(tim[s][n][tau], rule, r, bearings).mean()) for r in R_SECONDARY}
                          for rule in FIXED_RULES if rule in tim[s][n][tau]['A']},
                    'L_cmq_at_own_r': {str(r): float(losses(tim[s][n][tau], f'cmq{r}', r, bearings).mean())
                                       for r in R_SECONDARY},
                    'envelope': envelope(tim[s][n][tau], bearings,
                                         [x for x in FIXED_RULES if x in tim[s][n][tau]['A']]),
                    'late_majority_bearings': {rule: [b for b in bearings if tim[s][n][tau]['late'][rule][b] > 0.5]
                                               for rule in tim[s][n][tau]['late']}}
                for n in names}
        report['sets'][s] = block

        lines += ['', f'## {s} ({len(bearings)} bearings; smallest attainable p = {2 / 2 ** len(bearings):.4f})'
                  + ('' if s == 'all10' else ', descriptive'), '',
                  f'### P1: cost-matched quantile trigger, τ = {TAU:.2f}' + (' (primary)' if s == 'all10' else ''), '',
                  '| Model | ' + ' | '.join(f'L(r = {r})' for r in R_PRIMARY) + ' | '
                  + ' | '.join(f'Δ vs {v}, r = {r} (better on, p, Holm)' for v in VERSIONS for r in R_PRIMARY) + ' |',
                  '|---|' + '---|' * (len(R_PRIMARY) + len(VERSIONS) * len(R_PRIMARY))]
        for n in names:
            cells = [f"{block['losses_cmq'][str(r)][n]:.1f}" for r in R_PRIMARY]
            for v in VERSIONS:
                for r in R_PRIMARY:
                    if n in base_names:
                        c = block['P1'][f'{v} | r={r}'][n]
                        cells.append(f"{c['rel']:+.1f} % ({c['k']}/{c['n']}, {c['p_flip']:.3f}, {c['p_holm']:.3f})")
                    else:
                        cells.append('reference' if n == v else '–')
            lines.append(f'| {n} | ' + ' | '.join(cells) + ' |')

    # ---- secondary on all10: rules at tau = 0.20, Kendall tau, P1a, envelope --------------------
    a10 = SETS['all10']
    sec = report['sets']['all10']['secondary'][f'{TAU:.2f}']
    lines += ['', f'### all10, every rule at τ = {TAU:.2f} (secondary; PAVA offline)', '',
              '| Model | rule | ' + ' | '.join(f'L(r = {r})' for r in R_SECONDARY) + ' |',
              '|---|---|' + '---|' * len(R_SECONDARY)]
    for n in names:
        for rule in list(sec[n]['L']) + ['cmq (r = own)']:
            vals = (sec[n]['L_cmq_at_own_r'] if rule.startswith('cmq') else sec[n]['L'][rule])
            lines.append(f'| {n} | {rule} | ' + ' | '.join(f'{vals[str(r)]:.1f}' for r in R_SECONDARY) + ' |')
    lines += ['', 'Decision-optimal rule by cost ratio (lower envelope over raw, CES, PAVA, lb90res, lbmc), τ = 0.20:', '']
    for n in names:
        lines.append(f'- {n}: ' + '; '.join(f"{g['rule']} {g['r_from']:.2f}–{g['r_to']:.2f}" for g in sec[n]['envelope']))

    mae_rank = {n: ces_mae_bavg(models[n]['final'], a10) for n in names}
    report['ces_mae_bavg_all10'] = mae_rank
    kt = {}
    for r in R_PRIMARY:
        lv = report['sets']['all10']['losses_cmq'][str(r)]
        tau_k, p_k = stats.kendalltau([mae_rank[n] for n in names], [lv[n] for n in names])
        kt[str(r)] = {'tau': float(tau_k), 'p': float(p_k),
                      'ranking_ces_mae': sorted(names, key=mae_rank.get), 'ranking_loss': sorted(names, key=lv.get)}
    report['kendall_ces_mae_vs_loss'] = kt
    lines += ['', '### Ranking: bearing-averaged CES MAE against L(r) of the cost-matched trigger (all10, τ = 0.20)', '']
    for r in R_PRIMARY:
        lines.append(f"- r = {r}: Kendall tau {kt[str(r)]['tau']:.2f} (p {kt[str(r)]['p']:.3f}). CES MAE: "
                     + ', '.join(kt[str(r)]['ranking_ces_mae']) + '. Loss: ' + ', '.join(kt[str(r)]['ranking_loss']) + '.')

    p1a = {}
    lines += ['', '### P1a: cost-matched trigger against the best other deployable rule (all10, τ = 0.20)', '',
              '| Model | ' + ' | '.join(f'r = {r}: cmq / best other (rule), excess' for r in R_THEORY) + ' |',
              '|---|' + '---|' * len(R_THEORY)]
    for n in names:
        t = tim['all10'][n][TAU]
        cells, p1a[n] = [], {}
        for r in R_THEORY:
            lc = float(losses(t, f'cmq{r}', r, a10).mean())
            others = {rule: float(losses(t, rule, r, a10).mean()) for rule in DEPLOYABLE if rule in t['A']}
            best = min(others, key=others.get)
            exc = 100 * (lc - others[best]) / others[best] if others[best] else float('nan')
            p1a[n][str(r)] = {'cmq': lc, 'best_other': best, 'best_other_L': others[best], 'excess_pct': exc}
            cells.append(f'{lc:.1f} / {others[best]:.1f} ({best}), {exc:+.1f} %')
        lines.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    report['P1a'] = p1a

    # ---- P2: interval quality ------------------------------------------------------------------
    p2 = {'mc': {}, 'residual': {}, 'tests': {}}
    for n in drop_names:
        c = cal[n]['c_star']
        p2['mc'][n] = intervals(models[n]['final'], a10, sorted(models[n]['final']),
                                lambda d, c=c: (d['mu'] - Z * c * d['s'], d['mu'] + Z * c * d['s']))
    for n in names:
        q = cal[n]['q']
        p2['residual'][n] = intervals(models[n]['final'], a10, sorted(models[n]['final']),
                                      lambda d, q=q: (d['raw'] - q[0.95], d['raw'] - q[0.05]))
    drop_base = [b for b in base_names if b in drop_names]
    for v in VERSIONS:
        ref = np.array([p2['mc'][v]['per_bearing'][b]['interval_score'] for b in a10])
        rows = {b: compare(ref, np.array([p2['mc'][b]['per_bearing'][x]['interval_score'] for x in a10]))
                for b in drop_base}
        p2['tests'][v] = add_holm(rows)
    report['P2'] = p2
    lines += ['', '## P2: calibrated two-sided 90 % intervals (all10)', '',
              '| Model | interval | c* | coverage, pooled | coverage, late life | mean width | interval score, bearing avg | '
              'Δ vs adopted (better on, p, Holm) | Δ vs submitted (better on, p, Holm) |', '|---|---|---|---|---|---|---|---|---|']
    for kind in ('mc', 'residual'):
        for n in (drop_names if kind == 'mc' else names):
            e = p2[kind][n]
            isb = np.mean([e['per_bearing'][b]['interval_score'] for b in a10])
            wid = np.mean([e['per_bearing'][b]['width'] for b in a10])
            cells = []
            for v in VERSIONS:
                if kind == 'mc' and n in drop_base:
                    t = p2['tests'][v][n]
                    cells.append(f"{t['rel']:+.1f} % ({t['k']}/{t['n']}, {t['p_flip']:.3f}, {t['p_holm']:.3f})")
                else:
                    cells.append('reference' if (kind == 'mc' and n == v) else '–')
            cstr = f"{cal[n]['c_star']:.3f}" if kind == 'mc' else '–'
            lines.append(f"| {n} | {'MC dropout' if kind == 'mc' else 'residual quantile'} | {cstr} | "
                         f"{e['pooled_coverage']:.1f} % | {e['late_coverage']:.1f} % | {wid:.4f} | {isb:.4f} | "
                         + ' | '.join(cells) + ' |')

    # ---- P3: framework against standard practice -----------------------------------------------
    p3 = {}
    for r in R_PRIMARY:
        rows, decomp = {}, {}
        for n in names:
            seeds = ES_SEEDS
            t_es = timing(models[n]['es'], a10, cal[n], seeds)[TAU]
            t_fin = timing(models[n]['final'], a10, cal[n], seeds)[TAU]
            std = losses(t_es, 'raw', r, a10)
            fw = losses(t_fin, f'cmq{r}', r, a10)
            fin_raw = losses(t_fin, 'raw', r, a10)
            rows[n] = compare(fw, std)          # reference: framework; positive: framework better
            decomp[n] = {'selection': compare(fin_raw, std), 'decision_layer': compare(fw, fin_raw)}
        add_holm(rows)
        p3[str(r)] = {'framework_vs_standard': rows, 'decomposition': decomp,
                      'n_models_framework_lower': int(sum(rows[n]['other'] > rows[n]['ref'] for n in names))}
    report['P3'] = p3
    lines += ['', '## P3: framework against standard practice (all10, τ = 0.20, seeds 42, 123, 777, 999, 7)', '',
              'Standard practice: in-bearing early-stopping checkpoints, raw prediction as trigger. Framework: the final '
              'runs (budget selected on held-out bearings) with the cost-matched trigger. Δ positive: framework better.', '',
              '| Model | r | standard L | framework L | Δ (better on, p, Holm over nine) | selection only: Δ | '
              'decision layer only: Δ |', '|---|---|---|---|---|---|---|']
    for r in R_PRIMARY:
        for n in names:
            c, d = p3[str(r)]['framework_vs_standard'][n], p3[str(r)]['decomposition'][n]
            lines.append(f"| {n} | {r} | {c['other']:.1f} | {c['ref']:.1f} | {c['rel']:+.1f} % ({c['k']}/{c['n']}, "
                         f"{c['p_flip']:.3f}, {c['p_holm']:.3f}) | {d['selection']['rel']:+.1f} % | "
                         f"{d['decision_layer']['rel']:+.1f} % |")
        lines.append(f"| framework lower for | {r} | {p3[str(r)]['n_models_framework_lower']} of {len(names)} models | | | | |")

    report['provenance'] = provenance(('Experiments/r1_decision_models.py', 'Experiments/exp29_maintenance_timing.py',
                                       'Experiments/r1_strict_stats.py', 'Experiments/r1_analysis.py'))
    text = '\n'.join(lines) + '\n'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')
    out_path.with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(text if args.dry_run else '\n'.join(lines[:40]))
    print(f'written {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
