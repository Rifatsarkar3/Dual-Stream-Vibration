"""
r1_components_stats.py — Revision R1, analysis plan Amendment 10: statistics of the rebuilt
components. Run once, after the final runs; checked before on synthetic data (--dry-run).

Primary family (PRONOSTIA all10; per bearing the mean over the 20 final seeds; exact two-sided
sign-flip test over the per-bearing differences; Holm over the three):
  A10-E  bearing-averaged CES MAE, R (full) against rawonly
  A10-G  bearing-averaged CES MAE averaged over the noise levels 20, 10, 5, 0, -5 dB
         (Experiments/r1_components_noise.py), R against static
  A10-M  bearing-averaged CES MAE, R against no_pgmc
Replication family: the same three on the six XJTU-SY test and five Condition-3 bearings.
Claims (amendment): measurable effect = primary test significant after Holm with R lower;
replicated = the replication test too; only replicated components may become contributions.
Secondary (all10): (i) R against old_env, no_aug, adjacent, no_dann, spectral, Holm over five;
(ii) R against the submitted network and the 1D CNN, Holm over two.
Descriptive (unadjusted sign-flip p): calibration of R and its primary ablations on their own
validation runs (Section 4 rule, functions of Experiments/r1_decision_models.py): MC-dropout
coverage overall and in late life, interval score, ratio of the held-out to the test 0.9-quantile
of |error| (Experiments/r1_decision_inbearing.py), expected cost of the cost-matched trigger at
r = 5 and 10 (tau = 0.20). Mechanism: R's gate weight against SNR; share of increasing raw-prediction
pairs 5, 7.5 and 10 % of the life apart, R against no_pgmc; envelope stream alone, rebuilt against
submitted.
Reproduction check, before any test: recomputed with the same functions, the submitted network's
c*, CES MAE and cost-matched cost at r = 5 and 10 equal the stored values of
outputs/r1_analysis/decision_models_mc30.json within 1e-9; otherwise the analysis stops.

USAGE:
  python Experiments/r1_components_stats.py --dry-run    # synthetic data in a temporary folder
  python Experiments/r1_components_stats.py              # the analysis, once
Outputs: outputs/r1_analysis/components.md and components.json
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import r1_decision_models as dm  # noqa: E402
from r1_decision_inbearing import q90_abs, q90_abs_test  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import SETS, baseline_paths, holm, signflip_p  # noqa: E402
from r1_xjtu_stats import SETS as XSETS, load as xload  # noqa: E402

ALL10 = SETS['all10']
XJTU11 = XSETS['xjtu_test'] + XSETS['xjtu_zs']
NOISY = ['20dB', '10dB', '5dB', '0dB', '-5dB']
PRIMARY = {'A10-E': ('rawonly', 'E, envelope stream (R against raw-vibration stream only)'),
           'A10-G': ('static', 'G, gate under noise (R against static fusion, five noise levels)'),
           'A10-M': ('no_pgmc', 'M, pair PGMC (R against no PGMC)')}
SECONDARY_I = {'old_env': 'submitted envelope extractor', 'no_aug': 'no input noise',
               'adjacent': 'adjacent-window PGMC', 'no_dann': 'without DANN',
               'spectral': 'envelope stream only'}
DESCRIPTIVE = ('full', 'rawonly', 'static', 'no_pgmc')
GAPS = (0.05, 0.075, 0.10)
STORED = ROOT / 'outputs' / 'r1_analysis' / 'decision_models_mc30.json'
OUT = ROOT / 'outputs' / 'r1_analysis'


def real_paths():
    b = json.loads((ROOT / 'outputs/r1_components/budget_selection.json').read_text(encoding='utf-8'))['selected']
    lam = json.loads((ROOT / 'outputs/r1_components/selection.json').read_text(encoding='utf-8'))['selected']
    return {
        'final': {arm: f'outputs/r1_components_final/{arm}' for arm in
                  ['full', 'rawonly', 'static', 'no_pgmc'] + list(SECONDARY_I)},
        'noise': 'outputs/r1_components_final/noise.json',
        'val': {'full': f'outputs/r1_components/lam{lam}_B{b}_fold{{f}}',
                **{a: f'outputs/r1_components_val/{a}_fold{{f}}' for a in DESCRIPTIVE[1:]}},
        'xjtu_final': {arm: f'outputs/r1_components_xjtu_final/{arm}' for arm in DESCRIPTIVE},
        'xjtu_noise': 'outputs/r1_components_xjtu_final/noise.json',
        'submitted': dm.OURS[dm.SUBMITTED]['final'],
        'submitted_val': dm.OURS[dm.SUBMITTED]['val'],
        'submitted_spectral': 'outputs/r1_final_B10/abl20_spectral',
        'cnn': baseline_paths()['1D CNN'],
        'selected': {'budget': b, 'lambda': lam},
    }


# ---------------------------------------------------------------- statistics --------------------
def per_bearing_ces(data, bearings):
    return np.array([np.mean([np.mean(np.abs(dm.ces(data[s][b]['raw']) - data[s][b]['y'])) for s in sorted(data)])
                     for b in bearings])


def noise_per_bearing(noise, run, bearings):
    res = noise['results'][run]
    return np.array([np.mean([np.mean([res[s][lv][b]['ces_mae'] for lv in NOISY]) for s in res])
                     for b in bearings])


def test(ref, other):
    """ref: R; other: comparator; per-bearing arrays, lower is better. d > 0: R better."""
    d = other - ref
    return {'R': float(ref.mean()), 'other': float(other.mean()),
            'rel_pct': float(100 * (other.mean() - ref.mean()) / other.mean()),
            'R_better_on': int((d > 0).sum()), 'n': int(len(d)), 'p_flip': float(signflip_p(d))}


def with_holm(rows):
    adj = holm(np.array([r['p_flip'] for r in rows.values()]))
    for r, a in zip(rows.values(), adj):
        r['p_holm'] = float(a)
        r['significant_R_better'] = bool(a < 0.05 and r['R'] < r['other'])
    return rows


def increasing_pair_share(data, bearings):
    out = []
    for b in bearings:
        per_seed = []
        for s in sorted(data):
            raw = data[s][b]['raw']
            n = len(raw)
            per_seed.append(np.mean([np.mean(raw[k:] > raw[:-k]) for k in
                                     sorted({min(max(1, round(g * n)), n - 1) for g in GAPS})]))
        out.append(np.mean(per_seed))
    return np.array(out)


def describe_model(final, val_runs, bearings):
    cal = dm.calibrate({'val': val_runs, 'dropout': True})
    seeds = sorted(final)
    c = cal['c_star']
    iv = dm.intervals(final, bearings, seeds, lambda d: (d['mu'] - dm.Z * c * d['s'], d['mu'] + dm.Z * c * d['s']))
    t = dm.timing(final, bearings, cal, seeds)[0.2]
    cost = {r: dm.losses(t, f'cmq{r}', r, bearings) for r in (5, 10)}
    return {'c_star': c, 'coverage': iv['pooled_coverage'], 'late_coverage': iv['late_coverage'],
            'interval_score_per_bearing': np.array([iv['per_bearing'][b]['interval_score'] for b in bearings]),
            'tail_ratio': q90_abs(val_runs) / q90_abs_test(final, bearings, seeds),
            'cost_per_bearing': cost, 'ces_mae': float(dm.ces_mae_bavg(final, bearings))}


def val_files(pattern):
    return sum((sorted((ROOT / pattern.format(f=f)).glob('valpreds_seed*.npz')) for f in 'ABCD'), [])


def reproduction_check(paths):
    stored = json.loads(STORED.read_text(encoding='utf-8'))
    final = dm.load_preds(paths['submitted'])
    runs = dm.load_val(val_files(paths['submitted_val']))
    d = describe_model(final, runs, ALL10)
    got = {'c_star': d['c_star'], 'ces_mae': d['ces_mae'],
           'cost_r5': float(d['cost_per_bearing'][5].mean()), 'cost_r10': float(d['cost_per_bearing'][10].mean())}
    want = {'c_star': stored['calibration'][dm.SUBMITTED]['c_star'],
            'ces_mae': stored['ces_mae_bavg_all10'][dm.SUBMITTED],
            'cost_r5': stored['P1a'][dm.SUBMITTED]['5']['cmq'], 'cost_r10': stored['P1a'][dm.SUBMITTED]['10']['cmq']}
    diff = {k: abs(got[k] - want[k]) for k in got}
    if max(diff.values()) > 1e-9:
        raise SystemExit(f'reproduction check failed: {got} against stored {want}')
    return {'recomputed': got, 'stored': want, 'max_abs_diff': max(diff.values())}


def analyse(paths, check=True):
    out = {'paths': {k: v for k, v in paths.items() if k != 'selected'}, 'selected': paths.get('selected')}
    if check:
        out['reproduction_check'] = reproduction_check(paths)

    final = {arm: dm.load_preds(p) for arm, p in paths['final'].items()}
    for arm, f in final.items():
        if sorted(f) != sorted(dm.FINAL_SEEDS):
            raise SystemExit(f'{arm}: expected the 20 final seeds, found {sorted(f)}')
    noise = json.loads((ROOT / paths['noise']).read_text(encoding='utf-8'))
    ces = {arm: per_bearing_ces(f, ALL10) for arm, f in final.items()}

    prim = {}
    for key, (arm, label) in PRIMARY.items():
        if arm == 'static':
            a, b = noise_per_bearing(noise, 'full', ALL10), noise_per_bearing(noise, 'static', ALL10)
        else:
            a, b = ces['full'], ces[arm]
        prim[key] = {'component': label, 'comparator': arm, **test(a, b)}
    out['primary'] = with_holm(prim)

    xf = {arm: xload(p) for arm, p in paths['xjtu_final'].items()}
    xnoise = json.loads((ROOT / paths['xjtu_noise']).read_text(encoding='utf-8'))
    xces = {arm: per_bearing_ces(f, XJTU11) for arm, f in xf.items()}
    rep = {}
    for key, (arm, label) in PRIMARY.items():
        if arm == 'static':
            a, b = noise_per_bearing(xnoise, 'full', XJTU11), noise_per_bearing(xnoise, 'static', XJTU11)
        else:
            a, b = xces['full'], xces[arm]
        rep[key] = {'component': label, 'comparator': arm, **test(a, b)}
    out['replication'] = with_holm(rep)

    out['claims'] = {k: {'measurable_effect': out['primary'][k]['significant_R_better'],
                         'replicated': out['primary'][k]['significant_R_better']
                         and out['replication'][k]['significant_R_better']} for k in PRIMARY}
    for v in out['claims'].values():
        v['may_be_listed_as_contribution'] = v['replicated']

    out['secondary_i'] = with_holm({arm: {'change': lbl, **test(ces['full'], ces[arm])}
                                    for arm, lbl in SECONDARY_I.items()})
    sub, cnn = dm.load_preds(paths['submitted']), dm.load_preds(paths['cnn'])
    out['secondary_ii'] = with_holm({'submitted network': test(ces['full'], per_bearing_ces(sub, ALL10)),
                                     '1D CNN': test(ces['full'], per_bearing_ces(cnn, ALL10))})

    desc = {arm: describe_model(final[arm], dm.load_val(val_files(paths['val'][arm])), ALL10) for arm in DESCRIPTIVE}
    rows = {}
    for arm, d in desc.items():
        row = {k: v for k, v in d.items() if k not in ('interval_score_per_bearing', 'cost_per_bearing')}
        row['interval_score'] = float(d['interval_score_per_bearing'].mean())
        row['cost_r5'], row['cost_r10'] = (float(d['cost_per_bearing'][r].mean()) for r in (5, 10))
        if arm != 'full':
            row['p_interval_score'] = signflip_p(d['interval_score_per_bearing'] - desc['full']['interval_score_per_bearing'])
            row['p_cost_r5'], row['p_cost_r10'] = (float(signflip_p(d['cost_per_bearing'][r] - desc['full']['cost_per_bearing'][r]))
                                                   for r in (5, 10))
        rows[arm] = row
    out['descriptive'] = rows

    levels = noise['snr_levels_db']
    out['mechanism'] = {
        'gate_weight_by_level': {str(lv): float(np.mean([np.mean([noise['results']['full'][s][
            'clean' if lv == 'clean' else f'{lv}dB'][b]['alpha_mean'] for b in ALL10]) for s in noise['results']['full']]))
            for lv in levels},
        'increasing_pairs': {'R': float(increasing_pair_share(final['full'], ALL10).mean()),
                             'no_pgmc': float(increasing_pair_share(final['no_pgmc'], ALL10).mean()),
                             'p_flip': float(signflip_p(increasing_pair_share(final['no_pgmc'], ALL10)
                                                        - increasing_pair_share(final['full'], ALL10)))},
        'envelope_stream_alone': test(ces['spectral'], per_bearing_ces(dm.load_preds(paths['submitted_spectral']), ALL10)),
    }
    return out


# ---------------------------------------------------------------- report ------------------------
def fmt_family(rows, title):
    lines = [f'### {title}', '', '| Test | R | Comparator | R lower by | R better on | p | Holm p | Significant, R better |',
             '|---|---|---|---|---|---|---|---|']
    for k, r in rows.items():
        lines.append(f"| {k} {r.get('component', r.get('change', ''))} | {r['R']:.4f} | {r['other']:.4f} | "
                     f"{r['rel_pct']:+.1f} % | {r['R_better_on']}/{r['n']} | {r['p_flip']:.4f} | {r['p_holm']:.4f} | "
                     f"{'yes' if r['significant_R_better'] else 'no'} |")
    return lines + ['']


def report(out, dry):
    lines = ['# Rebuilt components (revision R1, analysis plan Amendment 10)' + (' — DRY RUN, synthetic data' if dry else ''), '']
    if 'reproduction_check' in out:
        rc = out['reproduction_check']
        lines += [f"Reproduction check: submitted network recomputed, max |difference| to the stored values "
                  f"{rc['max_abs_diff']:.1e} (c* {rc['recomputed']['c_star']:.4f}, CES MAE {rc['recomputed']['ces_mae']:.4f}, "
                  f"cost {rc['recomputed']['cost_r5']:.1f} / {rc['recomputed']['cost_r10']:.1f}).", '']
    if out.get('selected'):
        lines += [f"Selected setting of R: B = {out['selected']['budget']}, lambda = {out['selected']['lambda']}.", '']
    lines += fmt_family(out['primary'], 'Primary family, PRONOSTIA (ten test bearings)')
    lines += fmt_family(out['replication'], 'Replication family, XJTU-SY (six test and five Condition-3 bearings)')
    lines += ['### Claims', '', '| Component | Measurable effect | Replicated | May be listed as a contribution |', '|---|---|---|---|']
    for k, c in out['claims'].items():
        lines.append(f"| {k} | {'yes' if c['measurable_effect'] else 'no'} | {'yes' if c['replicated'] else 'no'} | "
                     f"{'yes' if c['may_be_listed_as_contribution'] else 'no'} |")
    lines += [''] + fmt_family(out['secondary_i'], 'Secondary (i): each change alone')
    lines += fmt_family(out['secondary_ii'], 'Secondary (ii): R against the submitted network and the 1D CNN')
    lines += ['### Descriptive: calibration and decisions (own validation runs; unadjusted p against R)', '',
              '| Model | c* | Coverage | Late life | Interval score | Tail ratio | Cost r = 5 | Cost r = 10 | CES MAE |',
              '|---|---|---|---|---|---|---|---|---|']
    for arm, r in out['descriptive'].items():
        p = '' if arm == 'full' else f" (p {r['p_interval_score']:.3f})"
        pc = '' if arm == 'full' else f" (p {r['p_cost_r5']:.3f})"
        pc10 = '' if arm == 'full' else f" (p {r['p_cost_r10']:.3f})"
        lines.append(f"| {arm} | {r['c_star']:.2f} | {r['coverage']:.1f} % | {r['late_coverage']:.1f} % | "
                     f"{r['interval_score']:.3f}{p} | {r['tail_ratio']:.2f} | {r['cost_r5']:.1f}{pc} | {r['cost_r10']:.1f}{pc10} | "
                     f"{r['ces_mae']:.4f} |")
    m = out['mechanism']
    lines += ['', '### Mechanism', '',
              'Gate weight of R by noise level: ' + ', '.join(f'{k} {v:.3f}' for k, v in m['gate_weight_by_level'].items()) + '.',
              f"Share of increasing raw-prediction pairs 5 to 10 % of the life apart: R {m['increasing_pairs']['R']:.3f}, "
              f"without PGMC {m['increasing_pairs']['no_pgmc']:.3f} (p {m['increasing_pairs']['p_flip']:.4f}).",
              f"Envelope stream alone, CES MAE: rebuilt {m['envelope_stream_alone']['R']:.4f}, submitted "
              f"{m['envelope_stream_alone']['other']:.4f} ({m['envelope_stream_alone']['R_better_on']}/10 bearings, "
              f"p {m['envelope_stream_alone']['p_flip']:.4f}).", '']
    return '\n'.join(lines)


# ---------------------------------------------------------------- dry run -----------------------
def synthetic(tmp, rng):
    """Synthetic final, validation and noise files in the real layout. Planted: rawonly worse
    than R on every bearing of both datasets; every other comparison null."""
    worse = {'rawonly': 0.04, 'spectral': 0.08}

    def bearings_preds(bearings, lengths, arm, seed_rng, with_mc=True):
        out = {}
        for b in bearings:
            y = np.linspace(1.0, 0.0, lengths[b])
            raw = np.clip(y + 0.03 * np.sin(lengths[b] / 50) + (0.08 + worse.get(arm, 0.0))
                          * seed_rng.standard_normal(len(y)), -0.2, 1.2)
            out[f'raw_{b}'], out[f'tgt_{b}'] = raw.astype(np.float32), y.astype(np.float32)
            if with_mc:
                out[f'mcmean_{b}'] = (raw + 0.01 * seed_rng.standard_normal(len(y))).astype(np.float32)
                out[f'mcstd_{b}'] = np.full(len(y), 0.03, np.float32)
        return out

    def write_final(path, bearings, lengths, arm, zs=None, zs_lengths=None):
        path.mkdir(parents=True, exist_ok=True)
        for s in dm.FINAL_SEEDS:
            np.savez(path / f'preds_seed{s}.npz', **bearings_preds(bearings, lengths, arm, rng))
            if zs:
                np.savez(path / f'zeroshot_preds_seed{s}.npz', **bearings_preds(zs, zs_lengths, arm, rng, with_mc=False))

    def write_val(pattern, arm):
        for f in 'ABCD':
            d = Path(pattern.format(f=f))
            d.mkdir(parents=True, exist_ok=True)
            for s in dm.ES_SEEDS:
                np.savez(d / f'valpreds_seed{s}.npz', **bearings_preds(['train_x'], {'train_x': 300}, arm, rng))

    def write_noise(path, final_dirs, bearings):
        res = {}
        for run in ('full', 'static'):
            res[run] = {}
            for s in dm.FINAL_SEEDS:
                d = np.load(final_dirs[run] / f'preds_seed{s}.npz')
                base = {b: float(np.mean(np.abs(dm.ces(d[f'raw_{b}'].astype(float)) - d[f'tgt_{b}']))) for b in bearings
                        if f'raw_{b}' in d.files}
                if len(base) < len(bearings):
                    z = np.load(final_dirs[run] / f'zeroshot_preds_seed{s}.npz')
                    base.update({b: float(np.mean(np.abs(dm.ces(z[f'raw_{b}'].astype(float)) - z[f'tgt_{b}'])))
                                 for b in bearings if b not in base})
                res[run][str(s)] = {lv: {b: {'ces_mae': base[b] * (1 + 0.1 * i) + 0.002 * rng.standard_normal(),
                                             'raw_mae': base[b], 'alpha_mean': 0.5 if run == 'static' else 0.45 - 0.02 * i}
                                         for b in bearings} for i, lv in enumerate(['clean'] + NOISY)}
        path.write_text(json.dumps({'snr_levels_db': ['clean', 20, 10, 5, 0, -5], 'results': res}), encoding='utf-8')

    lengths = {b: int(rng.integers(150, 400)) for b in ALL10}
    xlen = {b: int(rng.integers(150, 400)) for b in XJTU11}
    paths = {'final': {}, 'val': {}, 'xjtu_final': {}}
    for arm in ['full', 'rawonly', 'static', 'no_pgmc'] + list(SECONDARY_I):
        p = tmp / 'final' / arm
        write_final(p, ALL10, lengths, arm)
        paths['final'][arm] = str(p)
    for arm in DESCRIPTIVE:
        pattern = str(tmp / 'val' / f'{arm}_fold{{f}}')
        write_val(pattern, arm)
        paths['val'][arm] = pattern
        p = tmp / 'xjtu' / arm
        write_final(p, XSETS['xjtu_test'], xlen, arm, XSETS['xjtu_zs'], xlen)
        paths['xjtu_final'][arm] = str(p)
    write_noise(tmp / 'noise.json', {r: tmp / 'final' / r for r in ('full', 'static')}, ALL10)
    write_noise(tmp / 'xnoise.json', {r: tmp / 'xjtu' / r for r in ('full', 'static')}, XJTU11)
    for name, arm in (('submitted', 'full'), ('cnn', 'no_aug'), ('submitted_spectral', 'spectral')):
        p = tmp / name
        write_final(p, ALL10, lengths, arm)
        paths[name] = str(p)
    paths.update(noise=str(tmp / 'noise.json'), xjtu_noise=str(tmp / 'xnoise.json'))
    return paths


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            paths = synthetic(Path(tmp), np.random.default_rng(10))
            out = analyse(paths, check=False)
            text = report(out, dry=True)
        print(text)
        assert out['claims']['A10-E']['replicated'], 'planted effect (rawonly worse) not detected'
        assert out['primary']['A10-E']['R_better_on'] == 10 and out['replication']['A10-E']['R_better_on'] == 11
        assert abs(out['primary']['A10-E']['p_flip'] - 2 / 2 ** 10) < 1e-12
        assert abs(out['replication']['A10-E']['p_flip'] - 2 / 2 ** 11) < 1e-12
        print('DRY RUN: planted effect detected at the smallest attainable p on both datasets; '
              f"null comparisons: A10-G p {out['primary']['A10-G']['p_flip']:.3f}, "
              f"A10-M p {out['primary']['A10-M']['p_flip']:.3f}")
        return 0
    out = analyse(real_paths())
    out['date'] = time.strftime('%Y-%m-%d %H:%M')
    out['provenance'] = provenance(extra_code=('Experiments/r1_components_stats.py',))
    text = report(out, dry=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'components.md').write_text(text + '\n', encoding='utf-8')

    def jsonable(x):
        if isinstance(x, dict):
            return {str(k): jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [jsonable(v) for v in x]
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        return x
    (OUT / 'components.json').write_text(json.dumps(jsonable(out), indent=1), encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
