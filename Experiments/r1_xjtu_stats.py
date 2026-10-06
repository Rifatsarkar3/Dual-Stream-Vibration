"""
r1_xjtu_stats.py — Revision R1: XJTU-SY comparison tables with the statistics of analysis plan
Section 8 (plan Section 7, B2: after adoption, XJTU-SY and Sections 8 to 9 are rerun).

Models (test predictions saved per bearing):
  adopted variant        outputs/r1_final_temporal_xjtu/final20   (20 seeds)
  submitted architecture outputs/exp17_xjtu/final20                (20 seeds, as submitted)
  1D CNN, BiLSTM, CNN-LSTM  outputs/r1_reeval_xjtu/baseline_*     (5 seeds each, the submission's
                         XJTU baselines, re-evaluated by r1_reeval_xjtu.py)
Bearing sets: the six XJTU-SY test bearings (xjtu_test) and the five Condition-3 bearings (xjtu_zs).

Per bearing: mean over each model's own seeds (20 for ours, 5 for the baselines). Primary
quantity: bearing-averaged CES MAE; raw MAE co-reported; window-pooled metrics (mean and SD over
seeds) and offline PAVA as in r1_tables.py. Tests: exact two-sided sign-flip test over bearings,
Holm over the three baselines for each version of our model; the two versions compared
descriptively. With 6 (5) bearings the smallest attainable p is 0.031 (0.063), so with three
comparisons no Holm-adjusted p can fall below 0.094 (0.19): XJTU-SY results are descriptive.

USAGE:
  python Experiments/r1_xjtu_stats.py [--out outputs/r1_analysis/xjtu_temporal.md]
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
from r1_analysis import seed_metrics, variants_of  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import holm, signflip_p  # noqa: E402

ADOPTED = 'DSAFLite+PGMC with temporal context (adopted)'
SUBMITTED = 'DSAFLite+PGMC, submitted architecture'
MODELS = {ADOPTED: 'outputs/r1_final_temporal_xjtu/final20',
          SUBMITTED: 'outputs/exp17_xjtu/final20',
          '1D CNN': 'outputs/r1_reeval_xjtu/baseline_cnn',
          'BiLSTM': 'outputs/r1_reeval_xjtu/baseline_bilstm',
          'CNN-LSTM': 'outputs/r1_reeval_xjtu/baseline_cnnlstm'}
BASELINES = ('1D CNN', 'BiLSTM', 'CNN-LSTM')
SETS = {'xjtu_test': ['test_01_Bearing1_3', 'test_02_Bearing1_4', 'test_03_Bearing1_5',
                      'test_04_Bearing2_3', 'test_05_Bearing2_4', 'test_06_Bearing2_5'],
        'xjtu_zs': ['zeroshot_01_Bearing3_1', 'zeroshot_02_Bearing3_2', 'zeroshot_03_Bearing3_3',
                    'zeroshot_04_Bearing3_4', 'zeroshot_05_Bearing3_5']}


def load(path):
    """{seed: {bearing: {raw, y, mu, s}}} from preds_seed*.npz and zeroshot_preds_seed*.npz"""
    out = {}
    for pattern, offset in (('preds_seed*.npz', 10), ('zeroshot_preds_seed*.npz', 19)):
        for f in sorted(glob.glob(str(ROOT / path / pattern))):
            d = np.load(f)
            seed = int(os.path.basename(f)[offset:-4])
            for b in {k.split('_', 1)[1] for k in d.files if k.startswith('tgt_')}:
                raw = d[f'raw_{b}'].astype(np.float64)
                # exp17 stores no MC-dropout arrays for the zero-shot bearings; mu and s are only
                # used for coverage, which this script does not report
                has_mc = f'mcmean_{b}' in d.files
                out.setdefault(seed, {})[b] = {
                    'raw': raw, 'y': d[f'tgt_{b}'].astype(np.float64),
                    'mu': d[f'mcmean_{b}'].astype(np.float64) if has_mc else raw,
                    's': np.clip(d[f'mcstd_{b}'].astype(np.float64), 1e-4, None) if has_mc else np.ones_like(raw)}
    return out


def per_bearing(variant, bearings, method):
    """seed mean of the MAE of `method` per bearing, over all seeds of the model"""
    return np.array([np.mean([np.mean(np.abs(variants_of(variant[s][b]['raw'])[method] - variant[s][b]['y']))
                              for s in sorted(variant)]) for b in bearings])


def compare(ref, other, bearings, method):
    a, b = per_bearing(ref, bearings, method), per_bearing(other, bearings, method)
    d = b - a
    return {'ref': float(a.mean()), 'other': float(b.mean()),
            'rel': float(100 * (b.mean() - a.mean()) / b.mean()),
            'k': int((d > 0).sum()), 'n': len(d), 'p_flip': float(signflip_p(d))}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None,
                    help='default: outputs/r1_analysis/xjtu_temporal.md (xjtu_temporal_retuned.md with --baselines retuned)')
    ap.add_argument('--baselines', choices=['submission', 'retuned'], default='submission',
                    help="'retuned': the XJTU-SY baselines re-selected and retrained under plan Amendment 1, E6 "
                         "(outputs/r1_xjtu_baselines/final, 20 seeds)")
    args = ap.parse_args()
    retuned = args.baselines == 'retuned'
    out_path = args.out or str(ROOT / 'outputs' / 'r1_analysis' /
                               ('xjtu_temporal_retuned.md' if retuned else 'xjtu_temporal.md'))

    models = dict(MODELS)
    if retuned:
        models.update({'1D CNN': 'outputs/r1_xjtu_baselines/final/cnn', 'BiLSTM': 'outputs/r1_xjtu_baselines/final/bilstm',
                       'CNN-LSTM': 'outputs/r1_xjtu_baselines/final/cnnlstm'})
    n_base = 20 if retuned else 5
    data = {k: load(v) for k, v in models.items()}
    expected = {ADOPTED: 20, SUBMITTED: 20, **{b: n_base for b in BASELINES}}
    for k, n in expected.items():
        missing = [b for s in data[k] for b in sum(SETS.values(), []) if b not in data[k][s]]
        if len(data[k]) != n or missing:
            raise SystemExit(f'{k}: expected {n} seeds with all 11 bearings, found {len(data[k])} seeds, '
                             f'{len(missing)} missing seed-bearing entries')

    report = {'models': models, 'baselines': args.baselines, 'seeds': {k: sorted(v) for k, v in data.items()},
              'sets': {}}
    base_text = ('baselines 20, re-selected and retrained under plan Amendment 1, E6' if retuned else
                 "baselines 5, the submission's XJTU-SY baselines re-evaluated from their checkpoints")
    lines = ['# XJTU-SY comparison (revision R1; generated by Experiments/r1_xjtu_stats.py)', '',
             f'Per bearing: mean over each model\'s seeds (ours 20, {base_text}). Δ = (comparator − reference) / comparator on '
             'bearing-averaged CES MAE; positive: reference better. p: exact two-sided sign-flip test over '
             'bearings; Holm over the three baselines. Pooled: window-pooled, mean ± SD over seeds. '
             'PAVA: offline reference. Descriptive: no Holm-adjusted p can fall below 0.094 (six bearings) '
             'or 0.19 (five bearings).']
    for set_name, bearings in SETS.items():
        fams = {}
        for ref in (ADOPTED, SUBMITTED):
            res = {b: {m: compare(data[ref], data[b], bearings, m) for m in ('ces', 'raw')} for b in BASELINES}
            for m in ('ces', 'raw'):
                adj = holm(np.array([res[b][m]['p_flip'] for b in BASELINES]))
                for b, pa in zip(BASELINES, adj):
                    res[b][m]['p_holm'] = float(pa)
            fams[ref] = res
        two = {m: compare(data[ADOPTED], data[SUBMITTED], bearings, m) for m in ('ces', 'raw')}
        rows = {}
        for name, d in data.items():
            per_seed = [seed_metrics(d[s], bearings, 1.0) for s in sorted(d)]
            rows[name] = {
                'bearing_avg': {m: float(per_bearing(d, bearings, m).mean()) for m in ('ces', 'raw')},
                'pooled': {k: (float(np.mean([r[k] for r in per_seed])), float(np.std([r[k] for r in per_seed])))
                           for k in ('raw_mae', 'ces_mae', 'raw_rmse', 'raw_r2', 'pava_mae', 'pava_r2')},
                'per_bearing_ces': dict(zip(bearings, per_bearing(d, bearings, 'ces').tolist()))}
        report['sets'][set_name] = {'rows': rows, 'baselines': fams, 'adopted_vs_submitted': two}

        lines += ['', f'## {set_name} ({len(bearings)} bearings; smallest attainable p = {2 / 2 ** len(bearings):.4f})', '',
                  '| Model | seeds | CES MAE, bearing avg | Δ vs adopted (better on, p, Holm) | '
                  'Δ vs submitted (better on, p, Holm) | raw MAE, bearing avg | pooled raw MAE | pooled CES MAE | '
                  'pooled raw R² | PAVA MAE (offline) | PAVA R² |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for name, r in rows.items():
            cells = []
            for ref in (ADOPTED, SUBMITTED):
                if name == ref:
                    cells.append('reference')
                elif name in BASELINES:
                    f = fams[ref][name]['ces']
                    cells.append(f"{f['rel']:+.1f} % ({f['k']}/{f['n']}, {f['p_flip']:.3f}, {f['p_holm']:.3f})")
                elif ref == ADOPTED:          # comparator: submitted architecture
                    t = two['ces']
                    cells.append(f"{t['rel']:+.1f} % ({t['k']}/{t['n']}, {t['p_flip']:.3f}, –)")
                else:                         # comparator: adopted variant; reference: submitted
                    t = two['ces']
                    cells.append(f"{100 * (t['ref'] - t['other']) / t['ref']:+.1f} % "
                                 f"({t['n'] - t['k']}/{t['n']}, {t['p_flip']:.3f}, –)")
            p = r['pooled']
            lines.append(f"| {name} | {len(data[name])} | {r['bearing_avg']['ces']:.4f} | {cells[0]} | {cells[1]} | "
                         f"{r['bearing_avg']['raw']:.4f} | {p['raw_mae'][0]:.4f} ± {p['raw_mae'][1]:.4f} | "
                         f"{p['ces_mae'][0]:.4f} ± {p['ces_mae'][1]:.4f} | {p['raw_r2'][0]:.3f} | "
                         f"{p['pava_mae'][0]:.4f} | {p['pava_r2'][0]:.3f} |")
        lines += ['', 'Raw MAE, bearing-averaged, Holm over the three baselines: ' + '; '.join(
            f"{'adopted' if ref == ADOPTED else 'submitted'} vs {b} {fams[ref][b]['raw']['rel']:+.1f} % "
            f"({fams[ref][b]['raw']['k']}/{len(bearings)}, p {fams[ref][b]['raw']['p_flip']:.3f}, "
            f"Holm {fams[ref][b]['raw']['p_holm']:.3f})" for ref in (ADOPTED, SUBMITTED) for b in BASELINES) + '.']
    report['provenance'] = provenance(('Experiments/r1_xjtu_stats.py', 'Experiments/r1_strict_stats.py',
                                       'Experiments/r1_analysis.py'))
    text = '\n'.join(lines) + '\n'
    print(text)
    Path(out_path).write_text(text, encoding='utf-8')
    Path(out_path).with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'written {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
