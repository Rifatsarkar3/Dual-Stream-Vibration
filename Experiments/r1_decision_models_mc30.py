"""
r1_decision_models_mc30.py — Revision R1, analysis plan Amendment 4, A4: Amendment 2
(r1_decision_models.py, unchanged) repeated with the six dropout baselines' test predictions
re-scored at 30 MC-dropout passes (T4, outputs/r1_baseline_mc30/); our two versions keep their
stored 30-pass predictions. Then the sensitivity run (the submitted architecture re-scored by the
same seeded procedure, Amendment 3 T3) and the check that c*, the residual quantiles, P1 and P3,
which use validation or deterministic predictions only, reproduce Amendment 2.

USAGE:
  python Experiments/r1_decision_models_mc30.py --show-paths   # print the replaced paths and stop
  python Experiments/r1_decision_models_mc30.py                # A4, sensitivity, check
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import r1_decision_models as dm  # noqa: E402  (Amendment 2; used unchanged)
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import baseline_paths as original_paths  # noqa: E402

MC30 = {'1D CNN': 'cnn', 'BiLSTM': 'bilstm', 'CNN-LSTM': 'cnnlstm', '1D ResNet': 'resnet',
        'Transformer': 'transformer', 'MSCNN [7]': 'mscnn'}
SENS_SUBMITTED = 'outputs/r1_ablation_mc30/final20'
OUT = ROOT / 'outputs' / 'r1_analysis'
TOL = 1e-5


def mc30_paths():
    paths = dict(original_paths())
    for label, short in MC30.items():
        d = ROOT / 'outputs' / 'r1_baseline_mc30' / short
        meta = json.loads((d / 'meta.json').read_text(encoding='utf-8'))
        if meta['mc_dropout_passes'] != 30 or meta['n_checkpoints'] != 20 or meta['raw_check_max_abs_diff'] > TOL:
            raise SystemExit(f'{label}: T4 output {d} incomplete or failed its check ({meta})')
        paths[label] = f'outputs/r1_baseline_mc30/{short}'
    return paths


def run(out_name):
    sys.argv = ['r1_decision_models.py', '--out', str(OUT / out_name)]
    dm.main()
    return json.loads((OUT / out_name).with_suffix('.json').read_text(encoding='utf-8'))


def rel_diff(a, b):
    """largest relative difference between two JSON trees of numbers with the same structure"""
    if isinstance(a, dict):
        if set(a) != set(b):
            return float('inf')
        return max([rel_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, list):
        if len(a) != len(b):
            return float('inf')
        return max([rel_diff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return 0.0
        return abs(a - b) / max(abs(a), abs(b))
    return 0.0 if a == b else float('inf')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--show-paths', action='store_true')
    args = ap.parse_args()
    paths = mc30_paths()
    if args.show_paths:
        for k, v in paths.items():
            print(f'{k:16s} {v}')
        return 0
    dm.baseline_paths = mc30_paths          # r1_decision_models.real_models() looks the name up at call time
    new = run('decision_models_mc30.md')
    original_final = dm.OURS[dm.SUBMITTED]['final']
    dm.OURS[dm.SUBMITTED]['final'] = SENS_SUBMITTED
    try:
        sens = run('decision_models_mc30_sens.md')
    finally:
        dm.OURS[dm.SUBMITTED]['final'] = original_final

    old = json.loads((OUT / 'decision_models.json').read_text(encoding='utf-8'))
    parts = {
        'calibration (c*, residual quantiles)': (old['calibration'], new['calibration']),
        'P1 (cost-matched trigger, every set)': ({s: [old['sets'][s]['P1'], old['sets'][s]['losses_cmq']] for s in old['sets']},
                                                 {s: [new['sets'][s]['P1'], new['sets'][s]['losses_cmq']] for s in new['sets']}),
        'P3 (framework vs standard practice)': (old['P3'], new['P3']),
        'CES MAE ranking and Kendall tau': ([old['ces_mae_bavg_all10'], old['kendall_ces_mae_vs_loss']],
                                            [new['ces_mae_bavg_all10'], new['kendall_ces_mae_vs_loss']]),
    }
    check = {name: rel_diff(a, b) for name, (a, b) in parts.items()}
    ok = all(v <= 1e-3 for v in check.values())
    lines = ['# Amendment 4 check: results that use no MC-dropout predictions', '',
             '| Part | largest relative difference to Amendment 2 | status |', '|---|---|---|']
    lines += [f"| {k} | {v:.3g} | {'identical' if v == 0 else ('within 0.1 %' if v <= 1e-3 else 'DIFFERENT')} |"
              for k, v in check.items()]
    lines += ['', f"Overall: {'PASS' if ok else 'FAIL'}.", '',
              'Paths of the baselines\' final predictions in A4:', ''] + [f'- {k}: `{v}`' for k, v in paths.items()]
    text = '\n'.join(lines) + '\n'
    (OUT / 'decision_models_mc30_check.md').write_text(text, encoding='utf-8')
    (OUT / 'decision_models_mc30_check.json').write_text(json.dumps(
        {'check': check, 'pass': ok, 'paths': paths, 'sensitivity_submitted_final': SENS_SUBMITTED,
         'provenance': provenance(('Experiments/r1_decision_models_mc30.py', 'Experiments/r1_decision_models.py',
                                   'Experiments/r1_baseline_mc30.py'))}, indent=1), encoding='utf-8')
    print(text)
    print(f"sensitivity written: {OUT / 'decision_models_mc30_sens.md'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
