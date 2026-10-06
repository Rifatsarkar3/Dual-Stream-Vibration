"""
r1_baseline_final.py — Revision R1, analysis plan Section 5: a baseline whose budget
changed under the 5-seed selection (outputs/r1_baseline_sel5/<model>.json) is retrained
with the 20 final seeds at the new budget, with deterministic CUDA kernels, and evaluated
on the 13 evaluation bearings with 10 MC-dropout passes (as for every baseline).
Writes outputs/r1_baselines_final5/<model>/ preds_seed*.npz (r1_analysis format),
model_seed*.pth and results.json. Resumable per seed. The budget is read from the
selection file, never typed by hand.

USAGE:
  python Experiments/r1_baseline_final.py --model bilstm
"""

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Experiments.exp16_ress import CONFIG  # noqa: E402
from Experiments.exp19_baselines import train_baseline  # noqa: E402
from Experiments.r1_provenance import provenance  # noqa: E402

FINAL_SEEDS = [42, 123, 777, 999, 7, 8, 21, 34, 55, 89, 144, 233, 377, 610, 987,
               1597, 2584, 4181, 6765, 10946]
EVAL_BEARINGS = ['test_01_Bearing1_3', 'test_02_Bearing1_4', 'test_03_Bearing1_5',
                 'test_04_Bearing2_3', 'test_05_Bearing1_6', 'test_06_Bearing1_7',
                 'test_07_Bearing2_4', 'test_08_Bearing2_5', 'test_09_Bearing2_6',
                 'test_10_Bearing2_7', 'zs_01_Bearing3_1', 'zs_02_Bearing3_2',
                 'zs_03_Bearing3_3']


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--smoke', action='store_true',
                    help='pipeline check: seed 42, 1 epoch, written to a separate smoke folder')
    args = ap.parse_args()
    sel = json.loads((ROOT / 'outputs' / 'r1_baseline_sel5' / f'{args.model}.json').read_text())
    if not sel.get('budget_changed'):
        raise SystemExit(f"{args.model}: budget unchanged ({sel.get('K_5seed_4fold')}); "
                         'the plan keeps the existing final runs')
    k = 1 if args.smoke else int(sel['K_5seed_4fold'])
    out_dir = ROOT / 'outputs' / ('r1_baselines_final5_smoke' if args.smoke else 'r1_baselines_final5') / args.model
    if args.smoke:
        FINAL_SEEDS[:] = [42]
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = dict(CONFIG)
    cfg.update({'data_dir': str(ROOT / ('Processed_PRONOSTIA_v2hv' if args.model == 'tsmccnn'
                                        else 'Processed_PRONOSTIA_v2')),
                'val_mode': 'none', 'epochs': k, 'mc_dropout_passes': 10,
                'test_bearings': EVAL_BEARINGS})
    summary_f = out_dir / 'per_seed.json'
    per_seed = json.loads(summary_f.read_text()) if summary_f.exists() else []
    for seed in FINAL_SEEDS:
        if (out_dir / f'preds_seed{seed}.npz').exists():
            continue
        t0 = time.time()
        res, model = train_baseline(args.model, seed, cfg, device, verbose=False)
        arrays = res.pop('_arrays')
        np.savez_compressed(
            out_dir / f'preds_seed{seed}.npz',
            **{f'raw_{n}': v for n, v in arrays['raw_p'].items()},
            **{f'tgt_{n}': v for n, v in arrays['tgt'].items()},
            **{f'alpha_{n}': v for n, v in arrays['alphas'].items()},
            **{f'mcmean_{n}': v for n, v in arrays['mc_mean'].items()},
            **{f'mcstd_{n}': v for n, v in arrays['mc_std'].items()})
        torch.save(model.state_dict(), out_dir / f'model_seed{seed}.pth')
        per_seed.append({'seed': seed, 'raw_mae_13': res['raw']['mae'],
                         'ces_mae_13': res['ces']['mae'], 'minutes': round((time.time() - t0) / 60, 1)})
        summary_f.write_text(json.dumps(per_seed, indent=1))
        print(f"{args.model} seed {seed}: pooled raw MAE (13 bearings) = {res['raw']['mae']:.4f}", flush=True)
    (out_dir / 'results.json').write_text(json.dumps(
        {'model': args.model, 'K': k, 'selection': 'outputs/r1_baseline_sel5/' + args.model + '.json',
         'per_seed': per_seed, 'provenance': provenance(('Experiments/r1_baseline_final.py',))},
        indent=1), encoding='utf-8')
    print(f'{args.model}: final runs complete (K = {k})', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
