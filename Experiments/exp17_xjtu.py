"""
exp17_xjtu.py — DSAFLite+PGMC on XJTU-SY (leak-free protocol)
==============================================================

Reuses the exp16 pipeline unchanged on the XJTU-SY dataset:
  * train: Bearing1_1, Bearing1_2 (Cond 1, domain 0) + Bearing2_1, Bearing2_2
    (Cond 2, domain 1); validation = every 10th window of each training bearing
  * in-domain test: Bearing1_3..1_5, Bearing2_3..2_5 (6 held-out bearings)
  * zero-shot: Condition 3 bearings (Bearing3_1..3_5) — evaluated with the
    best-validation checkpoint, never seen during training. Real run-to-failure
    cross-condition transfer (replaces the weaker CWRU pseudo-RUL evidence).

RUN:
  .venv\\Scripts\\python.exe Experiments/exp17_xjtu.py --smoke
  .venv\\Scripts\\python.exe Experiments/exp17_xjtu.py --all-seeds
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from Experiments.exp16_ress import (  # noqa: E402
    CONFIG as BASE_CONFIG, train_one_seed, evaluate_on_test,
    load_bearings, aggregate)
from Experiments.r1_provenance import provenance  # noqa: E402

# Wrappers that change the model (e.g. Experiments/r1_temporal_xjtu.py) add their files here.
EXTRA_CODE = ()
PROVENANCE_EXTRA = {}

XJTU_CONFIG = dict(BASE_CONFIG)
XJTU_CONFIG.update({
    'data_dir': str(Path(__file__).parent.parent / 'Processed_XJTU'),
    'train_bearings': {
        'train_01_Bearing1_1': 0,
        'train_02_Bearing1_2': 0,
        'train_03_Bearing2_1': 1,
        'train_04_Bearing2_2': 1,
    },
    'test_bearings': [
        'test_01_Bearing1_3', 'test_02_Bearing1_4', 'test_03_Bearing1_5',
        'test_04_Bearing2_3', 'test_05_Bearing2_4', 'test_06_Bearing2_5',
    ],
    'zeroshot_bearings': [
        'zeroshot_01_Bearing3_1', 'zeroshot_02_Bearing3_2',
        'zeroshot_03_Bearing3_3', 'zeroshot_04_Bearing3_4',
        'zeroshot_05_Bearing3_5',
    ],
    'output_root': 'outputs/exp17_xjtu',
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--all-seeds', action='store_true')
    ap.add_argument('--variant-name', type=str, default=None)
    ap.add_argument('--pgmc-mode', choices=['aux-seq', 'soft', 'chunk'], default=None)
    ap.add_argument('--val-mode', choices=['window', 'bearing', 'none'], default=None)
    ap.add_argument('--val-bearings', type=str, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--scheduler-tmax', type=int, default=None)
    ap.add_argument('--proportional-warmup', action='store_true')
    ap.add_argument('--mc-passes', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--skip-zeroshot', action='store_true',
                    help='skip Condition-3 zero-shot eval (selection runs)')
    ap.add_argument('--lambda-pgmc', type=float, default=None)
    ap.add_argument('--seeds', type=str, default=None)
    ap.add_argument('--output-root', type=str, default=None)
    ap.add_argument('--skip-test', action='store_true',
                    help='selection runs: in-domain test bearings are neither loaded nor evaluated')
    ap.add_argument('--deterministic', action='store_true',
                    help='deterministic CUDA kernels, so a seed reproduces a run exactly')
    ap.add_argument('--save-val-preds', action='store_true',
                    help='also save final-epoch MC-dropout predictions on the validation bearing '
                         '(valpreds_seed*.npz) and the final-epoch weights, as exp16 does')
    args = ap.parse_args()

    if args.deterministic:
        # Must precede the first cuBLAS call; the CUDA context is created lazily.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config = dict(XJTU_CONFIG)
    if args.output_root:
        config['output_root'] = args.output_root
    if args.pgmc_mode:
        config['pgmc_mode'] = args.pgmc_mode
    if args.val_mode:
        config['val_mode'] = args.val_mode
    if args.val_bearings:
        config['val_bearings'] = [s.strip() for s in args.val_bearings.split(',')]
    if args.epochs:
        config['epochs'] = args.epochs
    if args.scheduler_tmax:
        config['scheduler_tmax'] = args.scheduler_tmax
    if args.proportional_warmup:
        config['proportional_warmup'] = True
    if args.mc_passes:
        config['mc_dropout_passes'] = args.mc_passes
    if args.patience:
        config['patience'] = args.patience
    if args.lambda_pgmc is not None:
        config['lambda_pgmc_max'] = args.lambda_pgmc
    if args.seeds:
        config['seeds'] = [int(s) for s in args.seeds.split(',')]
    if args.skip_test:
        config['skip_test'] = True
        config['test_bearings'] = []
    if args.save_val_preds:
        config['save_val_preds'] = True

    variant = args.variant_name or ('smoke' if args.smoke else 'full')
    out_dir = Path(config['output_root']) / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds, epochs_override = [42], 10
    elif args.all_seeds:
        seeds, epochs_override = config['seeds'], None
    else:
        seeds, epochs_override = [args.seed if args.seed is not None else 42], None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zeroshot_bearings = (None if args.skip_zeroshot else
                         load_bearings(config['data_dir'], config['zeroshot_bearings']))

    print(f"[exp17] XJTU-SY variant={variant} seeds={seeds}", flush=True)

    per_seed, per_seed_zs = [], []
    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        result, model, history = train_one_seed(
            seed, config, use_dann=True, use_pgmc=True, static_gate=False,
            epochs_override=epochs_override)

        arrays = result.pop('_arrays', None)
        if arrays is not None:
            np.savez_compressed(
                out_dir / f'preds_seed{seed}.npz',
                **{f'raw_{n}': v for n, v in arrays['raw_p'].items()},
                **{f'tgt_{n}': v for n, v in arrays['tgt'].items()},
                **{f'alpha_{n}': v for n, v in arrays['alphas'].items()},
                **{f'mcmean_{n}': v for n, v in arrays['mc_mean'].items()},
                **{f'mcstd_{n}': v for n, v in arrays['mc_std'].items()})
        val_arrays = result.pop('_val_arrays', None)
        final_state = result.pop('_final_state', None)
        if val_arrays is not None:
            np.savez_compressed(out_dir / f'valpreds_seed{seed}.npz', **val_arrays)
            torch.save(final_state, out_dir / f'model_final_seed{seed}.pth')
        torch.save(model.state_dict(), out_dir / f'model_seed{seed}.pth')

        if 'raw' in result:
            print(f"  TEST raw : MAE={result['raw']['mae']:.4f} R2={result['raw']['r2']:.4f} "
                  f"Mono={result['raw']['mono']:.1f}%")
            print(f"  TEST pava: MAE={result['pava']['mae']:.4f} R2={result['pava']['r2']:.4f}")
        elif result['val_history']:
            print(f"  final-epoch val MAE={result['val_history'][-1]:.4f} "
                  f"(test bearings not loaded)", flush=True)
        per_seed.append(result)

        # Zero-shot Condition 3 evaluation (never seen in training)
        if zeroshot_bearings is not None:
            zs = evaluate_on_test(model, zeroshot_bearings, config, device)
            zs_arrays = zs.pop('_arrays')
            np.savez_compressed(
                out_dir / f'zeroshot_preds_seed{seed}.npz',
                **{f'raw_{n}': v for n, v in zs_arrays['raw_p'].items()},
                **{f'tgt_{n}': v for n, v in zs_arrays['tgt'].items()})
            zs['seed'] = seed
            print(f"  ZS-C3 raw : MAE={zs['raw']['mae']:.4f} R2={zs['raw']['r2']:.4f}")
            print(f"  ZS-C3 pava: MAE={zs['pava']['mae']:.4f} R2={zs['pava']['r2']:.4f}",
                  flush=True)
            per_seed_zs.append(zs)

    final = {
        'experiment': 'exp17_xjtu',
        'variant': variant,
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'config': {k: v for k, v in config.items() if k != 'train_bearings'},
        'per_seed': per_seed,
        'per_seed_zeroshot': per_seed_zs,
        'aggregated': aggregate(per_seed) if per_seed and not config.get('skip_test') else {},
        'aggregated_zeroshot': aggregate(per_seed_zs) if per_seed_zs else {},
        'provenance': {**provenance(extra_code=('Experiments/exp17_xjtu.py',) + tuple(EXTRA_CODE)),
                       **PROVENANCE_EXTRA},
    }
    with open(out_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2)

    if len(per_seed) > 1 and final['aggregated']:
        a, az = final['aggregated'], final['aggregated_zeroshot']
        print(f"\n=== XJTU AGGREGATE ({len(per_seed)} seeds) ===")
        print(f"  in-domain raw : MAE={a['raw_mae_mean']:.4f}±{a['raw_mae_std']:.4f} "
              f"R2={a['raw_r2_mean']:.4f}")
        print(f"  in-domain pava: MAE={a['pava_mae_mean']:.4f}±{a['pava_mae_std']:.4f} "
              f"R2={a['pava_r2_mean']:.4f}")
        if az:
            print(f"  zero-shot C3 raw : MAE={az['raw_mae_mean']:.4f}±{az['raw_mae_std']:.4f} "
                  f"R2={az['raw_r2_mean']:.4f}")
    print(f"\n[exp17] results saved to {out_dir / 'results.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
