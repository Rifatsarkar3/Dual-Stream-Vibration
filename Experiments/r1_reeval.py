"""
r1_reeval.py — Revision R1: re-evaluate every saved checkpoint on the corrected
PRONOSTIA data (acc_*.csv only; Processed_PRONOSTIA_v2). Inference only, no training.

Training bearings are bitwise identical between the old and corrected processed
data, so the saved checkpoints remain valid; only the test evaluation changes.
Bearings evaluated: the original four test bearings, the six remaining
Condition 1-2 bearings of the official test set, and the three Condition-3
bearings (zero-shot). Metrics for any bearing subset are computed afterwards
from the saved per-bearing arrays (same key layout as exp16 preds_seed*.npz).

USAGE:
  python Experiments/r1_reeval.py --which all
  python Experiments/r1_reeval.py --which final20,abl20_static
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Experiments.exp16_ress import (CONFIG, evaluate_on_test, load_bearings,  # noqa: E402
                                    make_model)
from Experiments.exp19_baselines import MODELS, BaselineAdapter  # noqa: E402

EVAL_BEARINGS = [
    'test_01_Bearing1_3', 'test_02_Bearing1_4', 'test_03_Bearing1_5', 'test_04_Bearing2_3',
    'test_05_Bearing1_6', 'test_06_Bearing1_7', 'test_07_Bearing2_4', 'test_08_Bearing2_5',
    'test_09_Bearing2_6', 'test_10_Bearing2_7',
    'zs_01_Bearing3_1', 'zs_02_Bearing3_2', 'zs_03_Bearing3_3',
]

# variant -> (checkpoint dir, make_model kwargs or baseline name, MC passes used originally)
RUNS = {
    'final20': ('outputs/exp16_ress/final20', {}, 30),
    'abl20_static': ('outputs/exp16_ress/abl20_static', {'static_gate': True}, 10),
    'abl20_temporal': ('outputs/exp16_ress/abl20_temporal', {'stream': 'temporal'}, 10),
    'abl20_spectral': ('outputs/exp16_ress/abl20_spectral', {'stream': 'spectral'}, 10),
    'abl20_no_pgmc': ('outputs/exp16_ress/abl20_no_pgmc', {}, 10),
    'abl20_no_dann': ('outputs/exp16_ress/abl20_no_dann', {}, 10),
    'abl20_neither': ('outputs/exp16_ress/abl20_neither', {}, 10),
    'abl20_pgmc_soft': ('outputs/exp16_ress/abl20_pgmc_soft', {}, 10),
    'full_inbearing_es': ('outputs/exp16_ress/full', {}, 30),       # Sec. 4.2 demonstration
    'final21_lambda2': ('outputs/exp16_ress/final21', {}, 30),      # Sec. 6.1 diagnostic
    **{f'baseline_{m}': (f'outputs/exp19_baselines20/{m}', m, 10)
       for m in ('cnn', 'bilstm', 'cnnlstm', 'resnet', 'transformer')},
    # 4-fold selection changed these two baselines' budgets (cnnlstm 5 -> 10, transformer 10 -> 5)
    'baseline4f_cnnlstm': ('outputs/r1_baselines_4fold/cnnlstm', 'cnnlstm', 10),
    'baseline4f_transformer': ('outputs/r1_baselines_4fold/transformer', 'transformer', 10),
    # published model, Jiang et al. 2020 (ref [21]); two sensor channels
    'published_tsmccnn': ('outputs/r1_published/tsmccnn', 'tsmccnn', 10, 'hv'),
}
HV_RUNS = {name for name, spec in RUNS.items() if len(spec) > 3 and spec[3] == 'hv'}

SEED_RE = re.compile(r'model_seed(\d+)\.pth$')


def build(spec, device):
    if isinstance(spec, str):
        return BaselineAdapter(MODELS[spec](), needs_spectrogram=(spec == 'transformer')).to(device)
    return make_model(CONFIG, device, **spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', default='all')
    ap.add_argument('--data-dir', default=str(ROOT / 'Processed_PRONOSTIA_v2'))
    ap.add_argument('--data-dir-hv', default=str(ROOT / 'Processed_PRONOSTIA_v2hv'),
                    help='two-channel data (horizontal identical to --data-dir)')
    ap.add_argument('--out-root', default=str(ROOT / 'outputs' / 'r1_reeval'))
    args = ap.parse_args()

    names = list(RUNS) if args.which == 'all' else [s.strip() for s in args.which.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bearings_h = load_bearings(args.data_dir, EVAL_BEARINGS)
    bearings_hv = (load_bearings(args.data_dir_hv, EVAL_BEARINGS, channels='hv')
                   if any(n in HV_RUNS for n in names) else None)

    for name in names:
        ckpt_dir, spec, mc_passes = RUNS[name][:3]
        ckpts = sorted((ROOT / ckpt_dir).glob('model_seed*.pth'))
        if not ckpts:
            print(f'{name:20s} no checkpoints yet in {ckpt_dir}, skipping', flush=True)
            continue
        bearings = bearings_hv if name in HV_RUNS else bearings_h
        out_dir = Path(args.out_root) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(CONFIG, mc_dropout_passes=mc_passes)
        t0 = time.time()
        for ck in ckpts:
            seed = int(SEED_RE.search(ck.name).group(1))
            out_f = out_dir / f'preds_seed{seed}.npz'
            if out_f.exists():
                continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build(spec, device)
            model.load_state_dict(torch.load(ck, map_location=device))
            res = evaluate_on_test(model, bearings, cfg, device)
            a = res['_arrays']
            np.savez_compressed(
                out_f,
                **{f'raw_{n}': v for n, v in a['raw_p'].items()},
                **{f'tgt_{n}': v for n, v in a['tgt'].items()},
                **{f'alpha_{n}': v for n, v in a['alphas'].items()},
                **{f'mcmean_{n}': v for n, v in a['mc_mean'].items()},
                **{f'mcstd_{n}': v for n, v in a['mc_std'].items()})
        meta = {'source_checkpoints': ckpt_dir, 'n_checkpoints': len(ckpts),
                'mc_dropout_passes': mc_passes,
                'data_dir': args.data_dir_hv if name in HV_RUNS else args.data_dir,
                'bearings': EVAL_BEARINGS, 'date': time.strftime('%Y-%m-%d %H:%M')}
        (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
        print(f'{name:20s} {len(ckpts):2d} checkpoints  {time.time() - t0:6.1f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
