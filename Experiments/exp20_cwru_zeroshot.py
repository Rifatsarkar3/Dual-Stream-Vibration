"""
exp20_cwru_zeroshot.py — TRUE zero-shot CWRU evaluation
========================================================

The previous exp15_cwru.py TRAINED on CWRU with a random 75/25 window split
(same-recording windows in train and test), while the manuscript claimed
zero-shot transfer from PRONOSTIA. This script performs the actual claimed
experiment: PRONOSTIA-trained checkpoints (exp16 final4) evaluated on CWRU
with NO training, NO fine-tuning, NO threshold fitting.

Caveats reported alongside results: CWRU is sampled at 12 kHz (vs 25.6 kHz),
so a 2560-sample window spans 213 ms (vs 100 ms), and CWRU 'RUL' labels are a
pseudo-RUL constructed from seeded-fault damage sequences, not run-to-failure.

RUN:
  .venv\\Scripts\\python.exe Experiments/exp20_cwru_zeroshot.py ^
      --ckpt-dir outputs/exp16_ress/final4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from Experiments.exp16_ress import CONFIG, make_model, regression_metrics  # noqa: E402

CWRU_DIR = Path(__file__).parent.parent / 'data' / 'CWRU'
CONDITIONS = ['12k_Drive_End', '48k_Drive_End', 'Fan_End', 'Normal_Baseline']


def load_cwru():
    per_cond = {}
    for cond in CONDITIONS:
        cdir = CWRU_DIR / cond
        if not cdir.exists():
            continue
        vibs, ruls = [], []
        for f in sorted(cdir.glob('*.npz')):
            data = np.load(f, allow_pickle=True)
            vibs.append(data['vibration'].astype(np.float32))
            ruls.append(data['rul'].astype(np.float32))
        if vibs:
            per_cond[cond] = (np.concatenate(vibs), np.concatenate(ruls))
    return per_cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='outputs/exp16_ress/final4')
    ap.add_argument('--out', default='outputs/exp20_cwru_zeroshot/results.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    per_cond = load_cwru()
    print(f'CWRU conditions loaded: { {k: len(v[1]) for k, v in per_cond.items()} }')

    ckpts = sorted(Path(args.ckpt_dir).glob('model_seed*.pth'))
    per_seed = []
    for ck in ckpts:
        model = make_model(CONFIG, device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        all_p, all_t, cond_metrics = [], [], {}
        with torch.no_grad():
            for cond, (vib, rul) in per_cond.items():
                ps = []
                for s in range(0, len(vib), 256):
                    x = torch.from_numpy(vib[s:s + 256]).unsqueeze(1).to(device)
                    p, *_ = model(x)
                    ps.append(p.cpu().numpy())
                ps = np.concatenate(ps)
                cond_metrics[cond] = regression_metrics(ps, rul)
                all_p.append(ps)
                all_t.append(rul)
        pooled = regression_metrics(np.concatenate(all_p), np.concatenate(all_t))
        seed = int(ck.stem.replace('model_seed', ''))
        print(f'seed {seed}: pooled MAE={pooled["mae"]:.4f} R2={pooled["r2"]:.4f}')
        per_seed.append({'seed': seed, 'pooled': pooled, 'per_condition': cond_metrics})

    agg = {
        'mae_mean': float(np.mean([r['pooled']['mae'] for r in per_seed])),
        'mae_std': float(np.std([r['pooled']['mae'] for r in per_seed])),
        'r2_mean': float(np.mean([r['pooled']['r2'] for r in per_seed])),
        'r2_std': float(np.std([r['pooled']['r2'] for r in per_seed])),
    }
    print(f"\nTRUE zero-shot CWRU: MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f} "
          f"R2={agg['r2_mean']:.4f}±{agg['r2_std']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'exp20_cwru_zeroshot',
                   'date': time.strftime('%Y-%m-%d %H:%M'),
                   'ckpt_dir': str(args.ckpt_dir),
                   'note': ('True zero-shot: PRONOSTIA-trained checkpoints, no CWRU '
                            'training. Previous exp15 trained on CWRU (75/25 random '
                            'window split) and is NOT zero-shot.'),
                   'per_seed': per_seed, 'aggregated': agg}, f, indent=2)
    print(f'saved {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
