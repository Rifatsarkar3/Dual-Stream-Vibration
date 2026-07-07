"""
exp18_noise_gate.py — Noise-Robustness Study of the Adaptive Fusion Gate
=========================================================================

PURPOSE (RESS revision):
  The PRONOSTIA ablation shows the adaptive gate is MAE-neutral on clean test
  data (a transparently reported null result). The gate's design motivation is
  time-varying modality reliability, which clean benchmark data does not probe.
  This experiment injects additive white Gaussian noise at controlled SNR into
  the test-bearing vibration windows and measures, for every trained seed:

    1. MAE / R2 degradation vs SNR for the FULL model (adaptive gate)
       and the STATIC-GATE ablation checkpoints (alpha = 0.5).
    2. The gate response: mean alpha vs SNR. If the gate is functional it
       should shift stream weighting as the temporal stream's SNR collapses.

  Evaluation only — no training. Uses checkpoints written by exp16_ress.py:
    outputs/exp16_ress/<full-variant>/model_seed{S}.pth
    outputs/exp16_ress/<static-variant>/model_seed{S}.pth

RUN:
  .venv\\Scripts\\python.exe Experiments/exp18_noise_gate.py ^
      --full-dir outputs/exp16_ress/final ^
      --static-dir outputs/exp16_ress/abl_static_gate
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from Experiments.exp16_ress import (  # noqa: E402
    CONFIG, load_bearings, make_model, regression_metrics, pooled_eval)

SNR_LEVELS_DB = [None, 20, 10, 5, 0, -5]   # None = clean


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise at the target SNR per window."""
    p_sig = np.mean(x ** 2, axis=-1, keepdims=True)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return x + rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(p_noise)


def eval_checkpoint(model, bearings, device, snr_db, rng):
    preds, tgts, alphas = {}, {}, []
    model.eval()
    with torch.no_grad():
        for b in bearings:
            vib = b.vib if snr_db is None else add_awgn(b.vib, snr_db, rng)
            ps, als = [], []
            for s in range(0, b.n, 256):
                x = torch.from_numpy(vib[s:s + 256]).unsqueeze(1).to(device)
                p, a, *_ = model(x)
                ps.append(p.cpu().numpy())
                als.append(a.cpu().numpy())
            preds[b.name] = np.concatenate(ps)
            tgts[b.name] = b.rul
            alphas.append(np.concatenate(als))
    m = pooled_eval(preds, tgts)
    al = np.concatenate(alphas)
    m['alpha_mean'] = float(al.mean())
    m['alpha_std'] = float(al.std())
    return m


def run_variant(ckpt_dir: Path, static: bool, bearings, config, device) -> dict:
    out = {}
    ckpts = sorted(ckpt_dir.glob('model_seed*.pth'))
    if not ckpts:
        raise FileNotFoundError(f'no checkpoints in {ckpt_dir}')
    for snr in SNR_LEVELS_DB:
        per_seed = []
        for ck in ckpts:
            seed = int(ck.stem.replace('model_seed', ''))
            rng = np.random.default_rng(seed * 1000 + (0 if snr is None else snr + 100))
            model = make_model(config, device, static_gate=static)
            model.load_state_dict(torch.load(ck, map_location=device))
            per_seed.append(eval_checkpoint(model, bearings, device, snr, rng))
        key = 'clean' if snr is None else f'{snr}dB'
        out[key] = {
            'mae_mean': float(np.mean([m['mae'] for m in per_seed])),
            'mae_std': float(np.std([m['mae'] for m in per_seed])),
            'r2_mean': float(np.mean([m['r2'] for m in per_seed])),
            'alpha_mean': float(np.mean([m['alpha_mean'] for m in per_seed])),
            'alpha_std': float(np.mean([m['alpha_std'] for m in per_seed])),
            'n_seeds': len(per_seed),
        }
        print(f"  {key:>6}: MAE={out[key]['mae_mean']:.4f}±{out[key]['mae_std']:.4f} "
              f"R2={out[key]['r2_mean']:.4f} alpha={out[key]['alpha_mean']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full-dir', required=True)
    ap.add_argument('--static-dir', required=True)
    ap.add_argument('--out', default='outputs/exp18_noise_gate/results.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bearings = load_bearings(CONFIG['data_dir'], CONFIG['test_bearings'])

    print('[exp18] FULL model (adaptive gate):')
    full = run_variant(Path(args.full_dir), False, bearings, CONFIG, device)
    print('[exp18] STATIC gate (alpha=0.5):')
    static = run_variant(Path(args.static_dir), True, bearings, CONFIG, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'exp18_noise_gate',
                   'date': time.strftime('%Y-%m-%d %H:%M'),
                   'snr_levels_db': [s if s is not None else 'clean' for s in SNR_LEVELS_DB],
                   'adaptive_gate': full, 'static_gate': static}, f, indent=2)
    print(f'[exp18] saved {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
