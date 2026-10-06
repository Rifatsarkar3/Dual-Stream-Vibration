"""
r1_components_noise.py — Revision R1, analysis plan Amendment 10, fix G: test-time noise.

For each final seed of the rebuilt network R and of its static-fusion ablation, white Gaussian
noise is added to the test windows at 20, 10, 5, 0 and -5 dB SNR with the definition and random
streams of Experiments/exp18_noise_gate.py (per-window SNR; generator seeded 1000 * seed + SNR +
100, so both models see the same noise). Per bearing and seed, the raw and CES MAE and the mean
gate weight are recorded, for clean windows too. Evaluation only; no training.

Bearings: PRONOSTIA all10 (Amendment 10, primary family); XJTU-SY the six test and five
Condition-3 bearings (replication family).

USAGE:
  python Experiments/r1_components_noise.py --dataset pronostia
  python Experiments/r1_components_noise.py --dataset xjtu
Output: <final directory>/noise.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import Experiments.r1_components as rc  # noqa: E402
import Experiments.exp16_ress as e16  # noqa: E402
from Experiments.exp18_noise_gate import add_awgn  # noqa: E402

SNR_LEVELS_DB = [None, 20, 10, 5, 0, -5]
SETS = {
    'pronostia': {'data_dir': 'Processed_PRONOSTIA_v2', 'final': 'outputs/r1_components_final',
                  'bearings': [f'test_{i:02d}_Bearing{b}' for i, b in enumerate(
                      ['1_3', '1_4', '1_5', '2_3', '1_6', '1_7', '2_4', '2_5', '2_6', '2_7'], 1)]},
    'xjtu': {'data_dir': 'Processed_XJTU', 'final': 'outputs/r1_components_xjtu_final',
             'bearings': None},  # exp17's test and zero-shot bearings
}
MODELS = {'full': False, 'static': True}   # run directory -> static gate


def evaluate(model, bearings, device, snr_db, rng, config):
    out = {}
    model.eval()
    with torch.no_grad():
        for b in bearings:
            vib = b.vib if snr_db is None else add_awgn(b.vib, snr_db, rng)
            ps, als = [], []
            for s in range(0, b.n, 256):
                p, a, *_ = model(torch.from_numpy(vib[s:s + 256]).unsqueeze(1).to(device))
                ps.append(p.cpu().numpy())
                als.append(a.cpu().numpy())
            p = np.concatenate(ps)
            ces = e16.ConstrainedExponentialSmoothing(
                alpha=config['ces_alpha'], epsilon=config['ces_epsilon']).process(p)
            out[b.name] = {'raw_mae': float(np.mean(np.abs(p - b.rul))),
                           'ces_mae': float(np.mean(np.abs(np.asarray(ces) - b.rul))),
                           'alpha_mean': float(np.concatenate(als).mean())}
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=sorted(SETS), required=True)
    args = ap.parse_args()
    spec = SETS[args.dataset]
    rc.configure('band', 'awgn', 'gap', args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = dict(e16.CONFIG)
    names = spec['bearings']
    if names is None:
        from Experiments.exp17_xjtu import XJTU_CONFIG
        names = XJTU_CONFIG['test_bearings'] + XJTU_CONFIG['zeroshot_bearings']
    bearings = e16.load_bearings(str(ROOT / spec['data_dir']), names)
    final = ROOT / spec['final']

    results = {}
    for run, static in MODELS.items():
        ckpts = sorted((final / run).glob('model_seed*.pth'),
                       key=lambda p: int(p.stem.replace('model_seed', '')))
        if len(ckpts) != 20:
            raise SystemExit(f'{final / run}: expected 20 final checkpoints, found {len(ckpts)}')
        results[run] = {}
        for ck in ckpts:
            seed = int(ck.stem.replace('model_seed', ''))
            model = e16.make_model(config, device, static_gate=static)
            model.load_state_dict(torch.load(ck, map_location=device))
            results[run][seed] = {}
            for snr in SNR_LEVELS_DB:
                rng = np.random.default_rng(seed * 1000 + (0 if snr is None else snr + 100))
                key = 'clean' if snr is None else f'{snr}dB'
                results[run][seed][key] = evaluate(model, bearings, device, snr, rng, config)
            clean = np.mean([v['ces_mae'] for v in results[run][seed]['clean'].values()])
            print(f'[noise] {run} seed {seed}: clean CES MAE (bearing mean) {clean:.4f}', flush=True)

    out = final / 'noise.json'
    out.write_text(json.dumps({
        'analysis': 'R1 analysis plan, Amendment 10, fix G: test-time noise',
        'dataset': args.dataset, 'bearings': names,
        'snr_levels_db': ['clean' if s is None else s for s in SNR_LEVELS_DB],
        'results': results, 'date': time.strftime('%Y-%m-%d %H:%M'),
        'provenance': {**e16.provenance(),
                       'extra_code': {'Experiments/r1_components_noise.py': rc._sha256(Path(__file__))}},
    }, indent=1), encoding='utf-8')
    print(f'[noise] saved {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
