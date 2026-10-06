"""
r1_tsmc_randsplit.py: revision R1, analysis plan Amendment 8. TSMC-CNN (Jiang, Lee and Zeng 2020, ref [21])
evaluated under the random split of windows used in its publication, reproducibly and with saved outputs.

  R8-a  [21]'s data: Bearing2_1, 2_2, 2_3, 2_4 (4,414 windows), pooled, 20 random 70/30 splits
  R8-b  the protocol's pool: the 14 Condition 1-2 bearings (22,303 windows), pooled, 20 random 70/30 splits

Model and training exactly as in TSMC-CNN's protocol runs (exp19: BaselineAdapter(TSMCCNN()), input batch
normalization, horizontal and vertical channels, squared-error loss, Adam lr 1e-3, no weight decay, batch 32,
cosine annealing over the budget to 1e-6, gradient-norm clipping 1.0), budget 40 epochs, deterministic kernels,
final-epoch model, deterministic predictions. Split i uses numpy.random.default_rng(i); model seed = the i-th
final seed of the protocol.

USAGE:
  python Experiments/r1_tsmc_randsplit.py --smoke            # split 0, 2 epochs, both designs, scratch folder
  python Experiments/r1_tsmc_randsplit.py --design a|b       # the 20 runs of one design
  python Experiments/r1_tsmc_randsplit.py --summary          # outputs/r1_analysis/tsmc_randsplit.md and .json
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
import torch.nn as nn  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Experiments.exp16_ress import load_bearings  # noqa: E402
from Experiments.exp19_baselines import BaselineAdapter, MODEL_TRAINING  # noqa: E402
from Experiments.r1_provenance import provenance  # noqa: E402
from src.published_baselines import TSMCCNN  # noqa: E402

DATA = ROOT / 'Processed_PRONOSTIA_v2hv'
OUT = ROOT / 'outputs' / 'r1_tsmc_randsplit'
AN = ROOT / 'outputs' / 'r1_analysis'
FINAL_SEEDS = [42, 123, 777, 999, 7, 8, 21, 34, 55, 89, 144, 233, 377, 610, 987,
               1597, 2584, 4181, 6765, 10946]
EPOCHS, BATCH, TRAIN_FRACTION = 40, 32, 0.7
DESIGNS = {
    'a': ['train_03_Bearing2_1', 'train_04_Bearing2_2', 'test_04_Bearing2_3', 'test_07_Bearing2_4'],
    'b': ['train_01_Bearing1_1', 'train_02_Bearing1_2', 'train_03_Bearing2_1', 'train_04_Bearing2_2',
          'test_01_Bearing1_3', 'test_02_Bearing1_4', 'test_03_Bearing1_5', 'test_04_Bearing2_3',
          'test_05_Bearing1_6', 'test_06_Bearing1_7', 'test_07_Bearing2_4', 'test_08_Bearing2_5',
          'test_09_Bearing2_6', 'test_10_Bearing2_7'],
}
EXPECTED_WINDOWS = {'a': 4414, 'b': 22303}
CODE = ('Experiments/r1_tsmc_randsplit.py', 'Experiments/exp19_baselines.py', 'Experiments/exp16_ress.py',
        'src/published_baselines.py')


def pooled(design):
    bearings = load_bearings(str(DATA), DESIGNS[design], channels='hv')
    x = np.concatenate([b.vib for b in bearings]).astype(np.float32)          # (N, 2, 2560)
    y = np.concatenate([b.rul for b in bearings]).astype(np.float32)
    ids = np.concatenate([np.full(b.n, i) for i, b in enumerate(bearings)])
    return x, y, ids, [b.name for b in bearings]


def run_split(x, y, split, seed, epochs, device):
    settings = MODEL_TRAINING['tsmccnn']
    assert settings['loss'] == 'mse' and settings['optimizer'] == 'adam' and settings['lr'] == 1e-3
    idx = np.random.default_rng(split).permutation(len(y))
    ntr = int(TRAIN_FRACTION * len(y))
    tr, te = idx[:ntr], idx[ntr:]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = BaselineAdapter(TSMCCNN()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()
    xt = torch.from_numpy(x[tr]).unsqueeze(1)                                   # (n, 1, 2, 2560), as the loader
    yt = torch.from_numpy(y[tr])
    losses = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        total = 0.0
        for i in range(0, len(tr), BATCH):
            j = perm[i:i + BATCH]
            xb, yb = xt[j].to(device), yt[j].to(device)
            optimizer.zero_grad()
            pred, *_ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss.detach()) * len(j)
        scheduler.step()
        losses.append(total / len(tr))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te), 512):
            xb = torch.from_numpy(x[te[i:i + 512]]).unsqueeze(1).to(device)
            preds.append(model(xb)[0].cpu().numpy())
    p = np.concatenate(preds)
    err = p - y[te]
    mae = float(np.mean(np.abs(err)))
    r2 = float(1 - np.sum(err ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
    return {'split': split, 'seed': seed, 'n_train': int(ntr), 'n_test': int(len(te)), 'mae': mae, 'r2': r2,
            'pred_std': float(p.std()), 'train_loss_history': losses}, te, p


def run_design(design, splits, epochs, out_dir, device):
    x, y, ids, names = pooled(design)
    assert len(y) == EXPECTED_WINDOWS[design], (len(y), EXPECTED_WINDOWS[design])
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for split in splits:
        t0 = time.time()
        res, te, p = run_split(x, y, split, FINAL_SEEDS[split], epochs, device)
        res['seconds'] = round(time.time() - t0, 1)
        results.append(res)
        np.savez_compressed(out_dir / f'design_{design}_split{split:02d}.npz', test_index=te, y=y[te], pred=p,
                            bearing=ids[te])
        print(f'design {design} split {split:2d} seed {res["seed"]:5d}: MAE {res["mae"]:.4f}  R2 {res["r2"]:.3f}  '
              f'({res["seconds"]} s)', flush=True)
    meta = {'design': design, 'bearings': names, 'n_windows': int(len(y)), 'epochs': epochs, 'batch': BATCH,
            'train_fraction': TRAIN_FRACTION, 'training': MODEL_TRAINING['tsmccnn'], 'runs': results,
            'provenance': provenance(CODE)}
    (out_dir / f'design_{design}.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    return results


def summary():
    tab = json.loads((AN / 'tables_R1.json').read_text(encoding='utf-8'))
    ref = tab['sets']['all10']['rows']['TSMC-CNN [21]']['pooled']['raw_mae']
    rows, report = [], {}
    for design in ('a', 'b'):
        meta = json.loads((OUT / f'design_{design}.json').read_text(encoding='utf-8'))
        maes = np.array([r['mae'] for r in meta['runs']])
        assert len(maes) == 20
        report[design] = {'n_windows': meta['n_windows'], 'mae_mean': float(maes.mean()),
                          'mae_sd': float(maes.std(ddof=1)), 'mae_min': float(maes.min()),
                          'mae_max': float(maes.max()),
                          'r2_mean': float(np.mean([r['r2'] for r in meta['runs']]))}
    report['bearing_level_pooled_raw_mae'] = {'mean': ref[0], 'sd': ref[1]}
    report['ratio_bearing_level_to_b'] = ref[0] / report['b']['mae_mean']
    report['ratio_bearing_level_to_a'] = ref[0] / report['a']['mae_mean']
    report['published_21'] = 0.0214
    report['provenance'] = provenance(CODE)
    lines = ['# TSMC-CNN under a random split of windows (revision R1, Amendment 8)', '',
             'Generated by `Experiments/r1_tsmc_randsplit.py`; design in analysis plan Section 11, Amendment 8. '
             'Raw MAE on the held-out 30 % of windows, 20 random splits; descriptive.', '',
             '| Design | Bearings (windows) | MAE mean (SD) | range | mean R2 |', '|---|---|---|---|---|',
             f"| R8-a, [21]'s data | Bearing2_1 to 2_4 ({report['a']['n_windows']}) | {report['a']['mae_mean']:.4f} "
             f"({report['a']['mae_sd']:.4f}) | {report['a']['mae_min']:.4f} to {report['a']['mae_max']:.4f} | "
             f"{report['a']['r2_mean']:.3f} |",
             f"| R8-b, protocol pool | 14 Condition 1-2 bearings ({report['b']['n_windows']}) | "
             f"{report['b']['mae_mean']:.4f} ({report['b']['mae_sd']:.4f}) | {report['b']['mae_min']:.4f} to "
             f"{report['b']['mae_max']:.4f} | {report['b']['r2_mean']:.3f} |",
             f"| Bearing-level (protocol final runs, 20 seeds) | trained on 4, tested on 10 | {ref[0]:.4f} ({ref[1]:.4f}) | | |",
             '', f"Bearing-level over R8-b: {report['ratio_bearing_level_to_b']:.2f} times; over R8-a: "
             f"{report['ratio_bearing_level_to_a']:.2f} times. Published value for TSMC-CNN (K = 4, 70/30): 0.0214."]
    text = '\n'.join(lines) + '\n'
    (AN / 'tsmc_randsplit.md').write_text(text, encoding='utf-8')
    (AN / 'tsmc_randsplit.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(text)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--design', choices=['a', 'b'])
    ap.add_argument('--summary', action='store_true')
    args = ap.parse_args()
    if args.summary:
        summary()
        return 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        for d in ('a', 'b'):
            run_design(d, [0], 2, ROOT / 'outputs' / 'r1_tsmc_randsplit_smoke', device)
        return 0
    run_design(args.design, list(range(20)), EPOCHS, OUT, device)
    return 0


if __name__ == '__main__':
    sys.exit(main())
