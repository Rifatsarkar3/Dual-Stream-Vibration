"""
r1_mscnn.py — Revision R1, analysis plan Section 6: the multi-scale CNN of Li, Zhang and
Ding (2019, ref [7]; src/mscnn_li2019.py) under the leakage-free protocol.

  --stage smoke   seed 42, fold A, 10 epochs, validation only; checks the pre-registered
                  pass criteria and, on failure, repeats once with input batch normalization
  --stage select  budget selection: 4 folds x 5 selection seeds, B in {5, 10, 20, 40, 200}
                  (200 = published value), argmin of the mean final-epoch validation MAE
  --stage final   20 final seeds at the selected budget on all four training bearings;
                  predictions on the 13 evaluation bearings in the r1_analysis format

Training settings published in [7]: Adam, learning rate 1e-4 (constant), batch 32, MAE loss.
Protocol settings shared with every baseline: whole-life labels, gradient-norm clipping at
1.0 (training harness default), final-epoch model, 10 MC-dropout passes at evaluation.
History: window t uses recordings t-4 .. t; the first four windows of a bearing repeat its
first recording. Deterministic CUDA kernels; selection runs never load test bearings.

Because the learning rate is constant and nothing depends on the budget, a B-epoch run is
exactly the first B epochs of the 200-epoch run with the same seed; selection therefore
trains each (fold, seed) once for 200 epochs and reads the validation MAE after epochs 5,
10, 20, 40 and 200. One separate 10-epoch run is repeated to verify this equivalence.

USAGE:
  python Experiments/r1_mscnn.py --stage smoke|select|final
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
from Experiments.exp16_ress import (CONFIG, BearingArrays, WindowDataset,  # noqa: E402
                                    evaluate_on_test, load_bearings)
from Experiments.exp19_baselines import BaselineAdapter  # noqa: E402
from Experiments.r1_provenance import provenance  # noqa: E402
from src.mscnn_li2019 import MSCNN  # noqa: E402

DATA = ROOT / 'Processed_PRONOSTIA_v2'
OUT = ROOT / 'outputs' / 'r1_mscnn'
N_HIST = 5
LR, BATCH = 1e-4, 32
FOLDS = {'foldA': 'train_02_Bearing1_2', 'foldB': 'train_04_Bearing2_2',
         'foldC': 'train_01_Bearing1_1', 'foldD': 'train_03_Bearing2_1'}
SELECT_SEEDS = [7, 42, 123, 777, 999]
BUDGETS = [5, 10, 20, 40, 200]
FINAL_SEEDS = [42, 123, 777, 999, 7, 8, 21, 34, 55, 89, 144, 233, 377, 610, 987,
               1597, 2584, 4181, 6765, 10946]
EVAL_BEARINGS = ['test_01_Bearing1_3', 'test_02_Bearing1_4', 'test_03_Bearing1_5',
                 'test_04_Bearing2_3', 'test_05_Bearing1_6', 'test_06_Bearing1_7',
                 'test_07_Bearing2_4', 'test_08_Bearing2_5', 'test_09_Bearing2_6',
                 'test_10_Bearing2_7', 'zs_01_Bearing3_1', 'zs_02_Bearing3_2',
                 'zs_03_Bearing3_3']
EXTRA_CODE = ('src/mscnn_li2019.py', 'Experiments/r1_mscnn.py')


def with_history(bearings, k=N_HIST):
    """Window t -> recordings t-k+1 .. t (oldest first); causal, first recording repeated."""
    out = []
    for b in bearings:
        idx = np.clip(np.arange(b.n)[:, None] - np.arange(k - 1, -1, -1)[None, :], 0, None)
        out.append(BearingArrays(b.name, b.vib[idx], b.rul, b.condition))
    return out


def train(seed, epochs, device, val_bearing=None, input_norm=None, log_every=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    bearings = with_history(load_bearings(str(DATA), CONFIG['train_bearings']))
    tb = [b for b in bearings if b.name != val_bearing]
    vb = [b for b in bearings if b.name == val_bearing]
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        WindowDataset(tb, [np.arange(b.n) for b in tb]), batch_size=BATCH, shuffle=True,
        generator=gen, num_workers=0, drop_last=True)
    val_loader = (torch.utils.data.DataLoader(
        WindowDataset(vb, [np.arange(b.n) for b in vb]), batch_size=256, shuffle=False,
        num_workers=0) if vb else None)

    model = BaselineAdapter(MSCNN(n_hist=N_HIST, input_norm=input_norm)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    l1 = nn.L1Loss()
    loss_hist, val_hist = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, *_r = model(x)
            loss = l1(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(loss) * len(y)
            n += len(y)
        loss_hist.append(total / n)
        if val_loader is not None:
            vp, vt = predict(model, val_loader, device)
            val_hist.append(float(np.mean(np.abs(vp - vt))))
        if log_every and (epoch == 1 or epoch % log_every == 0):
            msg = f'  epoch {epoch:3d}/{epochs} train L1={loss_hist[-1]:.4f}'
            print(msg + (f' val MAE={val_hist[-1]:.4f}' if val_hist else ''), flush=True)
    return model, loss_hist, val_hist, tb, val_loader


def predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y, *_ in loader:
            pred, *_r = model(x.to(device))
            ps.append(pred.cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def stage_smoke(device):
    report = {'attempts': []}
    for input_norm in (None, 'bn'):
        model, loss_hist, val_hist, tb, val_loader = train(
            42, 10, device, val_bearing=FOLDS['foldA'], input_norm=input_norm, log_every=1)
        vp, vt = predict(model, val_loader, device)
        label_mean = float(np.mean(np.concatenate([b.rul for b in tb])))
        const_mae = float(np.mean(np.abs(label_mean - vt)))
        checks = {'train_loss_decreases': loss_hist[-1] < loss_hist[0],
                  'val_pred_std_gt_0.02': float(vp.std()) > 0.02,
                  'val_mae_below_constant_predictor': val_hist[-1] < const_mae}
        attempt = {'input_norm': input_norm, 'train_loss_history': loss_hist,
                   'val_history': val_hist, 'val_pred_std': float(vp.std()),
                   'constant_predictor_val_mae': const_mae, 'checks': checks,
                   'passed': all(checks.values())}
        report['attempts'].append(attempt)
        print(f"smoke input_norm={input_norm}: {checks} -> {'PASS' if attempt['passed'] else 'FAIL'}",
              flush=True)
        if attempt['passed']:
            break
    report['selected_input_norm'] = next((a['input_norm'] for a in report['attempts'] if a['passed']),
                                         'none passed')
    report['provenance'] = provenance(EXTRA_CODE)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'smoke.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    return 0 if report['selected_input_norm'] != 'none passed' else 1


def input_norm_from_smoke():
    rep = json.loads((OUT / 'smoke.json').read_text(encoding='utf-8'))
    if rep['selected_input_norm'] == 'none passed':
        raise SystemExit('smoke test failed for both input settings; see outputs/r1_mscnn/smoke.json')
    return rep['selected_input_norm']


def stage_select(device):
    input_norm = input_norm_from_smoke()
    path = OUT / 'selection.json'
    state = json.loads(path.read_text()) if path.exists() else {'runs': []}
    done = {(r['fold'], r['seed']) for r in state['runs']}
    for fold, val_bearing in FOLDS.items():
        for seed in SELECT_SEEDS:
            if (fold, seed) in done:
                continue
            t0 = time.time()
            _m, loss_hist, val_hist, _tb, _vl = train(seed, max(BUDGETS), device,
                                                      val_bearing=val_bearing, input_norm=input_norm)
            state['runs'].append({'fold': fold, 'seed': seed, 'val_history': val_hist,
                                  'train_loss_history': loss_hist,
                                  'minutes': round((time.time() - t0) / 60, 1)})
            path.write_text(json.dumps(state, indent=1), encoding='utf-8')
            print(f'{fold} seed {seed}: ' + '  '.join(f'B={B}:{val_hist[B - 1]:.4f}' for B in BUDGETS),
                  flush=True)

    # Equivalence check: a separate 10-epoch run equals the first 10 epochs of the 200-epoch run.
    _m, _l, val10, _tb, _vl = train(42, 10, device, val_bearing=FOLDS['foldA'], input_norm=input_norm)
    long_run = next(r for r in state['runs'] if r['fold'] == 'foldA' and r['seed'] == 42)
    equiv = float(np.max(np.abs(np.array(val10) - np.array(long_run['val_history'][:10]))))

    scores = {}
    for B in BUDGETS:
        v = np.array([r['val_history'][B - 1] for r in state['runs']])
        scores[B] = {'mean': float(v.mean()), 'se': float(v.std(ddof=1) / np.sqrt(len(v))), 'n': len(v)}
    k = min(BUDGETS, key=lambda b: scores[b]['mean'])
    state.update({'scores': {str(b): s for b, s in scores.items()}, 'K': k, 'input_norm': input_norm,
                  'equivalence_max_abs_diff_10ep': equiv, 'provenance': provenance(EXTRA_CODE)})
    path.write_text(json.dumps(state, indent=1), encoding='utf-8')
    for B in BUDGETS:
        print(f"B={B:3d}: 4-fold mean={scores[B]['mean']:.4f}  SE={scores[B]['se']:.4f}  n={scores[B]['n']}")
    print(f'SELECTED K = {k}; equivalence check max |diff| = {equiv:.2e}')
    return 0


def stage_final(device):
    sel = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))
    k, input_norm = sel['K'], sel['input_norm']
    out_dir = OUT / 'final'
    out_dir.mkdir(parents=True, exist_ok=True)
    test = with_history(load_bearings(str(DATA), EVAL_BEARINGS))
    cfg = dict(CONFIG)
    cfg['mc_dropout_passes'] = 10
    per_seed = []
    for seed in FINAL_SEEDS:
        if (out_dir / f'preds_seed{seed}.npz').exists():
            continue
        model, loss_hist, _v, _tb, _vl = train(seed, k, device, input_norm=input_norm)
        res = evaluate_on_test(model, test, cfg, device)
        arrays = res.pop('_arrays')
        np.savez_compressed(
            out_dir / f'preds_seed{seed}.npz',
            **{f'raw_{n}': v for n, v in arrays['raw_p'].items()},
            **{f'tgt_{n}': v for n, v in arrays['tgt'].items()},
            **{f'alpha_{n}': v for n, v in arrays['alphas'].items()},
            **{f'mcmean_{n}': v for n, v in arrays['mc_mean'].items()},
            **{f'mcstd_{n}': v for n, v in arrays['mc_std'].items()})
        torch.save(model.state_dict(), out_dir / f'model_seed{seed}.pth')
        res.update({'seed': seed, 'train_loss_history': loss_hist})
        per_seed.append(res)
        print(f"seed {seed}: pooled raw MAE (13 bearings) = {res['raw']['mae']:.4f}", flush=True)
    (out_dir / 'results.json').write_text(json.dumps(
        {'model': 'MSCNN (Li et al. 2019)', 'K': k, 'input_norm': input_norm,
         'n_params': sum(p.numel() for p in MSCNN(input_norm=input_norm).parameters()),
         'per_seed': per_seed, 'provenance': provenance(EXTRA_CODE)}, indent=1), encoding='utf-8')
    return 0


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['smoke', 'select', 'final'], required=True)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return {'smoke': stage_smoke, 'select': stage_select, 'final': stage_final}[args.stage](device)


if __name__ == '__main__':
    sys.exit(main())
