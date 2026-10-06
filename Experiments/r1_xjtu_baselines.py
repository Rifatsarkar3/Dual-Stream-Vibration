"""
r1_xjtu_baselines.py — Revision R1, analysis plan Amendment 1, E6: the three XJTU-SY baselines of the
submission (1D CNN, BiLSTM, CNN-LSTM) under the revised selection, as the PRONOSTIA baselines in
plan Section 5 but on the XJTU-SY folds.

  --stage select   budgets {5, 10, 20, 40} x folds A-D x seeds 7, 42, 123, 777, 999, final-epoch
                   validation MAE, argmin of the 20-run mean; deterministic kernels; test bearings
                   never loaded. Resumable (outputs/r1_xjtu_baselines/select/<model>.json).
  --stage final    the 20 final seeds at the selected budget (read from the selection file), 10
                   MC-dropout passes, evaluated on the six XJTU-SY test and five Condition-3 bearings;
                   one preds_seed*.npz per seed holding both sets (r1_xjtu_stats layout). Resumable.
  --smoke          select: fold A, seed 42, budget 1; final: seed 42, 1 epoch; separate folder

USAGE:
  python Experiments/r1_xjtu_baselines.py --stage select --model cnn
  python Experiments/r1_xjtu_baselines.py --stage final --model cnn
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
from Experiments.exp16_ress import evaluate_on_test, load_bearings  # noqa: E402
from Experiments.exp17_xjtu import XJTU_CONFIG  # noqa: E402
from Experiments.exp19_baselines import train_baseline  # noqa: E402
from Experiments.r1_provenance import provenance  # noqa: E402

MODELS = ['cnn', 'bilstm', 'cnnlstm']
SELECT_SEEDS = [7, 42, 123, 777, 999]
BUDGETS = [5, 10, 20, 40]
FOLDS = {'foldA': 'train_02_Bearing1_2', 'foldB': 'train_04_Bearing2_2',
         'foldC': 'train_01_Bearing1_1', 'foldD': 'train_03_Bearing2_1'}
FINAL_SEEDS = [42, 123, 777, 999, 7, 8, 21, 34, 55, 89, 144, 233, 377, 610, 987,
               1597, 2584, 4181, 6765, 10946]
EXTRA_CODE = ('Experiments/r1_xjtu_baselines.py', 'Experiments/exp17_xjtu.py')


def out_root(smoke):
    return ROOT / 'outputs' / ('r1_xjtu_baselines_smoke' if smoke else 'r1_xjtu_baselines')


def select(model, smoke, device):
    path = out_root(smoke) / 'select' / f'{model}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(path.read_text()) if path.exists() else {'model': model, 'runs': []}
    done = {(r['budget'], r['fold'], r['seed']) for r in state['runs']}
    budgets, folds, seeds = ([1], {'foldA': FOLDS['foldA']}, [42]) if smoke else (BUDGETS, FOLDS, SELECT_SEEDS)
    t0 = time.time()
    for B in budgets:
        for fold, val_bearing in folds.items():
            for seed in seeds:
                if (B, fold, seed) in done:
                    continue
                cfg = dict(XJTU_CONFIG)
                cfg.update({'val_mode': 'bearing', 'val_bearings': [val_bearing], 'epochs': B,
                            'patience': 10 ** 6, 'mc_dropout_passes': 5, 'skip_test': True, 'test_bearings': []})
                res, _m = train_baseline(model, seed, cfg, device, verbose=False)
                state['runs'].append({'budget': B, 'fold': fold, 'seed': seed,
                                      'final_val_mae': res['val_history'][-1], 'val_history': res['val_history']})
                path.write_text(json.dumps(state, indent=1))
        print(f'{model} B={B:2d}: runs done ({(time.time() - t0) / 60:.1f} min)', flush=True)
    scores = {}
    for B in budgets:
        v = np.array([r['final_val_mae'] for r in state['runs'] if r['budget'] == B])
        expected = len(folds) * len(seeds)
        if len(v) != expected:
            raise SystemExit(f'{model} B={B}: expected {expected} runs, found {len(v)}')
        scores[B] = {'mean': float(v.mean()), 'se': float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None,
                     'n': len(v)}
    k = min(budgets, key=lambda b: scores[b]['mean'])
    old = json.loads((ROOT / 'outputs' / 'exp19_xjtu' / model / 'results.json').read_text(encoding='utf-8'))
    state.update({'scores': {str(b): s for b, s in scores.items()}, 'K': k,
                  'K_submission': old['epoch_budget_K'], 'budget_changed': k != old['epoch_budget_K'],
                  'date': time.strftime('%Y-%m-%d %H:%M'), 'provenance': provenance(EXTRA_CODE)})
    path.write_text(json.dumps(state, indent=1))
    print(f'==> {model}: ' + '  '.join(f"B={b}:{scores[b]['mean']:.4f}" for b in budgets)
          + f"  K={k} (submission: {old['epoch_budget_K']})", flush=True)
    return 0


def final(model, smoke, device):
    sel = json.loads((out_root(smoke) / 'select' / f'{model}.json').read_text())
    k = 1 if smoke else int(sel['K'])
    seeds = [42] if smoke else FINAL_SEEDS
    out_dir = out_root(smoke) / 'final' / model
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(XJTU_CONFIG)
    cfg.update({'val_mode': 'none', 'epochs': k, 'mc_dropout_passes': 10})
    zeroshot = load_bearings(cfg['data_dir'], cfg['zeroshot_bearings'])
    summary_f = out_dir / 'per_seed.json'
    per_seed = json.loads(summary_f.read_text()) if summary_f.exists() else []
    for seed in seeds:
        if (out_dir / f'preds_seed{seed}.npz').exists():
            continue
        t0 = time.time()
        res, net = train_baseline(model, seed, cfg, device, verbose=False)
        zs = evaluate_on_test(net, zeroshot, cfg, device)
        arrays = {}
        for a in (res.pop('_arrays'), zs.pop('_arrays')):
            for key, tag in (('raw_p', 'raw'), ('tgt', 'tgt'), ('alphas', 'alpha'),
                             ('mc_mean', 'mcmean'), ('mc_std', 'mcstd')):
                arrays.update({f'{tag}_{b}': v for b, v in a[key].items()})
        np.savez_compressed(out_dir / f'preds_seed{seed}.npz', **arrays)
        torch.save(net.state_dict(), out_dir / f'model_seed{seed}.pth')
        per_seed.append({'seed': seed, 'test_raw_mae': res['raw']['mae'], 'test_ces_mae': res['ces']['mae'],
                         'zs_raw_mae': zs['raw']['mae'], 'minutes': round((time.time() - t0) / 60, 1)})
        summary_f.write_text(json.dumps(per_seed, indent=1))
        print(f"{model} seed {seed}: test raw MAE {res['raw']['mae']:.4f}, zero-shot raw MAE {zs['raw']['mae']:.4f} "
              f"({per_seed[-1]['minutes']} min)", flush=True)
    (out_dir / 'results.json').write_text(json.dumps(
        {'model': model, 'K': k, 'selection': str(out_root(smoke) / 'select' / f'{model}.json'),
         'per_seed': per_seed, 'provenance': provenance(EXTRA_CODE)}, indent=1), encoding='utf-8')
    print(f'{model}: XJTU-SY final runs complete (K = {k})', flush=True)
    return 0


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['select', 'final'], required=True)
    ap.add_argument('--model', required=True, help='one of ' + ', '.join(MODELS) + ', or all')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fn = select if args.stage == 'select' else final
    for m in (MODELS if args.model == 'all' else [args.model]):
        fn(m, args.smoke, device)
    return 0


if __name__ == '__main__':
    sys.exit(main())
