"""
r1_baseline_valpreds.py — Revision R1, analysis plan Amendment 2, V: final-epoch validation
predictions of the seven PRONOSTIA baselines at their selected budgets, the calibration data
for the cross-model analyses P1 to P3 (r1_decision_models.py).

For each baseline, its four-fold selection runs at the selected budget K are repeated with
seeds 7, 42, 123, 777 and 999, deterministic kernels and the original training settings. The
final-epoch model predicts its held-out training bearing: the deterministic prediction and 30
MC-dropout passes, as for our model (plan Section 4). exp19's train_baseline restores the
best-validation checkpoint after training, so its loop is reproduced here up to that restore
and the final-epoch state is kept; MSCNN's r1_mscnn.train already returns the final epoch.

Checks per run: (1) final-epoch validation MAE against the original selection run (exact for
originals run with deterministic kernels: seeds 7, 777, 999 of the exp19 models in
outputs/r1_baseline_sel5, every MSCNN run in outputs/r1_mscnn/selection.json; seeds 42 and
123 of the exp19 models are compared with their older runs, folds A/B as the mean of four
runs); (2) the saved deterministic validation predictions reproduce the final-epoch
validation MAE (rounded to five decimals, as in the training harness).

Output: outputs/r1_baseline_valpreds/<model>/fold{A..D}_seed{seed}.npz (raw_, tgt_, mcmean_,
mcstd_ arrays of the fold's validation bearing) and <model>/results.json. Resumable.

USAGE:
  python Experiments/r1_baseline_valpreds.py --model cnn --smoke
  python Experiments/r1_baseline_valpreds.py --model cnn
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
import Experiments.r1_mscnn as rm  # noqa: E402
from Experiments.exp16_ress import CONFIG, build_loaders, evaluate_on_test, load_bearings  # noqa: E402
from Experiments.exp19_baselines import MODELS as EXP19_MODELS  # noqa: E402
from Experiments.exp19_baselines import MODEL_TRAINING, BaselineAdapter, regression_metrics  # noqa: E402
from Experiments.r1_provenance import provenance  # noqa: E402

PLAN_K = {'cnn': 40, 'bilstm': 40, 'cnnlstm': 5, 'resnet': 10, 'transformer': 10, 'tsmccnn': 40, 'mscnn': 5}
MODELS = list(PLAN_K)
SEEDS = [7, 42, 123, 777, 999]
DETERMINISTIC_ORIGINAL = {7, 777, 999}           # exp19 models; every MSCNN run is deterministic
FOLDS = {'A': 'train_02_Bearing1_2', 'B': 'train_04_Bearing2_2',
         'C': 'train_01_Bearing1_1', 'D': 'train_03_Bearing2_1'}
PRIOR_AB = {'tsmccnn': 'outputs/r1_published'}  # default outputs/exp19_baselines
MC_PASSES = 30
EXTRA_CODE = ('Experiments/r1_baseline_valpreds.py', 'Experiments/r1_mscnn.py', 'src/mscnn_li2019.py')


def out_root(smoke):
    return ROOT / 'outputs' / ('r1_baseline_valpreds_smoke' if smoke else 'r1_baseline_valpreds')


def selected_k(model):
    """budget from the selection file, checked against the plan"""
    if model == 'mscnn':
        k = json.loads((ROOT / 'outputs' / 'r1_mscnn' / 'selection.json').read_text(encoding='utf-8'))['K']
    else:
        k = json.loads((ROOT / 'outputs' / 'r1_baseline_sel5' / f'{model}.json').read_text())['K_5seed_4fold']
    if k != PLAN_K[model]:
        raise SystemExit(f'{model}: selection file gives K = {k}, plan Amendment 2 says {PLAN_K[model]}')
    return k


def original_val_mae(model, k):
    """{(fold, seed): final-epoch validation MAE of the original selection run}, plus the A/B means"""
    if model == 'mscnn':
        sel = json.loads((ROOT / 'outputs' / 'r1_mscnn' / 'selection.json').read_text(encoding='utf-8'))
        return {(r['fold'][-1], r['seed']): r['val_history'][k - 1] for r in sel['runs']}, None
    runs = json.loads((ROOT / 'outputs' / 'r1_baseline_sel5' / f'{model}.json').read_text())['runs']
    ref = {(r['fold'][-1], r['seed']): r['final_val_mae'] for r in runs if r['budget'] == k}
    old_cd = json.loads((ROOT / 'outputs' / 'r1_lobo' / 'baselines' / f'{model}.json').read_text())['runs']
    ref.update({(r['fold'][-1], r['seed']): r['final_val_mae'] for r in old_cd
                if r['budget'] == k and r['seed'] in (42, 123)})
    prior = json.loads((ROOT / PRIOR_AB.get(model, 'outputs/exp19_baselines') / model / 'results.json').read_text())
    return ref, float(prior['budget_scores'][str(k)])   # mean of seeds 42, 123 on folds A and B


def selection_config(model, val_bearing, epochs):
    """the configuration of r1_baseline_select5.py for one fold run"""
    cfg = dict(CONFIG)
    cfg.update({'data_dir': str(ROOT / ('Processed_PRONOSTIA_v2hv' if model == 'tsmccnn' else 'Processed_PRONOSTIA_v2')),
                'val_mode': 'bearing', 'val_bearings': [val_bearing], 'epochs': epochs, 'patience': 10 ** 6,
                'mc_dropout_passes': MC_PASSES, 'skip_test': True, 'test_bearings': []})
    return cfg


def train_exp19_final(model_name, seed, cfg, device):
    """exp19_baselines.train_baseline up to its best-checkpoint restore; returns the final-epoch model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    config = {**cfg, **MODEL_TRAINING.get(model_name, {})}
    train_loader, val_loader, _seq, _test = build_loaders(config, seed)
    model = BaselineAdapter(EXP19_MODELS[model_name](),
                            needs_spectrogram=(model_name == 'transformer')).to(device)
    opt_cls = torch.optim.Adam if config.get('optimizer') == 'adam' else torch.optim.AdamW
    optimizer = opt_cls(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    criterion = nn.MSELoss() if config.get('loss') == 'mse' else nn.L1Loss()
    val_history = []
    for _epoch in range(1, config['epochs'] + 1):
        model.train()
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, *_r = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, y, *_ in val_loader:
                pred, *_r = model(x.to(device))
                vp.append(pred.cpu().numpy())
                vt.append(y.numpy())
        val_history.append(round(float(regression_metrics(np.concatenate(vp), np.concatenate(vt))['mae']), 5))
    return model, val_history, config


def one_run(model_name, fold, seed, k, device, smoke):
    val_bearing = FOLDS[fold]
    if model_name == 'mscnn':
        input_norm = json.loads((ROOT / 'outputs' / 'r1_mscnn' / 'selection.json').read_text(encoding='utf-8'))['input_norm']
        model, _loss, val_history, _tb, _vl = rm.train(seed, k, device, val_bearing=val_bearing, input_norm=input_norm)
        vb = rm.with_history(load_bearings(str(rm.DATA), [val_bearing]))
        cfg = dict(CONFIG, mc_dropout_passes=2 if smoke else MC_PASSES)
    else:
        cfg = selection_config(model_name, val_bearing, k)
        if smoke:
            cfg['mc_dropout_passes'] = 2
        model, val_history, cfg = train_exp19_final(model_name, seed, cfg, device)
        vb = load_bearings(cfg['data_dir'], [val_bearing], cfg.get('channels', 'h'))
    ev = evaluate_on_test(model, vb, cfg, device)
    a = ev['_arrays']
    arrays = {f'{tag}_{val_bearing}': a[key][val_bearing]
              for key, tag in (('raw_p', 'raw'), ('tgt', 'tgt'), ('mc_mean', 'mcmean'), ('mc_std', 'mcstd'))}
    pred_mae = float(np.mean(np.abs(a['raw_p'][val_bearing] - a['tgt'][val_bearing])))
    return arrays, val_history, pred_mae


def run(model_name, smoke, device):
    k = 1 if smoke else selected_k(model_name)
    folds, seeds = (['A'], [42]) if smoke else (list(FOLDS), SEEDS)
    out_dir = out_root(smoke) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_f = out_dir / 'results.json'
    state = json.loads(summary_f.read_text()) if summary_f.exists() else {'model': model_name, 'runs': []}
    done = {(r['fold'], r['seed']) for r in state['runs']}
    ref, ab_mean = ({}, None) if smoke else original_val_mae(model_name, k)
    for fold in folds:
        for seed in seeds:
            f_npz = out_dir / f'fold{fold}_seed{seed}.npz'
            if (fold, seed) in done and f_npz.exists():
                continue
            t0 = time.time()
            arrays, val_history, pred_mae = one_run(model_name, fold, seed, k, device, smoke)
            np.savez_compressed(f_npz, **arrays)
            orig = ref.get((fold, seed))
            state['runs'] = [r for r in state['runs'] if (r['fold'], r['seed']) != (fold, seed)]
            state['runs'].append({
                'fold': fold, 'seed': seed, 'final_val_mae': val_history[-1], 'val_history': val_history,
                'original_final_val_mae': orig, 'replay_abs_diff': None if orig is None else abs(val_history[-1] - orig),
                'original_deterministic': model_name == 'mscnn' or seed in DETERMINISTIC_ORIGINAL,
                'saved_prediction_mae': pred_mae, 'minutes': round((time.time() - t0) / 60, 2)})
            summary_f.write_text(json.dumps(state, indent=1))
            print(f"{model_name} fold {fold} seed {seed}: final val MAE {val_history[-1]:.5f} (original "
                  f"{'n/a' if orig is None else f'{orig:.5f}'}), saved predictions {pred_mae:.5f} "
                  f"({state['runs'][-1]['minutes']} min)", flush=True)
    runs = state['runs']
    det = [r['replay_abs_diff'] for r in runs if r['original_deterministic'] and r['replay_abs_diff'] is not None]
    other = [r['replay_abs_diff'] for r in runs if not r['original_deterministic'] and r['replay_abs_diff'] is not None]
    if ab_mean is not None:
        ab = [r['final_val_mae'] for r in runs if r['fold'] in 'AB' and r['seed'] in (42, 123)]
        other.append(abs(float(np.mean(ab)) - ab_mean) if len(ab) == 4 else float('nan'))
    consistency = max(abs(r['saved_prediction_mae'] - r['final_val_mae']) for r in runs)
    expected = len(folds) * len(seeds)
    state.update({'K': k, 'mc_dropout_passes': 2 if smoke else MC_PASSES, 'n_runs': len(runs),
                  'replay_max_abs_diff_deterministic_originals': max(det) if det else None,
                  'replay_max_abs_diff_other_originals': max(other) if other else None,
                  'max_abs_diff_saved_predictions_vs_val_history': consistency,
                  'date': time.strftime('%Y-%m-%d %H:%M'), 'provenance': provenance(EXTRA_CODE)})
    summary_f.write_text(json.dumps(state, indent=1))
    print(f"{model_name}: {len(runs)} of {expected} runs; replay max |diff| deterministic originals "
          f"{state['replay_max_abs_diff_deterministic_originals']}, other originals "
          f"{state['replay_max_abs_diff_other_originals']}; saved predictions vs val history {consistency:.1e}",
          flush=True)
    if len(runs) != expected:
        raise SystemExit(f'{model_name}: expected {expected} runs, found {len(runs)}')
    if not smoke and det and max(det) > 1e-9:
        raise SystemExit(f'{model_name}: repeated runs do not reproduce the deterministic originals')
    if consistency > 1e-4:
        raise SystemExit(f'{model_name}: saved validation predictions do not match the final-epoch validation MAE')
    print(f'{model_name}: validation predictions complete (K = {k})', flush=True)
    return 0


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='one of ' + ', '.join(MODELS) + ', or all')
    ap.add_argument('--smoke', action='store_true', help='fold A, seed 42, 1 epoch, 2 MC passes, separate folder')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for m in (MODELS if args.model == 'all' else [args.model]):
        run(m, args.smoke, device)
    return 0


if __name__ == '__main__':
    sys.exit(main())
