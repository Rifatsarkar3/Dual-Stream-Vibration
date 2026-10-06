"""
r1_bestckpt_valpreds.py — Revision R1, analysis plan Amendment 7, part B: best-epoch and final-epoch
validation predictions of the seven PRONOSTIA baselines' four-fold selection runs at their selected budgets
(1D CNN 40, BiLSTM 40, CNN-LSTM 5, 1D ResNet 10, Transformer 10, TSMC-CNN 40, MSCNN 5 epochs; seeds 7, 42,
123, 777, 999).

The training loop of Amendment 2, V (r1_baseline_valpreds.train_exp19_final; MSCNN: r1_mscnn.train) is
repeated with deterministic kernels, additionally keeping the state at the epoch with the lowest validation
MAE (exp19's restore rule: deterministic validation MAE, first minimum; the same rule on MSCNN's validation
history). The final-epoch state predicts the held-out bearing first, as in V, then the best-epoch state;
deterministic and 30 MC-dropout passes each.

Checks per run: (1) the final-epoch deterministic predictions reproduce V's stored ones within 1e-6;
(2) the best-epoch state's validation MAE equals the minimum of the run's validation history (within 1e-5:
exp19 rounds the history to five decimals).

Output: outputs/r1_bestckpt_valpreds/<model>/fold{F}_seed{s}_{final,best}.npz (raw_, tgt_, mcmean_,
mcstd_ of the fold's validation bearing), <model>/runs.json and <model>/meta.json. Resumable.

USAGE:
  python Experiments/r1_bestckpt_valpreds.py --model cnn           # or all
  python Experiments/r1_bestckpt_valpreds.py --smoke               # 1D CNN, fold A, seed 7, full length, 2 passes
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
import Experiments.r1_baseline_valpreds as vp  # noqa: E402  (Amendment 2, V; used unchanged)
import Experiments.r1_mscnn as rm  # noqa: E402
from Experiments.exp16_ress import CONFIG, WindowDataset, build_loaders, evaluate_on_test, load_bearings  # noqa: E402
from Experiments.exp19_baselines import MODELS as EXP19_MODELS  # noqa: E402
from Experiments.exp19_baselines import MODEL_TRAINING, BaselineAdapter, regression_metrics  # noqa: E402
from Experiments.r1_provenance import provenance  # noqa: E402
from src.mscnn_li2019 import MSCNN  # noqa: E402

MODELS = list(vp.PLAN_K)
MC_PASSES = 30
TOL_FINAL, TOL_BEST = 1e-6, 1e-5
OUT = ROOT / 'outputs' / 'r1_bestckpt_valpreds'
SMOKE_OUT = ROOT / 'outputs' / 'r1_bestckpt_valpreds_smoke'
EXTRA_CODE = ('Experiments/r1_bestckpt_valpreds.py', 'Experiments/r1_baseline_valpreds.py', 'Experiments/r1_mscnn.py',
              'src/mscnn_li2019.py')


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_exp19_both(model_name, seed, cfg, device):
    """vp.train_exp19_final with the best-epoch state kept as well (exp19: unrounded MAE, first minimum)"""
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
    val_history, best, best_state, best_epoch = [], float('inf'), None, 0
    for epoch in range(1, config['epochs'] + 1):
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
        vpred, vt = [], []
        with torch.no_grad():
            for x, y, *_ in val_loader:
                pred, *_r = model(x.to(device))
                vpred.append(pred.cpu().numpy())
                vt.append(y.numpy())
        mae = float(regression_metrics(np.concatenate(vpred), np.concatenate(vt))['mae'])
        val_history.append(round(mae, 5))
        if mae < best:
            best, best_state, best_epoch = mae, snapshot(model), epoch
    return model, best_state, best_epoch, val_history, config


def train_mscnn_both(seed, epochs, device, val_bearing, input_norm):
    """rm.train with the best-epoch state kept as well (first minimum of the validation MAE)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    bearings = rm.with_history(load_bearings(str(rm.DATA), CONFIG['train_bearings']))
    tb = [b for b in bearings if b.name != val_bearing]
    vb = [b for b in bearings if b.name == val_bearing]
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        WindowDataset(tb, [np.arange(b.n) for b in tb]), batch_size=rm.BATCH, shuffle=True,
        generator=gen, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        WindowDataset(vb, [np.arange(b.n) for b in vb]), batch_size=256, shuffle=False, num_workers=0)
    model = BaselineAdapter(MSCNN(n_hist=rm.N_HIST, input_norm=input_norm)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rm.LR, weight_decay=0.0)
    l1 = nn.L1Loss()
    val_history, best, best_state, best_epoch = [], float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, *_r = model(x)
            loss = l1(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        vpred, vt = rm.predict(model, val_loader, device)
        val_history.append(float(np.mean(np.abs(vpred - vt))))
        if val_history[-1] < best:
            best, best_state, best_epoch = val_history[-1], snapshot(model), epoch
    return model, best_state, best_epoch, val_history


def arrays_for(a, b):
    return {f'{tag}_{b}': a[key][b] for key, tag in (('raw_p', 'raw'), ('tgt', 'tgt'), ('mc_mean', 'mcmean'), ('mc_std', 'mcstd'))}


def one_run(model_name, fold, seed, k, device, passes):
    val_bearing = vp.FOLDS[fold]
    if model_name == 'mscnn':
        input_norm = json.loads((ROOT / 'outputs' / 'r1_mscnn' / 'selection.json').read_text(encoding='utf-8'))['input_norm']
        model, best_state, best_epoch, val_history = train_mscnn_both(seed, k, device, val_bearing, input_norm)
        vb = rm.with_history(load_bearings(str(rm.DATA), [val_bearing]))
        cfg = dict(CONFIG, mc_dropout_passes=passes)
    else:
        cfg = vp.selection_config(model_name, val_bearing, k)
        cfg['mc_dropout_passes'] = passes
        model, best_state, best_epoch, val_history, cfg = train_exp19_both(model_name, seed, cfg, device)
        vb = load_bearings(cfg['data_dir'], [val_bearing], cfg.get('channels', 'h'))
    final = arrays_for(evaluate_on_test(model, vb, cfg, device)['_arrays'], val_bearing)      # first, as in V
    model.load_state_dict({kk: v.to(device) for kk, v in best_state.items()})
    best = arrays_for(evaluate_on_test(model, vb, cfg, device)['_arrays'], val_bearing)
    return final, best, best_epoch, val_history, val_bearing


def mae_of(a, b):
    return float(np.mean(np.abs(a[f'raw_{b}'].astype(float) - a[f'tgt_{b}'].astype(float))))


def run(model_name, device, out_root, passes, runs, k=None):
    k = k or vp.selected_k(model_name)
    out_dir = out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    state_f = out_dir / 'runs.json'
    state = json.loads(state_f.read_text()) if state_f.exists() else []
    for fold, seed in runs:
        ff, fb = out_dir / f'fold{fold}_seed{seed}_final.npz', out_dir / f'fold{fold}_seed{seed}_best.npz'
        if ff.exists() and fb.exists() and any((r['fold'], r['seed']) == (fold, seed) for r in state):
            continue
        t0 = time.time()
        final, best, best_epoch, val_history, b = one_run(model_name, fold, seed, k, device, passes)
        np.savez_compressed(ff, **final)
        np.savez_compressed(fb, **best)
        v = np.load(ROOT / 'outputs' / 'r1_baseline_valpreds' / model_name / f'fold{fold}_seed{seed}.npz')
        final_diff = float(np.max(np.abs(final[f'raw_{b}'].astype(float) - v[f'raw_{b}'].astype(float))))
        mc_diff = float(np.max(np.abs(final[f'mcmean_{b}'].astype(float) - v[f'mcmean_{b}'].astype(float))))
        targets_equal = bool(np.array_equal(final[f'tgt_{b}'], v[f'tgt_{b}']))
        best_check = abs(mae_of(best, b) - float(min(val_history)))
        state = [r for r in state if (r['fold'], r['seed']) != (fold, seed)]
        state.append({'fold': fold, 'seed': seed, 'best_epoch': best_epoch, 'epochs': len(val_history),
                      'best_is_final': best_epoch == len(val_history), 'val_history': val_history,
                      'final_raw_diff_vs_V': final_diff, 'final_mcmean_diff_vs_V': mc_diff, 'targets_equal': targets_equal,
                      'best_mae': mae_of(best, b), 'final_mae': mae_of(final, b), 'best_mae_vs_history_min': best_check,
                      'minutes': round((time.time() - t0) / 60, 2)})
        state_f.write_text(json.dumps(state, indent=1))
        print(f"{model_name} fold {fold} seed {seed}: best epoch {best_epoch} of {len(val_history)}, validation MAE best "
              f"{state[-1]['best_mae']:.5f} / final {state[-1]['final_mae']:.5f}; final vs V {final_diff:.1e} "
              f"({state[-1]['minutes']} min)", flush=True)
        if final_diff > TOL_FINAL or not targets_equal:
            raise SystemExit(f'B7 {model_name} fold {fold} seed {seed}: final-epoch predictions do not reproduce V; '
                             f'stopping (Amendment 7)')
        if best_check > TOL_BEST:
            raise SystemExit(f'B7 {model_name} fold {fold} seed {seed}: best-epoch MAE {mae_of(best, b):.6f} differs from '
                             f'the minimum of the validation history {min(val_history):.6f}; stopping (Amendment 7)')
    meta = {'stage': 'B7', 'model': model_name, 'K': k, 'mc_dropout_passes': passes, 'n_runs': len(state),
            'max_final_raw_diff_vs_V': max(r['final_raw_diff_vs_V'] for r in state),
            'max_final_mcmean_diff_vs_V': max(r['final_mcmean_diff_vs_V'] for r in state),
            'max_best_mae_vs_history_min': max(r['best_mae_vs_history_min'] for r in state),
            'share_best_is_final': float(np.mean([r['best_is_final'] for r in state])),
            'mean_best_epoch': float(np.mean([r['best_epoch'] for r in state])),
            'date': time.strftime('%Y-%m-%d %H:%M'), 'provenance': provenance(EXTRA_CODE)}
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(f"{model_name}: B7 complete ({len(state)} runs; best epoch = final epoch in {100 * meta['share_best_is_final']:.0f} %; "
          f"final vs V max |diff| {meta['max_final_raw_diff_vs_V']:.1e})", flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', help='one of ' + ', '.join(MODELS) + ', or all')
    ap.add_argument('--smoke', action='store_true', help='1D CNN, fold A, seed 7, full length, 2 passes, separate folder')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        run('cnn', device, SMOKE_OUT, 2, [('A', 7)])
        print('Amendment 7 part B smoke test passed', flush=True)
        return 0
    if args.model not in MODELS + ['all']:
        raise SystemExit(f'--model must be one of {MODELS} or all')
    for m in (MODELS if args.model == 'all' else [args.model]):
        run(m, device, OUT, MC_PASSES, [(f, s) for f in vp.FOLDS for s in vp.SEEDS])
    return 0


if __name__ == '__main__':
    sys.exit(main())
