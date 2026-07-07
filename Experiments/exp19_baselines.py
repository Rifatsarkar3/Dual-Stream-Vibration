"""
exp19_baselines.py — Baselines on the Identical Leak-Free Protocol
===================================================================

Trains five published-architecture baselines on EXACTLY the same protocol as
the proposed model (exp16): same data split, same two-stage model selection
(epoch budget K chosen on 2-fold leave-one-bearing-out validation, then a
fixed-K retrain on all four training bearings), same five seeds, same one-shot
test evaluation. No PGMC, no DANN, no fusion gate — plain L1 regression.

Baselines:
  cnn          Vanilla1DCNN          (~0.5 M)
  bilstm       VanillaBiLSTM         (~0.8 M)
  cnnlstm      CNNLSTMBaseline
  transformer  VanillaTransformerBaseline
  resnet       DeepResNet1DBaseline

RUN (one baseline at a time; 'auto' stage does select->final automatically):
  .venv\\Scripts\\python.exe Experiments/exp19_baselines.py --model cnn
  .venv\\Scripts\\python.exe Experiments/exp19_baselines.py --model all
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from Experiments.exp16_ress import (  # noqa: E402
    CONFIG as BASE_CONFIG, build_loaders, evaluate_on_test,
    regression_metrics, aggregate)
from src.vanilla_baselines import Vanilla1DCNN, VanillaBiLSTM  # noqa: E402
from src.sota_baselines import (  # noqa: E402
    CNNLSTMBaseline, VanillaTransformerBaseline, DeepResNet1DBaseline)

MODELS = {
    'cnn': Vanilla1DCNN,
    'bilstm': VanillaBiLSTM,
    'cnnlstm': CNNLSTMBaseline,
    'transformer': VanillaTransformerBaseline,
    'resnet': DeepResNet1DBaseline,
}

SELECT_SEEDS = [42, 123]          # budget selection needs fewer seeds
FOLDS = {
    'foldA': ['train_02_Bearing1_2'],
    'foldB': ['train_04_Bearing2_2'],
}


class BaselineAdapter(nn.Module):
    """Normalizes baseline outputs to the exp16 5-tuple contract.

    needs_spectrogram: the transformer baseline operates on STFT images; we
    compute a (3, 64, 64) log-magnitude STFT from the raw window on the fly so
    the architecture matches its reference design without a separate pipeline.
    """

    def __init__(self, net: nn.Module, needs_spectrogram: bool = False):
        super().__init__()
        self.net = net
        self.needs_spectrogram = needs_spectrogram

    @staticmethod
    def _stft_image(x: torch.Tensor) -> torch.Tensor:
        sig = x.squeeze(1)                                     # (B, 2560)
        spec = torch.stft(sig, n_fft=126, hop_length=40,
                          window=torch.hann_window(126, device=x.device),
                          return_complex=True)                  # (B, 64, 65)
        mag = torch.log1p(spec.abs())
        mag = (mag - mag.amin(dim=(1, 2), keepdim=True)) / (
            mag.amax(dim=(1, 2), keepdim=True)
            - mag.amin(dim=(1, 2), keepdim=True) + 1e-8)
        img = torch.nn.functional.interpolate(
            mag.unsqueeze(1), size=(64, 64), mode='bilinear',
            align_corners=False)                                # (B, 1, 64, 64)
        return img.repeat(1, 3, 1, 1)                           # (B, 3, 64, 64)

    def forward(self, x):
        if self.needs_spectrogram:
            out = self.net(x, self._stft_image(x))
        else:
            out = self.net(x)
        pred = out[0] if isinstance(out, tuple) else out
        pred = pred.reshape(-1)
        alpha = torch.zeros(pred.shape[0], device=pred.device)
        return pred, alpha, None, None, None


def train_baseline(model_name: str, seed: int, config: Dict, device,
                   verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader, val_loader, _seq, test_bearings = build_loaders(config, seed)
    model = BaselineAdapter(MODELS[model_name](),
                            needs_spectrogram=(model_name == 'transformer')).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6)
    l1 = nn.L1Loss()

    best_val_mae, best_state, best_epoch, patience_cnt = float('inf'), None, 0, 0
    val_history = []
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, *_r = model(x)
            loss = l1(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        if val_loader is None:
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            continue

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, y, *_ in val_loader:
                pred, *_r = model(x.to(device))
                vp.append(pred.cpu().numpy())
                vt.append(y.numpy())
        vm = regression_metrics(np.concatenate(vp), np.concatenate(vt))
        val_history.append(round(float(vm['mae']), 5))
        if verbose and (epoch == 1 or epoch % 20 == 0):
            print(f"    epoch {epoch:3d} val MAE={vm['mae']:.4f}", flush=True)
        if vm['mae'] < best_val_mae:
            best_val_mae, best_epoch, patience_cnt = vm['mae'], epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    result = evaluate_on_test(model, test_bearings, config, device)
    result.update({'seed': seed, 'best_epoch': int(best_epoch),
                   'best_val_mae': (float(best_val_mae)
                                    if np.isfinite(best_val_mae) else None),
                   'n_params': int(n_params),
                   'val_history': val_history})
    return result, model


def run_model(model_name: str, out_root: Path, device, base_config: Dict,
              seeds=None, fixed_k: int = None):
    print(f"\n######## {model_name} ########", flush=True)
    out_dir = out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or base_config['seeds']

    # ---- Stage 1: budget selection (identical to proposed-model protocol) ----
    # Candidate budgets trained TO COMPLETION on 2-fold LOBO; select the budget
    # with the lowest mean final-epoch validation MAE. Validation data only.
    budget_scores = {}
    if fixed_k is not None:
        K = fixed_k
        print(f"  [select] using fixed K={K}", flush=True)
    else:
        budgets = [5, 10, 20, 40]
        for B in budgets:
            finals = []
            for fold, val_bearings in FOLDS.items():
                for seed in SELECT_SEEDS:
                    cfg = dict(base_config)
                    cfg.update({'val_mode': 'bearing', 'val_bearings': val_bearings,
                                'epochs': B, 'patience': 10 ** 6,
                                'mc_dropout_passes': 5})
                    res, _m = train_baseline(model_name, seed, cfg, device,
                                             verbose=False)
                    finals.append(res['val_history'][-1])
            budget_scores[B] = float(np.mean(finals))
            print(f"  [select] budget {B}: mean final val MAE = "
                  f"{budget_scores[B]:.4f}", flush=True)
        K = min(budget_scores, key=budget_scores.get)
        print(f"  [select] -> budget K={K}", flush=True)

    # ---- Stage 2: fixed-K retrain on all training bearings ----
    per_seed = []
    for seed in seeds:
        cfg = dict(base_config)
        cfg.update({'val_mode': 'none', 'epochs': K, 'mc_dropout_passes': 10})
        print(f"  [final] seed {seed} (K={K})", flush=True)
        res, model = train_baseline(model_name, seed, cfg, device, verbose=False)
        res.pop('_arrays', None)
        print(f"    raw MAE={res['raw']['mae']:.4f} R2={res['raw']['r2']:.4f} "
              f"pava MAE={res['pava']['mae']:.4f}", flush=True)
        torch.save(model.state_dict(), out_dir / f'model_seed{seed}.pth')
        per_seed.append(res)

    final = {
        'experiment': 'exp19_baselines', 'model': model_name,
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'epoch_budget_K': K, 'budget_scores': budget_scores,
        'n_params': per_seed[0]['n_params'],
        'per_seed': per_seed, 'aggregated': aggregate(per_seed),
    }
    with open(out_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2)
    a = final['aggregated']
    print(f"  ==> {model_name}: raw MAE={a['raw_mae_mean']:.4f}±{a['raw_mae_std']:.4f} "
          f"pava MAE={a['pava_mae_mean']:.4f}±{a['pava_mae_std']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=list(MODELS) + ['all'])
    ap.add_argument('--out-root', default='outputs/exp19_baselines')
    ap.add_argument('--dataset', choices=['pronostia', 'xjtu'], default='pronostia')
    ap.add_argument('--seeds', type=str, default=None)
    ap.add_argument('--fixed-k', type=int, default=None,
                    help='skip the budget sweep and use this K')
    args = ap.parse_args()

    base_config = dict(BASE_CONFIG)
    if args.dataset == 'xjtu':
        from Experiments.exp17_xjtu import XJTU_CONFIG
        base_config = dict(XJTU_CONFIG)
    seeds = ([int(s) for s in args.seeds.split(',')] if args.seeds else None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names = list(MODELS) if args.model == 'all' else [args.model]
    for name in names:
        run_model(name, Path(args.out_root), device, base_config,
                  seeds=seeds, fixed_k=args.fixed_k)
    return 0


if __name__ == '__main__':
    sys.exit(main())
