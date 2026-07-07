"""
exp16_ress.py — Leak-Free DSAFLite+PGMC Pipeline for RESS Submission
=====================================================================

WHY THIS EXPERIMENT EXISTS (replaces exp15_recovery results):
  Audit of the previous pipeline found two methodology defects that make the
  published numbers indefensible at a Q1 venue:

  DEFECT 1 — Test-set leakage via model selection:
    `create_dataloaders` returned the TEST bearings as `val_loader`, so early
    stopping selected the checkpoint with the best TEST MAE. The manuscript
    claims a 10% hold-out from training bearings; the code did not implement it.

  DEFECT 2 — Degenerate DANN:
    The training loop only ever fed SOURCE samples with domain label 0 to the
    discriminator. The target loader was constructed but never consumed, so the
    "domain-adversarial" loss was a constant-label BCE — a no-op.

  THIS SCRIPT fixes both:
    * Validation = every 10th window of each TRAINING bearing (deterministic,
      spans the full degradation trajectory). Test bearings are touched exactly
      once, at final evaluation.
    * DANN = real mixed-domain adversarial training. Domain label is derived
      from the operating condition of each TRAINING bearing
      (Bearing1_* = Condition 1 -> 0, Bearing2_* = Condition 2 -> 1).
      Backbone receives the GRL-reversed domain loss on every sample; the
      discriminator is trained on detached features with true labels.
      No test data is used during training in any form.

  ADDITIONS for RESS:
    * MC-dropout uncertainty quantification at evaluation (30 stochastic
      passes; PICP / MPIW / Gaussian NLL) — no retraining required.
    * PGMC 'sequential' mode option using within-bearing consecutive windows
      (chunk-based batches), matching the manuscript's Eq. for L_PGMC.
    * Per-window fusion-gate alphas saved for the noise-robustness study.

HOW TO RUN:
  cd "E:\\Yolo-Thermal\\Dual-Stream Vibration-Vision"
  .venv\\Scripts\\python.exe Experiments/exp16_ress.py --smoke            # 10-epoch smoke test, seed 42
  .venv\\Scripts\\python.exe Experiments/exp16_ress.py --seed 42          # one full seed
  .venv\\Scripts\\python.exe Experiments/exp16_ress.py --all-seeds        # manuscript statistics
  Ablations: --no-dann | --no-pgmc | --static-gate | --pgmc-mode soft

OUTPUTS: outputs/exp16_ress/<variant>/
  results.json, model_seed{S}.pth, preds_seed{S}.npz, summary printed.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dsaf_v2_lite import DSAFLite
from src.monotone_rul_head import upgrade_dsaflite_regressor
from src.mini_dann import DANNModule
from src.online_filters import RealTimeRunningMinimum, ConstrainedExponentialSmoothing
from src.isotonic_eval import IsotonicRegressor, compute_monotonicity


# -----------------------------------------------------------------------------
# Configuration (hyperparameters identical to the manuscript, Section 4.3)
# -----------------------------------------------------------------------------

CONFIG = {
    'data_dir': str(Path(__file__).parent.parent / 'Processed_PRONOSTIA'),
    'train_bearings': {  # bearing name -> condition label (0 = Cond1, 1 = Cond2)
        'train_01_Bearing1_1': 0,
        'train_02_Bearing1_2': 0,
        'train_03_Bearing2_1': 1,
        'train_04_Bearing2_2': 1,
    },
    'test_bearings': [
        'test_01_Bearing1_3', 'test_02_Bearing1_4',
        'test_03_Bearing1_5', 'test_04_Bearing2_3',
    ],
    'val_every_k': 10,        # every k-th window of each training bearing -> validation
    # Validation strategy (all leak-free; test bearings never touched):
    #  'window' : every k-th window of each training bearing (in-bearing val)
    #  'bearing': hold out whole training bearings (cross-bearing val measures
    #             the generalization the test actually demands)
    #  'none'   : no validation / no early stopping; train exactly `epochs`
    #             (use after selecting the epoch budget with 'bearing' mode)
    'val_mode': 'window',
    'val_bearings': ['train_02_Bearing1_2', 'train_04_Bearing2_2'],

    'feat_dim': 128,
    'dropout': 0.2,

    'batch_size': 32,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'epochs': 120,
    'patience': 30,

    'lambda_pgmc_max': 0.08,
    'pgmc_warmup_epochs': 10,
    # 'aux-seq': random batches for regression + auxiliary within-bearing sequential
    #            chunk pass for the PGMC penalty (matches manuscript Eq. exactly).
    # 'soft':    random batches, target-sorted soft prior (legacy behaviour).
    # 'chunk':   chunk-only batches (slow convergence; kept for reference).
    'pgmc_mode': 'aux-seq',
    'pgmc_chunk_len': 32,

    'lambda_dann_max': 0.08,
    'dann_gamma': 10.0,
    'dann_warmup_epochs': 20,

    'mc_dropout_passes': 30,
    'ces_alpha': 0.2,
    'ces_epsilon': 0.05,

    'seeds': [42, 123, 777, 999, 7],
    'output_root': 'outputs/exp16_ress',
}


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

class BearingArrays:
    """All windows of one bearing, in chronological order."""

    def __init__(self, name: str, vib: np.ndarray, rul: np.ndarray, condition: int):
        assert len(vib) == len(rul)
        self.name = name
        self.vib = vib.astype(np.float32)          # (N, 2560)
        self.rul = rul.astype(np.float32)          # (N,)
        self.condition = condition
        self.n = len(rul)


def load_bearings(data_dir: str, patterns: Dict[str, int] | List[str]) -> List[BearingArrays]:
    data_dir = Path(data_dir)
    if isinstance(patterns, dict):
        items = patterns.items()
    else:
        items = [(p, -1) for p in patterns]
    bearings = []
    for pattern, cond in items:
        files = sorted(data_dir.glob(f'{pattern}*.npz'))
        if not files:
            raise FileNotFoundError(f'No npz for pattern {pattern} in {data_dir}')
        for f in files:
            data = np.load(f, allow_pickle=True)
            bearings.append(BearingArrays(pattern, data['vibration'], data['rul'], cond))
    return bearings


class WindowDataset(torch.utils.data.Dataset):
    """Flat view over selected (bearing, window) pairs with identity tracking."""

    def __init__(self, bearings: List[BearingArrays], index_lists: List[np.ndarray]):
        self.bearings = bearings
        self.entries = []  # (bearing_idx, window_idx)
        for b_idx, idxs in enumerate(index_lists):
            for w in idxs:
                self.entries.append((b_idx, int(w)))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        b_idx, w = self.entries[i]
        b = self.bearings[b_idx]
        x = torch.from_numpy(b.vib[w]).unsqueeze(0)              # (1, 2560)
        y = torch.tensor(b.rul[w], dtype=torch.float32)
        return x, y, b_idx, w, b.condition


class ChunkBatchSampler(torch.utils.data.Sampler):
    """Each batch = `batch_size` consecutive windows of a single bearing.

    Enables the true within-bearing sequential PGMC constraint. Chunk order is
    reshuffled every epoch; the partial tail chunk of each bearing is kept if
    it has at least 4 windows (PGMC needs >=2 consecutive pairs).
    """

    def __init__(self, dataset: WindowDataset, batch_size: int, generator: torch.Generator):
        self.batch_size = batch_size
        self.generator = generator
        # Build chunks of positions (indices into dataset.entries).
        by_bearing: Dict[int, List[int]] = {}
        for pos, (b_idx, w) in enumerate(dataset.entries):
            by_bearing.setdefault(b_idx, []).append(pos)
        # entries were appended in chronological order per bearing
        self.chunks = []
        for b_idx, positions in by_bearing.items():
            for s in range(0, len(positions), batch_size):
                chunk = positions[s:s + batch_size]
                if len(chunk) >= 4:
                    self.chunks.append(chunk)

    def __iter__(self):
        order = torch.randperm(len(self.chunks), generator=self.generator).tolist()
        for i in order:
            yield self.chunks[i]

    def __len__(self):
        return len(self.chunks)


def build_loaders(config: Dict, seed: int) -> Tuple[torch.utils.data.DataLoader,
                                                    torch.utils.data.DataLoader,
                                                    List[BearingArrays]]:
    """Train / val loaders from TRAINING bearings only; test bearings returned raw."""
    train_bearings = load_bearings(config['data_dir'], config['train_bearings'])
    test_bearings = load_bearings(config['data_dir'], config['test_bearings'])

    mode = config.get('val_mode', 'window')
    if mode == 'bearing':
        val_names = set(config['val_bearings'])
        tb = [b for b in train_bearings if b.name not in val_names]
        vb = [b for b in train_bearings if b.name in val_names]
        assert tb and vb, 'bearing val mode needs non-empty train and val bearing sets'
        train_ds = WindowDataset(tb, [np.arange(b.n) for b in tb])
        val_ds = WindowDataset(vb, [np.arange(b.n) for b in vb])
    elif mode == 'none':
        train_ds = WindowDataset(train_bearings,
                                 [np.arange(b.n) for b in train_bearings])
        val_ds = None
    else:  # 'window'
        k = config['val_every_k']
        train_idx, val_idx = [], []
        for b in train_bearings:
            all_idx = np.arange(b.n)
            val_mask = (all_idx % k) == (k - 1)  # deterministic, spans whole life
            train_idx.append(all_idx[~val_mask])
            val_idx.append(all_idx[val_mask])
        train_ds = WindowDataset(train_bearings, train_idx)
        val_ds = WindowDataset(train_bearings, val_idx)

    gen = torch.Generator()
    gen.manual_seed(seed)

    if config['pgmc_mode'] == 'chunk':
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_sampler=ChunkBatchSampler(train_ds, config['batch_size'], gen),
            num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config['batch_size'], shuffle=True,
            generator=gen, num_workers=0, drop_last=True)

    val_loader = (torch.utils.data.DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=0)
        if val_ds is not None else None)

    # Auxiliary sequential-chunk loader for the aux-seq PGMC penalty
    seq_loader = None
    if config['pgmc_mode'] == 'aux-seq':
        seq_loader = torch.utils.data.DataLoader(
            train_ds, batch_sampler=ChunkBatchSampler(
                train_ds, config['pgmc_chunk_len'], gen),
            num_workers=0)

    return train_loader, val_loader, seq_loader, test_bearings


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def pooled_eval(per_bearing_preds: Dict[str, np.ndarray],
                per_bearing_targets: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Pooled MAE/RMSE/R2 + per-bearing-pair monotonicity (matches prior protocol)."""
    all_p = np.concatenate(list(per_bearing_preds.values()))
    all_t = np.concatenate(list(per_bearing_targets.values()))
    m = regression_metrics(all_p, all_t)
    dec, tot = 0, 0
    for p in per_bearing_preds.values():
        d = np.diff(p)
        dec += int((d <= 0).sum())
        tot += len(d)
    m['mono'] = 100.0 * dec / max(tot, 1)
    return m


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

class FixedGate(nn.Module):
    """Constant-alpha fusion: alpha=0.5 (static), 1.0 (temporal-only), 0.0 (spectral-only)."""

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, fa, fb):
        fused = self.alpha * fa + (1.0 - self.alpha) * fb
        return fused, torch.full((fa.shape[0],), self.alpha, device=fa.device)


def make_model(config: Dict, device, static_gate: bool = False,
               stream: str = 'dual') -> nn.Module:
    model = DSAFLite(feat_dim=config['feat_dim'], dropout=config['dropout'])
    upgrade_dsaflite_regressor(model, feat_dim=config['feat_dim'], dropout=config['dropout'])
    if stream == 'temporal':
        model.fusion = FixedGate(1.0)
    elif stream == 'spectral':
        model.fusion = FixedGate(0.0)
    elif static_gate:
        model.fusion = FixedGate(0.5)
    return model.to(device)


class _FrozenBN:
    """Context manager: switch BatchNorm layers to eval so an auxiliary forward
    pass does not contaminate running statistics (gradients still flow)."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.bns = [m for m in model.modules()
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]

    def __enter__(self):
        self.prev = [bn.training for bn in self.bns]
        for bn in self.bns:
            bn.eval()

    def __exit__(self, *exc):
        for bn, prev in zip(self.bns, self.prev):
            bn.train(prev)


def sequential_pgmc_penalty(model: nn.Module, seq_iter, seq_loader, device):
    """Manuscript Eq.: L_PGMC = mean over consecutive within-bearing pairs of
    max(0, y_hat[t+1] - y_hat[t])^2, computed on a chronological chunk of one
    training bearing. Returns (penalty, refreshed iterator)."""
    try:
        batch = next(seq_iter)
    except StopIteration:
        seq_iter = iter(seq_loader)
        batch = next(seq_iter)
    x, _y, _b, w_idx, _c = batch
    x = x.to(device)
    with _FrozenBN(model):
        pred, *_ = model(x)
    # Chunks are chronological within a single bearing; penalize any later
    # prediction exceeding an earlier adjacent one (squared hinge).
    order = torch.argsort(w_idx.to(device))
    p = pred[order]
    diffs = p[1:] - p[:-1]
    return torch.relu(diffs).pow(2).mean(), seq_iter


def train_one_seed(seed: int, config: Dict, use_dann: bool, use_pgmc: bool,
                   static_gate: bool, epochs_override: Optional[int] = None,
                   stream: str = 'dual', verbose: bool = True) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = epochs_override or config['epochs']
    train_loader, val_loader, seq_loader, test_bearings = build_loaders(config, seed)
    seq_iter = iter(seq_loader) if seq_loader is not None else None

    model = make_model(config, device, static_gate=static_gate, stream=stream)

    dann = None
    dann_opt = None
    if use_dann:
        dann = DANNModule(feat_dim=config['feat_dim'], gamma=config['dann_gamma']).to(device)
        dann_opt = torch.optim.Adam(dann.parameters(), lr=config['lr'] * 3, weight_decay=0.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    # scheduler_tmax lets a short fixed-K retrain reproduce the LR trajectory
    # under which the budget K was selected (cosine over the full 120 epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('scheduler_tmax') or epochs, eta_min=1e-6)

    l1 = nn.L1Loss()

    # Warmup schedules: absolute epochs by default (matches the 120-epoch
    # design); with proportional_warmup they scale with the actual budget so
    # short fixed-budget training still activates PGMC and DANN at the same
    # relative points (10/120 and 20/120 of the budget).
    pgmc_warm = config['pgmc_warmup_epochs']
    dann_warm = config['dann_warmup_epochs']
    if config.get('proportional_warmup'):
        pgmc_warm = max(2, round(epochs * 10 / 120))
        dann_warm = max(3, round(epochs * 20 / 120))

    def pgmc_lambda(epoch: int) -> float:
        if not use_pgmc:
            return 0.0
        ramp = min(1.0, epoch / pgmc_warm)
        return config['lambda_pgmc_max'] * ramp

    best_val_mae, best_state, patience_cnt, best_epoch = float('inf'), None, 0, 0
    history = {'val_mae': [], 'val_r2': []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        lam_pgmc = pgmc_lambda(epoch)
        dann_active = (use_dann and epoch > dann_warm)
        if dann is not None:
            # progress drives the GRL lambda schedule (0 after warmup start ramp)
            prog = max(0.0, (epoch - dann_warm) /
                       max(1, epochs - dann_warm))
            dann.set_progress(prog)

        model.train()
        if dann is not None:
            dann.train()

        for x, y, b_idx, w_idx, cond in train_loader:
            x, y = x.to(device), y.to(device)
            cond = cond.to(device).float()
            optimizer.zero_grad()

            pred, alpha, fused, fa, fb = model(x)
            total = l1(pred, y)

            if lam_pgmc > 0:
                mode = config['pgmc_mode']
                if mode == 'aux-seq':
                    pen, seq_iter = sequential_pgmc_penalty(
                        model, seq_iter, seq_loader, device)
                elif mode == 'chunk':
                    order = torch.argsort(w_idx.to(device))
                    p_sorted = pred[order]
                    pen = torch.relu(p_sorted[1:] - p_sorted[:-1]).pow(2).mean()
                else:  # 'soft' (legacy): target-sorted directional prior
                    order = torch.argsort(y, descending=True)
                    p_sorted = pred[order]
                    pen = torch.relu(p_sorted[1:] - p_sorted[:-1]).mean()
                total = total + lam_pgmc * pen

            if dann_active:
                # Backbone step: GRL-reversed domain loss on TRUE mixed labels
                domain_logit = dann(fused)
                d_loss = dann.domain_loss(domain_logit, cond)
                total = total + config['lambda_dann_max'] * d_loss

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if dann_active:
                # Discriminator step on detached features, true labels
                dann_opt.zero_grad()
                with torch.no_grad():
                    _, _, fused_d, _, _ = model(x)
                d_logit = dann.classifier(fused_d)
                d_loss_disc = dann.domain_loss(d_logit, cond)
                d_loss_disc.backward()
                dann_opt.step()

        scheduler.step()

        # ---- validation (training-bearing hold-out ONLY; never test data) ----
        if val_loader is None:
            # fixed-budget mode: keep the final state, no early stopping
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_mae = float('nan')
            if verbose and (epoch == 1 or epoch % 10 == 0):
                print(f"  epoch {epoch:3d}/{epochs}  (fixed budget, no val)", flush=True)
            continue

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for x, y, *_ in val_loader:
                pred, *_rest = model(x.to(device))
                vp.append(pred.cpu().numpy())
                vt.append(y.numpy())
        vp, vt = np.concatenate(vp), np.concatenate(vt)
        vm = regression_metrics(vp, vt)
        history['val_mae'].append(vm['mae'])
        history['val_r2'].append(vm['r2'])

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  epoch {epoch:3d}/{epochs}  val MAE={vm['mae']:.4f}  R2={vm['r2']:.4f}"
                  f"  (best {min(best_val_mae, vm['mae']):.4f})", flush=True)

        if vm['mae'] < best_val_mae:
            best_val_mae = vm['mae']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val MAE {best_val_mae:.4f})")
                break

    train_minutes = (time.time() - t0) / 60.0
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    result = evaluate_on_test(model, test_bearings, config, device)
    result.update({
        'seed': seed,
        'best_val_mae': (float(best_val_mae)
                         if np.isfinite(best_val_mae) else None),
        'best_epoch': int(best_epoch),
        'epochs_trained': epoch,
        'train_minutes': round(train_minutes, 1),
        'val_history': [round(float(v), 5) for v in history['val_mae']],
    })
    return result, model, history


# -----------------------------------------------------------------------------
# Final evaluation: raw / PAVA / RTRM / CES + MC-dropout UQ + gate alphas
# -----------------------------------------------------------------------------

def _enable_mc_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def evaluate_on_test(model: nn.Module, test_bearings: List[BearingArrays],
                     config: Dict, device) -> Dict:
    model.eval()
    raw_p, tgt, alphas = {}, {}, {}
    mc_mean, mc_std = {}, {}

    for b in test_bearings:
        preds, als = [], []
        with torch.no_grad():
            for s in range(0, b.n, 256):
                x = torch.from_numpy(b.vib[s:s + 256]).unsqueeze(1).to(device)
                p, a, *_ = model(x)
                preds.append(p.cpu().numpy())
                als.append(a.cpu().numpy())
        raw_p[b.name] = np.concatenate(preds)
        alphas[b.name] = np.concatenate(als)
        tgt[b.name] = b.rul.copy()

        # MC dropout (dropout layers stochastic, BN frozen in eval stats)
        model.eval()
        _enable_mc_dropout(model)
        mc = []
        with torch.no_grad():
            for _ in range(config['mc_dropout_passes']):
                ps = []
                for s in range(0, b.n, 256):
                    x = torch.from_numpy(b.vib[s:s + 256]).unsqueeze(1).to(device)
                    p, *_rest = model(x)
                    ps.append(p.cpu().numpy())
                mc.append(np.concatenate(ps))
        model.eval()
        mc = np.stack(mc)                      # (T_passes, N)
        mc_mean[b.name] = mc.mean(axis=0)
        mc_std[b.name] = mc.std(axis=0)

    # Post-processing variants
    pava_p, rtrm_p, ces_p = {}, {}, {}
    for name, p in raw_p.items():
        pava_p[name] = IsotonicRegressor(increasing=False).fit_transform(p)
        rtrm_p[name] = RealTimeRunningMinimum().process(p)
        ces_p[name] = ConstrainedExponentialSmoothing(
            alpha=config['ces_alpha'], epsilon=config['ces_epsilon']).process(p)

    out = {
        'raw': pooled_eval(raw_p, tgt),
        'pava': pooled_eval(pava_p, tgt),
        'rtrm': pooled_eval(rtrm_p, tgt),
        'ces': pooled_eval(ces_p, tgt),
    }

    # UQ metrics (90% Gaussian intervals from MC dropout)
    z = 1.6449
    all_mu = np.concatenate([mc_mean[n] for n in raw_p])
    all_sd = np.clip(np.concatenate([mc_std[n] for n in raw_p]), 1e-4, None)
    all_t = np.concatenate([tgt[n] for n in raw_p])
    lo, hi = all_mu - z * all_sd, all_mu + z * all_sd
    picp = float(np.mean((all_t >= lo) & (all_t <= hi)) * 100)
    mpiw = float(np.mean(hi - lo))
    nll = float(np.mean(0.5 * np.log(2 * np.pi * all_sd ** 2)
                        + (all_t - all_mu) ** 2 / (2 * all_sd ** 2)))
    out['uq'] = {'picp90': picp, 'mpiw90': mpiw, 'nll': nll,
                 'mc_mae': regression_metrics(all_mu, all_t)['mae']}

    out['per_bearing'] = {
        name: {
            'raw': regression_metrics(raw_p[name], tgt[name]),
            'pava_mae': regression_metrics(pava_p[name], tgt[name])['mae'],
            'mono_raw': compute_monotonicity(raw_p[name]),
            'alpha_mean': float(alphas[name].mean()),
            'alpha_std': float(alphas[name].std()),
            'n': int(len(tgt[name])),
        } for name in raw_p
    }
    out['_arrays'] = {'raw_p': raw_p, 'pava_p': pava_p, 'rtrm_p': rtrm_p,
                      'ces_p': ces_p, 'tgt': tgt, 'alphas': alphas,
                      'mc_mean': mc_mean, 'mc_std': mc_std}
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def aggregate(per_seed: List[Dict]) -> Dict:
    agg = {}
    for variant in ('raw', 'pava', 'rtrm', 'ces'):
        for metric in ('mae', 'rmse', 'r2', 'mono'):
            vals = [r[variant][metric] for r in per_seed]
            agg[f'{variant}_{metric}_mean'] = float(np.mean(vals))
            agg[f'{variant}_{metric}_std'] = float(np.std(vals))
    for metric in ('picp90', 'mpiw90', 'nll', 'mc_mae'):
        vals = [r['uq'][metric] for r in per_seed]
        agg[f'uq_{metric}_mean'] = float(np.mean(vals))
        agg[f'uq_{metric}_std'] = float(np.std(vals))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='10-epoch smoke test, seed 42')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--all-seeds', action='store_true')
    ap.add_argument('--no-dann', action='store_true')
    ap.add_argument('--no-pgmc', action='store_true')
    ap.add_argument('--static-gate', action='store_true')
    ap.add_argument('--stream', choices=['dual', 'temporal', 'spectral'], default='dual')
    ap.add_argument('--pgmc-mode', choices=['aux-seq', 'soft', 'chunk'], default=None)
    ap.add_argument('--val-mode', choices=['window', 'bearing', 'none'], default=None)
    ap.add_argument('--val-bearings', type=str, default=None,
                    help='comma-separated bearing patterns for --val-mode bearing')
    ap.add_argument('--epochs', type=int, default=None,
                    help='override epoch budget (use with --val-mode none)')
    ap.add_argument('--scheduler-tmax', type=int, default=None,
                    help='cosine T_max override (keep 120 for fixed-K retrains)')
    ap.add_argument('--proportional-warmup', action='store_true',
                    help='scale PGMC/DANN warmups with the epoch budget')
    ap.add_argument('--mc-passes', type=int, default=None,
                    help='MC-dropout passes (reduce for selection runs)')
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--seeds', type=str, default=None,
                    help='comma-separated seed list overriding the default 5')
    ap.add_argument('--lambda-pgmc', type=float, default=None,
                    help='override lambda_pgmc_max (for val-based strength sweep)')
    ap.add_argument('--variant-name', type=str, default=None)
    args = ap.parse_args()

    config = dict(CONFIG)
    if args.pgmc_mode:
        config['pgmc_mode'] = args.pgmc_mode
    if args.val_mode:
        config['val_mode'] = args.val_mode
    if args.val_bearings:
        config['val_bearings'] = [s.strip() for s in args.val_bearings.split(',')]
    if args.epochs:
        config['epochs'] = args.epochs
    if args.scheduler_tmax:
        config['scheduler_tmax'] = args.scheduler_tmax
    if args.proportional_warmup:
        config['proportional_warmup'] = True
    if args.mc_passes:
        config['mc_dropout_passes'] = args.mc_passes
    if args.patience:
        config['patience'] = args.patience
    if args.seeds:
        config['seeds'] = [int(s) for s in args.seeds.split(',')]
    if args.lambda_pgmc is not None:
        config['lambda_pgmc_max'] = args.lambda_pgmc

    use_dann = not args.no_dann
    use_pgmc = not args.no_pgmc
    static_gate = args.static_gate

    variant = args.variant_name or (
        'smoke' if args.smoke else
        ('full' if (use_dann and use_pgmc and not static_gate) else
         f"dann{int(use_dann)}_pgmc{int(use_pgmc)}_gate{int(not static_gate)}"))
    if config['pgmc_mode'] == 'soft' and not args.variant_name:
        variant += '_soft'

    out_dir = Path(config['output_root']) / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds, epochs_override = [42], 10
    elif args.all_seeds:
        seeds, epochs_override = config['seeds'], None
    else:
        seeds, epochs_override = [args.seed if args.seed is not None else 42], None

    print(f"[exp16] variant={variant} seeds={seeds} pgmc_mode={config['pgmc_mode']} "
          f"dann={use_dann} pgmc={use_pgmc} adaptive_gate={not static_gate}", flush=True)

    per_seed = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        result, model, history = train_one_seed(
            seed, config, use_dann, use_pgmc, static_gate,
            epochs_override=epochs_override, stream=args.stream)

        arrays = result.pop('_arrays')
        np.savez_compressed(
            out_dir / f'preds_seed{seed}.npz',
            **{f'raw_{n}': v for n, v in arrays['raw_p'].items()},
            **{f'tgt_{n}': v for n, v in arrays['tgt'].items()},
            **{f'alpha_{n}': v for n, v in arrays['alphas'].items()},
            **{f'mcmean_{n}': v for n, v in arrays['mc_mean'].items()},
            **{f'mcstd_{n}': v for n, v in arrays['mc_std'].items()})
        torch.save(model.state_dict(), out_dir / f'model_seed{seed}.pth')

        print(f"  TEST raw : MAE={result['raw']['mae']:.4f} R2={result['raw']['r2']:.4f} "
              f"Mono={result['raw']['mono']:.1f}%")
        print(f"  TEST pava: MAE={result['pava']['mae']:.4f} R2={result['pava']['r2']:.4f}")
        print(f"  TEST ces : MAE={result['ces']['mae']:.4f}  rtrm: MAE={result['rtrm']['mae']:.4f}")
        print(f"  UQ: PICP90={result['uq']['picp90']:.1f}% MPIW={result['uq']['mpiw90']:.3f} "
              f"NLL={result['uq']['nll']:.3f}", flush=True)
        per_seed.append(result)

    final = {
        'experiment': 'exp16_ress',
        'variant': variant,
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'config': {k: v for k, v in config.items() if k != 'train_bearings'},
        'protocol_notes': [
            'Validation = every 10th window of each training bearing (no test data).',
            'Early stopping monitors training-bearing validation MAE only.',
            'DANN: true mixed-domain adversarial training, Cond1 vs Cond2 train bearings.',
            'Test bearings evaluated once with the best-validation checkpoint.',
        ],
        'use_dann': use_dann, 'use_pgmc': use_pgmc, 'adaptive_gate': not static_gate,
        'per_seed': per_seed,
        'aggregated': aggregate(per_seed) if per_seed else {},
    }
    with open(out_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2)

    if len(per_seed) > 1:
        a = final['aggregated']
        print(f"\n=== AGGREGATE ({len(per_seed)} seeds) ===")
        print(f"  raw : MAE={a['raw_mae_mean']:.4f}±{a['raw_mae_std']:.4f} "
              f"R2={a['raw_r2_mean']:.4f}±{a['raw_r2_std']:.4f} Mono={a['raw_mono_mean']:.1f}%")
        print(f"  pava: MAE={a['pava_mae_mean']:.4f}±{a['pava_mae_std']:.4f} "
              f"R2={a['pava_r2_mean']:.4f}±{a['pava_r2_std']:.4f}")
        print(f"  ces : MAE={a['ces_mae_mean']:.4f}±{a['ces_mae_std']:.4f}")
        print(f"  UQ  : PICP90={a['uq_picp90_mean']:.1f}% MPIW={a['uq_mpiw90_mean']:.3f}")
    print(f"\n[exp16] results saved to {out_dir / 'results.json'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
