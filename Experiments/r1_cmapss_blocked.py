"""
r1_cmapss_blocked.py - Revision R1, Amendment 12 (plan Section 11): blocked in-unit calibration (SB)
on the C-MAPSS turbofan engines, built like the protocol (PR) of Amendment 11 but with folds of time
blocks inside every fleet engine instead of folds of engines.

SB: the W windows of each fleet engine are divided into four contiguous blocks (block k: indices
floor(kW/4) to floor((k+1)W/4) - 1). Fold k validates on block k of every fleet engine and trains on
the windows that share no cycle with the block (index at least 30 below its first or at least 30 above
its last; up to 29 buffer windows on each side are used by neither). Each fold is trained for 8,000
steps and predicts its blocks after 500, 1,000, 2,000, 4,000 and 8,000 steps; S*_SB is the budget with
the lowest four-fold mean block MAE (ties: the smaller); the deployed model is trained on the whole
fleet for S*_SB steps, as in PR; calibration on the fold networks' block errors after S*_SB steps.
Fold scaling: minimum and maximum over the cycles of the fold's training windows.
PR is not rerun (Amendment 11 outputs are reused by the analysis).

Usage:
  python Experiments/r1_cmapss_blocked.py --smoke      (FD001, n = 4, repeat 0, fold 1, 500 steps; test files not read)
  python Experiments/r1_cmapss_blocked.py --dataset FD001 [--repeats 0-19] [--fleets 4,10,25]
Outputs: outputs/r1_cmapss_blocked/<dataset>/n<NN>_r<RR>.npz and .json; smoke: outputs/r1_cmapss_blocked_smoke/.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Experiments'))
import r1_cmapss as C  # noqa: E402
from r1_provenance import provenance  # noqa: E402

OUT = ROOT / 'outputs' / 'r1_cmapss_blocked'
SMOKE_OUT = ROOT / 'outputs' / 'r1_cmapss_blocked_smoke'
FLEETS = [4, 10, 25]
N_BLOCKS = 4
CODE = ['Experiments/r1_cmapss.py', 'Experiments/r1_cmapss_blocked.py']


# ---------------------------------------------------------------- blocks ------------------------
def blocks(W, n_blocks=N_BLOCKS):
    """[start, end) of the contiguous blocks of an engine with W windows"""
    edges = [(k * W) // n_blocks for k in range(n_blocks + 1)]
    return [(edges[k], edges[k + 1]) for k in range(n_blocks)]


def block_split(W, k):
    """(training indices, validation indices) of fold k: validation = block k; training = windows
    sharing no cycle with any window of the block (window i covers cycles i + 1 to i + 30)"""
    start, end = blocks(W)[k]
    i = np.arange(W)
    val = i[(i >= start) & (i < end)]
    if len(val) == 0:
        return i, val
    train = i[(i <= start - C.WINDOW) | (i >= end - 1 + C.WINDOW)]
    return train, val


def n_windows(engines, u):
    return max(len(engines[int(u)]) - C.WINDOW + 1, 0)


def sb_specs(engines, units):
    """fold k -> (training spec, validation spec); a spec is a list of (unit, window indices)"""
    specs = []
    for k in range(N_BLOCKS):
        tr, va = [], []
        for u in units:
            t_idx, v_idx = block_split(n_windows(engines, u), k)
            tr.append((int(u), t_idx))
            va.append((int(u), v_idx))
        specs.append((tr, va))
    return specs


def engine_specs(engines, units):
    """the folds of run_pr (engine at position j in fold j mod 4) in the same spec form, same order"""
    parts = C.folds(units)
    specs = []
    for k in range(C.N_FOLDS):
        tr = np.concatenate([parts[j] for j in range(C.N_FOLDS) if j != k])
        specs.append(([(int(u), np.arange(n_windows(engines, u))) for u in tr],
                      [(int(u), np.arange(n_windows(engines, u))) for u in parts[k]]))
    return specs


# ---------------------------------------------------------------- cross-validation --------------
def windows_of(engines, spec):
    """unscaled windows, labels, units and window indices of a spec, in spec order"""
    xs, ys, us, idx = [], [], [], []
    for u, ii in spec:
        x = engines[int(u)]
        ii = np.asarray(ii, dtype=int)
        xs.append(C.windows(x)[ii])
        ys.append(C.train_labels(len(x))[ii])
        us.append(np.full(len(ii), int(u)))
        idx.append(ii)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(us), np.concatenate(idx)


def scaler_from_windows(w):
    """minimum and maximum over the cycles contained in the windows"""
    flat = w.reshape(-1, w.shape[-1])
    return flat.min(0), flat.max(0)


def run_cv(engines, fold_specs, seed, device, budgets=C.BUDGETS):
    """each fold: scaler on its training windows' cycles, network trained max(budgets) steps (seed and
    procedure as in run_pr), predictions on the validation windows after each budget"""
    held = {}
    for k, (tr_spec, va_spec) in enumerate(fold_specs):
        Xtr_raw, ytr, _, _ = windows_of(engines, tr_spec)
        lo, hi = scaler_from_windows(Xtr_raw)
        Xva_raw, yva, uva, iva = windows_of(engines, va_spec)
        X = C.scale(Xtr_raw, lo, hi).astype(np.float32)
        Xv = C.scale(Xva_raw, lo, hi).astype(np.float32)
        Xt, yt_, Xvt = C.tensors(device, X, ytr, Xv)
        model = C.make_model(seed, device)
        preds = {}

        def callback(step, model=model, Xvt=Xvt, preds=preds):
            if step in budgets:
                preds[step] = C.predict(model, Xvt)
            return True

        C.train_steps(model, Xt, yt_, seed, max(budgets), callback)
        held[k] = {'pred': np.stack([preds[s] for s in budgets]), 'y': yva, 'unit': uva, 'index': iva,
                   'train_label_mean': float(ytr.mean()), 'n_train': int(len(ytr)), 'n_val': int(len(yva))}
    mae = np.array([np.mean([np.mean(np.abs(held[k]['pred'][b] - held[k]['y'])) for k in held])
                    for b in range(len(budgets))])
    b_star = int(np.argmin(mae))                         # first minimum: the smaller budget on ties
    return held, mae, b_star


def deploy(engines, units, seed, device, steps):
    """the deployed model of PR: whole fleet, fleet scaler, `steps` optimizer steps, final state"""
    lo, hi = C.fit_scaler(engines, units)
    X, y, _, _ = C.make_set(engines, units, lo, hi, lambda unit, T: C.train_labels(T))
    Xt, yt_ = C.tensors(device, X, y)
    model = C.make_model(seed, device)
    C.train_steps(model, Xt, yt_, seed, steps)
    return model, lo, hi


# ---------------------------------------------------------------- cells -------------------------
def run_cell(ds, engines, test, n, r, device, out_dir):
    path = out_dir / f'n{n:02d}_r{r:02d}'
    if path.with_suffix('.json').exists():
        print(f'{ds} n={n} r={r}: exists, skipped', flush=True)
        return
    t0 = time.time()
    seed, units = C.FINAL_SEEDS[r], C.fleet(r, n)
    held, mae, b_star = run_cv(engines, sb_specs(engines, units), seed, device)
    s_star = C.BUDGETS[b_star]
    cal_e = np.concatenate([held[k]['pred'][b_star] - held[k]['y'] for k in held])
    model, lo, hi = deploy(engines, units, seed, device, s_star)
    test_eng, test_rul = test
    Xte, yte, ute, ite = C.make_set(test_eng, sorted(test_eng), lo, hi,
                                    lambda unit, T: C.test_labels(T, test_rul[unit]))
    (Xte_t,) = C.tensors(device, Xte)
    sb_test = C.predict(model, Xte_t)                    # the single test evaluation
    last = np.r_[ute[1:] != ute[:-1], True]
    arrays = {
        'units': units, 'seed': np.array(seed),
        'sb_budgets': np.array(C.BUDGETS), 'sb_fold_mae': mae, 'sb_s_star': np.array(s_star),
        'sb_cal_e': cal_e,
        'sb_cal_unit': np.concatenate([held[k]['unit'] for k in held]),
        'sb_cal_index': np.concatenate([held[k]['index'] for k in held]),
        'sb_cal_fold': np.concatenate([np.full(len(held[k]['y']), k) for k in held]),
        'sb_test_pred': sb_test,
        'test_y': yte, 'test_unit': ute, 'test_index': ite, 'test_last': last,
    }
    for k, h in held.items():
        arrays[f'sb_fold{k}_pred'] = h['pred']
        arrays[f'sb_fold{k}_y'] = h['y']
    tmp = path.with_name(path.name + '_tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path.with_suffix('.npz'))
    meta = {
        'dataset': ds, 'n': n, 'repeat': r, 'seed': seed, 'fleet': units.tolist(),
        'sb': {'s_star': s_star, 'fold_mae': mae.tolist(),
               'n_train_windows': [held[k]['n_train'] for k in held],
               'n_val_windows': [held[k]['n_val'] for k in held]},
        'n_cal': int(len(cal_e)), 'n_test_windows': int(len(yte)),
        'seconds': round(time.time() - t0, 1),
        'data_sha256': {k: v for k, v in C.SHA256.items() if ds in k},
        'provenance': provenance(CODE),
    }
    path.with_suffix('.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(f"{ds} n={n} r={r}: SB S* {s_star} (block MAE " + ', '.join(f'{m:.2f}' for m in mae)
          + f"); {meta['seconds']} s", flush=True)


def smoke(device):
    """FD001, n = 4, repeat 0; fold 1 of SB (an interior block), 500 steps; test files are not read"""
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    engines = C.load_train('FD001')
    units, seed = C.fleet(0, 4), C.FINAL_SEEDS[0]
    t0 = time.time()
    held, mae, _ = run_cv(engines, [sb_specs(engines, units)[1]], seed, device, budgets=[500])
    h = held[0]
    pred = h['pred'][0]
    res = {
        'fleet': units.tolist(), 'seed': seed, 'fold': 1,
        'block_mae': float(np.mean(np.abs(pred - h['y']))),
        'mean_predictor_mae': float(np.mean(np.abs(h['train_label_mean'] - h['y']))),
        'pred_sd': float(np.std(pred)), 'finite': bool(np.isfinite(pred).all()),
        'n_train_windows': h['n_train'], 'n_val_windows': h['n_val'], 'seconds': round(time.time() - t0, 1),
    }
    res['pass'] = bool(res['finite'] and res['block_mae'] < res['mean_predictor_mae'] and res['pred_sd'] > 2.5)
    res['provenance'] = provenance(CODE)
    (SMOKE_OUT / 'smoke.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k != 'provenance'}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--dataset', choices=['FD001', 'FD003'])
    ap.add_argument('--repeats', default=f'0-{C.N_REPEATS - 1}')
    ap.add_argument('--fleets', default=','.join(map(str, FLEETS)))
    args = ap.parse_args()
    C.deterministic()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        smoke(device)
        return
    if args.dataset is None:
        raise SystemExit('--dataset or --smoke')
    out_dir = OUT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    engines, test = C.load_train(args.dataset), C.load_test(args.dataset)
    for r in C.parse_range(args.repeats):
        for n in [int(x) for x in args.fleets.split(',')]:
            run_cell(args.dataset, engines, test, n, r, device, out_dir)
    print('done', flush=True)


if __name__ == '__main__':
    main()
