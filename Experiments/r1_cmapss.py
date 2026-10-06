"""
r1_cmapss.py - Revision R1, Amendment 11 (plan Section 11): in-unit against held-out-unit
calibration on the C-MAPSS turbofan engines (FD001 primary, FD003 replication), with fleets of
4, 10, 25 and 80 training engines and 20 repeats.

Standard practice (SP): every 10th window of each fleet engine (i mod 10 = 9) validates; the
validation MAE is computed every 50 steps; the first-minimum state is kept; training stops after
30 evaluations without a strict improvement or at 8,000 steps; calibration on those validation
windows; the deployed model is the kept state.
Protocol (PR): four engine folds by position (j mod 4); each fold is trained for 8,000 steps and
predicts its held-out engines after 500, 1,000, 2,000, 4,000 and 8,000 steps; S* is the budget
with the lowest four-fold mean held-out MAE (ties: the smaller); the deployed model is trained on
the whole fleet for S* steps; calibration on the fold networks' held-out errors after S* steps.

Network: Li, Ding and Sun (RESS 172, 2018) as described by Javanmardi and Hullermeier (2023):
four Conv(10 filters, 10 x 1) and one Conv(1 filter, 3 x 1) along time with size-keeping zero
padding and tanh; flatten (420); dropout 0.5; fully connected 100 (tanh); linear output.
Adam with constant learning rate 0.001, batch 512, mean squared error on RUL / 125.

Usage:
  python Experiments/r1_cmapss.py --smoke          (FD001, n = 4, repeat 0, 500 steps; test files not read)
  python Experiments/r1_cmapss.py --dataset FD001 [--repeats 0-19] [--fleets 4,10,25,80]
Outputs: outputs/r1_cmapss/<dataset>/n<NN>_r<RR>.npz and .json; smoke: outputs/r1_cmapss_smoke/.
"""

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Experiments'))
from r1_provenance import provenance  # noqa: E402

DATA = ROOT / 'data_cmapss' / 'CMAPSSData'
OUT = ROOT / 'outputs' / 'r1_cmapss'
SMOKE_OUT = ROOT / 'outputs' / 'r1_cmapss_smoke'
SHA256 = {   # plan Section 11, Amendment 11
    'train_FD001.txt': '963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8',
    'test_FD001.txt': '3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851',
    'RUL_FD001.txt': 'a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca',
    'train_FD003.txt': '2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2',
    'test_FD003.txt': '299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790',
    'RUL_FD003.txt': 'df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb',
}
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
SENSOR_COLS = [4 + s for s in SENSORS]   # columns: unit, cycle, three settings, sensors 1 to 21
WINDOW, CAP, BATCH, LR = 30, 125.0, 512, 1e-3
FLEETS = [4, 10, 25, 80]
N_ENGINES, N_REPEATS, N_FOLDS = 100, 20, 4
BUDGETS = [500, 1000, 2000, 4000, 8000]
MAX_STEPS = 8000
EVAL_EVERY, PATIENCE, VAL_EVERY_K = 50, 30, 10
FINAL_SEEDS = [42, 123, 777, 999, 7, 8, 21, 34, 55, 89, 144, 233, 377, 610, 987,
               1597, 2584, 4181, 6765, 10946]
CODE = ['Experiments/r1_cmapss.py']


# ---------------------------------------------------------------- data --------------------------
def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_files(names):
    for name in names:
        got = sha256(DATA / name)
        if got != SHA256[name]:
            raise SystemExit(f'{name}: SHA-256 {got} differs from the plan')


def engines_from_table(a):
    """{unit: T x 14 array of the kept sensors}, rows in cycle order; cycles must be 1..T"""
    out = {}
    for u in np.unique(a[:, 0]).astype(int):
        rows = a[a[:, 0] == u]
        rows = rows[np.argsort(rows[:, 1], kind='stable')]
        if not np.array_equal(rows[:, 1], np.arange(1, len(rows) + 1)):
            raise ValueError(f'unit {u}: cycles are not 1..T')
        out[int(u)] = rows[:, SENSOR_COLS].astype(np.float64)
    return out


def load_train(ds):
    check_files([f'train_{ds}.txt'])
    eng = engines_from_table(np.loadtxt(DATA / f'train_{ds}.txt'))
    if sorted(eng) != list(range(1, N_ENGINES + 1)):
        raise ValueError(f'{ds}: training units are not 1..{N_ENGINES}')
    return eng


def load_test(ds):
    check_files([f'test_{ds}.txt', f'RUL_{ds}.txt'])
    eng = engines_from_table(np.loadtxt(DATA / f'test_{ds}.txt'))
    rul = np.atleast_1d(np.loadtxt(DATA / f'RUL_{ds}.txt'))
    if sorted(eng) != list(range(1, len(rul) + 1)):
        raise ValueError(f'{ds}: test units do not match the RUL file')
    return eng, {u: float(rul[u - 1]) for u in eng}


def train_labels(T):
    """remaining life at the last cycle of each window of a run-to-failure record, capped"""
    i = np.arange(max(T - WINDOW + 1, 0))
    return np.minimum(T - (i + WINDOW), CAP).astype(np.float64)


def test_labels(T, rul_last):
    """remaining life at each window's last cycle of a record that ends rul_last cycles before failure"""
    i = np.arange(max(T - WINDOW + 1, 0))
    return np.minimum(rul_last + (T - (i + WINDOW)), CAP).astype(np.float64)


def windows(x):
    """T x s record -> (T - 29) x 30 x s windows, stride 1"""
    if len(x) < WINDOW:
        return np.empty((0, WINDOW, x.shape[1]))
    v = np.lib.stride_tricks.sliding_window_view(x, WINDOW, axis=0)   # (T - 29, s, 30)
    return np.ascontiguousarray(v.transpose(0, 2, 1))


def val_mask(n_windows):
    """the exp16 'window' split: every 10th window (index i with i mod 10 = 9)"""
    return (np.arange(n_windows) % VAL_EVERY_K) == (VAL_EVERY_K - 1)


def fleet(r, n):
    """first n engines of the permutation of repeat r (nested in n)"""
    return np.random.default_rng(r).permutation(np.arange(1, N_ENGINES + 1))[:n]


def folds(units):
    """engine at position j goes to fold j mod 4"""
    return [np.asarray(units[j::N_FOLDS]) for j in range(N_FOLDS)]


def fit_scaler(engines, units):
    x = np.concatenate([engines[int(u)] for u in units])
    return x.min(0), x.max(0)


def scale(w, lo, hi):
    span = hi - lo
    safe = np.where(span > 0, span, 1.0)
    return np.where(span > 0, 2.0 * (w - lo) / safe - 1.0, 0.0)


def make_set(engines, units, lo, hi, labels):
    """scaled windows of the given engines; labels(u, T) -> per-window labels"""
    xs, ys, us, idx = [], [], [], []
    for u in units:
        x = engines[int(u)]
        w = windows(x)
        xs.append(scale(w, lo, hi))
        ys.append(labels(int(u), len(x)))
        us.append(np.full(len(w), int(u)))
        idx.append(np.arange(len(w)))
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys), np.concatenate(us),
            np.concatenate(idx))


# ---------------------------------------------------------------- network -----------------------
class DCNN(nn.Module):
    """Li, Ding and Sun (2018) as described by Javanmardi and Hullermeier (2023)"""

    def __init__(self, n_sensors=len(SENSORS), window=WINDOW):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, 10, (10, 1), padding='same')]
                                   + [nn.Conv2d(10, 10, (10, 1), padding='same') for _ in range(3)])
        self.conv_out = nn.Conv2d(10, 1, (3, 1), padding='same')
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(window * n_sensors, 100)
        self.out = nn.Linear(100, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):                       # x: (batch, 30, 14)
        h = x.unsqueeze(1)
        for conv in self.convs:
            h = torch.tanh(conv(h))
        h = torch.tanh(self.conv_out(h))
        h = self.drop(h.flatten(1))
        h = torch.tanh(self.fc(h))
        return self.out(h).squeeze(1)


def deterministic():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(seed, device):
    torch.manual_seed(seed)
    return DCNN().to(device)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(X[i:i + 8192]) for i in range(0, len(X), 8192)])
    return out.double().cpu().numpy() * CAP


def train_steps(model, X, y, seed, total_steps, callback=None):
    """Adam (constant lr), batch 512, MSE on y / 125. callback(step) after every step may return
    False to stop. Returns the number of steps done."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    yt = y / CAP
    step, n = 0, len(X)
    while step < total_steps:
        perm = torch.randperm(n, generator=gen).to(X.device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(X[idx]), yt[idx])
            loss.backward()
            opt.step()
            step += 1
            if step % EVAL_EVERY == 0 and not np.isfinite(loss.item()):
                raise FloatingPointError(f'loss not finite at step {step}')
            if callback is not None and callback(step) is False:
                return step
            if step >= total_steps:
                break
    return step


class EarlyStopping:
    """first-minimum state; stop after PATIENCE evaluations without a strict improvement"""

    def __init__(self, patience=PATIENCE):
        self.patience, self.best, self.best_step, self.bad, self.state = patience, np.inf, 0, 0, None

    def update(self, step, mae, model=None):
        if mae < self.best:
            self.best, self.best_step, self.bad = mae, step, 0
            if model is not None:
                self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return True
        self.bad += 1
        return self.bad < self.patience


def tensors(device, *arrays):
    return [torch.as_tensor(a, dtype=torch.float32, device=device) for a in arrays]


# ---------------------------------------------------------------- pipelines ---------------------
def run_sp(engines, units, seed, device, max_steps=MAX_STEPS):
    lo, hi = fit_scaler(engines, units)
    X, y, u, i = make_set(engines, units, lo, hi, lambda unit, T: train_labels(T))
    vm = np.concatenate([val_mask(int((u == k).sum())) for k in units])
    assert np.array_equal(vm, (i % VAL_EVERY_K) == VAL_EVERY_K - 1)
    Xtr, ytr, Xv = tensors(device, X[~vm], y[~vm], X[vm])
    yv = y[vm]
    model = make_model(seed, device)
    stopper, history = EarlyStopping(), []

    def callback(step):
        if step % EVAL_EVERY:
            return True
        mae = float(np.mean(np.abs(predict(model, Xv) - yv)))
        if not np.isfinite(mae):
            raise FloatingPointError(f'validation MAE not finite at step {step}')
        history.append((step, mae))
        return stopper.update(step, mae, model)

    steps_done = train_steps(model, Xtr, ytr, seed, max_steps, callback)
    model.load_state_dict(stopper.state)
    val_pred = predict(model, Xv)
    info = {'lo': lo, 'hi': hi, 'val_pred': val_pred, 'val_y': yv, 'val_unit': u[vm],
            'train_label_mean': float(y[~vm].mean()), 'best_step': stopper.best_step,
            'best_val_mae': stopper.best, 'steps_done': steps_done,
            'history': np.array(history, dtype=np.float64), 'n_train': int((~vm).sum()), 'n_val': int(vm.sum())}
    return model, info


def run_pr(engines, units, seed, device, budgets=BUDGETS, fold_ids=range(N_FOLDS), final=True):
    parts = folds(units)
    held = {}
    for k in fold_ids:
        tr = np.concatenate([parts[j] for j in range(N_FOLDS) if j != k])
        lo, hi = fit_scaler(engines, tr)
        X, y, _, _ = make_set(engines, tr, lo, hi, lambda unit, T: train_labels(T))
        Xh, yh, uh, _ = make_set(engines, parts[k], lo, hi, lambda unit, T: train_labels(T))
        Xt, yt_, Xht = tensors(device, X, y, Xh)
        model = make_model(seed, device)
        preds = {}

        def callback(step, model=model, Xht=Xht, preds=preds):
            if step in budgets:
                preds[step] = predict(model, Xht)
            return True

        train_steps(model, Xt, yt_, seed, max(budgets), callback)
        held[k] = {'pred': np.stack([preds[s] for s in budgets]), 'y': yh, 'unit': uh,
                   'train_label_mean': float(y.mean())}
    mae = np.array([np.mean([np.mean(np.abs(held[k]['pred'][b] - held[k]['y'])) for k in held])
                    for b in range(len(budgets))])
    b_star = int(np.argmin(mae))                         # first minimum: the smaller budget on ties
    s_star = budgets[b_star]
    info = {'budgets': np.array(budgets), 'fold_mae': mae, 's_star': s_star, 'held': held,
            'cal_e': np.concatenate([held[k]['pred'][b_star] - held[k]['y'] for k in held])}
    if not final:
        return None, info
    lo, hi = fit_scaler(engines, units)
    X, y, _, _ = make_set(engines, units, lo, hi, lambda unit, T: train_labels(T))
    Xt, yt_ = tensors(device, X, y)
    model = make_model(seed, device)
    train_steps(model, Xt, yt_, seed, s_star)
    info.update(lo=lo, hi=hi)
    return model, info


# ---------------------------------------------------------------- cells -------------------------
def run_cell(ds, engines, test, n, r, device, out_dir):
    path = out_dir / f'n{n:02d}_r{r:02d}'
    if path.with_suffix('.json').exists():
        print(f'{ds} n={n} r={r}: exists, skipped', flush=True)
        return
    t0 = time.time()
    seed, units = FINAL_SEEDS[r], fleet(r, n)
    sp_model, sp = run_sp(engines, units, seed, device)
    pr_model, pr = run_pr(engines, units, seed, device)
    assert np.array_equal(sp['lo'], pr['lo']) and np.array_equal(sp['hi'], pr['hi'])
    test_eng, test_rul = test
    Xte, yte, ute, ite = make_set(test_eng, sorted(test_eng), sp['lo'], sp['hi'],
                                  lambda unit, T: test_labels(T, test_rul[unit]))
    (Xte_t,) = tensors(device, Xte)
    sp_test, pr_test = predict(sp_model, Xte_t), predict(pr_model, Xte_t)   # the single test evaluation
    last = np.r_[ute[1:] != ute[:-1], True]
    arrays = {
        'units': units, 'seed': np.array(seed),
        'sp_val_pred': sp['val_pred'], 'sp_val_y': sp['val_y'], 'sp_val_unit': sp['val_unit'],
        'sp_history': sp['history'], 'sp_test_pred': sp_test,
        'pr_budgets': pr['budgets'], 'pr_fold_mae': pr['fold_mae'], 'pr_cal_e': pr['cal_e'],
        'pr_test_pred': pr_test,
        'test_y': yte, 'test_unit': ute, 'test_index': ite, 'test_last': last,
    }
    for k, h in pr['held'].items():
        arrays[f'pr_fold{k}_pred'] = h['pred']
        arrays[f'pr_fold{k}_y'] = h['y']
        arrays[f'pr_fold{k}_unit'] = h['unit']
    tmp = path.with_name(path.name + '_tmp.npz')
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path.with_suffix('.npz'))
    meta = {
        'dataset': ds, 'n': n, 'repeat': r, 'seed': seed, 'fleet': units.tolist(),
        'folds': [p.tolist() for p in folds(units)],
        'sp': {'best_step': sp['best_step'], 'best_val_mae': sp['best_val_mae'], 'steps_done': sp['steps_done'],
               'n_train_windows': sp['n_train'], 'n_val_windows': sp['n_val']},
        'pr': {'s_star': pr['s_star'], 'fold_mae': pr['fold_mae'].tolist()},
        'n_test_windows': int(len(yte)), 'n_test_engines_with_windows': int(len(np.unique(ute))),
        'n_test_engines_short': int(sum(len(x) < WINDOW for x in test_eng.values())),
        'seconds': round(time.time() - t0, 1),
        'data_sha256': {k: v for k, v in SHA256.items() if ds in k},
        'provenance': provenance(CODE),
    }
    path.with_suffix('.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(f"{ds} n={n} r={r}: SP best step {sp['best_step']} (val MAE {sp['best_val_mae']:.2f}, "
          f"{sp['steps_done']} steps); PR S* {pr['s_star']} (fold MAE "
          + ', '.join(f'{m:.2f}' for m in pr['fold_mae']) + f"); {meta['seconds']} s", flush=True)


def smoke(device):
    """FD001, n = 4, repeat 0; SP and fold 0 of PR, 500 steps each; test files are not read"""
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    engines = load_train('FD001')
    units, seed = fleet(0, 4), FINAL_SEEDS[0]
    t0 = time.time()
    _, sp = run_sp(engines, units, seed, device, max_steps=500)
    t_sp = time.time() - t0
    t0 = time.time()
    _, pr = run_pr(engines, units, seed, device, budgets=[500], fold_ids=[0], final=False)
    t_pr = time.time() - t0
    h = pr['held'][0]
    pr_pred = h['pred'][0]
    res = {
        'fleet': units.tolist(), 'seed': seed,
        'sp': {'best_step': sp['best_step'], 'val_mae': sp['best_val_mae'],
               'mean_predictor_mae': float(np.mean(np.abs(sp['train_label_mean'] - sp['val_y']))),
               'pred_sd': float(np.std(sp['val_pred'])), 'finite': bool(np.isfinite(sp['val_pred']).all()),
               'seconds': round(t_sp, 1), 'n_train_windows': sp['n_train'], 'n_val_windows': sp['n_val']},
        'pr_fold0': {'heldout_mae': float(np.mean(np.abs(pr_pred - h['y']))),
                     'mean_predictor_mae': float(np.mean(np.abs(h['train_label_mean'] - h['y']))),
                     'pred_sd': float(np.std(pr_pred)), 'finite': bool(np.isfinite(pr_pred).all()),
                     'seconds': round(t_pr, 1)},
    }
    res['pass'] = bool(res['sp']['finite'] and res['pr_fold0']['finite']
                       and res['sp']['val_mae'] < res['sp']['mean_predictor_mae']
                       and res['pr_fold0']['heldout_mae'] < res['pr_fold0']['mean_predictor_mae']
                       and res['sp']['pred_sd'] > 2.5 and res['pr_fold0']['pred_sd'] > 2.5)
    res['provenance'] = provenance(CODE)
    (SMOKE_OUT / 'smoke.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k != 'provenance'}, indent=1))


def parse_range(text):
    out = []
    for part in text.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--dataset', choices=['FD001', 'FD003'])
    ap.add_argument('--repeats', default=f'0-{N_REPEATS - 1}')
    ap.add_argument('--fleets', default=','.join(map(str, FLEETS)))
    args = ap.parse_args()
    deterministic()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        smoke(device)
        return
    if args.dataset is None:
        raise SystemExit('--dataset or --smoke')
    out_dir = OUT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    engines, test = load_train(args.dataset), load_test(args.dataset)
    for r in parse_range(args.repeats):
        for n in [int(x) for x in args.fleets.split(',')]:
            run_cell(args.dataset, engines, test, n, r, device, out_dir)
    print('done', flush=True)


if __name__ == '__main__':
    main()
