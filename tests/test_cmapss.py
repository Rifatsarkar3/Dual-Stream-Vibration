"""
Unit tests for Experiments/r1_cmapss.py (Amendment 11), synthetic data only.

Run: python tests/test_cmapss.py   (pytest also collects the test_ functions)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Experiments'))
import r1_cmapss as C  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def synthetic_table(lengths, rng, offset=0.0):
    rows = []
    for u, T in enumerate(lengths, start=1):
        for c in range(1, T + 1):
            rows.append([u, c, 0.0, 0.0, 100.0] + list(offset + rng.normal(size=21) + 0.01 * c))
    return np.array(rows)


def synthetic_engines(lengths, seed=0):
    rng = np.random.default_rng(seed)
    return C.engines_from_table(synthetic_table(lengths, rng))


def test_sensor_columns():
    row = np.array([[1, 1, 0.1, 0.2, 0.3] + [float(s) for s in range(1, 22)]])
    eng = C.engines_from_table(row)
    assert eng[1].tolist() == [[float(s) for s in C.SENSORS]]
    assert C.SENSORS == [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]


def test_cycles_must_be_contiguous():
    rng = np.random.default_rng(0)
    t = synthetic_table([5], rng)
    t = t[[0, 1, 3, 4]]                      # cycle 3 missing
    try:
        C.engines_from_table(t)
    except ValueError:
        return
    raise AssertionError('missing cycle not detected')


def test_train_windows_and_labels():
    x = np.arange(45 * 14, dtype=float).reshape(45, 14)
    w = C.windows(x)
    assert w.shape == (16, 30, 14)
    for i in (0, 7, 15):
        assert np.array_equal(w[i], x[i:i + 30])
    y = C.train_labels(45)
    assert y.tolist() == list(range(15, -1, -1))


def test_label_cap():
    y = C.train_labels(200)
    assert len(y) == 171 and y[0] == 125 and y[-1] == 0
    raw = 200 - (np.arange(171) + 30)
    assert np.array_equal(y, np.minimum(raw, 125))


def test_test_labels():
    assert C.test_labels(40, 50).tolist() == list(range(60, 49, -1))
    y = C.test_labels(40, 120)
    assert y[0] == 125 and y[-1] == 120


def test_short_engine():
    assert C.windows(np.zeros((20, 14))).shape == (0, 30, 14)
    assert len(C.train_labels(20)) == 0 and len(C.test_labels(20, 10)) == 0


def test_val_mask():
    m = C.val_mask(35)
    assert np.flatnonzero(m).tolist() == [9, 19, 29]
    k = 10
    idx = np.arange(123)
    assert np.array_equal(C.val_mask(123), (idx % k) == (k - 1))    # exp16 'window' split


def test_folds():
    f4 = C.folds(np.array([11, 22, 33, 44]))
    assert [p.tolist() for p in f4] == [[11], [22], [33], [44]]
    units = C.fleet(3, 10)
    f10 = C.folds(units)
    assert [len(p) for p in f10] == [3, 3, 2, 2]
    assert sorted(np.concatenate(f10).tolist()) == sorted(units.tolist())
    for j, u in enumerate(units):
        assert u in f10[j % 4]
    f80 = C.folds(C.fleet(0, 80))
    assert [len(p) for p in f80] == [20, 20, 20, 20]


def test_nested_fleets():
    for r in range(20):
        big = C.fleet(r, 80)
        assert len(set(big.tolist())) == 80 and big.min() >= 1 and big.max() <= 100
        for n in (4, 10, 25):
            assert np.array_equal(C.fleet(r, n), big[:n])
    assert not np.array_equal(C.fleet(0, 10), C.fleet(1, 10))


def test_scaler_training_engines_only():
    eng = {1: np.linspace(0, 10, 50)[:, None] * np.ones((1, 14)),
           2: np.linspace(100, 200, 50)[:, None] * np.ones((1, 14))}
    lo, hi = C.fit_scaler(eng, [1])
    assert np.allclose(lo, 0) and np.allclose(hi, 10)
    s1 = C.scale(eng[1], lo, hi)
    assert np.isclose(s1.min(), -1) and np.isclose(s1.max(), 1)
    assert C.scale(eng[2], lo, hi).min() > 1               # held-out engine not used in the fit
    const = np.ones((5, 14))
    assert np.all(C.scale(const, np.ones(14), np.ones(14)) == 0)


def test_parameter_count():
    assert sum(p.numel() for p in C.DCNN().parameters()) == 45372


def test_output_shapes():
    model = C.DCNN().eval()
    shapes = []
    hooks = [m.register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape)))
             for m in list(model.convs) + [model.conv_out]]
    out = model(torch.zeros(5, 30, 14))
    for h in hooks:
        h.remove()
    assert out.shape == (5,)
    assert shapes == [(5, 10, 30, 14)] * 4 + [(5, 1, 30, 14)]


def test_early_stopping():
    es = C.EarlyStopping(patience=3)
    seq = [5.0, 4.0, 4.0, 3.0, 3.5, 3.0, 3.2]          # 3.0 repeated: not a strict improvement
    cont = [es.update(50 * (i + 1), m) for i, m in enumerate(seq)]
    assert es.best == 3.0 and es.best_step == 200       # first minimum
    assert cont == [True, True, True, True, True, True, False]


def test_prefix_equivalence():
    """the state after S steps of a longer run equals that of a run of exactly S steps"""
    C.deterministic()
    eng = synthetic_engines([80, 90, 70], seed=1)
    lo, hi = C.fit_scaler(eng, [1, 2, 3])
    X, y, _, _ = C.make_set(eng, [1, 2, 3], lo, hi, lambda u, T: C.train_labels(T))
    Xt, yt = C.tensors(DEVICE, X, y)
    old = C.BATCH
    C.BATCH = 32                                         # several steps per epoch, cut mid-epoch
    try:
        long_model, snap = C.make_model(42, DEVICE), {}

        def cb(step):
            if step == 37:
                snap.update({k: v.detach().clone() for k, v in long_model.state_dict().items()})
            return True

        C.train_steps(long_model, Xt, yt, 42, 100, cb)
        short_model = C.make_model(42, DEVICE)
        C.train_steps(short_model, Xt, yt, 42, 37)
    finally:
        C.BATCH = old
    diff = max(float((snap[k] - v).abs().max()) for k, v in short_model.state_dict().items())
    assert diff == 0.0, diff


def test_pipelines_end_to_end():
    """SP and PR on synthetic engines: shapes, masks, budget rule, calibration errors"""
    C.deterministic()
    eng = synthetic_engines([60, 75, 90, 65, 80], seed=2)
    units = np.array([3, 1, 5, 2, 4])
    _, sp = C.run_sp(eng, units, 7, DEVICE, max_steps=100)
    n_win = sum(len(eng[u]) - 29 for u in units)
    assert sp['n_train'] + sp['n_val'] == n_win
    assert sp['n_val'] == sum(int(C.val_mask(len(eng[u]) - 29).sum()) for u in units)
    assert sp['best_step'] in (50, 100) and len(sp['history']) == 2
    model, pr = C.run_pr(eng, units, 7, DEVICE, budgets=[20, 40])
    assert pr['s_star'] == [20, 40][int(np.argmin(pr['fold_mae']))]
    held = np.concatenate([pr['held'][k]['unit'] for k in range(4)])
    assert sorted(set(held.tolist())) == sorted(units.tolist())
    assert len(pr['cal_e']) == n_win
    assert np.array_equal(pr['lo'], sp['lo']) and np.array_equal(pr['hi'], sp['hi'])
    assert model is not None


if __name__ == '__main__':
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS {name}')
        except Exception as exc:                       # report every test, then fail at the end
            failed += 1
            print(f'FAIL {name}: {type(exc).__name__}: {exc}')
    print(f'{len(tests) - failed}/{len(tests)} passed on {DEVICE}')
    sys.exit(1 if failed else 0)
