"""
Unit tests for Experiments/r1_cmapss_blocked.py (Amendment 12), synthetic data only.

Run: python tests/test_cmapss_blocked.py   (pytest also collects the test_ functions)
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Experiments'))
sys.path.insert(0, str(ROOT / 'tests'))
import r1_cmapss as C  # noqa: E402
import r1_cmapss_blocked as B  # noqa: E402
from test_cmapss import synthetic_engines  # noqa: E402

CPU = torch.device('cpu')
WIDTHS = [4, 7, 31, 99, 100, 117, 177, 333]


def test_blocks_partition():
    for W in WIDTHS:
        bl = B.blocks(W)
        assert len(bl) == 4 and bl[0][0] == 0 and bl[-1][1] == W
        for (a, b), (c, d) in zip(bl[:-1], bl[1:]):
            assert b == c                                  # contiguous, no gap, no overlap
        sizes = [b - a for a, b in bl]
        assert max(sizes) - min(sizes) <= 1


def test_no_shared_cycle_and_buffers():
    for W in WIDTHS:
        for k in range(4):
            train, val = B.block_split(W, k)
            if len(val) == 0:
                continue
            cycles_val = set()
            for v in val:
                cycles_val.update(range(v, v + C.WINDOW))  # 0-based cycles of window v
            for t in train:
                assert not cycles_val.intersection(range(t, t + C.WINDOW)), (W, k, t)
            start, end = val[0], val[-1] + 1
            outside = [i for i in range(W) if i < start - (C.WINDOW - 1) or i > end - 1 + (C.WINDOW - 1)]
            assert train.tolist() == outside, (W, k)     # every window outside block and buffers trains
            left = [i for i in range(W) if start - (C.WINDOW - 1) <= i < start]
            right = [i for i in range(W) if end <= i <= end - 1 + (C.WINDOW - 1)]
            assert len(left) <= 29 and len(right) <= 29
            assert set(train).isdisjoint(left + right + val.tolist())


def test_each_window_validated_once():
    for W in WIDTHS:
        seen = np.concatenate([B.block_split(W, k)[1] for k in range(4)])
        assert sorted(seen.tolist()) == list(range(W))


def test_fold_scaler_uses_training_windows_only():
    eng = synthetic_engines([150, 160, 170, 180], seed=3)
    units = np.array([1, 2, 3, 4])
    W = B.n_windows(eng, 1)
    start, end = B.blocks(W)[1]
    eng[1] = eng[1].copy()
    eng[1][start + 1, :] = 1e6                          # a spike in a cycle covered only by the block
    tr_spec, _ = B.sb_specs(eng, units)[1]
    x, _, _, _ = B.windows_of(eng, tr_spec)
    lo, hi = B.scaler_from_windows(x)
    assert hi.max() < 1e5
    tr_spec2, _ = B.sb_specs(eng, units)[2]            # fold 2 trains on the windows before block 2,
    x2, _, _, _ = B.windows_of(eng, tr_spec2)          # whose cycles include the spike
    assert B.scaler_from_windows(x2)[1].max() >= 1e6


def test_windows_of_matches_make_set():
    eng = synthetic_engines([90, 75, 120], seed=5)
    units = np.array([2, 3, 1])
    lo, hi = C.fit_scaler(eng, units)
    X, y, u, i = C.make_set(eng, units, lo, hi, lambda unit, T: C.train_labels(T))
    spec = [(int(k), np.arange(B.n_windows(eng, k))) for k in units]
    xr, yr, ur, ir = B.windows_of(eng, spec)
    assert np.array_equal(C.scale(xr, lo, hi).astype(np.float32), X)
    assert np.array_equal(yr, y) and np.array_equal(ur, u) and np.array_equal(ir, i)
    lo2, hi2 = B.scaler_from_windows(xr)
    assert np.array_equal(lo2, lo) and np.array_equal(hi2, hi)


def test_engine_folds_reproduce_run_pr():
    C.deterministic()
    eng = synthetic_engines([70, 82, 64, 90, 77, 85, 73, 68], seed=11)
    units = np.array([5, 2, 8, 1, 7, 3, 6, 4])
    budgets = [10, 20]
    _, info = C.run_pr(eng, units, 42, CPU, budgets=budgets, final=False)
    held, mae, b_star = B.run_cv(eng, B.engine_specs(eng, units), 42, CPU, budgets=budgets)
    for k in range(4):
        assert np.array_equal(held[k]['pred'], info['held'][k]['pred']), k
        assert np.array_equal(held[k]['y'], info['held'][k]['y'])
    assert np.allclose(mae, info['fold_mae']) and budgets[b_star] == info['s_star']


def test_sb_calibration_covers_every_window_once():
    eng = synthetic_engines([130, 145, 160, 175], seed=7)
    units = np.array([3, 1, 4, 2])
    specs = B.sb_specs(eng, units)
    pairs = sorted((u, int(i)) for _, va in specs for u, ii in va for i in ii)
    expected = sorted((int(u), i) for u in units for i in range(B.n_windows(eng, u)))
    assert pairs == expected


def test_deploy_equals_pr_final():
    C.deterministic()
    eng = synthetic_engines([70, 82, 64, 90], seed=13)
    units = np.array([4, 1, 3, 2])
    model_pr, info = C.run_pr(eng, units, 42, CPU, budgets=[10, 20])
    model_sb, lo, hi = B.deploy(eng, units, 42, CPU, info['s_star'])
    for (a, pa), (b, pb) in zip(model_pr.state_dict().items(), model_sb.state_dict().items()):
        assert a == b and torch.equal(pa, pb), a
    assert np.array_equal(lo, info['lo']) and np.array_equal(hi, info['hi'])


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} tests passed')
