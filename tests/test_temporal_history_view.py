"""Unit tests for the strided window history of Experiments/r1_temporal_xjtu_final.py: it must
hold exactly the windows of the materialized history of Experiments/r1_temporal.py, give the
same batches through the exp16 data paths, and the same model outputs."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import Experiments.r1_temporal as rt  # noqa: E402
import Experiments.r1_temporal_xjtu_final as xf  # noqa: E402
import Experiments.exp16_ress as e16  # noqa: E402


def materialized(vib, k):
    idx = np.clip(np.arange(len(vib))[:, None] - np.arange(k - 1, -1, -1)[None, :], 0, None)
    return vib[idx]


def test_view_equals_materialized_history():
    rng = np.random.default_rng(0)
    for n in (1, 3, 5, 13):
        vib = rng.standard_normal((n, 7)).astype(np.float32)
        view = xf.history_view(vib, 5)
        assert view.shape == (n, 5, 7) and not view.flags.writeable
        assert np.array_equal(view, materialized(vib, 5))


def test_real_bearing_matches_r1_temporal_loader():
    args = (str(ROOT / 'Processed_PRONOSTIA_v2'), ['test_03_Bearing1_5'])
    a = rt.load_bearings_with_history(*args)[0]
    b = xf.load_bearings_with_history_view(*args)[0]
    assert a.name == b.name and a.n == b.n and a.condition == b.condition
    assert np.array_equal(a.rul, b.rul)
    assert np.array_equal(a.vib, b.vib)


def test_batches_and_model_outputs_identical():
    args = (str(ROOT / 'Processed_PRONOSTIA_v2'), ['test_03_Bearing1_5'])
    a = rt.load_bearings_with_history(*args)[0]
    b = xf.load_bearings_with_history_view(*args)[0]
    # evaluation path (exp16.evaluate_on_test): contiguous slices of 256 windows
    xa = torch.from_numpy(a.vib[0:256]).unsqueeze(1)
    xb = torch.from_numpy(np.ascontiguousarray(b.vib[0:256])).unsqueeze(1)
    xb_strided = torch.from_numpy(b.vib[0:256]).unsqueeze(1)
    assert torch.equal(xa, xb) and torch.equal(xa, xb_strided)
    # training path (exp16.WindowDataset): one window at a time
    da = e16.WindowDataset([a], [np.array([0, 4, 900])])
    db = e16.WindowDataset([b], [np.array([0, 4, 900])])
    for i in range(3):
        assert torch.equal(da[i][0], db[i][0])
    torch.manual_seed(0)
    model = rt.make_temporal_model(dict(e16.CONFIG), torch.device('cpu')).eval()
    with torch.no_grad():
        assert torch.equal(model(xa[:8])[0], model(xb_strided[:8])[0])
