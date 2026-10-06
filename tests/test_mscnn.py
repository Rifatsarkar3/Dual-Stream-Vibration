"""Unit tests for the MSCNN reproduction (analysis plan Section 6): shapes, parameter
count against a hand count, STFT geometry, and causality of the history input."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.mscnn_li2019 import MSCNN  # noqa: E402


def hand_count(n_hist=5, filters=5, bins=64, frames=11, fc=128):
    conv = lambda c_in, c_out: c_in * c_out * 9 + c_out  # noqa: E731
    per_recording = (conv(1, filters) + conv(filters, filters) + conv(filters, filters)
                     + conv(3 * filters, filters))
    return n_hist * per_recording + conv(filters, 1) + (bins * frames * fc + fc) + (fc + 1)


def test_parameter_count_matches_hand_count():
    model = MSCNN()
    assert sum(p.numel() for p in model.parameters()) == hand_count() == 96365


def test_stft_map_geometry():
    model = MSCNN()
    maps = model.tf_maps(torch.randn(3, 2560))
    assert maps.shape == (3, 1, 64, 11)       # 128 bins pooled 2:1, 11 centred frames


def test_output_shape_and_range():
    model = MSCNN().eval()
    y = model(torch.randn(4, 1, 5, 2560))
    assert y.shape == (4,)
    assert torch.all((y >= 0) & (y <= 1))


def test_history_is_causal():
    sys.path.insert(0, str(ROOT / 'Experiments'))
    from Experiments.exp16_ress import BearingArrays
    from Experiments.r1_mscnn import with_history
    rng = np.random.default_rng(0)
    vib = rng.standard_normal((12, 2560)).astype(np.float32)
    rul = np.linspace(1, 0, 12, dtype=np.float32)
    hist = with_history([BearingArrays('b', vib, rul, 0)])[0].vib
    assert hist.shape == (12, 5, 2560)
    for t in range(12):
        expected = [max(0, s) for s in range(t - 4, t + 1)]
        assert np.array_equal(hist[t], vib[expected])
    changed = vib.copy()
    changed[7:] += 100.0                         # alter the future of t = 6
    hist2 = with_history([BearingArrays('b', changed, rul, 0)])[0].vib
    assert np.array_equal(hist[:7], hist2[:7])
