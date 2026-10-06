"""Unit tests for the B2 temporal-context wrapper (analysis plan Section 7): no new
parameters, output shapes, exact reduction to the single-window model when all K windows
are identical, and compatibility with the fixed-gate ablations."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dsaf_v2_lite import DSAFLite  # noqa: E402
from src.monotone_rul_head import upgrade_dsaflite_regressor  # noqa: E402
from src.temporal_context import TemporalContext  # noqa: E402


def make_base():
    torch.manual_seed(0)
    base = DSAFLite(feat_dim=128, dropout=0.2)
    upgrade_dsaflite_regressor(base, feat_dim=128, dropout=0.2)
    return base


def test_no_new_parameters():
    base = make_base()
    n = sum(p.numel() for p in base.parameters())
    assert sum(p.numel() for p in TemporalContext(base, 5).parameters()) == n


def test_shapes():
    model = TemporalContext(make_base(), 5).eval()
    pred, alpha, fused, fa, fb = model(torch.randn(3, 1, 5, 2560))
    assert pred.shape == (3,) and alpha.shape == (3,)
    assert fused.shape == fa.shape == fb.shape == (3, 128)


def test_identical_windows_reduce_to_single_window_model():
    base = make_base().eval()
    model = TemporalContext(base, 5).eval()
    w = torch.randn(2, 1, 2560)
    with torch.no_grad():
        single = base(w)[0]
        stacked = model(w.unsqueeze(2).expand(2, 1, 5, 2560).contiguous())[0]
    assert torch.allclose(single, stacked, atol=1e-5)


def test_fixed_gate_ablation():
    sys.path.insert(0, str(ROOT / 'Experiments'))
    from Experiments.exp16_ress import FixedGate
    base = make_base()
    base.fusion = FixedGate(1.0)
    pred, alpha, *_ = TemporalContext(base, 5).eval()(torch.randn(2, 1, 5, 2560))
    assert pred.shape == (2,) and torch.all(alpha == 1.0)
