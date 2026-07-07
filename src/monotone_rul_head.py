"""
monotone_rul_head.py - Simplified Monotone RUL Head
Simple design: standard MLP + PAV post-processing for guaranteed monotonicity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MonotoneRULHead(nn.Module):
    """Standard regressor head with PAV post-processing for monotonicity."""

    def __init__(self, feat_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1)
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        rul = self.encoder(fused).squeeze(-1)
        return rul


class IsotonicPostProcessor:
    """Pool-Adjacent Violators (PAV) algorithm for monotone decreasing RUL."""

    @staticmethod
    def _pav_decreasing(y: np.ndarray) -> np.ndarray:
        """PAV for monotone DECREASING sequence."""
        n = len(y)
        y_neg = -y.copy()

        blocks = [[y_neg[0], 1]]
        for i in range(1, n):
            blocks.append([y_neg[i], 1])
            while len(blocks) >= 2 and blocks[-1][0] < blocks[-2][0]:
                last  = blocks.pop()
                prev  = blocks.pop()
                n1, n2 = prev[1], last[1]
                merged_mean = (prev[0] * n1 + last[0] * n2) / (n1 + n2)
                blocks.append([merged_mean, n1 + n2])

        y_iso_neg = np.repeat([b[0] for b in blocks], [b[1] for b in blocks])
        return -y_iso_neg

    def transform(self, bearing_preds: dict) -> dict:
        """Apply PAV to each bearing's predictions."""
        smoothed = {}
        for bid, preds in bearing_preds.items():
            smoothed[bid] = self._pav_decreasing(np.array(preds))
        return smoothed

    def compute_monotonicity(self, bearing_preds: dict) -> float:
        """Compute raw monotonicity score."""
        decreasing = 0
        total = 0
        for preds in bearing_preds.values():
            preds = np.array(preds)
            diffs = np.diff(preds)
            decreasing += (diffs <= 0).sum()
            total += len(diffs)
        return 100 * decreasing / total if total > 0 else 0.0


def upgrade_dsaflite_regressor(model, feat_dim: int = 128, dropout: float = 0.2) -> None:
    """Replace DSAFLite regressor with MonotoneRULHead."""
    model.regressor = MonotoneRULHead(feat_dim=feat_dim, dropout=dropout)
    n_params = sum(p.numel() for p in model.regressor.parameters())
    print(f"[upgrade_dsaflite_regressor] Replaced regressor with MonotoneRULHead "
          f"({n_params:,} params)")

    import torch
    x = torch.randn(4, 1, 2560)
    with torch.no_grad():
        rul, alpha, fused, fa, fb = model(x)
        assert rul.shape == (4,), f"Expected (4,), got {rul.shape}"
        assert not torch.isnan(rul).any(), "NaN in RUL output"
    print(f"[upgrade_dsaflite_regressor] Forward pass OK: rul range "
          f"[{rul.min():.3f}, {rul.max():.3f}]")


if __name__ == '__main__':
    print("=" * 60)
    print("TESTING MonotoneRULHead")
    print("=" * 60)

    # Test 1: Forward pass
    print("\n[Test 1] MonotoneRULHead forward pass")
    head = MonotoneRULHead(feat_dim=128, dropout=0.2)
    feat = torch.randn(8, 128)
    with torch.no_grad():
        rul = head(feat)
    print(f"  Input: (8, 128) -> Output: {rul.shape}")
    print(f"  RUL range: [{rul.min():.4f}, {rul.max():.4f}]")
    print(f"  NaN check: {torch.isnan(rul).any().item()} (should be False)")

    # Test 2: Isotonic post-processor
    print("\n[Test 2] IsotonicPostProcessor")
    pav = IsotonicPostProcessor()
    np.random.seed(42)
    bear0 = 1.0 - np.linspace(0, 1, 50) + np.random.randn(50) * 0.1
    bear1 = 1.0 - np.linspace(0, 1, 30) + np.random.randn(30) * 0.15
    preds = {0: bear0, 1: bear1}

    before_mono = pav.compute_monotonicity(preds)
    smoothed = pav.transform(preds)
    after_mono = pav.compute_monotonicity(smoothed)

    print(f"  Monotonicity before PAV: {before_mono:.1f}%")
    print(f"  Monotonicity after PAV:  {after_mono:.1f}% (should be 100%)")
    assert after_mono == 100.0, f"FAIL: PAV did not achieve 100% monotonicity"

    # Test 3: Integration
    print("\n[Test 3] upgrade_dsaflite_regressor")
    try:
        from src.dsaf_v2_lite import DSAFLite
        model = DSAFLite(feat_dim=128)
        upgrade_dsaflite_regressor(model, feat_dim=128)
        print("  [PASS] upgrade_dsaflite_regressor succeeded")
    except ImportError:
        print("  [SKIP] DSAFLite not available")

    print("\n[ALL TESTS PASSED] MonotoneRULHead is ready")
