"""
temporal_context.py — Revision R1, analysis plan Section 7, candidate B2.

Temporal context for DSAFLite: both streams and the fusion gate are applied with shared
weights to the K most recent windows (t-K+1 .. t, causal), the K fused feature vectors
are averaged, and the unchanged regression head maps the average to RUL. The wrapper adds
no parameters; K = 5 follows Li et al. [7] and is not tuned. Gate weights and stream
features are averaged the same way, so the DANN branch and gate diagnostics keep their
shapes. Works unchanged for the ablation variants (fixed gates, single streams).
"""

import torch
import torch.nn as nn


class TemporalContext(nn.Module):
    def __init__(self, base: nn.Module, k: int = 5):
        super().__init__()
        self.base = base
        self.k = k

    def forward(self, x: torch.Tensor, x_2d=None, use_checkpointing=False):
        if x.dim() == 4:                     # (B, 1, K, L) from the window dataset
            x = x.squeeze(1)
        b, k, length = x.shape
        if k != self.k:
            raise ValueError(f'expected {self.k} windows per sample, got {k}')
        flat = x.reshape(b * k, 1, length)
        feat_a = self.base.stream_a(flat)
        feat_b = self.base.stream_b(flat)
        fused, alpha = self.base.fusion(feat_a, feat_b)
        d = fused.shape[-1]
        fused = fused.reshape(b, k, d).mean(dim=1)
        feat_a = feat_a.reshape(b, k, d).mean(dim=1)
        feat_b = feat_b.reshape(b, k, d).mean(dim=1)
        alpha = alpha.reshape(b, k).mean(dim=1)
        rul_pred = self.base.regressor(fused).squeeze(-1)
        return rul_pred, alpha, fused, feat_a, feat_b
