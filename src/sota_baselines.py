"""
sota_baselines.py
=================
SOTA comparison baselines for DSAF paper revision.

Three architectures evaluated on the same PRONOSTIA split as DSAF:
  F_CNN_LSTM          — CNN feature extraction + bidirectional LSTM
  G_VanillaTransformer — Self-attention transformer on STFT patches
  H_DeepResNet1D      — 8-block deep 1D residual network

All models expose the same forward signature as DSAF:
    forward(x_1d, x_2d, use_checkpointing=False)
    → (rul_pred, gate_alpha, features, feat_1d, feat_2d)

This allows drop-in use with the existing Trainer and ablation framework.
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# F: CNN-LSTM Baseline
# ─────────────────────────────────────────────────────────────────────────────

class CNNLSTMBaseline(nn.Module):
    """
    Classic CNN-LSTM for bearing RUL prediction.
    CNN extracts local features; LSTM models temporal degradation dynamics.
    Operates on raw 1D vibration (2560 samples).

    Reference architecture type: Li et al. (2019), Zhu et al. (2019).
    """
    def __init__(self, cnn_channels=128, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64,          kernel_size=15, stride=4,  padding=7),
            nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64, cnn_channels, kernel_size=7,  stride=4,  padding=3),
            nn.BatchNorm1d(cnn_channels), nn.GELU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(cnn_channels), nn.GELU(),
        )
        # After stem: (B, cnn_channels, T) where T = 2560 // (4*4*2) = 80
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0
        )
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x_1d, x_2d=None, use_checkpointing=False):
        # x_1d: (B, 1, 2560)
        feat = self.cnn(x_1d)                 # (B, cnn_channels, T)
        feat = feat.transpose(1, 2)            # (B, T, cnn_channels)
        out, _ = self.lstm(feat)               # (B, T, lstm_hidden*2)
        last = out[:, -1, :]                   # (B, lstm_hidden*2)
        rul_pred = self.regressor(last)        # (B, 1)
        dummy_2d = torch.zeros(x_1d.shape[0], 1, 1, device=x_1d.device)
        return rul_pred.squeeze(-1), torch.tensor(0.0), last, feat, dummy_2d


# ─────────────────────────────────────────────────────────────────────────────
# G: Vanilla Transformer Baseline
# ─────────────────────────────────────────────────────────────────────────────

class VanillaTransformerBaseline(nn.Module):
    """
    Vanilla Transformer encoder on STFT spectrogram patches.
    Patch embedding → positional encoding → transformer encoder → CLS token → RUL.
    No dual-stream, no adaptive gate. Pure attention on spectrograms.

    Reference architecture type: Vaswani et al. (2017) applied to PHM.
    """
    def __init__(self, patch_size=8, embed_dim=256, nhead=4,
                 num_layers=4, dropout=0.1):
        super().__init__()
        # Patch embedding: conv2d with stride = patch_size
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            # output: (B, embed_dim, H/patch_size, W/patch_size)
        )
        # CLS token + positional embedding (learned)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop   = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x_1d, x_2d, use_checkpointing=False):
        # x_2d: (B, 3, H, W) — STFT spectrogram
        B = x_2d.shape[0]
        p = self.patch_embed(x_2d)             # (B, embed_dim, nh, nw)
        p = p.flatten(2).transpose(1, 2)       # (B, N_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        tokens = torch.cat([cls, p], dim=1)    # (B, N+1, embed_dim)
        tokens = self.pos_drop(tokens)
        out = self.transformer(tokens)         # (B, N+1, embed_dim)
        out = self.norm(out)
        cls_out = out[:, 0, :]                 # (B, embed_dim) — CLS token
        rul_pred = self.regressor(cls_out)     # (B, 1)
        dummy_1d = torch.zeros(B, 1, 1, device=x_2d.device)
        return rul_pred.squeeze(-1), torch.tensor(0.5), cls_out, dummy_1d, p


# ─────────────────────────────────────────────────────────────────────────────
# H: Deep 1D ResNet Baseline
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """Standard pre-activation 1D residual block."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class DeepResNet1DBaseline(nn.Module):
    """
    Deep 1D ResNet with 8 residual blocks operating on raw vibration signals.
    Substantially deeper than the 3-block 1D-CNN in DSAF Variant A.
    Provides a strong purely-vibration baseline for SOTA comparison.

    Reference architecture type: He et al. (2016) adapted to 1D PHM signals.
    """
    def __init__(self, n_res_blocks=8, base_channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(base_channels), nn.GELU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=7, stride=4,
                      padding=3, bias=False),
            nn.BatchNorm1d(base_channels), nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock1D(base_channels) for _ in range(n_res_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(base_channels, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x_1d, x_2d=None, use_checkpointing=False):
        feat = self.stem(x_1d)                 # (B, base_channels, T)
        feat = self.res_blocks(feat)           # (B, base_channels, T)
        pooled = self.pool(feat).squeeze(-1)   # (B, base_channels)
        rul_pred = self.regressor(pooled)      # (B, 1)
        dummy_2d = torch.zeros(x_1d.shape[0], 1, 1, device=x_1d.device)
        return rul_pred.squeeze(-1), torch.tensor(0.0), pooled, feat, dummy_2d
