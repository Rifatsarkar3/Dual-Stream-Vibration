"""
vanilla_baselines.py — Minimal, Canonical Baseline Architectures
==================================================================

Two clean, well-established baseline architectures for bearing RUL prediction
on the PRONOSTIA benchmark. These are intentionally simple to provide fair
comparison points for DSAF+PGMC.

Both expose the same forward signature as DSAFLite for trainer compatibility:
  forward(x_1d, x_2d=None, use_checkpointing=False)
  → (rul_pred, gate_alpha_dummy, feat_pool, feat_1d, feat_2d_dummy)
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Vanilla 1D CNN
# ─────────────────────────────────────────────────────────────────────────────

class Vanilla1DCNN(nn.Module):
    """
    Simple multi-channel 1D CNN for bearing RUL prediction.

    Architecture:
      - 4-5 conv blocks (each: Conv1d → BatchNorm → ReLU → MaxPool)
      - Global average pooling
      - Linear regression head → RUL output

    Takes raw 1D vibration signals (2560 samples) directly.
    ~0.5M parameters, no frills.

    Reference: Standard industry baseline for time-series regression
    (e.g., Li et al. 2019, Zhu et al. 2019).
    """

    def __init__(self, in_channels: int = 1, num_conv_blocks: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.num_conv_blocks = num_conv_blocks

        # Build conv blocks dynamically
        conv_layers = []
        channels = [in_channels, 32, 64, 128, 256, 256][:num_conv_blocks + 1]
        kernel_sizes = [15, 7, 5, 3, 3]
        strides = [4, 4, 2, 2, 1]
        paddings = [7, 3, 2, 1, 1]

        for i in range(num_conv_blocks):
            conv_layers.append(
                nn.Conv1d(
                    channels[i], channels[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i]
                )
            )
            conv_layers.append(nn.BatchNorm1d(channels[i + 1]))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))

        self.conv_blocks = nn.Sequential(*conv_layers)

        # Global average pooling → regression head
        self.pool = nn.AdaptiveAvgPool1d(1)
        final_dim = channels[-1]
        self.regressor = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x_1d, x_2d=None, use_checkpointing=False):
        """
        Args:
            x_1d: (B, 1, 2560) raw vibration
            x_2d: ignored
            use_checkpointing: ignored

        Returns:
            rul_pred: (B,) RUL predictions
            gate_alpha_dummy: scalar 0.0 (for API compatibility)
            feat_pool: (B, final_dim) pooled features
            feat_1d: (B, final_dim, T) conv output
            feat_2d_dummy: dummy tensor
        """
        feat = self.conv_blocks(x_1d)  # (B, C, T)
        feat_pool = self.pool(feat).squeeze(-1)  # (B, C)
        rul_pred = self.regressor(feat_pool)  # (B, 1)

        dummy_2d = torch.zeros(x_1d.shape[0], 1, 1, device=x_1d.device)
        return rul_pred.squeeze(-1), torch.tensor(0.0, device=x_1d.device), \
               feat_pool, feat, dummy_2d


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: BiLSTM (Bidirectional LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class VanillaBiLSTM(nn.Module):
    """
    Standard bidirectional LSTM for time-series RUL prediction.

    Architecture:
      - 1D CNN feature extractor (optional, minimal)
      - BiLSTM encoder (2 layers, 128 hidden)
      - Last hidden state → linear regressor

    Captures temporal degradation dynamics without explicit dual-stream fusion.
    ~0.8M parameters.

    Reference: Widely used in prognostics (e.g., Huang et al. 2018).
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Lightweight CNN feature extractor
        self.cnn_feat = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        # After CNN: (B, 128, ~80) for 2560-length input

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x_1d, x_2d=None, use_checkpointing=False):
        """
        Args:
            x_1d: (B, 1, 2560) raw vibration
            x_2d: ignored
            use_checkpointing: ignored

        Returns:
            rul_pred: (B,) RUL predictions
            gate_alpha_dummy: scalar 0.0 (for API compatibility)
            feat_pool: (B, hidden_dim*2) LSTM output at last timestep
            feat_1d: (B, T, hidden_dim*2) full LSTM output sequence
            feat_2d_dummy: dummy tensor
        """
        # CNN feature extraction
        x_feat = self.cnn_feat(x_1d)  # (B, 128, T)
        x_feat = x_feat.transpose(1, 2)  # (B, T, 128)

        # BiLSTM
        lstm_out, (h_n, c_n) = self.lstm(x_feat)  # (B, T, 256), tuple of (2, B, 128)

        # Last hidden state (bidirectional: concat forward + backward)
        feat_pool = lstm_out[:, -1, :]  # (B, 256)

        rul_pred = self.regressor(feat_pool)  # (B, 1)

        dummy_2d = torch.zeros(x_1d.shape[0], 1, 1, device=x_1d.device)
        return rul_pred.squeeze(-1), torch.tensor(0.0, device=x_1d.device), \
               feat_pool, lstm_out, dummy_2d


# ─────────────────────────────────────────────────────────────────────────────
# Factory function for trainer compatibility
# ─────────────────────────────────────────────────────────────────────────────

def create_vanilla_baseline(model_type: str = 'cnn', **kwargs) -> nn.Module:
    """
    Factory to instantiate vanilla baselines.

    Args:
        model_type: 'cnn' or 'lstm'
        **kwargs: passed to model constructor

    Returns:
        model: PyTorch nn.Module
    """
    if model_type.lower() in ['cnn', '1dcnn', 'vanilla1dcnn']:
        return Vanilla1DCNN(**kwargs)
    elif model_type.lower() in ['lstm', 'bilstm', 'vanillabilstm']:
        return VanillaBiLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'cnn' or 'lstm'.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test Vanilla1DCNN
    print("=" * 60)
    print("Testing Vanilla1DCNN")
    print("=" * 60)
    cnn_model = Vanilla1DCNN(num_conv_blocks=4).to(device)
    x_test = torch.randn(8, 1, 2560).to(device)
    rul_pred, gate, feat_pool, feat_1d, feat_2d = cnn_model(x_test)
    print(f"Input shape:        {x_test.shape}")
    print(f"Output RUL shape:   {rul_pred.shape}")
    print(f"Feature pool shape: {feat_pool.shape}")
    print(f"Num parameters:     {sum(p.numel() for p in cnn_model.parameters()):,}")

    # Test VanillaBiLSTM
    print("\n" + "=" * 60)
    print("Testing VanillaBiLSTM")
    print("=" * 60)
    lstm_model = VanillaBiLSTM(hidden_dim=128, num_layers=2).to(device)
    rul_pred, gate, feat_pool, feat_1d, feat_2d = lstm_model(x_test)
    print(f"Input shape:        {x_test.shape}")
    print(f"Output RUL shape:   {rul_pred.shape}")
    print(f"Feature pool shape: {feat_pool.shape}")
    print(f"LSTM seq shape:     {feat_1d.shape}")
    print(f"Num parameters:     {sum(p.numel() for p in lstm_model.parameters()):,}")
