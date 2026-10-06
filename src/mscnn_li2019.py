"""
mscnn_li2019.py — Multi-scale CNN of Li, Zhang and Ding (2019), "Deep learning-based
remaining useful life estimation of bearings using multi-scale feature extraction",
Reliability Engineering and System Safety 182:208-218 (ref [7]), reproduced under the
leakage-free protocol (revision R1, reviewer comment 6; analysis plan Section 6).

As published (Sections 2.1, 3.3 and 3.5, Table 2, Fig. 2 of [7]):
  input: horizontal vibration, the N_hist = 5 most recent recordings (2560 samples each)
  STFT: Hamming window 20 ms (512 samples at 25.6 kHz), frame shift 10 ms (256);
        lower half of the frequency range (128 bins); max-pooling 1 x 2 along frequency
  per recording: three 3 x 3 convolutions with 5 filters in sequence, their three output
        maps concatenated (multi-scale), a fourth 3 x 3 convolution with 5 filters
  the five per-recording high-level maps are added element-wise; a 3 x 3 convolution with
  one filter; flatten; fully connected layer with 128 units; one sigmoid output unit
  Leaky ReLU activations, dropout 0.5, zero padding, Xavier normal initialization
  (training: Adam, learning rate 1e-4, batch 32, MAE loss; see Experiments/r1_mscnn.py)

Not specified in [7], chosen here (plan Section 6): centred STFT frames with reflect
padding (11 frames per recording, map 64 x 11); periodic Hamming window (the STFT
convention); separate layers for each recording (Fig. 2 indexes each recording's layers
separately); Leaky ReLU slope 0.01; dropout after each of the four per-recording
convolutions (Fig. 2) and none after the fully connected layer; Leaky ReLU after the
one-filter convolution; zero biases.

input_norm='bn' is the pre-registered fallback if the smoke test fails: non-affine batch
normalization of the STFT magnitude (statistics from training batches only).
Parameters with the defaults: 96,365.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCNN(nn.Module):
    def __init__(self, n_hist: int = 5, window: int = 2560, n_fft: int = 512, hop: int = 256,
                 n_bins: int = 128, filters: int = 5, fc_units: int = 128,
                 dropout: float = 0.5, slope: float = 0.01, input_norm: str = None):
        super().__init__()
        self.n_hist, self.n_fft, self.hop, self.n_bins = n_hist, n_fft, hop, n_bins
        self.register_buffer('stft_window', torch.hamming_window(n_fft), persistent=False)
        self.frames = 1 + window // hop
        self.norm = nn.BatchNorm2d(1, affine=False) if input_norm == 'bn' else None

        def conv(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                                 nn.LeakyReLU(slope), nn.Dropout(dropout))

        self.extractors = nn.ModuleList(
            nn.ModuleDict({'c1': conv(1, filters), 'c2': conv(filters, filters),
                           'c3': conv(filters, filters), 'c4': conv(3 * filters, filters)})
            for _ in range(n_hist))
        self.compress = nn.Sequential(nn.Conv2d(filters, 1, kernel_size=3, padding=1),
                                      nn.LeakyReLU(slope))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear((n_bins // 2) * self.frames, fc_units),
                                nn.LeakyReLU(slope))
        self.out = nn.Linear(fc_units, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def tf_maps(self, x: torch.Tensor) -> torch.Tensor:
        """(N, window) raw recordings -> (N, 1, n_bins // 2, frames) STFT magnitude maps."""
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.stft_window,
                          center=True, return_complex=True)
        mag = spec.abs()[:, :self.n_bins, :].unsqueeze(1)
        return F.max_pool2d(mag, kernel_size=(2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:                       # (B, 1, N_hist, window) from the window dataset
            x = x.squeeze(1)
        b, k, length = x.shape
        if k != self.n_hist:
            raise ValueError(f'expected {self.n_hist} recordings per sample, got {k}')
        maps = self.tf_maps(x.reshape(b * k, length))
        if self.norm is not None:
            maps = self.norm(maps)
        maps = maps.reshape(b, k, 1, self.n_bins // 2, self.frames)
        high = 0
        for i, ext in enumerate(self.extractors):   # recordings ordered oldest -> newest
            f1 = ext['c1'](maps[:, i])
            f2 = ext['c2'](f1)
            f3 = ext['c3'](f2)
            high = high + ext['c4'](torch.cat([f1, f2, f3], dim=1))
        return torch.sigmoid(self.out(self.fc(self.compress(high)))).squeeze(-1)
