"""
published_baselines.py — Published bearing-RUL architectures reproduced under the
leakage-free protocol (revision R1, reviewer comment 6).

TSMCCNN: the time-series multi-channel CNN of Jiang, Lee and Zeng (2020), "Time series
multiple channel convolutional neural network with attention-based long short-term
memory for predicting bearing remaining useful life", Sensors 20(1):166 (ref [21]).
Layer specification from [21] (K = 4 channels):
  input 2 x 2560 (horizontal, vertical) scaled to [0, 1], split into K = 4 consecutive
  640-sample segments stacked as input channels -> tensor (4, 2, 640)
  Conv 64 @ 2x10, stride 1, no padding -> LeakyReLU(0.2) -> AvgPool 1x10
  Conv 32 @ 1x10, stride 1, no padding -> LeakyReLU(0.2) -> AvgPool 1x10
  Flatten (160) -> Dense(1)
[21] trains with MSE and Adam (set in exp19's MODEL_TRAINING). Choices for details [21]
leaves unspecified: Adam learning rate 1e-3 (library default); batch size 32 and epoch
budget from the protocol; linear output; LeakyReLU after the second convolution as
after the first. The second convolution gives width 54 where [21]'s table prints 53;
the flattened size (160) is unchanged.

Input scaling (documented deviation): [21] scales inputs to [0, 1]. Read as a fixed or
dataset-level min-max, that scaling leaves healthy-state variation at ~0.004 around 0.5
and the network collapses to the mean predictor, even under [21]'s own random 70/30
window split (test MAE 0.246, prediction std 0). Per-sensor standardization trains
(random-split MAE 0.046 after 100 epochs vs 0.0214 reported). input_norm='bn' implements
it as a non-affine BatchNorm over the two sensor channels: statistics come from training
batches only, so no validation or test information enters it.
"""

import torch
import torch.nn as nn


class TSMCCNN(nn.Module):
    def __init__(self, n_segments: int = 4, window: int = 2560, input_norm: str = 'bn',
                 full_scale_g: float = 50.0):
        super().__init__()
        if window % n_segments:
            raise ValueError('window must split evenly into segments')
        self.k = n_segments
        self.full_scale = full_scale_g
        self.input_norm = input_norm
        self.norm = nn.BatchNorm1d(2, affine=False) if input_norm == 'bn' else None
        self.features = nn.Sequential(
            nn.Conv2d(n_segments, 64, kernel_size=(2, 10)),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=(1, 10)),
            nn.Conv2d(64, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=(1, 10)),
        )
        width = window // n_segments - 9            # 631
        width = (width // 10 - 9) // 10              # 63 -> 54 -> 5
        self.head = nn.Linear(32 * width, 1)         # 160

    def forward(self, x):
        if x.dim() == 4:                             # (B, 1, 2, 2560) from the loader
            x = x.squeeze(1)
        b, sensors, n = x.shape
        if self.norm is not None:
            x = self.norm(x)                                   # per-sensor standardization
        else:                                                  # [21]'s [0, 1] scaling, fixed range
            x = (x + self.full_scale) / (2 * self.full_scale)
        x = x.reshape(b, sensors, self.k, n // self.k).permute(0, 2, 1, 3)  # (B, K, 2, 640)
        return self.head(self.features(x).flatten(1)).squeeze(1)
