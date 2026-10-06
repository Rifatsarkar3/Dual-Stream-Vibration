"""
components_r2.py — Revision R1, analysis plan Amendment 10: the three components of the
dual-stream network rebuilt so that each can do what the submitted version was meant to do.

Corrected ablations (R1_corrected_results.md, Sections 4, 8 and 9d) showed that no submitted
component has a measurable effect. The code shows why:

  E  The envelope stream demodulated the whole 0 to 12.8 kHz range and divided every
     window's spectrum by its maximum (src/dsaf_v2_lite.py, EnvelopeSpectrumExtractor).
     Envelope analysis demodulates a resonance band excited by the defect impacts, and the
     division removed the growth of vibration amplitude with wear. Here: BandEnvelope-
     SpectrumExtractor, a fixed band chosen on the training bearings
     (Experiments/r1_band_select.py), log amplitude, no per-window normalization.
  G  The fusion gate was trained on clean windows only, so nothing in training made the
     reliability of the two streams differ and the gate weight stayed constant. Here:
     InputNoise, additive white Gaussian noise at a random per-window SNR during training
     only, shared by both streams (WithInputNoise).
  M  PGMC penalized increases between adjacent windows, where the true remaining life falls
     by 1/T (0.0004 to 0.0013) and the prediction scatter dominates. Here: PairBatchSampler
     and pair_penalty, the same squared hinge on window pairs 5 to 10 % of the bearing life
     apart.

Experiments/r1_components.py plugs these into Experiments/exp16_ress.py without modifying it.
"""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


class BandEnvelopeSpectrumExtractor(nn.Module):
    """Envelope spectrum of one resonance band, in log amplitude.

    The analytic signal is formed from the positive-frequency bins inside [f_lo, f_hi] only,
    which band-passes and Hilbert-transforms in one step. The one-sided magnitude spectrum of
    its envelope is divided by the window length (bin 0 is then the mean envelope) and log10 is
    taken. Nothing is normalized per window: a window with twice the envelope amplitude has a
    spectrum higher by log10(2) in every bin above the floor. BatchNorm over the batch
    standardizes the input of the convolutional stream without removing these differences.
    """

    def __init__(self, band: Sequence[float], fs: float = 25600.0, signal_len: int = 2560,
                 n_fft_out: int = 512, eps: float = 1e-6):
        super().__init__()
        lo, hi = float(band[0]), float(band[1])
        if not 0.0 < lo < hi <= fs / 2:
            raise ValueError(f'band {band} is not inside (0, {fs / 2}] Hz')
        freqs = torch.fft.fftfreq(signal_len, d=1.0 / fs)
        self.register_buffer('h', 2.0 * ((freqs >= lo) & (freqs <= hi)).float())
        self.register_buffer('band_hz', torch.tensor([lo, hi]))
        self.n_fft_out = n_fft_out
        self.eps = eps
        self.norm = nn.BatchNorm1d(1)

    def spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, L) raw windows -> (B, n_fft_out) log10 envelope spectrum, before BatchNorm."""
        x = x.squeeze(1)
        analytic = torch.fft.ifft(torch.fft.fft(x, dim=-1) * self.h, dim=-1)
        mag = torch.fft.rfft(analytic.abs(), dim=-1).abs()[:, :self.n_fft_out] / x.shape[-1]
        return torch.log10(mag + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.spectrum(x).unsqueeze(1))


class InputNoise(nn.Module):
    """Training-only additive white Gaussian noise at a random SNR per window.

    With probability p a window receives noise of power P_signal / 10^(SNR/10), the SNR drawn
    uniformly from snr_db; the definition of SNR is that of Experiments/exp18_noise_gate.py.
    Identity in eval mode, so validation, test and MC-dropout predictions (which switch only the
    Dropout layers back to training) see clean windows.
    """

    def __init__(self, snr_db: Tuple[float, float] = (0.0, 30.0), p: float = 0.5):
        super().__init__()
        self.snr_db = (float(snr_db[0]), float(snr_db[1]))
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        n = x.shape[0]
        lo, hi = self.snr_db
        snr = lo + (hi - lo) * torch.rand(n, device=x.device, dtype=x.dtype)
        apply = (torch.rand(n, device=x.device) < self.p).to(x.dtype)
        p_sig = x.pow(2).mean(dim=tuple(range(1, x.dim())))
        sd = torch.sqrt(p_sig / torch.pow(10.0, snr / 10.0)) * apply
        return x + torch.randn_like(x) * sd.view((n,) + (1,) * (x.dim() - 1))


class WithInputNoise(nn.Module):
    """A model with InputNoise in front: both streams see the same noisy window, as they would
    see the same noisy sensor signal. Outputs are those of the wrapped model."""

    def __init__(self, base: nn.Module, noise: InputNoise):
        super().__init__()
        self.base = base
        self.noise = noise

    def forward(self, x: torch.Tensor, x_2d=None, use_checkpointing=False):
        return self.base(self.noise(x))


class PairBatchSampler(torch.utils.data.Sampler):
    """Batches of window pairs from the training set for the pair PGMC penalty.

    Each batch lists n_pairs earlier positions followed by the n_pairs matching later
    positions. A pair is drawn by choosing a bearing with probability proportional to its
    number of windows, a gap k = round(u * n_b) with u uniform in gap_frac, where n_b is the
    bearing's number of training windows, and a start t uniform in {0, ..., n_b - k - 1};
    the pair is (window t, window t + k) of that bearing in chronological order.
    """

    def __init__(self, entries: List[Tuple[int, int]], n_pairs: int,
                 gap_frac: Tuple[float, float], n_batches: int, generator: torch.Generator):
        by_bearing: Dict[int, List[Tuple[int, int]]] = {}
        for pos, (b_idx, w) in enumerate(entries):
            by_bearing.setdefault(int(b_idx), []).append((int(w), pos))
        self.positions = [[pos for _w, pos in sorted(v)] for _b, v in sorted(by_bearing.items())]
        if any(len(p) < 2 for p in self.positions):
            raise ValueError('every bearing needs at least two training windows')
        sizes = torch.tensor([len(p) for p in self.positions], dtype=torch.float64)
        self.weights = sizes / sizes.sum()
        self.n_pairs = int(n_pairs)
        self.gap_frac = (float(gap_frac[0]), float(gap_frac[1]))
        self.n_batches = int(n_batches)
        self.generator = generator

    def __iter__(self):
        g = self.generator
        lo, hi = self.gap_frac
        for _ in range(self.n_batches):
            early, late = [], []
            chosen = torch.multinomial(self.weights, self.n_pairs, replacement=True, generator=g)
            for b in chosen.tolist():
                pos = self.positions[b]
                n = len(pos)
                u = lo + (hi - lo) * torch.rand(1, generator=g, dtype=torch.float64).item()
                k = min(max(1, round(u * n)), n - 1)
                t = int(torch.randint(0, n - k, (1,), generator=g).item())
                early.append(pos[t])
                late.append(pos[t + k])
            yield early + late

    def __len__(self):
        return self.n_batches


def pair_penalty(pred: torch.Tensor, n_pairs: int) -> torch.Tensor:
    """Squared hinge of the PGMC equation on pairs: mean of max(0, y_hat_late - y_hat_early)^2,
    where pred holds the n_pairs earlier predictions followed by the n_pairs later ones."""
    return torch.relu(pred[n_pairs:2 * n_pairs] - pred[:n_pairs]).pow(2).mean()
