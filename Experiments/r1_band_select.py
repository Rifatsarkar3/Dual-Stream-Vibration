"""
r1_band_select.py — Revision R1, analysis plan Amendment 10, fix E: the demodulation band of the
envelope stream (src/components_r2.py), chosen for one dataset on its four training bearings only.

Rule (Amendment 10):
  candidates  bands [lo, lo + W] with W in {800, 1600, 3200, 6400} Hz and lo a positive multiple
              of W/2 with lo + W <= 12,800 Hz: 52 bands on the half-overlapping dyadic grid of the
              kurtogram;
  windows     of each training bearing, those of its last 20 % of life (y <= 0.2), where defect
              impacts excite the structural resonances;
  statistic   spectral kurtosis of the band's analytic signal a (positive-frequency FFT bins in the
              band, doubled, exactly as the extractor forms it): K = mean|a|^4 / (mean|a|^2)^2 - 2,
              0 for Gaussian noise and larger the more impulsive the band;
  score       mean over the four training bearings of the bearing's mean K;
  choice      the band with the largest score (ties: narrower band, then lower edge).
Test and zero-shot bearings are not loaded.

USAGE:
  python Experiments/r1_band_select.py --dataset pronostia
  python Experiments/r1_band_select.py --dataset xjtu
Output: outputs/r1_components/band_<dataset>.json, with the score of every candidate.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FS = 25600.0
WIDTHS = (800, 1600, 3200, 6400)
LATE = 0.2
DATA = {'pronostia': 'Processed_PRONOSTIA_v2', 'xjtu': 'Processed_XJTU'}
OUT = ROOT / 'outputs' / 'r1_components'


def candidate_bands(fs: float = FS, widths=WIDTHS):
    bands = []
    for w in widths:
        lo = w // 2
        while lo + w <= fs / 2:
            bands.append((float(lo), float(lo + w)))
            lo += w // 2
    return bands


def band_kurtosis(spec: np.ndarray, band, fs: float = FS) -> np.ndarray:
    """Spectral kurtosis of the band's analytic signal, per window.
    spec: (n, L) complex FFT of the windows; returns (n,)."""
    f = np.fft.fftfreq(spec.shape[-1], d=1.0 / fs)
    h = 2.0 * ((f >= band[0]) & (f <= band[1]))
    power = np.abs(np.fft.ifft(spec * h, axis=-1)) ** 2
    return (power ** 2).mean(axis=-1) / power.mean(axis=-1) ** 2 - 2.0


def select_band(late_windows: dict, fs: float = FS, widths=WIDTHS):
    """late_windows: bearing name -> (n, L) array. Returns (selected band, ranked candidates)."""
    bands = candidate_bands(fs, widths)
    per_bearing = {}
    for name, w in late_windows.items():
        spec = np.fft.fft(w.astype(np.float64), axis=-1)
        per_bearing[name] = [float(band_kurtosis(spec, b, fs).mean()) for b in bands]
    rows = []
    for i, b in enumerate(bands):
        rows.append({'band_hz': list(b), 'width_hz': b[1] - b[0],
                     'score': float(np.mean([v[i] for v in per_bearing.values()])),
                     'per_bearing': {n: v[i] for n, v in per_bearing.items()}})
    rows.sort(key=lambda r: (-r['score'], r['width_hz'], r['band_hz'][0]))
    return tuple(rows[0]['band_hz']), rows


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=sorted(DATA), required=True)
    args = ap.parse_args()

    from Experiments.exp16_ress import CONFIG, load_bearings
    from Experiments.r1_provenance import provenance
    if args.dataset == 'xjtu':
        from Experiments.exp17_xjtu import XJTU_CONFIG as cfg
    else:
        cfg = CONFIG
    data_dir = ROOT / DATA[args.dataset]
    train = load_bearings(str(data_dir), cfg['train_bearings'])  # training bearings only
    late = {b.name: b.vib[b.rul <= LATE] for b in train}
    for name, w in late.items():
        if len(w) == 0:
            raise ValueError(f'{name}: no windows with y <= {LATE}')

    band, rows = select_band(late)
    print(f'[band] {args.dataset}: late-life windows ' +
          ', '.join(f'{n} {len(w)}' for n, w in late.items()))
    print('[band] best five candidates (score = mean over bearings of mean spectral kurtosis):')
    for r in rows[:5]:
        print(f"  {r['band_hz'][0]:7.0f} to {r['band_hz'][1]:7.0f} Hz  score {r['score']:.3f}")
    print(f'[band] selected: {band[0]:.0f} to {band[1]:.0f} Hz')

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'band_{args.dataset}.json'
    out.write_text(json.dumps({
        'analysis': 'R1 analysis plan, Amendment 10, fix E: demodulation band of the envelope stream',
        'dataset': args.dataset,
        'data_dir': DATA[args.dataset],
        'late_life_threshold': LATE,
        'late_windows': {n: int(len(w)) for n, w in late.items()},
        'band_hz': list(band),
        'candidates': rows,
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'provenance': provenance(extra_code=('Experiments/r1_band_select.py',)),
    }, indent=2), encoding='utf-8')
    print(f'[band] saved {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
