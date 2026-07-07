"""
generate_fig1.py — Architecture block diagram (Figure 1) for manuscript_v2.

Re-renders Figure 1 to match the ACTUAL lightweight model (src/dsaf_v2_lite.py,
manuscript Section 3) — replacing the stale old-draft figure that depicted a
2D-STFT -> ResNet-18 spectral stream. The true model:

  * Temporal stream : 1D CNN on raw vibration (k 15/7/3, ch 1->64->128->128)
  * Spectral stream : FIXED non-learnable Hilbert-envelope-spectrum extractor
                      (Hilbert -> |envelope| -> FFT -> 512 bins) + 3-layer 1D CNN
                      (k 7/5/3, ch 1->64->64->64)
  * Adaptive scalar fusion gate alpha_t
  * Regression head -> RUL
  * Training-time only (dashed, discarded at inference): PGMC auxiliary
    monotonic penalty + binary DANN domain discriminator
  * 192,130 inference parameters; 1.48 ms FP32 latency / 100 ms window

Output: paper/figures_v2/Fig1_architecture.png (600 dpi).
RUN: .venv\\Scripts\\python.exe files/generate_fig1.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'paper' / 'figures_v2' / 'Fig1_architecture.png'
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans',
                     'mathtext.fontset': 'dejavusans'})

C_INPUT = '#d9e8f5'
C_TEMP = '#cfe8d4'
C_SPEC = '#fde0c2'
C_FIXED = '#f6f0c0'
C_FUSE = '#e3d4f0'
C_HEAD = '#f5d0d0'
C_TRAIN = '#fbfbfb'
EDGE = '#333333'
GREY = '#8a8a8a'


def box(ax, x, y, w, h, text, fc, fontsize=7.4, ls='-', lw=1.1, weight='normal'):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.6',
                       linewidth=lw, edgecolor=EDGE, facecolor=fc,
                       linestyle=ls, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, zorder=3, weight=weight)


def arrow(ax, p0, p1, ls='-', color=EDGE, lw=1.1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls,
                                 zorder=1, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis('off')

    # ---------- forward (inference) path ----------
    box(ax, 1.5, 23, 16, 12,
        'Raw vibration\nwindow\n$x_t \\in \\mathbb{R}^{1 \\times 2560}$\n(100 ms @ 25.6 kHz)',
        C_INPUT, fontsize=7.2)

    # temporal lane (top)
    box(ax, 22, 38, 30, 12,
        'Temporal stream — 1D CNN\nConv1d $\\times$3  (k 15/7/3, str 4/4/2)\n'
        'ch 1$\\to$64$\\to$128$\\to$128, BN + GELU\nGAP $\\to$ Linear',
        C_TEMP, fontsize=7.0)

    # spectral lane (bottom): fixed extractor -> CNN
    box(ax, 16.5, 7, 20, 12,
        'Fixed envelope\nextractor (non-learnable)\nHilbert $\\to$ |env| $\\to$ FFT\n'
        '$\\to$ 512 bins\n(BPFO/BPFI/BSF/FTF)',
        C_FIXED, fontsize=6.6)
    box(ax, 38, 7, 28, 12,
        'Spectral stream — 1D CNN\nConv1d $\\times$3  (k 7/5/3, str 2)\n'
        'ch 1$\\to$64$\\to$64$\\to$64, BN + GELU\nGAP $\\to$ Linear',
        C_SPEC, fontsize=7.0)

    # fusion gate
    box(ax, 70, 27, 17, 13,
        'Adaptive scalar\nfusion gate\n$\\alpha_t = \\sigma(\\cdot)$\n'
        '$F_t = \\alpha_t F_t^{\\mathrm{temp}}$\n$+ (1-\\alpha_t) F_t^{\\mathrm{spec}}$',
        C_FUSE, fontsize=6.9)

    # regression head
    box(ax, 70, 9, 17, 13,
        'Regression head\nLN $\\to$ Drop\n$\\to$ Lin(128,64) $\\to$ GELU\n'
        '$\\to$ Drop $\\to$ Lin(64,1)',
        C_HEAD, fontsize=6.9)

    # output
    box(ax, 90.5, 12, 8.5, 7, '$\\hat{y}_t$\n(RUL)', C_INPUT, fontsize=9,
        weight='bold')

    # ---------- training-time only (dashed band, top) ----------
    ax.text(50, 62.3, 'Training-time only  —  discarded at inference',
            ha='center', va='center', fontsize=7.3, style='italic', color=GREY)
    box(ax, 16, 51, 31, 9,
        'PGMC auxiliary monotonic penalty\n'
        'chronological within-bearing chunk\n'
        '(BN frozen), $\\lambda_{\\mathrm{PGMC}}(e)$ warm-up',
        C_TRAIN, fontsize=6.5, ls=(0, (4, 2)))
    box(ax, 57, 51, 33, 9,
        'DANN binary domain discriminator (GRL)\n'
        'Lin(128,64) $\\to$ Lin(64,1), 8.3k params\n'
        'backbone gets GRL-reversed gradient',
        C_TRAIN, fontsize=6.5, ls=(0, (4, 2)))

    # ---------- arrows: forward ----------
    arrow(ax, (17.5, 30), (22, 43))            # input -> temporal
    arrow(ax, (17.5, 28), (18, 15))            # input -> envelope extractor
    arrow(ax, (36.5, 13), (38, 13))            # extractor -> spectral CNN
    arrow(ax, (52, 44), (70, 35))              # temporal -> gate
    arrow(ax, (66, 14), (70, 30))              # spectral -> gate
    arrow(ax, (78.5, 27), (78.5, 22))          # gate -> head
    arrow(ax, (87, 15.5), (90.5, 15.5))        # head -> output

    # feature-dim labels on the fusion inputs
    ax.text(60.5, 41, '$F_t^{\\mathrm{temp}}\\in\\mathbb{R}^{128}$',
            ha='center', va='center', fontsize=6.7, color='#2f6b3a')
    ax.text(63.5, 21.5, '$F_t^{\\mathrm{spec}}\\in\\mathbb{R}^{128}$',
            ha='center', va='center', fontsize=6.7, color='#9a5a18')

    # ---------- arrows: training-time (dashed) ----------
    arrow(ax, (74, 22), (38, 51), ls=(0, (4, 2)), color=GREY)   # y_hat path -> PGMC
    arrow(ax, (83, 40), (78, 51), ls=(0, (4, 2)), color=GREY)   # fused -> DANN

    ax.text(50, -1.5,
            'Inference graph: 192,130 parameters  ·  1.48 ms FP32 latency per 100 ms window',
            ha='center', va='center', fontsize=7.2, style='italic', color='#444444')

    fig.savefig(OUT, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
