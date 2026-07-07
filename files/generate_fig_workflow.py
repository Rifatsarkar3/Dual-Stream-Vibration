"""
generate_fig_workflow.py — Reliability-oriented workflow diagram, new Figure 1
for the RESS-framed manuscript (paper/latex/manuscript.tex).

Per the RESS revision plan, the manuscript's first figure must communicate the
paper's full reliability-oriented contribution (leakage-free validation ->
estimator -> degradation-consistency -> uncertainty calibration -> maintenance
decision assessment), NOT the neural-network architecture. The architecture
diagram (files/generate_fig1.py) becomes Figure 2 instead.

Layout: a 3-column x 2-row "snake" flow (data -> validation -> estimator,
wrapping down to consistency -> calibration -> decision), matching the six
pipeline blocks specified in the revision plan, with a dashed boundary around
the four middle blocks ("model development without test leakage") and a solid
boundary around the decision block ("decision-level evaluation on held-out
bearings"), ending in a single output callout.

Output: paper/figures_v2/Fig1_workflow.png (600 dpi), copied to
paper/latex/media/image1_workflow.png for LaTeX inclusion.
RUN: .venv\\Scripts\\python.exe files/generate_fig_workflow.py
"""

from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'paper' / 'figures_v2' / 'Fig1_workflow.png'
MEDIA_OUT = ROOT / 'paper' / 'latex' / 'media' / 'image1_workflow.png'
OUT.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans',
                     'mathtext.fontset': 'dejavusans'})

C_DATA = '#d9e8f5'
C_VALID = '#fde0c2'
C_EST = '#cfe8d4'
C_CONSIST = '#f6f0c0'
C_UQ = '#e3d4f0'
C_DECIDE = '#f5d0d0'
C_OUT = '#dcdcdc'
EDGE = '#333333'
GREY = '#7a7a7a'


def box(ax, x, y, w, h, title, sub, fc, title_fs=7.6, sub_fs=6.1, lw=1.2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.02,rounding_size=0.7',
                       linewidth=lw, edgecolor=EDGE, facecolor=fc, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 3.4, title, ha='center', va='top',
            fontsize=title_fs, weight='bold', zorder=4)
    ax.text(x + w / 2, y + h - 7.0, sub, ha='center', va='top',
            fontsize=sub_fs, zorder=4, linespacing=1.55)


def arrow(ax, p0, p1, lw=1.3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=13,
                                 linewidth=lw, color=EDGE, zorder=2,
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    ax.set_xlim(-2, 100)
    ax.set_ylim(-14, 70)
    ax.axis('off')

    row1_y, row2_y, h = 46, 12, 20
    col1_x, col2_x, col3_x, w = 0, 35, 70, 28

    # ---- dashed "model development without test leakage" boundary ----
    dash = Rectangle((col2_x - 3, row2_y - 3), (col3_x + w) - (col2_x - 3) + 3,
                      (row1_y + h) - (row2_y - 3) + 3,
                      fill=False, linestyle=(0, (5, 3)), edgecolor=GREY, lw=1.3,
                      zorder=1)
    ax.add_patch(dash)
    ax.text(col2_x + ((col3_x + w) - col2_x) / 2, row1_y + h + 4,
            'model development without test leakage',
            ha='center', va='bottom', fontsize=7.0, style='italic', color=GREY)

    # ---- solid "decision-level evaluation on held-out bearings" boundary ----
    solid = Rectangle((col1_x - 1, row2_y - 3), w + 2, h + 6,
                       fill=False, linestyle='-', edgecolor=EDGE, lw=1.3, zorder=1)
    ax.add_patch(solid)
    ax.text(col1_x + w / 2, row2_y + h + 4,
            'decision-level evaluation\non held-out bearings',
            ha='center', va='bottom', fontsize=7.0, style='italic', color=GREY,
            linespacing=1.4)

    # ---- row 1: data -> leak-free validation -> estimator ----
    box(ax, col1_x, row1_y, w, h,
        'Run-to-failure\nbearing data',
        'vibration windows\nbearing ID\noperating condition\nRUL labels',
        C_DATA)
    box(ax, col2_x, row1_y, w, h,
        'Leakage-free validation\nprotocol',
        'bearing-level train/val/test split\nindependent budget sweep\nno test-bearing model selection',
        C_VALID)
    box(ax, col3_x, row1_y, w, h,
        'Compact RUL estimator',
        'raw-vibration branch\nenvelope-spectrum branch\nDSAFLite+PGMC\npoint RUL prediction',
        C_EST)

    # ---- row 2 (reversed): consistency -> calibration -> decision ----
    box(ax, col3_x, row2_y, w, h,
        'Degradation-consistency\nlayer',
        'raw trajectory\noffline PAVA (retrospective only)\ncausal CES / RTRM (deployment path)',
        C_CONSIST)
    box(ax, col2_x, row2_y, w, h,
        'Uncertainty calibration',
        'MC-dropout samples\ncalibration data only\nlower-bound coverage (LBC)\nno test-set calibration',
        C_UQ)
    box(ax, col1_x, row2_y, w, h,
        'Maintenance decision\nassessment',
        'alarm threshold ' + r'$\tau$' + '\nmissed / false alarm rate\nalarm timing error ' + r'$\Delta t$' + '\nasymmetric decision loss',
        C_DECIDE)

    # ---- output callout ----
    out_w, out_h = 60, 9
    out_x = (100 - out_w) / 2 - 2
    box(ax, out_x, -12, out_w, out_h,
        'Reliability-valid RUL decision support', '', C_OUT, title_fs=8.4)

    # ---- flow arrows ----
    arrow(ax, (col1_x + w, row1_y + h / 2), (col2_x, row1_y + h / 2))
    arrow(ax, (col2_x + w, row1_y + h / 2), (col3_x, row1_y + h / 2))
    arrow(ax, (col3_x + w / 2, row1_y), (col3_x + w / 2, row2_y + h))
    arrow(ax, (col3_x, row2_y + h / 2), (col2_x + w, row2_y + h / 2))
    arrow(ax, (col2_x, row2_y + h / 2), (col1_x + w, row2_y + h / 2))
    arrow(ax, (col1_x + w / 2, row2_y - 6), (out_x + out_w / 2, -12 + out_h))

    fig.savefig(OUT, dpi=600, bbox_inches='tight')
    plt.close(fig)
    shutil.copyfile(OUT, MEDIA_OUT)
    print(f'saved {OUT}')
    print(f'copied to {MEDIA_OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
