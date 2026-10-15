"""
make_figures_R2.py: figures of the revised manuscript (JRESS-D-26-05118, revision 1).

Reads only saved analysis outputs (outputs/r1_analysis/*.json) and the saved test predictions of
the paper's model (outputs/r1_final_B10/final20/preds_seed42.npz); no model is trained or run.
Writes to paper/revision_R2/manuscript_R2/figures/ each figure as a PNG at 1000 dpi (opaque RGB, resolution
recorded in the file; the manuscript, the supplement and the upload package use these) and as a vector PDF master.

Style (reviewer comment on figure readability; authors' requests of 14 and 15 Oct 2026, values unchanged):
- journal style: every plot in a closed frame with inward ticks on all four sides and minor ticks on continuous
  axes, framed legends, panel labels (a), (b), ..., plain block diagrams with black outlines;
- every figure 7.2 in wide, printed at 100 % across the two-column text block (7.26 in): Arial, 7.5 pt tick labels,
  8 pt axis labels, nothing below 6.3 pt; mathematics in STIX Sans;
- colours from the ColorBrewer "Paired" scheme, one meaning each in every figure: blue held-out bearings, red
  within-bearing windows, grey raw or uncalibrated output, green CES, orange PAVA, purple the calibrated lower
  bounds (dark: on the CES mean, light: on the raw mean).

USAGE:
  python paper/revision_R2/manuscript_R2/make_figures_R2.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Polygon, Rectangle
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.online_filters import ConstrainedExponentialSmoothing  # noqa: E402
from src.isotonic_eval import IsotonicRegressor  # noqa: E402

AN = ROOT / 'outputs' / 'r1_analysis'
OUT = Path(__file__).resolve().parent / 'figures'
W = 7.2                                   # in; text block of the two-column layout: 7.26 in
DPI = 1000                                # PNG resolution (authors, 15 Oct 2026: PNG, at least 1000 dpi)

# ---- colours: ColorBrewer "Paired" (dark/light pairs), one meaning each ----------------------------------------
BLUE, BLUE_L = '#1F78B4', '#A6CEE3'
RED, RED_L = '#E31A1C', '#FB9A99'
GREEN = '#33A02C'
ORANGE, ORANGE_L = '#FF7F00', '#FDBF6F'
PURPLE, PURPLE_L = '#6A3D9A', '#CAB2D6'
BROWN = '#B15928'
GRAY, GRAY_L = '#808080', '#D9D9D9'
HO, HO_L = BLUE, BLUE_L                   # held-out bearings (the protocol)
WB, WB_L = RED, RED_L                     # within-bearing validation windows
RAW = GRAY                                # raw prediction, uncalibrated intervals
CES_C = GREEN                             # CES (causal filter)
PAVA_C, PAVA_L = ORANGE, ORANGE_L         # PAVA (offline reference)
LBC, LBR = PURPLE, PURPLE_L               # calibrated lower bound on the CES mean, on the raw mean
FS_TICK, FS_LAB, FS_NOTE, FS_SMALL = 7.5, 8, 7.5, 7

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans', 'font.size': FS_NOTE, 'axes.labelsize': FS_LAB,
    'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK, 'legend.fontsize': FS_SMALL,
    'text.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': 0.7,
    'axes.spines.top': True, 'axes.spines.right': True,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3.2, 'ytick.major.size': 3.2, 'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.minor.width': 0.55, 'ytick.minor.width': 0.55,
    'xtick.minor.visible': True, 'ytick.minor.visible': True, 'xtick.major.pad': 3.0, 'ytick.major.pad': 3.0,
    'axes.labelpad': 3.0, 'lines.linewidth': 1.0, 'hatch.linewidth': 0.6,
    'legend.frameon': True, 'legend.fancybox': False, 'legend.edgecolor': 'black', 'legend.framealpha': 1.0,
    'legend.borderpad': 0.45, 'legend.handletextpad': 0.5, 'legend.borderaxespad': 0.5,
    'pdf.fonttype': 42,
})

NAMES = {'submitted architecture': 'Ours', '1D CNN': '1D CNN', 'BiLSTM': 'BiLSTM', 'CNN-LSTM': 'CNN-LSTM',
         '1D ResNet': '1D ResNet', 'Transformer': 'Transformer', 'TSMC-CNN [21]': 'TSMC-CNN',
         'MSCNN [7]': 'MSCNN'}


def load(name):
    return json.loads((AN / name).read_text(encoding='utf-8'))


def png(fig, path):
    """Opaque RGB PNG at DPI with the resolution recorded in the file, so that LaTeX gives it its 7.2 in width."""
    fig.savefig(path, dpi=DPI, facecolor='white')
    with Image.open(path) as im:
        rgb = im.convert('RGB')
    rgb.save(path, dpi=(DPI, DPI), optimize=True)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{stem}.pdf')
    png(fig, OUT / f'{stem}.png')
    plt.close(fig)
    print(f'written {stem}')


def axes_at(fig, x, y, w, h, **kw):
    """Axes placed in inches from the lower left corner of the figure."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh], **kw)


def canvas(fig, zorder=0):
    """Frameless axes over the whole figure in inches, for diagrams and labels."""
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1], zorder=zorder)
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.axis('off')
    return ax


def letter(fig, x, y, s):
    """Panel label, e.g. (a), at a position in inches (top left corner of the text)."""
    fw, fh = fig.get_size_inches()
    fig.text(x / fw, y / fh, s, fontsize=9, weight='bold', ha='left', va='top')


def framed(leg):
    leg.get_frame().set_linewidth(0.6)
    return leg


def categorical(ax, axis='y'):
    """Category axis: major ticks at the categories only."""
    ax.tick_params(axis=axis, which='minor', left=False, right=False, top=False, bottom=False)


def dotted_grid(ax, axis='x'):
    ax.grid(axis=axis, which='major', color='#B0B0B0', lw=0.45, ls=(0, (1, 2)))
    ax.set_axisbelow(True)


def connect(ax, pts, color='black', ls='-', lw=0.8):
    """Orthogonal connector through points (inches) with an arrow head at the last point. The shaft takes the line
    style; the head is always solid (a dashed head shows notches at high resolution)."""
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    d = np.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / d, (y1 - y0) / d
    head = 0.061                                    # head length in inches: 0.55 x 8 pt
    xs, ys = zip(*(list(pts[:-1]) + [(x1 - 0.8 * head * ux, y1 - 0.8 * head * uy)]))
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_joinstyle='miter', solid_capstyle='butt',
            dash_capstyle='butt', zorder=1)
    ax.annotate('', xy=(x1, y1), xytext=(x1 - head * ux, y1 - head * uy), zorder=1,
                arrowprops=dict(arrowstyle='-|>,head_length=0.55,head_width=0.26', color=color, lw=lw,
                                shrinkA=0, shrinkB=0, mutation_scale=8))


def block(ax, x, y, w, h, title=None, body=(), ec='black', fc='white', ls='-', lw=0.7, tcolor='black',
          bcolor='black', tsize=7.2, bsize=6.8, top=None, lead=1.3):
    """Block of a diagram: rectangle with a bold title and regular body lines, centred. top: inches from the top
    edge to the first line (default: the text block is centred vertically)."""
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    lines = ([(t, True) for t in title.split('\n')] if title else []) + [(b, False) for b in body]
    hts = [(tsize if bold else bsize) / 72 * lead for _, bold in lines]
    gap = 0.03 if title and body else 0.0
    total = sum(hts) + gap
    yy = y + h - top if top is not None else y + (h + total) / 2
    for i, ((s, bold), ht) in enumerate(zip(lines, hts)):
        ax.text(x + w / 2, yy - ht / 2, s, ha='center', va='center', fontsize=tsize if bold else bsize,
                weight='bold' if bold else 'normal', color=tcolor if bold else bcolor, zorder=4)
        yy -= ht + (gap if bold and i + 1 < len(lines) and not lines[i + 1][1] else 0)


def first_cross(x, tau):
    idx = np.flatnonzero(x <= tau)
    return int(idx[0]) if idx.size else len(x) - 1


def records(ax, x0, x1, ytop, kinds, pitch=0.14, h=0.085, label=True, n_val=8, size=6.5):
    """Run-to-failure records of the training bearings as outlined bars (start left, failure right): 'train' grey,
    'interleaved' grey with validation windows (red), 'held' the held-out bearing (blue)."""
    for i, kind in enumerate(kinds):
        yy = ytop - i * pitch
        ax.add_patch(Rectangle((x0, yy), x1 - x0, h, fc=HO if kind == 'held' else GRAY_L, ec='black', lw=0.5,
                               zorder=3))
        if kind == 'interleaved':
            for k in range(n_val):
                xc = x0 + (k + 0.5) * (x1 - x0) / n_val
                ax.add_patch(Rectangle((xc - 0.013, yy), 0.026, h, fc=WB, ec='none', zorder=4))
        if label:
            ax.text(x0 - 0.05, yy + h / 2, f'B{i + 1}', fontsize=size, ha='right', va='center')


# ---------------------------------------------------------------------------------------------------------------
# Concept figure (Section 1): where the calibration errors come from, what they say about the error tail, and what
# an understated decision quantile does to the alarm (Corollary 1). Saved values only.
# ---------------------------------------------------------------------------------------------------------------
def fig_concept():
    d = load('decision_inbearing.json')
    key = 'submitted architecture'
    u = d['secondary']['understatement'][key]
    q56_wb = d['calibration_SP'][key]['q']['0.8333333333333334']
    q56_ho = d['calibration_PR'][key]['q']['0.8333333333333334']
    z = np.load(ROOT / 'outputs' / 'r1_final_B10' / 'final20' / 'preds_seed42.npz')
    name = 'test_03_Bearing1_5'
    y, raw = z['tgt_' + name], z['raw_' + name]
    t = np.arange(len(y)) * 10 / 3600.0
    tau = 0.20

    H = 2.55
    fig = plt.figure(figsize=(W, H))
    cv = canvas(fig)

    # (a) the two sources of calibration errors, one bar per training bearing
    letter(fig, 0.0, H - 0.03, '(a)')
    cv.text(0.36, H - 0.05, 'Within-bearing calibration', fontsize=FS_NOTE, weight='bold', va='top')
    records(cv, 0.36, 2.20, 2.06, ['interleaved'] * 4)
    cv.text(0.36, 1.53, 'validation windows interleaved with\nthe training windows of each bearing',
            fontsize=FS_SMALL, va='top', linespacing=1.25)
    cv.text(0.36, 1.18, 'Held-out calibration', fontsize=FS_NOTE, weight='bold', va='top')
    records(cv, 0.36, 2.20, 0.90, ['train'] * 3 + ['held'])
    cv.text(0.36, 0.37, 'a whole bearing left out of training', fontsize=FS_SMALL, va='top')
    for x0, cols, lab in ((0.36, [GRAY_L], 'training windows'), (1.30, [WB, HO], 'calibration data')):
        for j, c in enumerate(cols):
            cv.add_patch(Rectangle((x0 + j * 0.10, 0.06), 0.075, 0.075, fc=c, ec='black', lw=0.4))
        cv.text(x0 + len(cols) * 0.10 + 0.03, 0.097, lab, fontsize=FS_SMALL, va='center')

    # (b) 0.9-quantile of the absolute error: calibration data against the test bearings (same checkpoints)
    letter(fig, 2.42, H - 0.03, '(b)')
    b = axes_at(fig, 3.30, 0.50, 1.02, 1.80)
    rows = [('within-bearing', u['SP_calibration_q90'], u['SP_test_q90'], WB, WB_L),
            ('held-out', u['PR_calibration_q90'], u['PR_test_q90'], HO, HO_L)]
    for i, (lab, qc, qt, col, light) in enumerate(rows):
        for off, v, fc in ((-0.19, qc, col), (0.19, qt, light)):
            b.barh(i + off, v, height=0.34, color=fc, edgecolor='black', lw=0.5, zorder=2)
            b.text(v + 0.01, i + off, f'{v:.2f}', va='center', fontsize=FS_SMALL)
    b.set_yticks([0, 1])
    b.set_yticklabels(['Within-\nbearing', 'Held-out'], linespacing=1.1)
    categorical(b, 'y')
    b.set_ylim(2.08, -0.55)
    b.set_xlim(0, 0.46)
    b.set_xticks([0, 0.2, 0.4])
    b.set_xticklabels(['0', '0.2', '0.4'])
    b.set_xlabel('0.9-quantile of\nabsolute error', linespacing=1.15)
    framed(b.legend(handles=[Patch(facecolor='#606060', edgecolor='black', lw=0.5, label='calibration data'),
                             Patch(facecolor='#D0D0D0', edgecolor='black', lw=0.5, label='test bearings')],
                    loc='lower right', fontsize=6.4, handlelength=1.0, borderaxespad=0.3, labelspacing=0.25,
                    borderpad=0.35))

    # (c) the cost-matched trigger at r = 5 with each estimate of the decision quantile
    letter(fig, 4.56, H - 0.03, '(c)')
    c = axes_at(fig, 5.04, 0.50, 2.08, 1.95)
    k_true = first_cross(y, tau)
    k_wb = first_cross(raw - q56_wb, tau)
    k_ho = first_cross(raw - q56_ho, tau)
    c.plot(t, raw, color=RAW, lw=0.5, zorder=2)
    c.plot(t, y, color='black', lw=1.3, zorder=3)
    c.text(4.50, 0.285, 'true RUL', fontsize=FS_SMALL, ha='right', va='center')
    c.text(4.10, 0.66, 'prediction', fontsize=FS_SMALL, color='#505050', ha='left', va='bottom')
    c.axhline(tau, color='black', lw=0.6, ls=(0, (4, 2)), zorder=1)
    c.text(0.15, tau + 0.02, r'$\tau$', fontsize=FS_NOTE, va='bottom')
    ym = -0.11
    c.annotate('', xy=(t[k_wb], ym + 0.06), xytext=(t[k_true], ym + 0.06),
               arrowprops=dict(arrowstyle='<|-|>,head_length=0.35,head_width=0.16', color=WB, lw=0.7,
                               shrinkA=0, shrinkB=0, mutation_scale=7))
    c.text((t[k_true] + t[k_wb]) / 2, ym + 0.08, 'late', fontsize=6.5, color=WB, ha='center', va='bottom')
    marks = [(k_true, 'black', 'maintenance due'), (k_ho, HO, 'alarm, held-out'), (k_wb, WB, 'alarm, within-bearing')]
    for k, col, lab in marks:
        c.plot([t[k]], [ym], marker='v', ms=6, color=col, mec='black', mew=0.4, ls='', zorder=4, label=lab)
    c.set_ylim(-0.18, 1.18)
    c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    c.set_xlim(0, t[-1] + 0.12)
    c.set_xlabel('Operating time (h)')
    c.set_ylabel('Normalized RUL')
    framed(c.legend(loc='upper right', fontsize=6.5, handlelength=1.0, labelspacing=0.3, borderpad=0.4))
    save(fig, 'fig0_concept')
    return {'q56_within': q56_wb, 'q56_heldout': q56_ho, 'due_h': float(t[k_true]),
            'alarm_heldout_h': float(t[k_ho]), 'alarm_within_h': float(t[k_wb]),
            'q90': {r[0]: (r[1], r[2]) for r in rows}}


# ---------------------------------------------------------------------------------------------------------------
# Workflow and evaluation protocol (Section 3)
# ---------------------------------------------------------------------------------------------------------------
def tail_sketch(ax, x0, x1, y0, h, color):
    """Error distribution with the tail beyond a quantile shaded (sketch)."""
    u = np.linspace(0, 1, 240)
    f = (u * 6.5) ** 1.7 * np.exp(-u * 6.5)
    f = f / f.max()
    xs = x0 + (x1 - x0) * u
    m = u >= 0.56
    ax.fill_between(xs[m], y0, y0 + h * f[m], color=color, lw=0, zorder=3)
    ax.plot(xs, y0 + h * f, color='black', lw=0.7, zorder=4)
    ax.plot([x0, x1], [y0, y0], color='black', lw=0.7, zorder=4)
    xq = x0 + (x1 - x0) * 0.56
    ax.plot([xq, xq], [y0, y0 + h * 0.70], color='black', lw=0.6, ls=(0, (2, 1.5)), zorder=4)
    ax.text(xq, y0 + h * 0.74, '$q_p$', fontsize=FS_SMALL, ha='center', va='bottom', zorder=4)


def frame_label(ax, x, y, bold, regular, color='black'):
    """Group label ending at x (bold lead, regular rest)."""
    t = ax.text(x, y, regular, fontsize=FS_NOTE, ha='right', va='top')
    r = ax.figure.canvas.get_renderer()
    x0 = ax.transData.inverted().transform((t.get_window_extent(renderer=r).x0, 0))[0]
    ax.text(x0 - 0.06, y, bold, fontsize=FS_NOTE, weight='bold', color=color, ha='right', va='top')


def fig_framework():
    H = 3.92
    fig = plt.figure(figsize=(W, H))
    cv = canvas(fig)
    # group frames solid: in the caption, dashed boxes are the alternatives
    cv.add_patch(Rectangle((0.06, 2.13), W - 0.12, 1.75, fc='none', ec='#808080', lw=0.8, zorder=1))
    frame_label(cv, 7.04, 3.83, 'Development:', 'training bearings only')
    cv.add_patch(Rectangle((0.06, 1.10), W - 0.12, 0.93, fc='none', ec='#808080', lw=0.8, zorder=1))
    frame_label(cv, 7.04, 1.98, 'Evaluation:', 'test bearings, used once')
    # development
    block(cv, 0.18, 2.98, 1.24, 0.64, 'Training bearings', ['four bearings,', 'two operating conditions'])
    block(cv, 1.67, 2.98, 1.70, 0.64, 'Leave-one-bearing-out runs', ['four folds, fixed budget,', 'final epoch kept'])
    block(cv, 1.67, 2.24, 1.70, 0.54, 'Selection', ['budget by mean validation MAE;',
                                                    r'$\lambda$ by the one-standard-error rule'])
    block(cv, 3.62, 2.98, 1.66, 0.64, 'Calibration data', ['errors on the held-out', 'bearing of each run'])
    block(cv, 5.53, 2.24, 1.49, 1.38, 'Error distribution on\nunseen bearings, $G$',
          ['quantiles $q_p$, interval factor $c$*'], top=0.09)
    tail_sketch(cv, 5.72, 6.84, 2.36, 0.42, HO)
    connect(cv, [(1.42, 3.30), (1.67, 3.30)])
    connect(cv, [(2.52, 2.98), (2.52, 2.78)])
    connect(cv, [(3.37, 3.30), (3.62, 3.30)])
    connect(cv, [(5.28, 3.30), (5.53, 3.30)])
    # evaluation
    block(cv, 0.18, 1.18, 1.24, 0.64, 'Final model', ['all four bearings,', '$B$* epochs, 20 seeds'])
    block(cv, 1.67, 1.18, 1.70, 0.64, 'Test bearings (+ zero-shot)', ['PRONOSTIA 10 (+3),', 'XJTU-SY 6 (+5)'])
    block(cv, 3.62, 1.18, 1.66, 0.64, 'Alarm rule (Proposition 1)',
          [r'$\hat{y}_t - q_{r/(1+r)} \leq \tau$', r'interval $\mu_t \pm z\,c$*$s_t$'])
    block(cv, 5.53, 1.18, 1.49, 0.64, 'Inference', ['bearing as unit,', 'exact sign-flip test,', 'Holm correction'])
    connect(cv, [(0.80, 2.98), (0.80, 1.82)])
    connect(cv, [(1.42, 1.40), (1.67, 1.40)])
    connect(cv, [(3.37, 1.50), (3.62, 1.50)])
    connect(cv, [(5.28, 1.50), (5.53, 1.50)])
    connect(cv, [(1.67, 2.51), (1.545, 2.51), (1.545, 1.64), (1.42, 1.64)])
    cv.text(1.50, 2.40, r'$B$*, $\lambda$', fontsize=FS_SMALL, va='center', ha='right')
    connect(cv, [(5.70, 2.24), (5.70, 2.08), (4.45, 2.08), (4.45, 1.82)])
    cv.text(4.51, 1.93, r'$q_p$, $c$*', fontsize=FS_SMALL, va='center', ha='left')
    # alternatives evaluated against the protocol (dashed)
    t = cv.text(0.12, 0.98, 'Alternatives evaluated against the protocol', fontsize=FS_NOTE, weight='bold',
                color=WB, va='top')
    r = fig.canvas.get_renderer()
    x1 = cv.transData.inverted().transform((t.get_window_extent(renderer=r).x1, 0))[0]
    cv.text(x1 + 0.06, 0.98, '(Sections 4.2 and 6.1)', fontsize=FS_NOTE, va='top')
    alts = [(['Within-bearing validation windows for', 'early stopping and calibration:'],
             'error tail understated 3 to 9 times'),
            (['Checkpoint selected on the', 'calibration bearing:'], 'coverage 1.4 points farther from the target'),
            (['Seed-level tests and window pooling:'], '29 instead of 7 of 44 comparisons significant')]
    for i, (body, effect) in enumerate(alts):
        x = 0.12 + i * 2.38
        block(cv, x, 0.06, 2.20, 0.74, None, body + [''], ec=WB, ls=(0, (3.5, 2)), lw=0.8)
        n = len(body)
        cv.text(x + 1.10, 0.06 + 0.37 - (n - 0.2) * 6.8 / 72 * 1.3 / 2 - 0.05, effect, fontsize=6.9,
                weight='bold', color=WB, ha='center', va='center', zorder=4)
    save(fig, 'fig1_framework')


# ---------------------------------------------------------------------------------------------------------------
# Architecture of the network (the submitted diagram redrawn, same content)
# ---------------------------------------------------------------------------------------------------------------
def slab(ax, x, yc, t, h, front, top, side, dx=0.085, dy=0.055):
    """Feature map drawn as a slab: thickness t (channels), height h (time steps), shallow perspective."""
    y0, y1 = yc - h / 2, yc + h / 2
    kw = dict(closed=True, ec='black', lw=0.4, zorder=3)
    ax.add_patch(Polygon([(x, y1), (x + dx, y1 + dy), (x + t + dx, y1 + dy), (x + t, y1)], fc=top, **kw))
    ax.add_patch(Polygon([(x + t, y0), (x + t + dx, y0 + dy), (x + t + dx, y1 + dy), (x + t, y1)], fc=side, **kw))
    ax.add_patch(Rectangle((x, y0), t, h, fc=front, ec='black', lw=0.4, zorder=3))


def waveform(ax, x0, x1, yc, amp, color):
    """Vibration window (sketch)."""
    u = np.linspace(0, 1, 500)
    rng = np.random.default_rng(3)
    s = np.sin(2 * np.pi * 13 * u) * (0.55 + 0.45 * np.sin(2 * np.pi * 1.6 * u + 0.5)) + 0.22 * rng.standard_normal(u.size)
    ax.plot(x0 + (x1 - x0) * u, yc + amp * s / np.abs(s).max(), color=color, lw=0.45, zorder=4)


def spectrum(ax, x0, x1, y0, hmax, color, peak):
    """Envelope spectrum with peaks at a fault frequency and its harmonics (sketch)."""
    n = 30
    rng = np.random.default_rng(5)
    base = 0.12 + 0.14 * rng.random(n)
    peaks = {4: 1.0, 9: 0.72, 14: 0.55, 19: 0.42, 24: 0.32}
    cw = (x1 - x0) / n
    for i in range(n):
        ax.add_patch(Rectangle((x0 + i * cw, y0), cw * 0.62, hmax * peaks.get(i, base[i]),
                               fc=peak if i in peaks else color, ec='none', zorder=4))
    ax.plot([x0, x1], [y0, y0], color='black', lw=0.5, zorder=4)


def fig_architecture():
    H = 3.35
    fig = plt.figure(figsize=(W, H))
    cv = canvas(fig)
    blue = (BLUE, BLUE_L, '#135182')
    orange = (ORANGE, ORANGE_L, '#B35900')
    # input window and the bus to both streams
    cv.add_patch(Rectangle((0.08, 0.98), 0.94, 1.00, fc='white', ec='black', lw=0.7, zorder=2))
    waveform(cv, 0.16, 0.94, 1.70, 0.17, 'black')
    cv.text(0.55, 1.44, 'Raw vibration\nwindow', fontsize=6.8, weight='bold', va='top', ha='center',
            linespacing=1.15, zorder=4)
    cv.text(0.55, 1.20, r'$x_t \in \mathbb{R}^{1\times 2560}$' + '\n100 ms at 25.6 kHz', fontsize=6.5,
            va='top', ha='center', linespacing=1.4, zorder=4)
    cv.plot([1.02, 1.14], [1.48, 1.48], color='black', lw=0.8, zorder=1)
    cv.plot([1.14, 1.14], [0.70, 2.22], color='black', lw=0.8, zorder=1, solid_capstyle='butt')
    connect(cv, [(1.14, 2.22), (2.60, 2.22)])
    cv.text(1.86, 2.27, '$x_t$', fontsize=FS_SMALL, ha='center', va='bottom')
    connect(cv, [(1.14, 0.70), (1.28, 0.70)])
    # temporal stream
    cv.text(1.27, 2.92, 'Temporal stream: 1D CNN', fontsize=7.2, weight='bold', va='top')
    cv.text(1.27, 2.78, 'Conv1d ×3, BN + GELU', fontsize=6.5, va='top')
    for (x, tt, hh, k, s, ch) in ((2.62, 0.07, 0.62, 15, 4, 64), (2.98, 0.12, 0.42, 7, 4, 128),
                                  (3.36, 0.12, 0.30, 3, 2, 128)):
        slab(cv, x, 2.22, tt, hh, *blue)
        cv.text(x + tt / 2 + 0.04, 1.82, f'k{k}, s{s}\n{ch} ch', fontsize=6.3, ha='center', va='top',
                linespacing=1.25)
    # envelope extractor and spectral stream
    cv.text(1.27, 1.41, 'Fixed envelope extractor', fontsize=7.2, weight='bold', va='top')
    cv.text(1.27, 1.27, '(non-learnable)', fontsize=6.5, va='top')
    cv.add_patch(Rectangle((1.28, 0.24), 1.14, 0.88, fc='white', ec='black', lw=0.7, zorder=2))
    spectrum(cv, 1.36, 2.34, 0.83, 0.22, '#BDBDBD', orange[0])
    cv.text(1.36, 0.76, 'Hilbert → |env| → FFT\n→ 512 bins\n(BPFO/BPFI/BSF/FTF)', fontsize=6.3, va='top',
            linespacing=1.3, zorder=4)
    connect(cv, [(2.42, 0.70), (2.60, 0.70)])
    cv.text(2.62, 1.41, 'Spectral stream: 1D CNN', fontsize=7.2, weight='bold', va='top')
    cv.text(2.62, 1.27, 'Conv1d ×3, BN + GELU', fontsize=6.5, va='top')
    for (x, hh, k) in ((2.64, 0.50, 7), (2.94, 0.40, 5), (3.24, 0.30, 3)):
        slab(cv, x, 0.70, 0.07, hh, *orange)
        cv.text(x + 0.075, 0.38, f'k{k}, s2\n64 ch', fontsize=6.3, ha='center', va='top', linespacing=1.25)
    # global average pooling and projection, feature vectors
    for yc in (2.22, 0.70):
        x_last = 3.36 + 0.12 + 0.085 if yc > 1 else 3.24 + 0.07 + 0.085
        connect(cv, [(x_last + 0.02, yc), (3.66, yc)])
        cv.add_patch(Rectangle((3.66, yc - 0.13), 0.72, 0.26, fc='white', ec='black', lw=0.7, zorder=3))
        cv.text(4.02, yc, 'GAP → Linear', fontsize=6.3, ha='center', va='center', zorder=4)
    connect(cv, [(4.38, 2.22), (5.03, 2.22), (5.03, 1.84)])
    cv.text(4.42, 2.28, r'$F_t^{\mathrm{temp}} \in \mathbb{R}^{128}$', fontsize=6.8, ha='left', va='bottom')
    connect(cv, [(4.38, 0.70), (5.03, 0.70), (5.03, 1.10)])
    cv.text(4.42, 0.63, r'$F_t^{\mathrm{spec}} \in \mathbb{R}^{128}$', fontsize=6.8, ha='left', va='top')
    # fusion gate, regression head, output
    block(cv, 4.48, 1.10, 1.10, 0.74, 'Adaptive scalar\nfusion gate',
          [r'$\alpha_t = \sigma(\cdot)$', r'$F_t = \alpha_t F_t^{\mathrm{temp}}$',
           r'$+\,(1-\alpha_t)\,F_t^{\mathrm{spec}}$'], tsize=6.9, bsize=6.6, lead=1.25)
    connect(cv, [(5.58, 1.47), (5.78, 1.47)])
    block(cv, 5.78, 1.00, 0.86, 0.94, 'Regression\nhead', ['LN → Drop', '→ Lin(128,64)', '→ GELU → Drop',
                                                         '→ Lin(64,1)'], tsize=6.9, bsize=6.3, lead=1.25)
    connect(cv, [(6.64, 1.47), (6.80, 1.47)])
    cv.add_patch(Rectangle((6.80, 1.19), 0.34, 0.56, fc='white', ec='black', lw=0.9, zorder=3))
    cv.text(6.97, 1.53, r'$\hat{y}_t$', fontsize=9, ha='center', va='center', zorder=4)
    cv.text(6.97, 1.33, 'RUL', fontsize=6.5, ha='center', va='center', weight='bold', zorder=4)
    # training-only components (dashed), discarded at inference
    cv.text(5.38, 3.27, 'Training only, discarded at inference', fontsize=6.8, style='italic', ha='center',
            va='center')
    dash = (0, (3, 2))
    block(cv, 3.62, 2.56, 1.78, 0.60, 'DANN discriminator (GRL)',
          ['binary domain classifier, 8.3k params:', 'Lin(128,64) → Lin(64,1); backbone',
           'gets GRL-reversed gradient'], ec='#505050', ls=dash, tsize=6.8, bsize=6.3, lead=1.25)
    block(cv, 5.55, 2.56, 1.59, 0.60, 'PGMC monotonic penalty',
          ['auxiliary; chronological within-', 'bearing chunk (BN frozen),', r'warm-up $\lambda_{\mathrm{PGMC}}(j)$'],
          ec='#505050', ls=dash, tsize=6.8, bsize=6.3, lead=1.25)
    connect(cv, [(5.32, 1.84), (5.32, 2.56)], color='#505050', ls=dash)
    connect(cv, [(6.97, 1.75), (6.97, 2.56)], color='#505050', ls=dash)
    cv.text(3.60, 0.07, 'Inference graph: 192,130 parameters; 1.48 ms FP32 latency per 100 ms window',
            fontsize=6.8, style='italic', ha='center', va='bottom')
    save(fig, 'fig2_architecture')


# ---------------------------------------------------------------------------------------------------------------
# Within-bearing against held-out calibration (Amendments 6 and 7)
# ---------------------------------------------------------------------------------------------------------------
def dumbbell(ax, ys, lo, hi, c_lo, c_hi, m_lo, m_hi, s=30):
    for yy, a, b in zip(ys, lo, hi):
        ax.plot([a, b], [yy, yy], color='#9A9A9A', lw=1.0, zorder=1)
    ax.scatter(lo, ys, s=s, color=c_lo, marker=m_lo, zorder=3, edgecolors='black', linewidths=0.5)
    ax.scatter(hi, ys, s=s, color=c_hi, marker=m_hi, zorder=3, edgecolors='black', linewidths=0.5)


def fig_calibration_source():
    pr = load('decision_inbearing.json')
    xj = load('decision_xjtu_inbearing.json')
    rows = []
    for ds, d in (('PRONOSTIA', pr), ('XJTU-SY', xj)):
        for key in d['secondary']['understatement']:
            u = d['secondary']['understatement'][key]
            sp = d['residual_intervals']['SP'][key]['pooled_coverage']
            ho = d['residual_intervals']['PR'][key]['pooled_coverage']
            rows.append((ds, NAMES[key], u['SP_ratio'], u['PR_ratio'], sp, ho))
    n_pr = sum(r[0] == 'PRONOSTIA' for r in rows)
    ys = np.array([i if i < n_pr else i + 0.8 for i in range(len(rows))], dtype=float)
    mc = load('pooled_framework.json')['A5_D']
    names = dict(NAMES, **{'adopted variant': 'Temporal-context variant'})
    keys = ['submitted architecture', 'adopted variant'] + [k for k in NAMES if k in mc and k != 'submitted architecture']
    mrows = [(names[k], mc[k]['raw']['pooled_coverage'], mc[k]['calibrated']['pooled_coverage'],
              mc[k]['raw']['late_coverage'], mc[k]['calibrated']['late_coverage']) for k in keys]   # TSMC-CNN: no dropout
    my = np.arange(len(mrows), dtype=float)

    H = 5.30
    fig = plt.figure(figsize=(W, H))
    xa, wa, xb, wb = 1.62, 2.58, 4.52, 2.58
    a1 = axes_at(fig, xa, 2.80, wa, 2.02)
    a2 = axes_at(fig, xb, 2.80, wb, 2.02, sharey=a1)
    b1 = axes_at(fig, xa, 0.50, wa, 1.52)
    b2 = axes_at(fig, xb, 0.50, wb, 1.52, sharey=b1)
    a1.set_ylim(ys[-1] + 0.7, -0.7)
    b1.set_ylim(my[-1] + 0.7, -0.7)
    dumbbell(a1, ys, [r[2] for r in rows], [r[3] for r in rows], WB, HO, 'o', 's')
    dumbbell(a2, ys, [r[4] for r in rows], [r[5] for r in rows], WB, HO, 'o', 's')
    dumbbell(b1, my, [r[1] for r in mrows], [r[2] for r in mrows], RAW, HO, 'D', 's', s=26)
    dumbbell(b2, my, [r[3] for r in mrows], [r[4] for r in mrows], RAW, HO, 'D', 's', s=26)

    a1.set_xscale('log')
    a1.set_xlim(0.08, 1.7)
    a1.set_xticks([0.1, 0.2, 0.5, 1.0])
    a1.set_xticklabels(['0.1', '0.2', '0.5', '1'])
    a1.set_xlabel('0.9-quantile of absolute error,\ncalibration data / test bearings', linespacing=1.15)
    a2.set_xlim(15, 100)
    a2.set_xlabel('Coverage of the 90% interval\non test bearings (%)', linespacing=1.15)
    for ax in (b1, b2):
        ax.set_xlim(0, 102)
    b1.set_xlabel('Coverage of the 90% MC-dropout interval (%)')
    b2.set_xlabel('Coverage in late life, true RUL < 0.2 (%)')
    for ax, x0, lab in ((a1, 1.0, 'no understatement'), (a2, 90, 'target'), (b1, 90, 'target'), (b2, 90, 'target')):
        ax.axvline(x0, color='black', lw=0.7, ls=(0, (4, 2)), zorder=2)
        ax.text(x0 * (0.97 if ax is a1 else 1), 1.015, lab, transform=ax.get_xaxis_transform(), fontsize=FS_SMALL,
                ha='right' if ax is a1 else 'center', va='bottom')
    sep = (ys[n_pr - 1] + ys[n_pr]) / 2
    for ax in (a1, a2):
        ax.axhline(sep, color='black', lw=0.5)
    for ax in (a1, a2, b1, b2):
        categorical(ax, 'y')
        dotted_grid(ax, 'x')
    a1.set_yticks(ys)
    a1.set_yticklabels([r[1] for r in rows])
    b1.set_yticks(my)
    b1.set_yticklabels([r[0] for r in mrows])
    plt.setp(a2.get_yticklabels(), visible=False)
    plt.setp(b2.get_yticklabels(), visible=False)
    for ax, ds, yy in ((a1, 'PRONOSTIA', ys[:n_pr].mean()), (a1, 'XJTU-SY', ys[n_pr:].mean()),
                       (b1, 'PRONOSTIA', my.mean())):
        ax.text(-0.535, yy, ds, transform=ax.get_yaxis_transform(), rotation=90, ha='center', va='center',
                fontsize=FS_NOTE, weight='bold')
    handles = [Line2D([], [], color=WB, marker='o', mec='black', mew=0.5, ls='', ms=5.2,
                      label='within-bearing calibration'),
               Line2D([], [], color=HO, marker='s', mec='black', mew=0.5, ls='', ms=5.0, label='held-out calibration'),
               Line2D([], [], color=RAW, marker='D', mec='black', mew=0.5, ls='', ms=4.6,
                      label='uncalibrated MC dropout')]
    framed(fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=((xa + xb + wb) / 2 / W, 0.998),
                      columnspacing=1.6))
    letter(fig, 0.02, 4.98, '(a)')
    letter(fig, xb - 0.32, 4.98, '(b)')
    letter(fig, 0.02, 2.26, '(c)')
    letter(fig, xb - 0.32, 2.26, '(d)')
    save(fig, 'fig3_calibration_source')
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Components of the network (Section 5.3): relative difference of each variant to the full model on four endpoints,
# from the saved ablation analysis. Exact values: Supplementary Material.
# ---------------------------------------------------------------------------------------------------------------
ABL = [('static fusion', 'Static fusion (no gate)'), ('temporal stream only', 'Raw-vibration stream only'),
       ('spectral stream only', 'Envelope-spectrum stream only'), ('without PGMC', 'Without PGMC'),
       ('without DANN', 'Without DANN'), ('without PGMC and DANN', 'Without PGMC and DANN'),
       ('PGMC batch ordering (soft)', 'PGMC on sorted batches')]


def component_values():
    d = load('decision_ablations.json')
    mae = d['ces_mae_bavg_all10']
    full = mae['full model']
    isc = d['A3_P2']['tests']                                # primary interval-score family (as in the table)
    out = []
    for key, label in ABL:
        out.append((label, 100 * (mae[key] - full) / mae[key], isc[key]['rel'],
                    d['A3_P1']['5']['tests'][key]['rel'], d['A3_P1']['10']['tests'][key]['rel']))
    return out


def fig_components():
    vals = component_values()
    H = 2.50
    fig = plt.figure(figsize=(W, H))
    heads = ['CES MAE', 'Interval score', 'Cost, $r$ = 5', 'Cost, $r$ = 10']
    x0, pw, gap = 1.80, 1.26, 0.10
    ys = np.arange(len(vals), dtype=float)
    for j, head in enumerate(heads):
        ax = axes_at(fig, x0 + j * (pw + gap), 0.58, pw, 1.62)
        ax.barh(ys, [row[1 + j] for row in vals], height=0.58, color=BLUE, edgecolor='black', lw=0.5, zorder=2)
        ax.axvline(0, color='black', lw=0.7, zorder=3)
        ax.set_xlim(-22, 52)
        ax.set_xticks([-20, 0, 20, 40])
        ax.set_xticklabels(['−20', '0', '20', '40'])
        ax.set_ylim(len(vals) - 0.5, -0.5)
        categorical(ax, 'y')
        dotted_grid(ax, 'x')
        ax.set_title(head, fontsize=FS_NOTE, pad=4)
        ax.set_yticks(ys)
        if j == 0:
            ax.set_yticklabels([v[0] for v in vals])
        else:
            ax.set_yticklabels([])
    xc = (x0 + (4 * pw + 3 * gap) / 2) / W
    fig.text(xc, 0.20 / H, 'Relative difference to the full model, (variant − full) / variant (%)', fontsize=FS_LAB,
             ha='center', va='bottom')
    fig.text(xc, 0.06 / H, 'positive: full model better', fontsize=FS_SMALL, ha='center', va='bottom')
    save(fig, 'fig_components')
    return vals


# ---------------------------------------------------------------------------------------------------------------
# Alarm timing (Section 6.3): mean timing error and per-bearing share of late alarms at tau = 0.20 for the
# corrections and the calibrated lower bounds of the paper's model. Exact values: Supplementary Material.
# ---------------------------------------------------------------------------------------------------------------
RULES = [('raw', 'Raw prediction (and RTRM)', RAW), ('ces', 'CES', CES_C), ('pava', 'PAVA (offline)', PAVA_C),
         ('lb_raw', 'Lower bound, raw mean', LBR), ('lb_ces', 'Lower bound, CES mean', LBC)]


def swarm(v, dy=0.12, dx=2.6, per_col=3, tol=1.5):
    """Positions of a compact swarm: equal values (within tol) packed in columns of per_col dots about the
    value, so that no dot leaves its row."""
    x, y = np.array(v, dtype=float), np.zeros(len(v))
    order = np.argsort(v, kind='stable')
    groups = []
    for i in order:
        if groups and abs(v[i] - v[groups[-1][-1]]) <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    for g in groups:
        ncol = int(np.ceil(len(g) / per_col))
        for j, i in enumerate(g):
            col, row = divmod(j, per_col)
            n_in = min(per_col, len(g) - col * per_col)
            x[i] = v[i] + (col - (ncol - 1) / 2) * dx
            y[i] = (row - (n_in - 1) / 2) * dy
    return x, y


def fig_alarm_timing():
    t = load('decision_final20_all10.json')['timing']['0.20']
    assert t['stats']['raw'] == t['stats']['rtrm']                   # RTRM alarms exactly when raw does
    H = 2.35
    fig = plt.figure(figsize=(W, H))
    xa, wa, xb, wb = 1.72, 2.30, 4.48, 2.62
    a = axes_at(fig, xa, 0.52, wa, 1.58)
    b = axes_at(fig, xb, 0.52, wb, 1.58, sharey=a)
    out = {}
    ys = np.arange(len(RULES), dtype=float)
    for i, (key, label, col) in enumerate(RULES):
        st = t['stats'][key]
        dt_h = st['dt'] * 10 / 3600.0
        a.barh(i, dt_h, height=0.56, color=col, edgecolor='black', lw=0.5, zorder=2)
        a.text(dt_h + (0.08 if dt_h >= 0 else -0.08), i, f'{dt_h:+.2f} h'.replace('-', '−'), va='center',
               ha='left' if dt_h >= 0 else 'right', fontsize=FS_SMALL)
        fr = np.array(list(t['late_fraction_per_bearing'][key].values())) * 100
        sx, sy = swarm(fr)
        b.scatter(sx, i + sy, s=15, color=col, zorder=3, edgecolors='black', linewidths=0.35, clip_on=False)
        b.plot([st['late'] * 100] * 2, [i - 0.36, i + 0.36], color='black', lw=1.8, zorder=4, solid_capstyle='butt')
        out[key] = (round(dt_h, 3), round(st['late'] * 100, 1))
    a.axvline(0, color='black', lw=0.7, zorder=3)
    a.set_xlim(-3.2, 1.62)
    a.set_xticks([-2, -1, 0, 1])
    a.set_xticklabels(['−2', '−1', '0', '1'])
    a.set_xlabel('Mean alarm timing error (h)')
    a.text(-3.12, -0.62, '← early', fontsize=FS_SMALL, ha='left', va='center')
    a.text(1.55, -0.62, 'late →', fontsize=FS_SMALL, ha='right', va='center')
    b.set_xlim(-6, 106)
    b.set_xticks([0, 20, 40, 60, 80, 100])
    b.set_xlabel('Late alarms per bearing (% of seeds)')
    a.set_yticks(ys)
    a.set_yticklabels([r[1] for r in RULES])
    a.set_ylim(len(RULES) - 0.45, -0.95)
    plt.setp(b.get_yticklabels(), visible=False)
    for ax in (a, b):
        categorical(ax, 'y')
        dotted_grid(ax, 'x')
    framed(b.legend(handles=[Line2D([], [], color='#909090', marker='o', mec='black', mew=0.35, ls='', ms=4,
                                    label='one test bearing'),
                             Line2D([], [], color='black', lw=1.8, label='all bearings')],
                    loc='upper left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.5, borderpad=0.35,
                    handlelength=1.2, columnspacing=1.0, borderaxespad=0.3))
    letter(fig, 0.02, H - 0.04, '(a)')
    letter(fig, xb - 0.32, H - 0.04, '(b)')
    save(fig, 'fig_alarm_timing')
    return out


# ---------------------------------------------------------------------------------------------------------------
# Decision-optimal rule against the cost ratio (paper's model, ten test bearings)
# ---------------------------------------------------------------------------------------------------------------
VARIANTS = {'pava': ('PAVA', PAVA_L, 'black'), 'ces': ('CES', CES_C, 'white'), 'raw': ('raw', RAW, 'white'),
            'rtrm': ('raw', RAW, 'white'), 'lb_ces': ('LB-CES', LBC, 'white'), 'lb_raw': ('LB-raw', LBR, 'black')}


def fig_decision_map():
    d = load('decision_final20_all10.json')
    taus = ['0.30', '0.20', '0.10']
    H = 2.10
    fig = plt.figure(figsize=(W, H))
    ax = axes_at(fig, 0.80, 0.74, 6.28, 1.12)
    for i, tau in enumerate(taus):
        for s in d['timing'][tau]['optimal_by_ratio']:
            lab, col, tcol = VARIANTS[s['variant']]
            a, b = max(s['r_from'], 0.5), min(s['r_to'], 100.0)
            if b <= a:
                continue
            la, lb = np.log10(a), np.log10(b)
            if s['variant'] == 'pava':
                ax.barh(i, lb - la, left=la, height=0.6, color=col, edgecolor=PAVA_C, lw=0, hatch='//////',
                        zorder=2)
            else:
                ax.barh(i, lb - la, left=la, height=0.6, color=col, lw=0, zorder=2)
            ax.barh(i, lb - la, left=la, height=0.6, color='none', edgecolor='black', lw=0.6, zorder=3)
            if lb - la > 0.17:
                ax.text((la + lb) / 2, i, lab, ha='center', va='center', fontsize=FS_SMALL, color=tcol,
                        weight='bold', zorder=6,
                        bbox=dict(facecolor=PAVA_L, edgecolor='none', pad=1.2) if s['variant'] == 'pava' else None)
    for r in (2, 5, 10):                     # behind the bars: visible between rows, never across a label
        ax.axvline(np.log10(r), color='black', lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.text(np.log10(r), 1.025, f'$r$ = {r}', transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=FS_SMALL)
    ax.set_ylim(2.5, -0.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f'$\\tau$ = {t}' for t in taus])
    categorical(ax, 'y')
    ticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    minor = [m * 10 ** e for e in (-1, 0, 1) for m in range(2, 10) if 0.5 < m * 10 ** e < 100 and m * 10 ** e not in ticks]
    ax.set_xticks(np.log10(minor), minor=True)
    ax.set_xlim(np.log10(0.5), np.log10(100))
    ax.set_xlabel('Cost ratio $r$ (cost per window of late / of early maintenance)')
    handles = [Patch(facecolor=PAVA_L, hatch='//////', edgecolor=PAVA_C, lw=0, label='PAVA (offline reference)'),
               Patch(facecolor=CES_C, edgecolor='black', lw=0.5, label='CES (causal)'),
               Patch(facecolor=RAW, edgecolor='black', lw=0.5, label='raw prediction'),
               Patch(facecolor=LBC, edgecolor='black', lw=0.5, label='LB-CES: lower bound on CES mean'),
               Patch(facecolor=LBR, edgecolor='black', lw=0.5, label='LB-raw: lower bound on raw mean')]
    framed(fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.005), fontsize=6.6,
                      handlelength=1.3, handleheight=0.9, handletextpad=0.4, columnspacing=1.0))
    save(fig, 'fig5_decision_map')


# ---------------------------------------------------------------------------------------------------------------
# One test trajectory of the paper's model (median-error bearing, seed 42)
# ---------------------------------------------------------------------------------------------------------------
def fig_trajectory():
    z = np.load(ROOT / 'outputs' / 'r1_final_B10' / 'final20' / 'preds_seed42.npz')
    names = sorted(k[len('raw_'):] for k in z.files if k.startswith('raw_test_'))
    ces = {n: ConstrainedExponentialSmoothing(alpha=0.2, epsilon=0.05).process(z['raw_' + n]) for n in names}
    mae = {n: float(np.mean(np.abs(ces[n] - z['tgt_' + n]))) for n in names}
    med = float(np.median(list(mae.values())))
    pick = min(names, key=lambda n: (abs(mae[n] - med), n))
    cal = load('decision_inbearing.json')['calibration_PR']['submitted architecture']
    c_star = cal['c_star']
    q56 = cal['q'][str(5 / 6)] if str(5 / 6) in cal['q'] else cal['q']['0.8333333333333334']
    y = z['tgt_' + pick]
    raw = z['raw_' + pick]
    mu = z['mcmean_' + pick]
    sd = z['mcstd_' + pick]
    pv = IsotonicRegressor(increasing=False).fit_transform(raw)
    t = np.arange(len(y)) * 10 / 3600.0
    tau = 0.20
    zq = 1.6449
    H = 3.05
    fig = plt.figure(figsize=(W, H))
    ax = axes_at(fig, 1.02, 1.12, 6.08, 1.86)
    st = axes_at(fig, 1.02, 0.46, 6.08, 0.56, sharex=ax)
    ax.fill_between(t, np.clip(mu - zq * c_star * sd, 0.0, 1.3), np.clip(mu + zq * c_star * sd, 0.0, 1.3),
                    color=HO_L, alpha=0.75, lw=0, zorder=1, label='calibrated 90% interval (MC dropout)')
    ax.plot(t, raw, color=RAW, lw=0.5, zorder=2, label='raw prediction')
    ax.plot(t, pv, color=PAVA_C, lw=1.2, ls=(0, (4, 1.6)), zorder=4, label='PAVA (offline reference)')
    ax.plot(t, ces[pick], color=CES_C, lw=1.1, zorder=3, label='CES (causal)')
    ax.plot(t, y, color='black', lw=1.4, zorder=5, label='true normalized RUL')
    ax.axhline(tau, color='black', lw=0.6, ls=(0, (4, 2)), zorder=2)
    ax.text(0.06, tau + 0.02, r'$\tau$ = 0.20', ha='left', va='bottom', fontsize=FS_SMALL, zorder=6)
    h, lab = ax.get_legend_handles_labels()
    order = [4, 3, 2, 1, 0]
    framed(ax.legend([h[i] for i in order], [lab[i] for i in order], loc='upper right', ncol=2, fontsize=6.6,
                     columnspacing=1.2, handlelength=1.8))
    ax.set_ylim(-0.02, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xlim(0, t[-1] + 0.06)
    ax.set_ylabel('Normalized RUL')
    plt.setp(ax.get_xticklabels(), visible=False)
    alarms = [('true crossing', first_cross(y, tau), 'black'),
              ('cost-matched trigger, $r$ = 5', first_cross(raw - q56, tau), HO),
              ('CES point prediction', first_cross(ces[pick], tau), CES_C),
              ('PAVA (offline)', first_cross(pv, tau), PAVA_C)]
    short = ['true crossing', 'trigger, $r$ = 5', 'CES', 'PAVA']
    for row, (lab, k, col) in enumerate(alarms):
        st.plot([t[k]], [row], marker='v', ms=6, color=col, mec='black', mew=0.4, ls='', zorder=3, clip_on=False)
    st.set_ylim(3.6, -0.6)
    st.set_yticks(range(4))
    st.set_yticklabels(short, fontsize=FS_SMALL)
    categorical(st, 'y')
    dotted_grid(st, 'y')
    st.set_xlabel('Operating time (h)')
    st.set_ylabel('Alarm', fontsize=FS_SMALL)
    save(fig, 'fig4_trajectory')
    times = {lab: round(float(t[k]), 2) for lab, k, _ in alarms}
    return pick, mae[pick], med, times


# ---------------------------------------------------------------------------------------------------------------
# Graphical abstract
# ---------------------------------------------------------------------------------------------------------------
def graphical_abstract():
    """Elsevier graphical abstract: 13.28 x 5.31 cm, readable at that size; PNG and TIFF at 1000 dpi, PDF master."""
    pr = load('decision_inbearing.json')
    xj = load('decision_xjtu_inbearing.json')
    rows = []
    for ds, d in (('PRONOSTIA', pr), ('XJTU-SY', xj)):
        for key in d['secondary']['understatement']:
            rows.append((d['residual_intervals']['SP'][key]['pooled_coverage'],
                         d['residual_intervals']['PR'][key]['pooled_coverage']))
    w, h = 13.28 / 2.54, 5.31 / 2.54
    fig = plt.figure(figsize=(w, h))
    cv = canvas(fig)
    cv.text(0.10, h - 0.10, 'A maintenance alarm needs the error tail\non bearings the model has never seen',
            fontsize=8, weight='bold', va='top', linespacing=1.25)
    cv.text(0.10, h - 0.50, r'cost-optimal alarm:  $\hat{y}_t - q_{r/(1+r)} \leq \tau$', fontsize=7, va='top')
    for x0, title, kinds, res, col in (
            (0.10, 'Calibration on windows\nof training bearings', ['interleaved'] * 3,
             'tail understated 3 to 9\ntimes on PRONOSTIA', WB),
            (1.55, 'Calibration on whole\nheld-out bearings', ['train', 'train', 'held'],
             'coverage 27 to 62 points\nhigher on PRONOSTIA', HO)):
        cv.text(x0, 1.20, title, fontsize=6.6, va='top', weight='bold', linespacing=1.15)
        records(cv, x0, x0 + 1.25, 0.72, kinds, pitch=0.11, h=0.07, label=False, n_val=6)
        cv.text(x0, 0.42, res, fontsize=6.6, color=col, va='top', weight='bold', linespacing=1.15)
    ax = axes_at(fig, 3.12, 0.40, 1.98, 1.32)
    ys = np.arange(len(rows), dtype=float)
    dumbbell(ax, ys, [r[0] for r in rows], [r[1] for r in rows], WB, HO, 'o', 's', s=13)
    ax.axvline(90, color='black', lw=0.6, ls=(0, (4, 2)))
    ax.text(90, 1.02, 'target', transform=ax.get_xaxis_transform(), fontsize=6.3, ha='center', va='bottom')
    ax.set_xlim(15, 100)
    ax.set_ylim(len(rows) - 0.4, -1.75)
    ax.set_yticks([])
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_xlabel('Coverage of 90% intervals (%)', fontsize=6.6, labelpad=2)
    ax.tick_params(axis='x', labelsize=6.3)
    dotted_grid(ax, 'x')
    ax.text(rows[0][0], -0.7, 'training windows', fontsize=6.0, color=WB, ha='center', va='bottom', weight='bold')
    ax.text(rows[0][1] + 1.5, -0.7, 'held-out', fontsize=6.0, color=HO, ha='right', va='bottom', weight='bold')
    cv.text(3.12, h - 0.10, '12 network–dataset pairs', fontsize=6.6, va='top', weight='bold')
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'graphical_abstract.pdf', facecolor='white')
    png(fig, OUT / 'graphical_abstract.png')
    plt.close(fig)
    with Image.open(OUT / 'graphical_abstract.png') as im:
        im.save(OUT / 'graphical_abstract.tif', compression='tiff_lzw', dpi=(DPI, DPI))
    print('written graphical_abstract (13.28 x 5.31 cm)')


# ---------------------------------------------------------------------------------------------------------------
# Supplementary Fig. S1 (Section S10, Amendment 9)
# ---------------------------------------------------------------------------------------------------------------
def fig_s10_mechanism():
    """Per-run ratio of the in-bearing to the test 0.9-quantile against the epoch of the checkpoint selected by
    in-bearing early stopping, from mechanism.json."""
    d = load('mechanism.json')
    assert not d['dry_run'] and d['results']['A9_P1']['T'] == -43
    models = list(NAMES)
    palette = [BLUE, RED, GREEN, ORANGE, PURPLE, BROWN, '#E7298A', '#404040']
    colors = dict(zip(models, palette))
    markers = dict(zip(models, 'osD^vPXh'))
    H = 2.85
    fig = plt.figure(figsize=(W, H))
    axes = [axes_at(fig, 0.80, 0.52, 3.00, 1.86), axes_at(fig, 4.08, 0.52, 3.00, 1.86)]
    for ax, ds in zip(axes, ('PRONOSTIA', 'XJTU-SY')):
        for c in (c for c in d['combinations'] if c['dataset'] == ds):
            e = np.array([120.0 if cen else x for x, cen in zip(c['e'], c['censored'])])
            rho = np.array(c['q_cal']) / np.array(c['q_test'])
            order = np.argsort(e, kind='stable')
            col, mk = colors[c['model']], markers[c['model']]
            ax.plot(e[order], rho[order], '-', color=col, lw=0.8, zorder=2)
            cen = np.array(c['censored'])
            ax.scatter(e[~cen], rho[~cen], s=17, color=col, marker=mk, zorder=3, linewidths=0.4,
                       edgecolors='black')
            if cen.any():
                ax.scatter(e[cen], rho[cen], s=17, facecolors='white', edgecolors=col, marker=mk, zorder=3,
                           linewidths=0.9)
        ax.set_xscale('log')
        ax.set_xlim(1.6, 150)
        ax.set_xticks([2, 5, 10, 20, 50, 100])
        ax.set_xticklabels(['2', '5', '10', '20', '50', '100'])
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.axhline(1.0, color='black', lw=0.6, ls=(0, (4, 2)))
        ax.text(0.03, 0.96, ds, transform=ax.transAxes, ha='left', va='top', fontsize=FS_NOTE, weight='bold')
    axes[1].set_yticklabels([])
    axes[0].set_ylabel('Within-bearing / test 0.9-quantile\nof the absolute error', linespacing=1.15)
    fig.text(3.94 / W, 0.08 / H, 'Epoch of the checkpoint selected by within-bearing early stopping (log scale)',
             fontsize=FS_LAB, ha='center', va='bottom')
    handles = [Line2D([], [], color=colors[m], marker=markers[m], lw=0.8, ms=4.2, mec='black', mew=0.4,
                      label=NAMES[m]) for m in models]
    framed(fig.legend(handles=handles, loc='upper center', ncol=8, handletextpad=0.3, columnspacing=0.9,
                      fontsize=FS_SMALL, bbox_to_anchor=(3.94 / W, 0.995), handlelength=1.6))
    save(fig, 'figS10_mechanism')


def main():
    graphical_abstract()
    print('  concept:', fig_concept())
    print('  components:', [tuple(round(x, 1) if isinstance(x, float) else x for x in v) for v in fig_components()])
    print('  alarm timing (h, late %):', fig_alarm_timing())
    fig_framework()
    fig_s10_mechanism()
    rows = fig_calibration_source()
    for r in rows:
        print('  ', r[0], r[1], f'ratio SP {r[2]:.2f} PR {r[3]:.2f} cov SP {r[4]:.1f} PR {r[5]:.1f}')
    fig_decision_map()
    pick, m, med, times = fig_trajectory()
    print(f'  trajectory bearing {pick}: CES MAE {m:.4f} (median over bearings {med:.4f}); alarms (h) {times}')
    fig_architecture()


if __name__ == '__main__':
    main()
