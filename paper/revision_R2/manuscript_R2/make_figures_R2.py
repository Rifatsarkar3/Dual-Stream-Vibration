"""
make_figures_R2.py: figures of the revised manuscript (JRESS-D-26-05118, revision 1).

Reads only saved analysis outputs (outputs/r1_analysis/*.json) and the saved test predictions of
the paper's model (outputs/r1_final_B10/final20/preds_seed42.npz); no model is trained or run.
Writes vector PDFs (and PNG previews) to paper/revision_R2/manuscript_R2/figures/.

Shared style (reviewer comment on figure readability): Times-like serif at 8 pt for a 6.3 in
text width, one colour per scheme in every figure, no titles inside figures.

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
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.online_filters import ConstrainedExponentialSmoothing  # noqa: E402
from src.isotonic_eval import IsotonicRegressor  # noqa: E402

AN = ROOT / 'outputs' / 'r1_analysis'
OUT = Path(__file__).resolve().parent / 'figures'
WIDTH = 6.3

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'TeX Gyre Termes', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'savefig.dpi': 300,
})

# one colour per scheme, used in every figure
C_INB = '#D55E00'      # in-bearing validation (training-bearing windows)
C_HO = '#0072B2'       # held-out bearings (the protocol)
C_TRUE = '#000000'
C_RAW = '#9A9A9A'
C_CES = '#0072B2'
C_PAVA = '#7B3294'
C_LBCES = '#1B7837'
C_LBRAW = '#7FBC41'

NAMES = {'submitted architecture': 'Ours', '1D CNN': '1D CNN', 'BiLSTM': 'BiLSTM', 'CNN-LSTM': 'CNN-LSTM',
         '1D ResNet': '1D ResNet', 'Transformer': 'Transformer', 'TSMC-CNN [21]': 'TSMC-CNN',
         'MSCNN [7]': 'MSCNN'}


def load(name):
    return json.loads((AN / name).read_text(encoding='utf-8'))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{stem}.png', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'written {stem}')


# ---------------------------------------------------------------------------------------------
# Concept figure (Section 1): where the calibration errors come from, what they say about the error tail, and
# what an understated decision quantile does to the alarm (Corollary 1). Saved values only.
# ---------------------------------------------------------------------------------------------
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

    fig = plt.figure(figsize=(WIDTH, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.85, 1.3], wspace=0.38)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) two sources of calibration errors
    a.set_xlim(0, 40)
    a.set_ylim(-1.6, 9.6)
    a.axis('off')
    a.text(0, 9.2, 'Within-bearing calibration', fontsize=7.2, weight='bold', color=C_INB, va='center')
    for row in range(4):
        yy = 7.85 - row * 0.8
        for k in range(40):
            a.add_patch(plt.Rectangle((k, yy), 0.82, 0.52, color=C_INB if k % 10 == 4 else '#C9C9C9', lw=0))
    a.text(0, 4.6, 'validation windows interleaved with\nthe training windows of each bearing', fontsize=6.3,
           va='center', color='#333333', linespacing=1.2)
    a.text(0, 3.05, 'Held-out calibration', fontsize=7.2, weight='bold', color=C_HO, va='center')
    for row in range(4):
        yy = 1.75 - row * 0.62
        for k in range(40):
            a.add_patch(plt.Rectangle((k, yy), 0.82, 0.42, color=C_HO if row == 3 else '#C9C9C9', lw=0))
    a.text(0, -1.15, 'a whole bearing left out of training', fontsize=6.3, va='center', color='#333333')
    a.text(0.0, 1.04, '(a)', transform=a.transAxes, fontsize=8, weight='bold', ha='right')

    # (b) 0.9-quantile of the absolute error: calibration data against test bearings
    rows = [('within-bearing', u['SP_calibration_q90'], u['SP_test_q90'], C_INB),
            ('held-out', u['PR_calibration_q90'], u['PR_test_q90'], C_HO)]
    for i, (lab, qc, qt, col) in enumerate(rows):
        y0 = i * 1.45
        b.text(0.0, y0 - 0.36, lab, fontsize=7, color=col, weight='bold', va='bottom', ha='left')
        b.barh(y0, qc, height=0.34, color=col, lw=0)
        b.barh(y0 + 0.4, qt, height=0.34, color='#E3E3E3', edgecolor='#444444', lw=0.6)
        b.text(qc + 0.008, y0, f'{qc:.2f}' + ('  calibration data' if i == 0 else ''), va='center', fontsize=6.2,
               color=col)
        b.text(qt + 0.008, y0 + 0.4, f'{qt:.2f}', va='center', fontsize=6.2, color='#333333')
        if i == 0:
            b.text(0.01, y0 + 0.4, 'test bearings', va='center', fontsize=6.0, color='#333333')
    b.set_yticks([])
    b.set_ylim(2.25, -0.55)
    b.set_xlim(0, 0.42)
    b.set_xticks([0, 0.2, 0.4])
    b.set_xlabel('0.9-quantile of absolute error')
    for s in ('top', 'right', 'left'):
        b.spines[s].set_visible(False)
    b.text(0.0, 1.04, '(b)', transform=b.transAxes, fontsize=8, weight='bold', ha='right')

    # (c) the cost-matched trigger at r = 5 with each estimate of the decision quantile
    c.plot(t, raw, color=C_RAW, lw=0.5, alpha=0.85, label='prediction')
    c.plot(t, y, color=C_TRUE, lw=1.0, label='true RUL')
    c.axhline(tau, color='#333333', lw=0.7, ls=':')
    c.text(0.3, tau + 0.03, r'$\tau$', fontsize=7, va='bottom')
    k_true = first_cross(y, tau)
    k_wb = first_cross(raw - q56_wb, tau)
    k_ho = first_cross(raw - q56_ho, tau)
    marks = [(k_true, C_TRUE, 'due'), (k_ho, C_HO, 'held-out'), (k_wb, C_INB, 'within-bearing')]
    for k, col, _ in marks:
        c.plot([t[k]], [-0.1], marker='v', color=col, ms=6, ls='', clip_on=False, zorder=4)
    c.set_ylim(-0.13, 1.15)
    c.set_xlim(0, t[-1] + 0.15)
    c.set_xlabel('operating time (h)')
    c.set_ylabel('normalized RUL')
    for s in ('top', 'right'):
        c.spines[s].set_visible(False)
    h = [Line2D([], [], color=C_TRUE, lw=1.0), Line2D([], [], color=C_RAW, lw=0.8)] + \
        [Line2D([], [], marker='v', color=col, ls='', ms=5) for _, col, _ in marks]
    lab = ['true RUL', 'prediction', 'maintenance due', 'alarm, held-out', 'alarm, within-bearing']
    c.legend(h, lab, loc='upper right', frameon=False, fontsize=6.0, handlelength=1.4, labelspacing=0.25,
             borderaxespad=0.1)
    c.text(0.0, 1.04, '(c)', transform=c.transAxes, fontsize=8, weight='bold', ha='right')
    save(fig, 'fig0_concept')
    return {'q56_within': q56_wb, 'q56_heldout': q56_ho, 'due_h': float(t[k_true]),
            'alarm_heldout_h': float(t[k_ho]), 'alarm_within_h': float(t[k_wb]),
            'q90': {r[0]: (r[1], r[2]) for r in rows}}


# ---------------------------------------------------------------------------------------------
# Figure 1: the evaluation, calibration and decision protocol
# ---------------------------------------------------------------------------------------------
def box(ax, x, y, w, h, text, edge='#333333', face='white', ls='-', lw=0.8, weight='normal', size=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=0.8',
                                linewidth=lw, edgecolor=edge, facecolor=face, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, weight=weight,
            linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color='#333333', ls='-', text=None, tx=0, ty=0, size=6.6):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=0.8, linestyle=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=7))
    if text:
        ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text, ha='center', va='center', fontsize=size,
                color=color, style='italic', bbox=dict(facecolor='white', edgecolor='none', pad=0.4))


def fig_framework():
    fig, ax = plt.subplots(figsize=(WIDTH, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58)
    ax.axis('off')
    # lane frames
    ax.add_patch(FancyBboxPatch((0.6, 31.2), 98.8, 26.0, boxstyle='round,pad=0,rounding_size=1.2',
                                linewidth=0.7, edgecolor='#666666', facecolor='#F4F8FB', linestyle='--'))
    ax.text(1.8, 55.8, 'Development: training bearings only', fontsize=7, weight='bold', va='center',
            color='#333333')
    ax.add_patch(FancyBboxPatch((0.6, 13.0), 98.8, 17.0, boxstyle='round,pad=0,rounding_size=1.2',
                                linewidth=0.7, edgecolor='#666666', facecolor='white'))
    ax.text(1.8, 14.5, 'Evaluation: test bearings, used once', fontsize=7, weight='bold', va='center',
            color='#333333')
    # development lane
    box(ax, 2.5, 35.5, 17.0, 16.5, 'Four training\nbearings\n(two operating\nconditions)')
    box(ax, 24.0, 43.6, 22.0, 9.6, 'Four-fold leave-one-\nbearing-out runs: fixed\nbudget, final epoch kept', size=6.9)
    box(ax, 24.0, 33.4, 22.0, 7.2, 'Selection: budget by mean\nvalidation MAE; $\\lambda$ by the\none-standard-error rule',
        face='#EAF2FA', size=6.7)
    box(ax, 51.0, 40.2, 22.0, 10.0, 'Calibration data:\nerrors on the held-out\nbearing of each run', face='#EAF2FA')
    box(ax, 77.5, 35.5, 20.5, 16.5, 'Error distribution on\nunseen bearings, $G$:\nquantiles $q_p$,\ninterval factor $c^*$',
        face='#EAF2FA')
    arrow(ax, 19.5, 48.4, 24.0, 48.4)
    arrow(ax, 35.0, 43.6, 35.0, 40.6)
    arrow(ax, 46.0, 47.0, 51.0, 45.2)
    arrow(ax, 73.0, 45.2, 77.5, 44.4)
    # evaluation lane
    box(ax, 2.5, 17.2, 17.0, 10.0, 'Final model:\nall four bearings,\n$B^*$ epochs, 20 seeds', size=6.9)
    box(ax, 24.0, 17.2, 22.0, 10.0, 'Test bearings (+ zero-shot):\nPRONOSTIA 10 (+3)\nXJTU-SY 6 (+5)', size=6.8)
    box(ax, 51.0, 17.2, 22.0, 10.0, 'Alarm rule (Proposition 1):\n$\\hat y_t - q_{r/(1+r)} \\leq \\tau$\ninterval $\\mu_t \\pm z\\,c^* s_t$', size=6.8)
    box(ax, 77.5, 17.2, 20.5, 10.0, 'Inference: bearing as\nunit, exact sign-flip\ntest, Holm correction', size=6.8)
    arrow(ax, 11.0, 35.5, 11.0, 27.2)
    arrow(ax, 19.5, 22.2, 24.0, 22.2)
    arrow(ax, 46.0, 22.2, 51.0, 22.2)
    arrow(ax, 73.0, 22.2, 77.5, 22.2)
    arrow(ax, 25.5, 33.4, 18.5, 27.2, color='#555555', text='$B^*$, $\\lambda$', tx=3.6, ty=-0.6)
    arrow(ax, 79.5, 35.5, 72.0, 27.2, color='#555555', text='$q_p$, $c^*$', tx=-4.4, ty=-0.4)
    # alternatives evaluated
    ax.text(1.8, 10.4, 'Alternatives evaluated against the protocol (Sections 4.2 and 6.1)', fontsize=7, weight='bold',
            color=C_INB, va='center')
    box(ax, 2.5, 0.8, 30.2, 7.6, 'Within-bearing validation windows for\nearly stopping and calibration:\nerror tail understated 1.8 to 9 times',
        edge=C_INB, ls='--', size=6.5)
    box(ax, 35.3, 0.8, 30.2, 7.6, 'Checkpoint selected on the\ncalibration bearing: coverage 1.4\npoints farther from the target',
        edge=C_INB, ls='--', size=6.5)
    box(ax, 68.1, 0.8, 29.9, 7.6, 'Seed-level tests and window\npooling: 29 instead of 7 of 44\ncomparisons significant',
        edge=C_INB, ls='--', size=6.5)
    save(fig, 'fig1_framework')


# ---------------------------------------------------------------------------------------------
# Figure 3: where the calibration errors come from (Amendments 6 and 7)
# ---------------------------------------------------------------------------------------------
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
    n = len(rows)
    ys = []
    y = 0.0
    prev = None
    for r in rows:
        if prev is not None and r[0] != prev:
            y += 0.8
        ys.append(y)
        y += 1.0
        prev = r[0]
    ys = np.array(ys)
    fig = plt.figure(figsize=(WIDTH, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.62, wspace=0.08)
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1], sharey=a1)
    plt.setp(a2.get_yticklabels(), visible=False)
    for (ds, name, rs, rp, cs, cp), yy in zip(rows, ys):
        a1.plot([rs, rp], [yy, yy], color='#BBBBBB', lw=0.9, zorder=1)
        a1.scatter(rs, yy, s=18, color=C_INB, zorder=2, marker='o')
        a1.scatter(rp, yy, s=18, color=C_HO, zorder=2, marker='s')
        a2.plot([cs, cp], [yy, yy], color='#BBBBBB', lw=0.9, zorder=1)
        a2.scatter(cs, yy, s=18, color=C_INB, zorder=2, marker='o')
        a2.scatter(cp, yy, s=18, color=C_HO, zorder=2, marker='s')
    a1.set_xscale('log')
    a1.set_xlim(0.08, 1.6)
    a1.axvline(1.0, color='#333333', lw=0.7, ls=':')
    a1.set_xticks([0.1, 0.2, 0.5, 1.0])
    a1.set_xticklabels(['0.1', '0.2', '0.5', '1'])
    a1.set_xlabel('0.9-quantile of absolute error:\ncalibration data / test bearings')
    a2.set_xlim(15, 100)
    a2.axvline(90, color='#333333', lw=0.7, ls=':')
    a2.set_xlabel('Coverage of the 90 % interval on\ntest bearings (%)')
    a1.set_yticks(ys)
    a1.set_yticklabels([r[1] for r in rows])
    a1.invert_yaxis()
    for ds in ('PRONOSTIA', 'XJTU-SY'):
        idx = [i for i, r in enumerate(rows) if r[0] == ds]
        ymid = ys[idx].mean()
        a1.text(-0.42, ymid, ds, transform=a1.get_yaxis_transform(), rotation=90, ha='center', va='center',
                fontsize=7.5, weight='bold')
    sep = (ys[[i for i, r in enumerate(rows) if r[0] == 'PRONOSTIA'][-1]] + ys[[i for i, r in enumerate(rows) if r[0] == 'XJTU-SY'][0]]) / 2
    for a in (a1, a2):
        a.axhline(sep, color='#DDDDDD', lw=0.6)
        a.tick_params(axis='y', length=0)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    a1.text(0.97, -0.9, 'no understatement', fontsize=6.8, ha='right', va='bottom', color='#333333')
    a2.text(90, -0.9, 'target', fontsize=6.8, ha='center', va='bottom', color='#333333')
    handles = [Line2D([], [], color=C_INB, marker='o', ls='', ms=4.5, label='within-bearing calibration'),
               Line2D([], [], color=C_HO, marker='s', ls='', ms=4.5, label='held-out calibration'),
               Line2D([], [], color=C_RAW, marker='D', ls='', ms=4.0, label='uncalibrated MC dropout')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.55, 0.985))
    a1.text(-0.02, 1.02, '(a)', transform=a1.transAxes, fontsize=8, weight='bold', ha='right')
    a2.text(-0.02, 1.02, '(b)', transform=a2.transAxes, fontsize=8, weight='bold', ha='right')

    # (c, d) deployed MC-dropout intervals of each network with dropout, before and after held-out calibration
    mc = load('pooled_framework.json')['A5_D']
    names = dict(NAMES, **{'adopted variant': 'Temporal-context variant'})
    keys = ['submitted architecture', 'adopted variant'] + [k for k in NAMES if k in mc and k != 'submitted architecture']
    mrows = [(names[k], mc[k]['raw']['pooled_coverage'], mc[k]['calibrated']['pooled_coverage'],
              mc[k]['raw']['late_coverage'], mc[k]['calibrated']['late_coverage']) for k in keys]   # TSMC-CNN: no dropout
    my = np.arange(len(mrows))
    b1 = fig.add_subplot(gs[1, 0])
    b2 = fig.add_subplot(gs[1, 1], sharey=b1)
    plt.setp(b2.get_yticklabels(), visible=False)
    for (name, raw_all, cal_all, raw_late, cal_late), yy in zip(mrows, my):
        for ax, lo, hi in ((b1, raw_all, cal_all), (b2, raw_late, cal_late)):
            ax.plot([lo, hi], [yy, yy], color='#BBBBBB', lw=0.9, zorder=1)
            ax.scatter(lo, yy, s=16, color=C_RAW, marker='D', zorder=2)
            ax.scatter(hi, yy, s=18, color=C_HO, marker='s', zorder=2)
    for ax, lab in ((b1, 'Coverage of the 90 % MC-dropout interval (%)'),
                    (b2, 'Coverage in late life, true RUL < 0.2 (%)')):
        ax.set_xlim(0, 102)
        ax.axvline(90, color='#333333', lw=0.7, ls=':')
        ax.set_xlabel(lab)
        ax.tick_params(axis='y', length=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    b1.set_yticks(my)
    b1.set_yticklabels([r[0] for r in mrows])
    b1.invert_yaxis()
    b1.text(-0.42, my.mean(), 'PRONOSTIA', transform=b1.get_yaxis_transform(), rotation=90, ha='center',
            va='center', fontsize=7.5, weight='bold')
    b1.text(-0.02, 1.04, '(c)', transform=b1.transAxes, fontsize=8, weight='bold', ha='right')
    b2.text(-0.02, 1.04, '(d)', transform=b2.transAxes, fontsize=8, weight='bold', ha='right')
    save(fig, 'fig3_calibration_source')
    return rows


# ---------------------------------------------------------------------------------------------
# Components of the network (Section 5.3): relative difference of each variant to the full model on four
# endpoints, from the saved ablation analysis. Exact values: Supplementary Material.
# ---------------------------------------------------------------------------------------------
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
    marks = [('CES MAE', 'o', '#0072B2'), ('interval score', 's', '#E69F00'),
             ('cost, $r$ = 5', '^', '#009E73'), ('cost, $r$ = 10', 'v', '#CC79A7')]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.45))
    off = np.linspace(-0.27, 0.27, len(marks))
    for i, row in enumerate(vals):
        ax.axhspan(i - 0.42, i + 0.42, color='#F4F4F4' if i % 2 == 0 else 'white', lw=0, zorder=0)
        for j, (lab, m, col) in enumerate(marks):
            ax.scatter(row[1 + j], i + off[j], marker=m, s=22, color=col, zorder=3, linewidths=0)
    ax.axvline(0, color='#333333', lw=0.7)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([v[0] for v in vals])
    ax.set_ylim(len(vals) - 0.5, -0.5)
    ax.set_xlim(-20, 50)
    ax.set_xlabel('Difference to the full model, (variant $-$ full) / variant (%); positive: full model better')
    ax.tick_params(axis='y', length=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    handles = [Line2D([], [], marker=m, color=col, ls='', ms=5, label=lab) for lab, m, col in marks]
    ax.legend(handles=handles, loc='lower right', frameon=False, ncol=2, fontsize=7, handletextpad=0.3,
              columnspacing=1.0)
    save(fig, 'fig_components')
    return vals


# ---------------------------------------------------------------------------------------------
# Alarm timing (Section 6.3): mean timing error and per-bearing share of late alarms at tau = 0.20 for the
# corrections and the calibrated lower bounds of the paper's model. Exact values: Supplementary Material.
# ---------------------------------------------------------------------------------------------
RULES = [('raw', 'Raw prediction (and RTRM)', C_RAW), ('ces', 'CES', C_CES), ('pava', 'PAVA (offline)', C_PAVA),
         ('lb_raw', 'Lower bound, raw mean', C_LBRAW), ('lb_ces', 'Lower bound, CES mean', C_LBCES)]


def fig_alarm_timing():
    t = load('decision_final20_all10.json')['timing']['0.20']
    assert t['stats']['raw'] == t['stats']['rtrm']                   # RTRM alarms exactly when raw does
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.15), sharey=True,
                               gridspec_kw={'width_ratios': [1, 1.15], 'wspace': 0.08})
    rng = np.random.default_rng(0)
    out = {}
    for i, (key, label, col) in enumerate(RULES):
        st = t['stats'][key]
        dt_h = st['dt'] * 10 / 3600.0
        a.barh(i, dt_h, height=0.5, color=col, lw=0)
        a.text(dt_h + (0.08 if dt_h >= 0 else -0.08), i, f'{dt_h:+.2f} h'.replace('-', '−'), va='center',
               ha='left' if dt_h >= 0 else 'right', fontsize=6.6)
        fr = np.array(list(t['late_fraction_per_bearing'][key].values())) * 100
        jit = rng.uniform(-0.17, 0.17, len(fr))
        b.scatter(fr, i + jit, s=13, color=col, alpha=0.85, linewidths=0, zorder=2)
        b.plot([st['late'] * 100] * 2, [i - 0.3, i + 0.3], color='#222222', lw=1.4, zorder=3)
        out[key] = (round(dt_h, 3), round(st['late'] * 100, 1))
    a.axvline(0, color='#333333', lw=0.7)
    a.set_xlim(-2.9, 1.3)
    a.set_xlabel('Mean alarm timing error (h); negative: early')
    a.set_yticks(range(len(RULES)))
    a.set_yticklabels([r[1] for r in RULES])
    a.set_ylim(len(RULES) - 0.5, -0.5)
    b.set_xlim(-4, 104)
    b.set_xlabel('Late alarms per bearing (% of seeds); bar: all bearings')
    for ax in (a, b):
        ax.tick_params(axis='y', length=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    a.text(-0.02, 1.03, '(a)', transform=a.transAxes, fontsize=8, weight='bold', ha='right')
    b.text(-0.02, 1.03, '(b)', transform=b.transAxes, fontsize=8, weight='bold', ha='right')
    save(fig, 'fig_alarm_timing')
    return out


# ---------------------------------------------------------------------------------------------
# Figure 4: decision-optimal rule against the cost ratio (paper's model, ten test bearings)
# ---------------------------------------------------------------------------------------------
C_PAVA_LIGHT = '#D9C2E3'
VARIANTS = {'pava': ('PAVA (offline)', C_PAVA_LIGHT), 'ces': ('CES', C_CES), 'raw': ('raw', C_RAW),
            'rtrm': ('raw', C_RAW), 'lb_ces': ('lower bound, CES mean', C_LBCES),
            'lb_raw': ('lower bound, raw mean', C_LBRAW)}


def fig_decision_map():
    d = load('decision_final20_all10.json')
    taus = ['0.30', '0.20', '0.10']
    fig, ax = plt.subplots(figsize=(WIDTH, 2.05))
    for i, tau in enumerate(taus):
        segs = d['timing'][tau]['optimal_by_ratio']
        for s in segs:
            lab, col = VARIANTS[s['variant']]
            a, b = max(s['r_from'], 0.5), min(s['r_to'], 100.0)
            if b <= a:
                continue
            ax.barh(i, np.log10(b) - np.log10(a), left=np.log10(a), height=0.62, color=col,
                    edgecolor=C_PAVA if s['variant'] == 'pava' else 'white', linewidth=0.8,
                    hatch='////' if s['variant'] == 'pava' else None)
            mid = (np.log10(a) + np.log10(b)) / 2
            short = {'pava': 'PAVA', 'ces': 'CES', 'raw': 'raw', 'rtrm': 'raw', 'lb_ces': 'LB-CES',
                     'lb_raw': 'LB-raw'}[s['variant']]
            if np.log10(b) - np.log10(a) > 0.17:
                ax.text(mid, i, short, ha='center', va='center', fontsize=7,
                        color='white' if s['variant'] in ('lb_ces', 'ces') else 'black',
                        bbox=dict(facecolor=C_PAVA_LIGHT if s['variant'] == 'pava' else col, edgecolor='none', pad=0.6), zorder=6)
    for r in (2, 5, 10):
        ax.axvline(np.log10(r), color='#222222', lw=0.8, ls='--', zorder=5)
        ax.text(np.log10(r) + 0.025, -0.52, f'$r$ = {r}', ha='left', va='bottom', fontsize=6.8, zorder=6,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
    ax.set_ylim(2.45, -0.75)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f'$\\tau$ = {t}' for t in taus])
    ticks = [0.5, 1, 2, 5, 10, 20, 50, 100]
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(np.log10(0.5), np.log10(100))
    ax.set_xlabel('cost ratio $r$ = cost per window of late maintenance / cost per window of early maintenance')
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    handles = [Patch(facecolor=C_PAVA_LIGHT, hatch='////', edgecolor=C_PAVA, label='PAVA (offline reference)'),
               Patch(facecolor=C_CES, label='CES (causal)'), Patch(facecolor=C_RAW, label='raw prediction'),
               Patch(facecolor=C_LBCES, label='calibrated lower bound on CES mean'),
               Patch(facecolor=C_LBRAW, label='calibrated lower bound on raw mean')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.32))
    save(fig, 'fig5_decision_map')


# ---------------------------------------------------------------------------------------------
# Figure 5: one test trajectory of the paper's model (median-error bearing, seed 42)
# ---------------------------------------------------------------------------------------------
def first_cross(x, tau):
    idx = np.flatnonzero(x <= tau)
    return int(idx[0]) if idx.size else len(x) - 1


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
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6))
    ax.fill_between(t, np.clip(mu - zq * c_star * sd, -0.2, 1.3), np.clip(mu + zq * c_star * sd, -0.2, 1.3),
                    color=C_HO, alpha=0.11, lw=0, label='calibrated 90 % interval (MC dropout)')
    ax.plot(t, raw, color=C_RAW, lw=0.5, alpha=0.8, label='raw prediction')
    ax.plot(t, ces[pick], color=C_CES, lw=1.1, label='CES (causal)')
    ax.plot(t, pv, color=C_PAVA, lw=1.1, ls='--', label='PAVA (offline reference)')
    ax.plot(t, y, color=C_TRUE, lw=1.1, label='true normalized RUL')
    ax.axhline(tau, color='#333333', lw=0.7, ls=':')
    ax.text(t[-1], tau + 0.02, '$\\tau$ = 0.20', ha='right', va='bottom', fontsize=7)
    alarms = [('true crossing', first_cross(y, tau), C_TRUE, 'v'),
              ('cost-matched trigger, $r$ = 5', first_cross(raw - q56, tau), C_LBCES, 'v'),
              ('CES point prediction', first_cross(ces[pick], tau), C_CES, 'v'),
              ('PAVA (offline)', first_cross(pv, tau), C_PAVA, 'v')]
    for lab, k, col, m in alarms:
        ax.plot([t[k]], [-0.12], marker=m, color=col, ms=6, ls='', clip_on=False)
    ax.set_ylim(-0.15, 1.25)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel('operating time (h)')
    ax.set_ylabel('normalized RUL')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2 = [Line2D([], [], marker='v', color=col, ls='', ms=5) for _, _, col, _ in alarms]
    l2 = ['alarm: ' + lab for lab, _, _, _ in alarms]
    ax.legend(h1 + h2, l1 + l2, loc='upper center', ncol=3, frameon=False, fontsize=6.6,
              bbox_to_anchor=(0.5, -0.2))
    save(fig, 'fig4_trajectory')
    times = {lab: round(float(t[k]), 2) for lab, k, _, _ in alarms}
    return pick, mae[pick], med, times


# ---------------------------------------------------------------------------------------------
# Figure 2: architecture of the network (the submitted diagram redrawn in the shared style, same content;
# the two training-only boxes swap places so that each dashed connector is short and crosses nothing)
# ---------------------------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(WIDTH, 3.85))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60.5)
    ax.axis('off')
    grey = '#555555'
    # training-only components (dashed), discarded at inference
    ax.text(70.0, 59.2, 'Training-time only, discarded at inference', ha='center', va='center', fontsize=7,
            style='italic', color=grey)
    box(ax, 41.0, 45.5, 27.0, 11.5, 'DANN binary domain\ndiscriminator (GRL)\nLin(128,64) $\\rightarrow$ Lin(64,1)\n'
        '8.3k params; backbone\ngets GRL-reversed gradient', edge=grey, face='#F7F7F7', ls='--', size=6.6)
    box(ax, 71.5, 45.5, 28.0, 11.5, 'PGMC auxiliary\nmonotonic penalty\nchronological within-bearing\n'
        'chunk (BN frozen)\n$\\lambda_{\\mathrm{PGMC}}(j)$ warm-up', edge=grey, face='#F7F7F7', ls='--', size=6.6)
    # inference graph
    box(ax, 1.0, 20.0, 15.5, 13.0, 'Raw vibration\nwindow\n$x_t \\in \\mathbb{R}^{1 \\times 2560}$\n100 ms at\n'
        '25.6 kHz', face='#EAF2FA', size=6.8)
    box(ax, 19.5, 33.0, 33.0, 10.5, 'Temporal stream: 1D CNN\nConv1d $\\times$3 (k 15/7/3, str 4/4/2)\n'
        'ch 1$\\rightarrow$64$\\rightarrow$128$\\rightarrow$128, BN + GELU\nGAP $\\rightarrow$ Linear',
        face='#E5F1E1', size=6.8)
    box(ax, 9.0, 4.5, 24.0, 12.5, 'Fixed envelope extractor\n(non-learnable)\n'
        'Hilbert $\\rightarrow$ |env| $\\rightarrow$ FFT\n$\\rightarrow$ 512 bins\n(BPFO/BPFI/BSF/FTF)',
        face='#FBF5D9', size=6.6)
    box(ax, 36.5, 4.5, 30.0, 12.5, 'Spectral stream: 1D CNN\nConv1d $\\times$3 (k 7/5/3, str 2)\n'
        'ch 1$\\rightarrow$64$\\rightarrow$64$\\rightarrow$64, BN + GELU\nGAP $\\rightarrow$ Linear',
        face='#FCEBD9', size=6.8)
    box(ax, 71.5, 24.5, 17.0, 13.0, 'Adaptive scalar\nfusion gate\n$\\alpha_t = \\sigma(\\cdot)$\n'
        '$F_t = \\alpha_t F_t^{\\mathrm{temp}}$\n$+\\,(1-\\alpha_t)\\,F_t^{\\mathrm{spec}}$', face='#EEE9F5', size=6.8)
    box(ax, 71.5, 6.0, 17.0, 13.0, 'Regression head\nLN $\\rightarrow$ Drop\n$\\rightarrow$ Lin(128,64)\n'
        '$\\rightarrow$ GELU $\\rightarrow$ Drop\n$\\rightarrow$ Lin(64,1)', face='#F7E3E3', size=6.8)
    box(ax, 91.5, 8.5, 8.0, 8.0, '$\\hat{y}_t$\n(RUL)', face='#EAF2FA', size=7.6, weight='bold')
    # data flow at inference
    arrow(ax, 16.5, 30.0, 19.5, 36.5)
    arrow(ax, 10.0, 20.0, 12.0, 17.0)
    arrow(ax, 33.0, 10.75, 36.5, 10.75)
    arrow(ax, 52.5, 37.0, 71.5, 32.5)
    arrow(ax, 66.5, 14.0, 71.5, 27.5)
    arrow(ax, 80.0, 24.5, 80.0, 19.0)
    arrow(ax, 88.5, 12.5, 91.5, 12.5)
    ax.text(61.5, 37.8, '$F_t^{\\mathrm{temp}} \\in \\mathbb{R}^{128}$', ha='center', va='center', fontsize=7,
            color='#1B7837')
    ax.text(62.0, 22.0, '$F_t^{\\mathrm{spec}} \\in \\mathbb{R}^{128}$', ha='center', va='center', fontsize=7,
            color='#8C510A')
    # training-only connectors: fused features to the discriminator, predictions to the monotonic penalty
    arrow(ax, 75.0, 37.5, 65.0, 45.5, color='#777777', ls='--')
    arrow(ax, 95.5, 16.5, 95.5, 45.5, color='#777777', ls='--')
    ax.text(50.0, 1.5, 'Inference graph: 192,130 parameters; 1.48 ms FP32 latency per 100 ms window',
            ha='center', va='center', fontsize=7, style='italic', color=grey)
    save(fig, 'fig2_architecture')


def graphical_abstract():
    """Elsevier graphical abstract: 13.28 x 5.31 cm, readable at that size; TIFF at 600 dpi plus PDF/PNG."""
    pr = load('decision_inbearing.json')
    xj = load('decision_xjtu_inbearing.json')
    rows = []
    for ds, d in (('PRONOSTIA', pr), ('XJTU-SY', xj)):
        for key in d['secondary']['understatement']:
            rows.append((d['residual_intervals']['SP'][key]['pooled_coverage'],
                         d['residual_intervals']['PR'][key]['pooled_coverage']))
    w, h = 13.28 / 2.54, 5.31 / 2.54
    fig = plt.figure(figsize=(w, h))
    ax0 = fig.add_axes([0.0, 0.0, 0.53, 1.0])
    ax0.set_xlim(0, 100)
    ax0.set_ylim(0, 100)
    ax0.axis('off')
    ax0.text(2, 95, 'A maintenance alarm needs the error tail\non bearings the model has never seen',
             fontsize=7.4, weight='bold', va='top', linespacing=1.3)
    ax0.text(2, 70, r'cost-optimal alarm: $\hat y_t - q_{r/(1+r)} \leq \tau$', fontsize=7.0, va='top',
             color='#222222')
    box(ax0, 3, 30, 44, 20, 'calibrate on windows of\ntraining bearings', edge=C_INB, ls='--', size=6.4)
    box(ax0, 53, 30, 44, 20, 'calibrate on whole\nheld-out bearings', edge=C_HO, size=6.4, face='#EAF2FA')
    ax0.text(25, 21, 'tail understated\n1.8 to 9 times', ha='center', va='top', fontsize=6.4, color=C_INB)
    ax0.text(75, 21, 'coverage 27 to 62 points\nhigher on PRONOSTIA', ha='center', va='top', fontsize=6.4, color=C_HO)
    ax = fig.add_axes([0.62, 0.22, 0.35, 0.68])
    ys = np.arange(len(rows))
    for (cs, cp), yy in zip(rows, ys):
        ax.plot([cs, cp], [yy, yy], color='#BBBBBB', lw=0.8, zorder=1)
        ax.scatter(cs, yy, s=10, color=C_INB, zorder=2, marker='o')
        ax.scatter(cp, yy, s=10, color=C_HO, zorder=2, marker='s')
    ax.axvline(90, color='#333333', lw=0.6, ls=':')
    ax.text(90, -0.9, 'target', fontsize=6, ha='center', va='bottom')
    ax.set_xlim(15, 100)
    ax.set_yticks([])
    ax.set_xlabel('coverage of 90% intervals (%)', fontsize=6.4)
    ax.tick_params(axis='x', labelsize=6)
    ax.invert_yaxis()
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.text(0.5, 1.1, '12 network-dataset pairs', transform=ax.transAxes, ha='center', fontsize=6.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'graphical_abstract.pdf', facecolor='white')
    fig.savefig(OUT / 'graphical_abstract.png', dpi=300, facecolor='white')
    tmp = OUT / '_ga_tmp.png'
    fig.savefig(tmp, dpi=600, facecolor='white')
    plt.close(fig)
    from PIL import Image
    Image.open(tmp).convert('RGB').save(OUT / 'graphical_abstract.tif', compression='tiff_lzw', dpi=(600, 600))
    tmp.unlink()
    print('written graphical_abstract (13.28 x 5.31 cm)')


def fig_s10_mechanism():
    """Supplementary Fig. S1 (Section S10, Amendment 9): per-run ratio of the in-bearing to the test 0.9-quantile
    against the epoch of the checkpoint selected by in-bearing early stopping, from mechanism.json."""
    d = load('mechanism.json')
    assert not d['dry_run'] and d['results']['A9_P1']['T'] == -43
    models = list(NAMES)
    colors = dict(zip(models, plt.get_cmap('tab10').colors))
    markers = dict(zip(models, 'osD^vPXh'))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    for ax, ds in zip(axes, ('PRONOSTIA', 'XJTU-SY')):
        for c in (c for c in d['combinations'] if c['dataset'] == ds):
            e = np.array([120.0 if cen else x for x, cen in zip(c['e'], c['censored'])])
            rho = np.array(c['q_cal']) / np.array(c['q_test'])
            order = np.argsort(e, kind='stable')
            col, mk = colors[c['model']], markers[c['model']]
            ax.plot(e[order], rho[order], '-', color=col, lw=0.7, alpha=0.6)
            cen = np.array(c['censored'])
            ax.scatter(e[~cen], rho[~cen], s=16, color=col, marker=mk, zorder=3, linewidths=0.4,
                       edgecolors='white')
            if cen.any():
                ax.scatter(e[cen], rho[cen], s=16, facecolors='none', edgecolors=col, marker=mk, zorder=3,
                           linewidths=0.8)
        ax.set_xscale('log')
        ax.set_xlim(1.6, 150)
        ax.set_xticks([2, 5, 10, 20, 50, 100])
        ax.set_xticklabels(['2', '5', '10', '20', '50', '100'])
        ax.minorticks_off()
        ax.axhline(1.0, color='#777777', lw=0.6, ls=':')
        ax.text(0.03, 0.99, ds, transform=ax.transAxes, ha='left', va='top', fontsize=7.5, weight='bold')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Within-bearing / test 0.9-quantile\nof the absolute error')
    axes[0].set_ylim(0, 1.15)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.supxlabel('Epoch of the checkpoint selected by within-bearing early stopping (logarithmic axis)', fontsize=8,
                  y=0.075)
    handles = [Line2D([], [], color=colors[m], marker=markers[m], lw=0.7, ms=4, label=NAMES[m]) for m in models]
    fig.legend(handles=handles, loc='lower center', ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.04),
               handletextpad=0.3, columnspacing=0.9, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
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
