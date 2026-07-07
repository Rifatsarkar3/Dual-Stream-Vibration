"""
generate_figures_v2.py — Manuscript figures from protocol-v4 results.

Produces (600 dpi PNG into paper/figures_v2/):
  Fig2_trajectories.png     GT vs raw vs PAVA + MC-dropout 90% band (Bearing1_3, seed 42)
  Fig3_ablation.png         component ablation bars (PAVA MAE, 5 seeds)
  Fig4_postprocessing.png   raw / RTRM / CES / PAVA trajectories (one bearing)
  Fig5_noise_gate.png       MAE vs SNR (adaptive vs static) + gate alpha response
  Fig6_xjtu.png             XJTU-SY per-bearing PAVA MAE + zero-shot C3
  Fig7_budget_sweep.png     LOBO budget sweep mean val MAE (selection transparency)
  (Fig_baselines generated separately once exp19 lands)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'paper' / 'figures_v2'
OUT.mkdir(parents=True, exist_ok=True)
E16 = ROOT / 'outputs' / 'exp16_ress'

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.dpi': 600, 'savefig.bbox': 'tight'})

sys.path.insert(0, str(ROOT))
from src.isotonic_eval import IsotonicRegressor  # noqa: E402
from src.online_filters import (  # noqa: E402
    RealTimeRunningMinimum, ConstrainedExponentialSmoothing)


def fig2_trajectories():
    d = np.load(E16 / 'final20' / 'preds_seed42.npz')
    b = 'test_01_Bearing1_3'
    raw, tgt = d[f'raw_{b}'], d[f'tgt_{b}']
    mu, sd = d[f'mcmean_{b}'], d[f'mcstd_{b}']
    pava = IsotonicRegressor(increasing=False).fit_transform(raw)
    t = np.arange(len(raw)) / 360.0  # hours (one window every 10 s)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.fill_between(t, mu - 1.645 * sd, mu + 1.645 * sd, alpha=0.25,
                    color='tab:blue', label='Uncalibrated MC-dropout 90% interval')
    ax.plot(t, tgt, 'k-', lw=1.4, label='Ground truth')
    ax.plot(t, raw, color='tab:blue', lw=0.6, alpha=0.85, label='Raw prediction')
    ax.plot(t, pava, color='tab:red', lw=1.4, label='PAVA-corrected')
    ax.set_xlabel('Operating time (h)')
    ax.set_ylabel('Normalized RUL')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / 'Fig2_trajectories.png')
    plt.close(fig)


def fig3_ablation():
    variants = [
        ('Full model', 'final20'),
        ('w/o PGMC', 'abl20_no_pgmc'),
        ('w/o DANN', 'abl20_no_dann'),
        ('w/o adaptive gate', 'abl20_static'),
        ('w/o PGMC & DANN', 'abl20_neither'),
        ('Temporal-only', 'abl20_temporal'),
        ('Spectral-only', 'abl20_spectral'),
        ('PGMC (batch-ordering)', 'abl20_pgmc_soft'),
    ]
    means, stds, labels = [], [], []
    for label, vdir in variants:
        with open(E16 / vdir / 'results.json', encoding='utf-8') as f:
            a = json.load(f)['aggregated']
        labels.append(label)
        means.append(a['pava_mae_mean'])
        stds.append(a['pava_mae_std'])

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    colors = ['tab:red'] + ['tab:blue'] * (len(labels) - 1)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.85, capsize=3, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('PAVA MAE (20 seeds, mean ± std)')
    for yi, m in zip(y, means):
        ax.text(m + 0.012, yi, f'{m:.4f}', va='center', fontsize=8)
    ax.set_xlim(0, max(means) * 1.25)
    fig.savefig(OUT / 'Fig3_ablation.png')
    plt.close(fig)


def fig4_postprocessing():
    d = np.load(E16 / 'final20' / 'preds_seed42.npz')
    b = 'test_02_Bearing1_4'
    raw, tgt = d[f'raw_{b}'], d[f'tgt_{b}']
    pava = IsotonicRegressor(increasing=False).fit_transform(raw)
    rtrm = RealTimeRunningMinimum().process(raw)
    ces = ConstrainedExponentialSmoothing(alpha=0.2, epsilon=0.05).process(raw)
    t = np.arange(len(raw)) / 360.0

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(t, tgt, 'k-', lw=1.4, label='Ground truth')
    ax.plot(t, raw, color='tab:blue', lw=0.5, alpha=0.6, label='Raw')
    ax.plot(t, rtrm, color='tab:green', lw=1.1, label='RTRM (causal)')
    ax.plot(t, ces, color='tab:orange', lw=1.1, label='CES (causal)')
    ax.plot(t, pava, color='tab:red', lw=1.4, label='PAVA (offline)')
    ax.set_xlabel('Operating time (h)')
    ax.set_ylabel('Normalized RUL')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(OUT / 'Fig4_postprocessing.png')
    plt.close(fig)


def fig5_noise_gate():
    with open(ROOT / 'outputs' / 'exp18_noise_gate' / 'results_n20.json',
              encoding='utf-8') as f:
        res = json.load(f)
    keys = ['clean', '20dB', '10dB', '5dB', '0dB', '-5dB']
    x = np.arange(len(keys))
    ad = [res['adaptive_gate'][k] for k in keys]
    st = [res['static_gate'][k] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax1.errorbar(x, [m['mae_mean'] for m in ad], yerr=[m['mae_std'] for m in ad],
                 marker='o', label='Adaptive gate', capsize=3)
    ax1.errorbar(x, [m['mae_mean'] for m in st], yerr=[m['mae_std'] for m in st],
                 marker='s', label='Static fusion (α=0.5)', capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['clean', '20', '10', '5', '0', '−5'])
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Raw MAE')
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title('(a) Accuracy under AWGN', fontsize=9)

    ax2.plot(x, [m['alpha_mean'] for m in ad], 'o-', color='tab:purple')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['clean', '20', '10', '5', '0', '−5'])
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Mean gate weight α (temporal share)')
    ax2.set_title('(b) Gate response', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'Fig5_noise_gate.png')
    plt.close(fig)


def fig6_xjtu():
    with open(ROOT / 'outputs' / 'exp17_xjtu' / 'final20' / 'results.json',
              encoding='utf-8') as f:
        res = json.load(f)
    # per-bearing PAVA MAE means across seeds
    names = list(res['per_seed'][0]['per_bearing'].keys())
    short = [n.split('_', 2)[-1] for n in names]
    means = [np.mean([s['per_bearing'][n]['pava_mae'] for s in res['per_seed']])
             for n in names]
    stds = [np.std([s['per_bearing'][n]['pava_mae'] for s in res['per_seed']])
            for n in names]
    zs_names = list(res['per_seed_zeroshot'][0]['per_bearing'].keys())
    zs_short = [n.split('_', 2)[-1] for n in zs_names]
    zs_means = [np.mean([s['per_bearing'][n]['pava_mae']
                         for s in res['per_seed_zeroshot']]) for n in zs_names]
    zs_stds = [np.std([s['per_bearing'][n]['pava_mae']
                       for s in res['per_seed_zeroshot']]) for n in zs_names]

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    x1 = np.arange(len(short))
    x2 = np.arange(len(short) + 0, len(short) + len(zs_short))
    ax.bar(x1, means, yerr=stds, capsize=3, color='tab:blue', alpha=0.85,
           label='Held-out (Cond. 1–2)')
    ax.bar(x2, zs_means, yerr=zs_stds, capsize=3, color='tab:orange', alpha=0.85,
           label='Zero-shot (Cond. 3)')
    ax.set_xticks(np.concatenate([x1, x2]))
    ax.set_xticklabels(short + zs_short, rotation=45, ha='right')
    ax.set_ylabel('PAVA MAE (20 seeds)')
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / 'Fig6_xjtu.png')
    plt.close(fig)


def fig7_budget_sweep():
    budgets = [5, 10, 20, 40]
    means, stds = [], []
    for B in budgets:
        finals = []
        for f_ in ('A', 'B'):
            with open(E16 / f'sweepseq_B{B}_fold{f_}' / 'results.json',
                      encoding='utf-8') as f:
                res = json.load(f)
            finals += [s['val_history'][-1] for s in res['per_seed']]
        means.append(np.mean(finals))
        stds.append(np.std(finals))

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(len(budgets))  # categorical positions: avoids log-scale minor
                                  # tick labels (e.g. "56x10^0") bleeding through
    ax.errorbar(x, means, yerr=stds, marker='o', capsize=3, color='tab:blue')
    ax.set_xlabel('Training budget B (epochs)')
    ax.set_ylabel('Validation MAE (final epoch)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.annotate('selected B*=5', xy=(x[0], means[0]), xytext=(x[0] + 0.6, means[0] - 0.01),
                arrowprops=dict(arrowstyle='->'), fontsize=8)
    fig.savefig(OUT / 'Fig7_budget_sweep.png')
    plt.close(fig)


if __name__ == '__main__':
    fig2_trajectories(); print('Fig2 ok')
    fig3_ablation(); print('Fig3 ok')
    fig4_postprocessing(); print('Fig4 ok')
    fig5_noise_gate(); print('Fig5 ok')
    fig6_xjtu(); print('Fig6 ok')
    fig7_budget_sweep(); print('Fig7 ok')
    print(f'figures saved to {OUT}')



