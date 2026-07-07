"""
exp29_maintenance_timing.py — Maintenance-threshold alarm-timing and decision-loss analysis
=============================================================================================

Extends the risk-decision analysis in exp23 from a *classification* framing
(false-alarm / missed-detection rate at one instant) to an *alarm-timing*
framing: for each bearing, when does each prediction variant first cross a
normalized-RUL maintenance threshold tau, relative to when the true RUL
actually crosses it?

NO retraining and NO new experiments. Pure deterministic post-processing over
arrays already on disk:
  * outputs/exp16_ress/final20/preds_seed*.npz  (per test bearing, time-ordered:
        tgt_*    = true normalized RUL (1 -> 0 over the run-to-failure sequence)
        raw_*    = raw deterministic point prediction (same array PAVA/RTRM/CES
                   are already applied to for the paper's headline MAE numbers,
                   see exp16_ress.py:571-577)
        mcmean_* = MC-dropout predictive mean   (mu)
        mcstd_*  = MC-dropout predictive std    (s) )
  * outputs/exp22_uq_calibration/results.json   (calibration scale c*)

Five prediction variants (matching Sections 3.9/5.4's offline-vs-causal split):
  raw          : raw_p, no correction
  pava         : offline isotonic regression over the full trajectory (retrospective;
                 NOT valid for online deployment)
  rtrm         : Real-Time Running Minimum (causal, online)
  ces          : Constrained Exponential Smoothing, alpha=0.2 eps=0.05 (causal, online)
  lower_bound  : calibrated 90% one-sided lower bound, mu - z*c**s (causal, online;
                 uses the same c* as exp22/exp23)

Alarm-timing definitions (normalized-RUL threshold tau):
  t_true(tau) = first index where y        <= tau
  t_hat(tau)  = first index where variant  <= tau  (last index, flagged
                censored, if the variant never crosses tau)
  delta_t     = t_hat - t_true   (windows; >0 late, <0 early)

  severe_late = censored, OR true RUL at the moment of the (late) predicted
                alarm has already fallen to <= tau/2 -- i.e. by the time the
                model would have told you, the bearing is already halfway to
                the *next* threshold down. This ties severity to the physical
                trajectory rather than an arbitrary window count. Stated here
                as an illustrative choice, not a validated industrial rule.

  decision loss(c_late, c_early) = c_late*max(0, delta_t) + c_early*max(0, -delta_t)
                headline c_late:c_early = 5:1 (late costlier), with 2:1 and
                10:1 reported as sensitivity settings per the RESS conversion
                plan Sec. 9.5. This is an illustrative asymmetric cost, not a
                universal industrial cost model.

Aggregation matches exp23: per-seed rate = macro-average (equal weight) over
the 4 test bearings, then mean +/- std over the 20 seeds.

PRONOSTIA windows are 10 s apart (Sec. 4.1: one 0.1 s snapshot every 10 s), so
delta_t in windows is also reported converted to seconds for interpretability.

RUN:
  .venv\\Scripts\\python.exe Experiments/exp29_maintenance_timing.py
"""

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.isotonic_eval import IsotonicRegressor
from src.online_filters import RealTimeRunningMinimum, ConstrainedExponentialSmoothing

ROOT = Path(__file__).parent.parent
FINAL = ROOT / 'outputs' / 'exp16_ress' / 'final20'
CALIB = ROOT / 'outputs' / 'exp22_uq_calibration' / 'results.json'
OUT_DIR = ROOT / 'outputs' / 'exp29_maintenance_timing'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG = ROOT / 'paper' / 'figures_v2' / 'Fig9_maintenance_timing.png'

Z90 = 1.6448536269514722          # one-sided 90% normal quantile
TAU_LIST = [0.30, 0.20, 0.10]
COST_RATIOS = [(2, 1), (5, 1), (10, 1)]   # (c_late, c_early); headline = (5, 1)
SECONDS_PER_WINDOW = 10.0          # PRONOSTIA snapshot cadence (Sec. 4.1)
CES_ALPHA, CES_EPSILON = 0.2, 0.05  # matches exp16_ress.py config


def load_seed(npz_path):
    """Return {bearing_name: dict(y, raw, mu, s)} for all test bearings in one seed file."""
    d = np.load(npz_path)
    bearings = sorted(k.split('_', 1)[1] for k in d.keys() if k.startswith('tgt_'))
    out = {}
    for b in bearings:
        out[b] = dict(
            y=d[f'tgt_{b}'].astype(np.float64),
            raw=d[f'raw_{b}'].astype(np.float64),
            mu=d[f'mcmean_{b}'].astype(np.float64),
            s=d[f'mcstd_{b}'].astype(np.float64),
        )
    return out


def build_variants(bearing, c_star):
    """Compute the five prediction-variant trajectories for one bearing."""
    raw = bearing['raw']
    pava = IsotonicRegressor(increasing=False).fit_transform(raw.copy())
    rtrm = RealTimeRunningMinimum().process(raw.copy())
    ces = ConstrainedExponentialSmoothing(alpha=CES_ALPHA, epsilon=CES_EPSILON).process(raw.copy())
    lower_bound = bearing['mu'] - Z90 * c_star * bearing['s']
    return {'raw': raw, 'pava': pava, 'rtrm': rtrm, 'ces': ces, 'lower_bound': lower_bound}


def first_crossing(sequence, tau):
    """Index of first element <= tau; (len-1, True) if it never crosses (censored)."""
    idx = np.argmax(sequence <= tau)
    if sequence[idx] > tau:  # argmax returned 0 because nothing satisfied the condition
        return len(sequence) - 1, True
    return int(idx), False


def bearing_deltas(y, variant_seq, tau):
    """(delta_t, severe_late) for one bearing/variant/tau."""
    t_true, _ = first_crossing(y, tau)
    t_hat, censored = first_crossing(variant_seq, tau)
    delta_t = t_hat - t_true
    severe_late = censored or (delta_t > 0 and y[t_hat] <= tau / 2.0)
    return delta_t, severe_late


def metrics_one_bearing(bearing, c_star, tau, variants):
    y = bearing['y']
    out = {}
    for name in variants:
        seq = variants[name]
        delta_t, severe = bearing_deltas(y, seq, tau)
        out[name] = {
            'delta_t': float(delta_t),
            'abs_delta_t': float(abs(delta_t)),
            'late': float(delta_t > 0),
            'severe_late': float(severe),
            'early': float(delta_t < 0),
            'loss': {f'{cl}:{ce}': float(cl * max(0, delta_t) + ce * max(0, -delta_t))
                     for cl, ce in COST_RATIOS},
        }
    return out


def macro_average(per_bearing_metrics, variant_names):
    """Equal-weight average across the 4 test bearings for one seed (matches exp23)."""
    out = {}
    for name in variant_names:
        rows = [pb[name] for pb in per_bearing_metrics]
        out[name] = {
            'delta_t': float(np.mean([r['delta_t'] for r in rows])),
            'abs_delta_t': float(np.mean([r['abs_delta_t'] for r in rows])),
            'late_rate': float(np.mean([r['late'] for r in rows])),
            'severe_late_rate': float(np.mean([r['severe_late'] for r in rows])),
            'early_rate': float(np.mean([r['early'] for r in rows])),
            'loss': {k: float(np.mean([r['loss'][k] for r in rows])) for k in rows[0]['loss']},
        }
    return out


def main():
    with open(CALIB, encoding='utf-8') as f:
        c_star = float(json.load(f)['c_star'])
    files = sorted(FINAL.glob('preds_seed*.npz'))
    assert files, f'no prediction files in {FINAL}'

    seeds_data = [load_seed(p) for p in files]
    variant_names = ['raw', 'pava', 'rtrm', 'ces', 'lower_bound']

    table = {}
    for tau in TAU_LIST:
        per_seed_macro = []
        for per_bearing in seeds_data:
            variants_per_bearing = [build_variants(b, c_star) for b in per_bearing.values()]
            per_bearing_metrics = [
                metrics_one_bearing(bearing, c_star, tau, variants)
                for bearing, variants in zip(per_bearing.values(), variants_per_bearing)
            ]
            per_seed_macro.append(macro_average(per_bearing_metrics, variant_names))

        entry = {}
        for name in variant_names:
            def collect(key):
                return [s[name][key] for s in per_seed_macro]
            entry[name] = {
                'mean_delta_t_windows': float(np.mean(collect('delta_t'))),
                'mean_delta_t_seconds': float(np.mean(collect('delta_t')) * SECONDS_PER_WINDOW),
                'mae_delta_t_windows': float(np.mean(collect('abs_delta_t'))),
                'mae_delta_t_windows_std': float(np.std(collect('abs_delta_t'))),
                'late_rate_mean': float(np.mean(collect('late_rate'))),
                'late_rate_std': float(np.std(collect('late_rate'))),
                'severe_late_rate_mean': float(np.mean(collect('severe_late_rate'))),
                'early_rate_mean': float(np.mean(collect('early_rate'))),
                'early_rate_std': float(np.std(collect('early_rate'))),
                'decision_loss': {
                    ratio: {
                        'mean': float(np.mean([s[name]['loss'][ratio] for s in per_seed_macro])),
                        'std': float(np.std([s[name]['loss'][ratio] for s in per_seed_macro])),
                    } for ratio in [f'{cl}:{ce}' for cl, ce in COST_RATIOS]
                },
            }
        table[f'{tau:.2f}'] = entry

    results = {
        'experiment': 'exp29_maintenance_timing',
        'n_seeds': len(seeds_data),
        'c_star': c_star,
        'z90': Z90,
        'seconds_per_window': SECONDS_PER_WINDOW,
        'tau_list': TAU_LIST,
        'cost_ratios': [f'{cl}:{ce}' for cl, ce in COST_RATIOS],
        'headline_cost_ratio': '5:1',
        'variants': variant_names,
        'table': table,
    }
    with open(OUT_DIR / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(json.dumps(table['0.20'], indent=2))
    print(f"\nWrote {OUT_DIR / 'results.json'}")

    # ---- figure: representative bearing/seed trajectory panel ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seed_idx = 0
    bearing_name = sorted(seeds_data[0].keys())[0]
    bearing = seeds_data[seed_idx][bearing_name]
    variants = build_variants(bearing, c_star)
    tau_plot = 0.20
    y = bearing['y']
    t = np.arange(len(y))

    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3,
                          'figure.dpi': 600, 'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(t, y, color='black', lw=1.6, label='True RUL')
    style = {'raw': ('tab:gray', '--'), 'pava': ('tab:green', '-'),
             'rtrm': ('tab:orange', '-'), 'ces': ('tab:purple', '-'),
             'lower_bound': ('tab:blue', ':')}
    labels = {'raw': 'Raw', 'pava': 'PAVA (offline)', 'rtrm': 'RTRM (causal)',
              'ces': 'CES (causal)', 'lower_bound': f'Calibrated lower bound ($c^*$={c_star:.2f})'}
    for name, seq in variants.items():
        col, ls = style[name]
        ax.plot(t, seq, color=col, ls=ls, lw=1.1, label=labels[name])
    ax.axhline(tau_plot, color='red', lw=0.9, ls='--', alpha=0.7, label=f'$\\tau$={tau_plot:.2f}')

    t_true, _ = first_crossing(y, tau_plot)
    ax.scatter([t_true], [y[t_true]], marker='*', s=90, color='black', zorder=5,
               label='True crossing')
    for name, seq in variants.items():
        t_hat, censored = first_crossing(seq, tau_plot)
        if not censored:
            col, _ = style[name]
            ax.scatter([t_hat], [seq[t_hat]], marker='o', s=28, color=col,
                       edgecolor='white', linewidth=0.6, zorder=6)

    ax.set_xlabel('Window index (time)')
    ax.set_ylabel('Normalized RUL')
    ax.set_title(f'Maintenance-threshold alarm behavior\n({bearing_name}, seed 1)')
    ax.legend(fontsize=6.5, loc='upper right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG)
    print(f"Wrote {FIG}")


if __name__ == '__main__':
    main()
