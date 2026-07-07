"""
exp23_risk_decision.py — Risk-informed maintenance decision analysis
====================================================================

Quantifies how *calibrated* vs *uncalibrated* predictive uncertainty changes the
realized error rates of an interval-triggered maintenance rule. This is the
reliability/risk artifact for the RESS framing: it translates the calibrated UQ
(Section 3.10 / 5.7) into the language of false alarms vs. missed detections.

NO retraining and NO new experiments. Pure deterministic post-processing over
arrays already on disk:
  * outputs/exp16_ress/final20/preds_seed*.npz  (per test bearing:
        tgt_*  = true normalized RUL
        mcmean_* = MC-dropout predictive mean   (mu)
        mcstd_*  = MC-dropout predictive std    (s) )
  * outputs/exp22_uq_calibration/results.json   (calibration scale c*)

Decision model
--------------
A deployed prognostic acts on a rule, not the point RUL. We use the canonical
interval-triggered rule: schedule maintenance when the *lower* bound of the
predictive interval falls at or below a lead-time threshold tau (normalized RUL).
Triggering on the lower bound biases toward early intervention — the conservative
choice when a missed detection is the costlier error (the typical case for
safety-critical rotating assets).

Three policies (z = 1.645 for a one-sided 90% bound):
  point        : trigger if  mu <= tau                       (no uncertainty)
  uncalibrated : trigger if  mu - z*s        <= tau          (narrow interval)
  calibrated   : trigger if  mu - z*(c* * s) <= tau          (wide, calibrated)

Ground truth: a window "should alarm" iff true RUL y <= tau.
  false-alarm rate     = P(trigger    | y >  tau)   premature maintenance
  missed-detection rate= P(no trigger | y <= tau)   safety-critical error

Rates are computed per seed (windows pooled over the 4 test bearings) then
reported as mean +/- std over the 20 seeds.

RUN:
  .venv\\Scripts\\python.exe Experiments/exp23_risk_decision.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
FINAL = ROOT / 'outputs' / 'exp16_ress' / 'final20'
CALIB = ROOT / 'outputs' / 'exp22_uq_calibration' / 'results.json'
OUT_DIR = ROOT / 'outputs' / 'exp23_risk_decision'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG = ROOT / 'paper' / 'figures_v2' / 'Fig8_risk_curve.png'

Z90 = 1.6448536269514722  # one-sided 90% normal quantile
TAU_TABLE = 0.20          # headline maintenance threshold (normalized RUL)
TAU_SWEEP = [0.10, 0.15, 0.20, 0.30]


def load_seed(npz_path):
    """Return {bearing_name: (y, mu, s)} for all test bearings in one seed file."""
    d = np.load(npz_path)
    bearings = sorted(k.split('_', 1)[1] for k in d.keys() if k.startswith('tgt_'))
    return {b: (d[f'tgt_{b}'].astype(np.float64), d[f'mcmean_{b}'].astype(np.float64),
                d[f'mcstd_{b}'].astype(np.float64)) for b in bearings}


def rates_one_bearing(y, mu, s, c_star, tau):
    """False-alarm and missed-detection rates for the three policies at tau,
    on a single bearing's pooled windows."""
    should = y <= tau                      # ground-truth alarm
    not_should = ~should
    n_pos = max(int(should.sum()), 1)
    n_neg = max(int(not_should.sum()), 1)

    triggers = {
        'point': mu <= tau,
        'uncalibrated': (mu - Z90 * s) <= tau,
        'calibrated': (mu - Z90 * c_star * s) <= tau,
    }
    out = {}
    for name, trig in triggers.items():
        fa = float((trig & not_should).sum()) / n_neg          # P(trig | y>tau)
        md = float((~trig & should).sum()) / n_pos             # P(no trig | y<=tau)
        out[name] = {'false_alarm': fa, 'missed_detection': md}
    return out


def rates_macro(per_bearing, c_star, tau, policies):
    """Macro-average false-alarm/missed-detection rates across bearings for one
    seed: compute each policy's rate per bearing, then average across bearings
    with equal bearing weight (matches Table 8/9's per-seed macro-average, not
    a window-count-weighted pool)."""
    per_bearing_rates = {p: [] for p in policies}
    for (y, mu, s) in per_bearing.values():
        r = rates_one_bearing(y, mu, s, c_star, tau)
        for p in policies:
            per_bearing_rates[p].append(r[p])
    return {p: {'false_alarm': float(np.mean([r['false_alarm'] for r in per_bearing_rates[p]])),
                'missed_detection': float(np.mean([r['missed_detection'] for r in per_bearing_rates[p]]))}
            for p in policies}


def coverage(y, mu, s, scale):
    """Empirical two-sided 90% interval coverage with std scaled by `scale`."""
    half = Z90 * scale * s
    inside = (y >= mu - half) & (y <= mu + half)
    return float(inside.mean())


def main():
    with open(CALIB, encoding='utf-8') as f:
        c_star = float(json.load(f)['c_star'])
    files = sorted(FINAL.glob('preds_seed*.npz'))
    assert files, f'no prediction files in {FINAL}'

    seeds = [load_seed(p) for p in files]  # list of {bearing: (y, mu, s)}
    policies = ['point', 'uncalibrated', 'calibrated']

    # ---- headline table at tau = TAU_TABLE (mean +/- std over seeds) ----
    # Per-seed rate = macro-average (equal weight) across the 4 test bearings,
    # matching Table 8/9's stated methodology, not a window-count-weighted pool.
    table = {}
    for tau in TAU_SWEEP:
        per_policy = {p: {'false_alarm': [], 'missed_detection': []} for p in policies}
        for per_bearing in seeds:
            r = rates_macro(per_bearing, c_star, tau, policies)
            for p in policies:
                per_policy[p]['false_alarm'].append(r[p]['false_alarm'])
                per_policy[p]['missed_detection'].append(r[p]['missed_detection'])
        table[f'{tau:.2f}'] = {
            p: {
                'false_alarm_mean': float(np.mean(per_policy[p]['false_alarm'])),
                'false_alarm_std': float(np.std(per_policy[p]['false_alarm'])),
                'missed_detection_mean': float(np.mean(per_policy[p]['missed_detection'])),
                'missed_detection_std': float(np.std(per_policy[p]['missed_detection'])),
            } for p in policies
        }

    # ---- empirical coverage (sanity vs exp22/Section 5.7: ~60% uncal, ~89% cal) ----
    # This is a pooled-window quantity (interval coverage), distinct from the
    # macro-averaged alarm-rule table above; pooling matches Section 5.7's own
    # PICP definition over all test windows.
    def pooled(per_bearing):
        y = np.concatenate([v[0] for v in per_bearing.values()])
        mu = np.concatenate([v[1] for v in per_bearing.values()])
        s = np.concatenate([v[2] for v in per_bearing.values()])
        return y, mu, s

    cov_uncal = float(np.mean([coverage(*pooled(pb), 1.0) for pb in seeds]))
    cov_cal = float(np.mean([coverage(*pooled(pb), c_star) for pb in seeds]))

    # ---- operating-characteristic curve: sweep tau, macro-average over bearings
    # then mean over seeds (same methodology as the headline table, so Fig. 8
    # is numerically consistent with Table 8's four marked thresholds) ----
    tau_grid = np.linspace(0.02, 0.50, 49)
    curve = {p: {'false_alarm': [], 'missed_detection': []} for p in policies}
    for tau in tau_grid:
        agg = {p: {'fa': [], 'md': []} for p in policies}
        for per_bearing in seeds:
            r = rates_macro(per_bearing, c_star, tau, policies)
            for p in policies:
                agg[p]['fa'].append(r[p]['false_alarm'])
                agg[p]['md'].append(r[p]['missed_detection'])
        for p in policies:
            curve[p]['false_alarm'].append(float(np.mean(agg[p]['fa'])))
            curve[p]['missed_detection'].append(float(np.mean(agg[p]['md'])))

    results = {
        'experiment': 'exp23_risk_decision',
        'n_seeds': len(seeds),
        'c_star': c_star,
        'z90': Z90,
        'tau_table': TAU_TABLE,
        'empirical_coverage': {'uncalibrated': cov_uncal, 'calibrated': cov_cal},
        'table': table,
        'curve': {'tau': tau_grid.tolist(), **curve},
    }
    with open(OUT_DIR / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # ---- figure ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3,
                         'figure.dpi': 600, 'savefig.bbox': 'tight'})
    style = {'point': ('tab:gray', 's', 'Point estimate ($\\mu$)'),
             'uncalibrated': ('tab:orange', '^', 'Uncalibrated 90% lower bound'),
             'calibrated': ('tab:blue', 'o', f'Calibrated 90% lower bound ($c^*$={c_star:.2f})')}
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for p in policies:
        col, mk, lab = style[p]
        ax.plot(np.array(curve[p]['false_alarm']) * 100,
                np.array(curve[p]['missed_detection']) * 100,
                marker=mk, ms=3, lw=1.2, color=col, label=lab)
    # Annotate the four headline thresholds (Table 8/9) on the calibrated curve
    # so the sweep can be visually tied to the specific tau values discussed in text.
    # Three of the four points (tau=0.10/0.15/0.20) sit almost on top of each
    # other near the curve's knee, so per-point offset labels collide. Instead,
    # stack all four labels in the empty region below the legend and route a
    # thin leader line from each label back to its own marker.
    label_anchors = {0.10: (24, 62), 0.15: (24, 53), 0.20: (24, 44), 0.30: (24, 35)}
    for tau in TAU_SWEEP:
        t = table[f'{tau:.2f}']['calibrated']
        fa_pt = t['false_alarm_mean'] * 100
        md_pt = t['missed_detection_mean'] * 100
        ax.scatter([fa_pt], [md_pt], marker='D', s=36, facecolor='white',
                   edgecolor='tab:blue', linewidth=1.3, zorder=5)
        ax_text, ay_text = label_anchors[tau]
        ax.annotate(f'$\\tau$={tau:.2f}', xy=(fa_pt, md_pt), xycoords='data',
                    xytext=(ax_text, ay_text), textcoords='data',
                    fontsize=7.5, color='tab:blue', ha='left', va='center',
                    arrowprops=dict(arrowstyle='-', color='tab:blue', lw=0.6,
                                     shrinkA=2, shrinkB=4))
    ax.set_xlabel('False-alarm rate (%)  — premature maintenance')
    ax.set_ylabel('Missed-detection rate (%)  — safety-critical')
    ax.set_title('Maintenance-rule operating characteristic\n(threshold $\\tau$ swept; 20 seeds)',
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    fig.savefig(FIG)
    plt.close(fig)

    # ---- console report ----
    print(f"[exp23] c* = {c_star:.4f}   coverage uncal={cov_uncal*100:.1f}%  "
          f"cal={cov_cal*100:.1f}%")
    t = table[f'{TAU_TABLE:.2f}']
    print(f"\n=== Alarm-rule error rates at tau={TAU_TABLE} (20 seeds, pooled bearings) ===")
    print(f"{'policy':<14}{'false-alarm %':>18}{'missed-detection %':>22}")
    for p in policies:
        fa = t[p]['false_alarm_mean'] * 100
        fas = t[p]['false_alarm_std'] * 100
        md = t[p]['missed_detection_mean'] * 100
        mds = t[p]['missed_detection_std'] * 100
        print(f"{p:<14}{fa:>10.1f}+/-{fas:<5.1f}{md:>14.1f}+/-{mds:<5.1f}")
    print(f"\n[exp23] saved {OUT_DIR / 'results.json'} and {FIG}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
