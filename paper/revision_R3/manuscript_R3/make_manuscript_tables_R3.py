"""
make_manuscript_tables_R3.py: tables of the manuscript rebuilt or restored in revision 2 (internal version R3), after
Reviewer #1 asked why tables of the submission had been removed instead of revised.

Reads saved predictions and analysis outputs only; no model is trained or run. Writes one LaTeX file per table to
supp_tables/m_<name>.tex (evidence for tools/number_provenance.py). The manuscript holds the same tables as typed text,
like its other tables; --check reports any data row of these files that is missing from manuscript_R3.tex. Every value
that the manuscript of revision 1 already printed is asserted against it or against its analysis file.

USAGE:
  python paper/revision_R3/manuscript_R3/make_manuscript_tables_R3.py [--check]
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import exp29_maintenance_timing as e29  # noqa: E402
from make_supplement_tables_R3 import OUT, ces, f4, load, table  # noqa: E402
from src.isotonic_eval import IsotonicRegressor  # noqa: E402

TEX = HERE / 'manuscript_R3.tex'
FINAL = ROOT / 'outputs' / 'r1_final_B10' / 'final20'
OURS = 'DSAFLite+PGMC, submitted architecture'
VARIANT = 'DSAFLite+PGMC with temporal context (adopted)'
PARAMS = {  # (label in the table, parameter file, key); the variant adds no parameter (Supplementary Material S5)
    'Ours': 192130,
    'Temporal-context variant': 192130,
    '1D CNN': json.loads((ROOT / 'outputs/exp19_baselines20/cnn/results.json').read_text())['n_params'],
    'CNN-LSTM': json.loads((ROOT / 'outputs/exp19_baselines20/cnnlstm/results.json').read_text())['n_params'],
    'BiLSTM': json.loads((ROOT / 'outputs/exp19_baselines20/bilstm/results.json').read_text())['n_params'],
    '1D ResNet': json.loads((ROOT / 'outputs/exp19_baselines20/resnet/results.json').read_text())['n_params'],
    'MSCNN': json.loads((ROOT / 'outputs/r1_mscnn/final/results.json').read_text())['n_params'],
    'TSMC-CNN': json.loads((ROOT / 'outputs/r1_published/tsmccnn/results.json').read_text())['n_params'],
    'Transformer': json.loads((ROOT / 'outputs/exp19_baselines20/transformer/results.json').read_text())['n_params'],
}
ABL = [('Static fusion (no gate)', 'static fusion', 'abl20_static'),
       ('Raw-vibration stream only', 'temporal stream only', 'abl20_temporal'),
       ('Envelope-spectrum stream only', 'spectral stream only', 'abl20_spectral'),
       ('Without PGMC', 'without PGMC', 'abl20_no_pgmc'),
       ('Without DANN', 'without DANN', 'abl20_no_dann'),
       ('Without PGMC and DANN', 'without PGMC and DANN', 'abl20_neither'),
       ('PGMC on sorted batches', 'PGMC batch ordering', 'abl20_pgmc_soft')]


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'm_{name}.tex').write_text(text, encoding='utf-8')
    print(f'written supp_tables/m_{name}.tex')


def signed(x, d=1):
    return f'+{x:.{d}f}\\%' if x >= 0 else f'$-${-x:.{d}f}\\%'


def holm(p):
    s = f'{p:.3f}' if p < 0.1 else f'{p:.2f}'
    if p >= 0.995:
        s = '1.0'
    return f'\\textbf{{{s}}}' if p < 0.05 else s


def pm(pair, d=4):
    return f'{pair[0]:.{d}f} $\\pm$ {pair[1]:.{d}f}'


# ----------------------------------------------------------------------------------------- accuracy (Tables 1 and 2)
def accuracy():
    rows = load('tables_R1.json')['sets']['all10']['rows']
    s = load('strict_B10.json')['sets']['all10']['ces']['baselines']
    keys = {'Ours': OURS, 'Temporal-context variant': VARIANT, '1D CNN': '1D CNN', 'CNN-LSTM': 'CNN-LSTM',
            'BiLSTM': 'BiLSTM', '1D ResNet': '1D ResNet', 'MSCNN': 'MSCNN [7]', 'TSMC-CNN': 'TSMC-CNN [21]',
            'Transformer': 'Transformer'}
    budget = {'Ours': 10, 'Temporal-context variant': 5, '1D CNN': 40, 'CNN-LSTM': 5, 'BiLSTM': 40, '1D ResNet': 10,
              'MSCNN': 5, 'TSMC-CNN': 40, 'Transformer': 10}
    tex_name = {'MSCNN': 'MSCNN \\cite{Li2019}', 'TSMC-CNN': 'TSMC-CNN \\cite{Jiang2020}'}
    out = []
    for lab, key in keys.items():
        r = rows[key]
        p = r['pooled']
        if lab == 'Ours':
            ces_avg, delta = f"\\textbf{{{r['bearing_avg']['ces']:.4f}}}", 'reference'
        elif lab == 'Temporal-context variant':
            ces_avg, delta = f"{r['bearing_avg']['ces']:.4f}", '+1.2\\% (6/10; 0.61)'      # outside the Holm family
        else:
            t = s[key]
            assert abs(t['other'] - r['bearing_avg']['ces']) < 1e-9, lab
            ces_avg = f"{r['bearing_avg']['ces']:.4f}"
            delta = f"{signed(t['rel'])} ({t['k']}/{t['n']}; {holm(t['p_holm'])})"
        out.append([tex_name.get(lab, lab), f'{PARAMS[lab] / 1e6:.2f}', str(budget[lab]), ces_avg, delta,
                    f4(p['ces_mae'][0]), f"{pm(p['raw_mae'])} ({p['raw_r2'][0]:.3f})", f4(p['pava_mae'][0]), f"{p['pava_r2'][0]:.3f}"])
    head = ('Model & Parameters (M) & $B$ & \\multicolumn{2}{c}{Averaged over bearings} & '
            '\\multicolumn{4}{c}{Pooled over windows} \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\n'
            ' & & & CES MAE & $\\Delta$ (bearings; $p_{\\mathrm{Holm}}$) & CES MAE & Raw MAE ($R^2$) & PAVA MAE (offline) & '
            'PAVA $R^2$ (offline)')
    write('accuracy', table(
        'tab:1',
        'Accuracy on the ten PRONOSTIA test bearings (20 seeds per model). Primary endpoint: CES MAE averaged over '
        'bearings, which gives each bearing equal weight. $\\Delta$ = (baseline $-$ ours)/baseline, positive when our '
        'error is lower, with the number of bearings on which ours is lower and the Holm-adjusted sign-flip $p$ (bold: '
        '$< 0.05$). Pooled means give long-lived bearings more weight and must not be compared with bearing averages; '
        'raw MAE with its standard deviation over seeds and its $R^2$. PAVA is an offline reference. The temporal-context variant of '
        'Section~\\ref{sec:two-stage} is listed for completeness; its comparison with ours is not part of the Holm '
        'family. Budget selection is noisy for the baselines too: three of the six PRONOSTIA baseline budgets that '
        'could be compared changed when five instead of two seeds per fold were used.', 'lcccccccc', head, out))


# ------------------------------------------------------------------------------------------------ ablations (Table 3)
def ablations():
    strict = load('strict_B10.json')['sets']['all10']
    summ = load('results_B10.json')['sets']['all10']['summary']
    full = summ['B10:final20']
    ref = load('tables_R1.json')['sets']['all10']['rows'][OURS]['pooled']
    assert abs(full['raw_mae'][0] - ref['raw_mae'][0]) < 1e-6 and abs(full['pava_mae'][0] - ref['pava_mae'][0]) < 1e-6
    c, r = strict['ces']['ablations'], strict['raw']['ablations']
    any_row = next(iter(c.values()))
    rows = [['Full model', f4(any_row['ref']), 'reference', f4(next(iter(r.values()))['ref']), 'reference',
             f"{pm(full['raw_mae'])} ({full['raw_r2'][0]:.3f})", f4(full['pava_mae'][0]), f"{full['pava_r2'][0]:.3f}"]]
    for lab, key, var in ABL:
        tc, tr, m = c[key], r[key], summ[f'B10:{var}']
        rows.append([lab, f4(tc['other']), f"{signed(tc['rel'])} ({tc['k']}/{tc['n']}; {holm(tc['p_holm'])})",
                     f4(tr['other']), f"{signed(tr['rel'])} ({tr['k']}/{tr['n']}; {holm(tr['p_holm'])})",
                     f"{pm(m['raw_mae'])} ({m['raw_r2'][0]:.3f})", f4(m['pava_mae'][0]), f"{m['pava_r2'][0]:.3f}"])
    assert abs(r['spectral stream only']['p_holm'] - 0.0273) < 1e-3          # 0.027 in Section 5.3
    head = ('Variant & \\multicolumn{2}{c}{CES MAE, bearing avg.} & \\multicolumn{2}{c}{Raw MAE, bearing avg.} & '
            'Raw MAE ($R^2$), pooled & PAVA MAE, pooled (offline) & PAVA $R^2$ (offline) \\\\\n'
            '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & Value & $\\Delta$ (bearings; $p_{\\mathrm{Holm}}$) & Value & '
            '$\\Delta$ (bearings; $p_{\\mathrm{Holm}}$) & & &')
    write('ablations', table(
        'tab:3',
        'Component ablations (each variant removes or replaces one component and is retrained, calibrated and evaluated by the same rules; $B = 10$, ten PRONOSTIA test bearings, 20 seeds per variant). '
        '$\\Delta$ = (variant $-$ full)/variant, positive when the full model is better, with the bearings on which the '
        'full model is better and the sign-flip $p$ over bearings, Holm-adjusted over the seven variants (bold: '
        '$< 0.05$). Pooled raw MAE with its standard deviation over seeds and its $R^2$. Interval scores and costs: '
        'Fig.~\\ref{fig:components} and Supplementary Material S13.', 'lccccccc', head, rows))


# ---------------------------------------------------------------------------------------- corrections (Tables 1, 4)
def correction_metrics():
    acc = {k: {m: [] for m in ('mae', 'rmse', 'r2', 'mono')} for k in ('raw', 'rtrm', 'ces', 'pava')}
    files = sorted(glob.glob(str(FINAL / 'preds_seed*.npz')))
    assert len(files) == 20
    for f in files:
        z = np.load(f)
        names = sorted(k[4:] for k in z.files if k.startswith('raw_test_'))
        assert len(names) == 10
        err = {k: [] for k in acc}
        mono = {k: [] for k in acc}
        ys = []
        for n in names:
            y, raw = z['tgt_' + n].astype(float), z['raw_' + n].astype(float)
            seq = {'raw': raw, 'rtrm': np.minimum.accumulate(raw), 'ces': np.asarray(ces(raw), float),
                   'pava': np.asarray(IsotonicRegressor(increasing=False).fit_transform(raw), float)}
            ys.append(y)
            for k, x in seq.items():
                err[k].append(x - y)
                mono[k].append(100.0 * np.mean(np.diff(x) <= 0))
        y = np.concatenate(ys)
        sst = np.sum((y - y.mean()) ** 2)
        for k in acc:
            e = np.concatenate(err[k])
            acc[k]['mae'].append(np.mean(np.abs(e)))
            acc[k]['rmse'].append(np.sqrt(np.mean(e ** 2)))
            acc[k]['r2'].append(1.0 - np.sum(e ** 2) / sst)
            acc[k]['mono'].append(np.mean(mono[k]))
    # np.std with its default ddof = 0, as in tables_R1.json, so that the same quantity prints the same value everywhere
    return {k: {m: (float(np.mean(v)), float(np.std(v))) for m, v in d.items()} for k, d in acc.items()}


def corrections():
    m = correction_metrics()
    ref = load('tables_R1.json')['sets']['all10']['rows'][OURS]['pooled']
    assert abs(m['raw']['mae'][0] - ref['raw_mae'][0]) < 1e-5 and abs(m['ces']['mae'][0] - ref['ces_mae'][0]) < 1e-5
    assert abs(m['pava']['mae'][0] - ref['pava_mae'][0]) < 1e-5 and abs(m['raw']['r2'][0] - ref['raw_r2'][0]) < 1e-5
    assert abs(m['raw']['rmse'][0] - ref['raw_rmse'][0]) < 1e-5 and abs(m['pava']['r2'][0] - ref['pava_r2'][0]) < 1e-5
    assert abs(m['rtrm']['mae'][0] - 0.1747) < 1e-4 and abs(m['raw']['mono'][0] - 50.89) < 0.01
    t = load('decision_final20_all10.json')['timing']['0.20']['stats']
    labels = [('Raw (none)', 'raw', 'yes', 'raw'), ('RTRM', 'rtrm', 'yes', 'rtrm'),
              ('CES ($\\beta = 0.2$, $\\varepsilon = 0.05$)', 'ces', 'yes', 'ces'),
              ('PAVA (offline reference)', 'pava', 'no', 'pava')]
    rows = []
    for lab, k, causal, rule in labels:
        dt = t[rule]['dt']
        rows.append([lab, causal, pm(m[k]['mae']), pm(m[k]['rmse']), pm(m[k]['r2'], 3),
                     f"{m[k]['mono'][0]:.1f}", f"{100 * t[rule]['late']:.1f}\\%",
                     f'+{dt:.1f}' if dt >= 0 else f'$-${-dt:.1f}'])
    write('corrections', table(
        'tab:4',
        'Corrections of the same raw predictions (PRONOSTIA, ten test bearings, 20 seeds). MAE, RMSE and $R^2$ pooled '
        'over test windows, mean $\\pm$ standard deviation over seeds; monotonicity: share of non-increasing consecutive '
        'predictions, averaged over bearings; late alarms: share of bearing-seed trajectories whose alarm at '
        '$\\tau = 0.20$ comes after maintenance became due; $\\Delta t$: mean alarm timing error in 10~s windows. RTRM '
        'alarms exactly when the raw prediction does (Section~\\ref{sec:postprocessing}).',
        'lccccccc', 'Correction & Causal & MAE, pooled & RMSE & $R^2$ & Monotonicity (\\%) & Late alarms & $\\Delta t$',
        rows))
    return m


# ---------------------------------------------------------------------------------------------------- noise (Table 5)
def noise():
    nz = load('noise_B10_all10.json')
    rows = []
    for snr in nz['snr_levels_db']:
        k = 'clean' if snr == 'clean' else f'{snr}dB'
        a, s = nz['adaptive_gate'][k], nz['static_gate'][k]
        adv = 100 * (s['mae_mean'] - a['mae_mean']) / s['mae_mean']
        name = 'clean' if snr == 'clean' else (f'{snr} dB' if int(snr) >= 0 else f'$-${-int(snr)} dB')
        rows.append([name, pm((a['mae_mean'], a['mae_std'])), pm((s['mae_mean'], s['mae_std'])), signed(adv),
                     f"{a['r2_mean']:.3f}".replace('-', '$-$'), f"{s['r2_mean']:.3f}".replace('-', '$-$'),
                     f"{a['alpha_mean']:.3f}"])
    write('noise', table(
        'tab:5',
        'Robustness to additive white noise on the test windows (our network, ten PRONOSTIA test bearings, 20 seeds, '
        'identical checkpoints): pooled raw MAE (mean $\\pm$ standard deviation over seeds) and $R^2$ with the adaptive '
        'gate and with static fusion, the relative advantage of the gate (negative: static fusion has the lower '
        'error) and the mean gate weight.',
        'lcccccc', 'SNR & Adaptive gate MAE & Static fusion MAE & Gate advantage & Adaptive $R^2$ & Static $R^2$ & '
        'Mean gate weight', rows))


# ---------------------------------------------------------------------------------------------------- XJTU-SY (Table 6)
def xjtu():
    x = load('xjtu_temporal_retuned.json')['sets']
    keys = [('Ours', OURS), ('1D CNN', '1D CNN'), ('CNN-LSTM', 'CNN-LSTM'), ('BiLSTM', 'BiLSTM')]
    budget = {'Ours': 5, '1D CNN': 20, 'CNN-LSTM': 5, 'BiLSTM': 20}
    delta = {'Ours': 'reference', '1D CNN': '$-$4.7\\% (2/6)', 'CNN-LSTM': '$-$0.1\\% (3/6)', 'BiLSTM': '+5.3\\% (4/6)'}

    def r2(v):
        return f'{v:.2f}'.replace('-', '$-$')

    rows = []
    for lab, key in keys:
        t, z = x['xjtu_test']['rows'][key], x['xjtu_zs']['rows'][key]
        rows.append([lab, str(budget[lab]), f4(t['bearing_avg']['ces']), delta[lab],
                     f"{t['pooled']['raw_mae'][0]:.4f} ({r2(t['pooled']['raw_r2'][0])})",
                     f4(t['pooled']['pava_mae'][0]), f4(z['bearing_avg']['ces']),
                     f"{z['pooled']['raw_mae'][0]:.4f} ({r2(z['pooled']['raw_r2'][0])})",
                     f4(z['pooled']['pava_mae'][0])])
    head = (' & & \\multicolumn{4}{c}{Test bearings (6)} & \\multicolumn{3}{c}{Condition 3, zero-shot (5)} \\\\\n'
            '\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\n'
            'Model & $B$ & CES MAE & $\\Delta$ (bearings) & Raw MAE ($R^2$) & PAVA MAE & CES MAE & Raw MAE ($R^2$) & '
            'PAVA MAE')
    write('xjtu', table(
        'tab:6',
        'XJTU-SY (20 seeds per model; baselines selected by the same procedure as ours). CES MAE: averaged over '
        'bearings; raw and PAVA MAE: pooled over windows; PAVA is an offline reference. $\\Delta$ as in '
        'Table~\\ref{tab:1}, with the number of test bearings on which ours is lower; every sign-flip $p \\ge 0.62$.',
        'lcccccccc', head, rows))


# ------------------------------------------------------------------------------------------- error rates (Tables 8, 9)
def rates():
    d = load('decision_final20_all10.json')
    assert abs(d['c_star'] - 3.947) < 1e-3
    pb = d['rates_tau020_per_bearing']
    pols = ('point', 'uncalibrated', 'calibrated')

    def cells(src):
        out = []
        for p in pols:
            out += [f"{src[p]['missed_detection'][0]:.1f} $\\pm$ {src[p]['missed_detection'][1]:.1f}",
                    f"{src[p]['false_alarm'][0]:.1f} $\\pm$ {src[p]['false_alarm'][1]:.1f}"]
        return out

    rows = []
    for b in sorted(pb, key=lambda k: k.split('_', 2)[-1]):
        rows.append([b.split('_', 2)[-1].replace('_', '\\_')] + cells(pb[b]))
    rows.append('\\midrule')
    rows.append(['Macro-average'] + cells(d['rates_tau020']))
    assert abs(d['rates_tau020']['point']['missed_detection'][0] - 79.5) < 0.05
    assert abs(d['rates_tau020']['calibrated']['false_alarm'][0] - 25.2) < 0.05
    head = ('Test bearing & \\multicolumn{2}{c}{Point estimate} & \\multicolumn{2}{c}{Uncalibrated lower bound} & '
            '\\multicolumn{2}{c}{Calibrated lower bound} \\\\\n'
            '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n'
            ' & MD (\\%) & FA (\\%) & MD (\\%) & FA (\\%) & MD (\\%) & FA (\\%)')
    write('rates', table(
        'tab:8',
        'Realized error rates of interval-triggered maintenance rules at $\\tau = 0.20$ (PRONOSTIA, ten test bearings, '
        '20 seeds; mean $\\pm$ standard deviation over seeds, which reflects seeds and not bearings). Point estimate: '
        '$\\mu_t \\le \\tau$; uncalibrated lower bound: $\\mu_t - z s_t \\le \\tau$; calibrated lower bound: '
        '$\\mu_t - z c^* s_t \\le \\tau$ with $z = 1.645$ and $c^* = 3.95$. MD: missed-detection rate, the share of windows '
        'with $y_t \\le \\tau$ that did not trigger. FA: false-alarm rate, the share of windows with $y_t > \\tau$ that '
        'triggered.', 'lcccccc', head, rows))
    return pb


# ------------------------------------------------------------------------------------------------ alarm timing (Table 10)
def constant_margin_rules():
    """Timing of raw - q on the ten test bearings for the residual 0.9 bound and the cost-matched trigger at r = 5."""
    q = load('decision_models_mc30.json')['calibration']['submitted architecture']['q']
    margins = {'lb90res': q['0.9'], 'cmq5': q['0.8333333333333334']}
    acc = {k: {m: [] for m in ('dt', 'late', 'severe', 'early', 'A', 'B')} for k in margins}
    for f in sorted(glob.glob(str(FINAL / 'preds_seed*.npz'))):
        z = np.load(f)
        names = sorted(k[4:] for k in z.files if k.startswith('raw_test_'))
        per = {k: {m: [] for m in acc[k]} for k in margins}
        for n in names:
            y, raw = z['tgt_' + n].astype(float), z['raw_' + n].astype(float)
            for k, q_ in margins.items():
                dt, severe = e29.bearing_deltas(y, raw - q_, 0.20)
                per[k]['dt'].append(dt)
                per[k]['late'].append(float(dt > 0))
                per[k]['severe'].append(float(severe))
                per[k]['early'].append(float(dt < 0))
                per[k]['A'].append(max(0.0, dt))
                per[k]['B'].append(max(0.0, -dt))
        for k in margins:
            for m in acc[k]:
                acc[k][m].append(np.mean(per[k][m]))
    out = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in acc.items()}
    p1a = load('decision_models_mc30.json')['P1a']['submitted architecture']['5']
    assert abs(5 * out['cmq5']['A'] + out['cmq5']['B'] - p1a['cmq']) < 0.01
    assert abs(5 * out['lb90res']['A'] + out['lb90res']['B'] - p1a['best_other_L']) < 0.01 and \
        p1a['best_other'] == 'lb90res'
    return out


def timing():
    st = load('decision_final20_all10.json')['timing']['0.20']['stats']
    st = dict(st) | constant_margin_rules()
    labels = [('Raw prediction', 'raw', 'yes'), ('RTRM', 'rtrm', 'yes'), ('CES', 'ces', 'yes'),
              ('PAVA (offline)', 'pava', 'no'), ('MC-dropout lower bound, raw mean', 'lb_raw', 'yes'),
              ('MC-dropout lower bound, CES mean', 'lb_ces', 'yes'), ('Residual 0.9 bound', 'lb90res', 'yes'),
              ('Cost-matched trigger, $r = 5$', 'cmq5', 'yes')]
    rows = []
    for lab, k, causal in labels:
        s = st[k]
        dt = s['dt']
        rows.append([lab, causal, f'+{dt:.1f}' if dt >= 0 else f'$-${-dt:.1f}', f"{100 * s['late']:.1f}",
                     f"{100 * s['severe']:.1f}", f"{100 * s['early']:.1f}", f"{s['A']:.1f}", f"{s['B']:.1f}",
                     f"{5 * s['A'] + s['B']:.1f}"])
    write('timing', table(
        'tab:10',
        'Alarm timing and decision loss of our network at $\\tau = 0.20$ (PRONOSTIA, ten test bearings, 20 seeds). '
        '$\\overline{\\Delta t}$: mean alarm timing error in windows of 10~s. Late, severe late and early: shares of '
        'bearing-seed trajectories (Section~\\ref{sec:decision-metrics}). $A$, $B$: mean lateness and earliness in '
        'windows. $L(5) = 5A + B$. Residual 0.9 bound and cost-matched trigger: raw prediction minus the 0.9- and the '
        '5/6-quantile of the errors on held-out bearings. These values describe the ten bearings; they are not evidence of cost savings '
        '(Section~\\ref{sec:discussion-decision}).',
        'lcccccccc', 'Rule & Causal & $\\overline{\\Delta t}$ & Late (\\%) & Severe late (\\%) & Early (\\%) & $A$ & $B$ & '
        '$L(5)$', rows))
    return st


def training_minutes():
    r = json.loads((FINAL / 'results.json').read_text(encoding='utf-8'))
    total = sum(p['train_minutes'] for p in r['per_seed'])
    assert len(r['per_seed']) == 20 and all(p['epochs_trained'] == 10 for p in r['per_seed'])
    write('training_time', f'% Full 20-seed training at B = 10: {total:.0f} GPU-minutes (sum of train_minutes).\n')
    return total


def stage_coverage():
    """Coverage of the calibrated MC-dropout interval by health stage (Section 5.7, Table S of S13). Reproduces the published
    overall and late-life values first, so that the method is the one the manuscript uses."""
    c = load('decision_final20_all10.json')['c_star']
    stages = (('all', lambda y: np.ones_like(y, bool)), ('early', lambda y: y > 0.5),
              ('mid', lambda y: (y >= 0.2) & (y <= 0.5)), ('late', lambda y: y < 0.2))
    files = sorted(glob.glob(str(FINAL / 'preds_seed*.npz')))
    assert len(files) == 20
    pooled = {k: [] for k, _ in stages}
    per = {}
    for f in files:
        z = np.load(f)
        names = sorted((k[4:] for k in z.files if k.startswith('tgt_test_')), key=lambda k: k.split('_', 2)[-1])
        assert len(names) == 10
        hits = {k: [] for k, _ in stages}
        for n in names:
            y, mu, s = (z[p + n].astype(float) for p in ('tgt_', 'mcmean_', 'mcstd_'))
            inside = np.abs(y - mu) <= 1.645 * c * s
            for k, sel in stages:
                m = sel(y)
                hits[k].append(inside[m])
                per.setdefault(n, {}).setdefault(k, []).append(100.0 * inside[m].mean())
        for k, _ in stages:
            pooled[k].append(100.0 * np.concatenate(hits[k]).mean())
    pool = {k: float(np.mean(v)) for k, v in pooled.items()}
    assert abs(pool['all'] - 88.0) < 0.05 and abs(pool['late'] - 58.9) < 0.05, pool       # Section 5.7, revision 1
    rows = [[n.split('_', 2)[-1].replace('_', '\\_')] + [f"{np.mean(per[n][k]):.1f}" for k, _ in stages] for n in per]
    rows.append('\\midrule')
    rows.append(['Pooled over windows'] + [f'{pool[k]:.1f}' for k, _ in stages])
    (OUT / 's13_stage.tex').write_text(table(
        'tab:s13-stage',
        'Coverage of the calibrated 90\\% MC-dropout interval by health stage (manuscript Section 5.7; our network, ten PRONOSTIA '
        'test bearings, 20 seeds, $c^* = 3.95$; mean over seeds). Descriptive, from the saved predictions.',
        'lcccc', 'Test bearing & Overall (\\%) & Early life, $y_t > 0.5$ (\\%) & Mid life, $0.2 \\le y_t \\le 0.5$ (\\%) & '
        'Late life, $y_t < 0.2$ (\\%)', rows), encoding='utf-8')
    print('written supp_tables/s13_stage.tex')
    return pool


def check():
    """Every data row of the generated tables must stand in the manuscript as written."""
    tex = TEX.read_text(encoding='utf-8').replace('\r\n', '\n')
    missing = 0
    for f in sorted(OUT.glob('m_*.tex')):
        body = f.read_text(encoding='utf-8')
        m = re.search(r'\\midrule\n(.*?)\\bottomrule', body, flags=re.S)
        if not m:
            continue
        for row in m.group(1).splitlines():
            if row.strip() and row not in tex:
                print(f'{f.name}: not in the manuscript: {row[:110]}')
                missing += 1
    print('rows of the generated tables missing from the manuscript:', missing)
    return missing


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check:
        sys.exit(1 if check() else 0)
    accuracy()
    ablations()
    m = corrections()
    noise()
    xjtu()
    rates()
    timing()
    print('coverage by health stage:', {k: round(v, 1) for k, v in stage_coverage().items()})
    print('corrections:', {k: {a: round(b[0], 4) for a, b in v.items()} for k, v in m.items()})
    print('training minutes, 20 seeds at B = 10:', training_minutes())
    print('parameters:', PARAMS)


if __name__ == '__main__':
    main()
