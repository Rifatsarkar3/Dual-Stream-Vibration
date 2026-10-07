"""
make_supplement_tables.py: tables of the Supplementary Material (JRESS-D-26-05118, revision 1).

Reads saved test predictions and analysis outputs only; no model is trained or run. Writes one LaTeX file per
table to paper/revision_R1/manuscript_R1/supp_tables/ and prints consistency checks against the numbers in the
manuscript (bearing-averaged CES MAE, coverage).

USAGE:
  python paper/revision_R1/manuscript_R1/make_supplement_tables.py
"""

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.online_filters import ConstrainedExponentialSmoothing  # noqa: E402
from src.isotonic_eval import IsotonicRegressor  # noqa: E402

AN = ROOT / 'outputs' / 'r1_analysis'
OUT = Path(__file__).resolve().parent / 'supp_tables'
# Reference numbers of the manuscript, read from its .aux after compilation, so that the supplement cites the
# current numbering. The result files label two models with the submission's numbers ('TSMC-CNN [21]',
# 'MSCNN [7]'); these labels stay data keys and are renumbered only in the written tables.
_AUX_PATH = Path(__file__).resolve().parent / 'manuscript_R1.aux'
# numbers of the reference list of revision 1, used where the manuscript source is not available (public release)
_FALLBACK = {'Jiang2020': '15', 'Li2019': '14', 'SaxenaCMAPSS2008': '76', 'Li2018RESS': '77', 'Javanmardi2023': '17',
             'Roberts2017': '26', 'LiGahrooeiKim2026': '32', 'Barua2026': '48', 'BenAbdennour2026': '49',
             'XuZio2026': '63'}
REFNUM = (dict(re.findall(r'\\bibcite\{([^}]*)\}\{\{(\d+)\}', _AUX_PATH.read_text(encoding='utf-8')))
          if _AUX_PATH.exists() else dict(_FALLBACK))
RENUMBER = {'[21]': f"[{REFNUM['Jiang2020']}]", '[7]': f"[{REFNUM['Li2019']}]"}
REF_MACROS = {'refTSMC': 'Jiang2020', 'refMSCNN': 'Li2019', 'refCMAPSS': 'SaxenaCMAPSS2008',
              'refLiDCNN': 'Li2018RESS', 'refJavanmardi': 'Javanmardi2023', 'refRoberts': 'Roberts2017',
              'refLiFleet': 'LiGahrooeiKim2026', 'refBarua': 'Barua2026', 'refBenAbdennour': 'BenAbdennour2026',
              'refXuZio': 'XuZio2026'}

PRON_MODELS = [  # (label, path, key in tables_R1.json)
    ('Ours', 'outputs/r1_final_B10/final20', 'DSAFLite+PGMC, submitted architecture'),
    ('Variant', 'outputs/r1_final_temporal_B5/final20', 'DSAFLite+PGMC with temporal context (adopted)'),
    ('1D CNN', 'outputs/r1_reeval/baseline_cnn', '1D CNN'),
    ('CNN-LSTM', 'outputs/r1_baselines_final5/cnnlstm', 'CNN-LSTM'),
    ('BiLSTM', 'outputs/r1_baselines_final5/bilstm', 'BiLSTM'),
    ('1D ResNet', 'outputs/r1_reeval/baseline_resnet', '1D ResNet'),
    ('MSCNN', 'outputs/r1_mscnn/final', 'MSCNN [7]'),
    ('TSMC-CNN', 'outputs/r1_reeval/published_tsmccnn', 'TSMC-CNN [21]'),
    ('Transformer', 'outputs/r1_baselines_final5/transformer', 'Transformer'),
]
XJTU_MODELS = [
    ('Ours', 'outputs/exp17_xjtu/final20', 'DSAFLite+PGMC, submitted architecture'),
    ('Variant', 'outputs/r1_final_temporal_xjtu/final20', 'DSAFLite+PGMC with temporal context (adopted)'),
    ('1D CNN', 'outputs/r1_xjtu_baselines/final/cnn', '1D CNN'),
    ('BiLSTM', 'outputs/r1_xjtu_baselines/final/bilstm', 'BiLSTM'),
    ('CNN-LSTM', 'outputs/r1_xjtu_baselines/final/cnnlstm', 'CNN-LSTM'),
]
ARCH = ['submitted architecture', '1D CNN', 'BiLSTM', 'CNN-LSTM', '1D ResNet', 'Transformer', 'TSMC-CNN [21]',
        'MSCNN [7]']
SHORT = {'submitted architecture': 'Ours', 'adopted variant': 'Variant', 'TSMC-CNN [21]': 'TSMC-CNN',
         'MSCNN [7]': 'MSCNN'}
RULE = {'raw': 'raw', 'ces': 'CES', 'pava': 'PAVA (offline)', 'rtrm': 'RTRM', 'lb_raw': 'LB, raw mean',
        'lb_ces': 'LB, CES mean', 'lb90res': 'residual 0.9 bound', 'lbmc': 'MC lower bound'}


def load(name):
    return json.loads((AN / name).read_text(encoding='utf-8'))


def bname(key):
    return key.split('_', 2)[-1].replace('_', '\\_')


def ces(x):
    return ConstrainedExponentialSmoothing(alpha=0.2, epsilon=0.05).process(x)


def per_bearing_mae(path, patterns=('preds_seed*.npz', 'zeroshot_preds_seed*.npz')):
    """{bearing key: {'ces': mean over seeds, 'raw': ..., 'pava': ...}} from all prediction files."""
    acc = {}
    for pattern in patterns:
        for f in sorted(glob.glob(str(ROOT / path / pattern))):
            z = np.load(f)
            for k in z.files:
                if not k.startswith('raw_'):
                    continue
                name = k[len('raw_'):]
                y, p = z['tgt_' + name], z[k]
                d = acc.setdefault(name, {'ces': [], 'raw': [], 'pava': []})
                d['raw'].append(np.mean(np.abs(p - y)))
                d['ces'].append(np.mean(np.abs(ces(p) - y)))
                d['pava'].append(np.mean(np.abs(IsotonicRegressor(increasing=False).fit_transform(p) - y)))
    return {k: {m: float(np.mean(v)) for m, v in d.items()} | {'n_seeds': len(d['ces'])} for k, d in acc.items()}


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    for old, new in RENUMBER.items():
        text = text.replace(old, new)
    (OUT / f'{name}.tex').write_text(text, encoding='utf-8')
    print(f'written supp_tables/{name}.tex')


def refnums():
    """macros with the manuscript's current reference numbers, for the text of the supplement"""
    lines = [f'\\newcommand{{\\{macro}}}{{{REFNUM[key]}}}' for macro, key in REF_MACROS.items()]
    (OUT / 'refnums.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('written supp_tables/refnums.tex:', ', '.join(f'{k} [{REFNUM[v]}]' for k, v in REF_MACROS.items()))


def table(label, caption, colspec, header, rows, resize=True, size='\\small'):
    body = '\n'.join(' & '.join(r) + ' \\\\' if not isinstance(r, str) else r for r in rows)
    tab = (f'\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n'
           f'\\end{{tabular}}')
    if resize:
        tab = '\\resizebox{\\linewidth}{!}{%\n' + tab + '\n}'
    return (f'\\begin{{table}}[htbp]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n{tab}\n'
            f'\\end{{table}}\n')


def f4(x):
    return f'{x:.4f}'


def pct(x, d=1):
    return f'{x:.{d}f}'


# -------------------------------------------------------------------------------------------- S2: per bearing
def s2_pronostia():
    ref = load('tables_R1.json')['sets']
    res = {lab: per_bearing_mae(p, ('preds_seed*.npz',)) for lab, p, _ in PRON_MODELS}
    test = [k for k in res['Ours'] if k.startswith('test_')]
    zs = [k for k in res['Ours'] if k.startswith('zs_')]
    rows = []
    for k in test + zs:
        if k == zs[0]:
            rows.append('\\midrule')
        rows.append([bname(k)] + [f4(res[lab][k]['ces']) for lab, _, _ in PRON_MODELS])
    rows.append('\\midrule')
    for setname, keys in (('Mean, ten test bearings', test), ('Mean, Condition 3', zs)):
        rows.append([setname] + [f4(np.mean([res[lab][k]['ces'] for k in keys])) for lab, _, _ in PRON_MODELS])
    for lab, _, key in PRON_MODELS:
        mine = np.mean([res[lab][k]['ces'] for k in test])
        theirs = ref['all10']['rows'][key]['bearing_avg']['ces']
        print(f'  check PRONOSTIA {lab:12s} bearing-avg CES {mine:.5f} vs tables_R1 {theirs:.5f}')
        assert abs(mine - theirs) < 5e-4, (lab, mine, theirs)
    write('s2_pronostia', table(
        'tab:s2-pronostia',
        'PRONOSTIA: CES MAE per bearing (mean over the 20 seeds of each model). Variant: the temporal-context '
        'variant of our model (Section S5). The bearing averages reproduce Table 4 of the manuscript.',
        'l' + 'c' * len(PRON_MODELS), 'Bearing & ' + ' & '.join(lab for lab, _, _ in PRON_MODELS), rows))
    return res


def s2_xjtu():
    ref = load('xjtu_temporal_retuned.json')['sets']
    res = {lab: per_bearing_mae(p) for lab, p, _ in XJTU_MODELS}
    test = [k for k in res['1D CNN'] if k.startswith('test_')]
    zs = [k for k in res['1D CNN'] if k.startswith('zeroshot_')]
    rows = []
    for k in test + zs:
        if k == zs[0]:
            rows.append('\\midrule')
        rows.append([bname(k)] + [f4(res[lab][k]['ces']) for lab, _, _ in XJTU_MODELS])
    rows.append('\\midrule')
    for setname, keys, s in (('Mean, six test bearings', test, 'xjtu_test'), ('Mean, Condition 3', zs, 'xjtu_zs')):
        rows.append([setname] + [f4(np.mean([res[lab][k]['ces'] for k in keys])) for lab, _, _ in XJTU_MODELS])
        for lab, _, key in XJTU_MODELS:
            mine = np.mean([res[lab][k]['ces'] for k in keys])
            theirs = ref[s]['rows'][key]['bearing_avg']['ces']
            print(f'  check XJTU {s} {lab:9s} bearing-avg CES {mine:.5f} vs retuned {theirs:.5f}')
            assert abs(mine - theirs) < 5e-4, (lab, s, mine, theirs)
    write('s2_xjtu', table(
        'tab:s2-xjtu',
        'XJTU-SY: CES MAE per bearing (mean over 20 seeds; baselines selected by the same four-fold procedure as '
        'our model). The bearing averages reproduce Table 5 of the manuscript.',
        'l' + 'c' * len(XJTU_MODELS), 'Bearing & ' + ' & '.join(lab for lab, _, _ in XJTU_MODELS), rows,
        resize=False))


def s2_ours(res_pron):
    cal = load('calibration_v2.json')['test']
    xc = load('xjtu_calibration.json')['versions']['submitted']
    xres = per_bearing_mae('outputs/exp17_xjtu/final20', ('preds_seed*.npz',))
    rows = []
    for setname in ('all10', 'zs3'):
        unc = cal[setname]['uncalibrated_two_sided']['per_bearing']
        calb = cal[setname]['two_sided_c_star']['per_bearing']
        for k in unc:
            r = res_pron['Ours'][k]
            rows.append(['PRONOSTIA', bname(k), f4(r['raw']), f4(r['ces']), f4(r['pava']), pct(unc[k]), pct(calb[k])])
        rows.append(['', 'pooled', '', '', '', pct(cal[setname]['uncalibrated_two_sided']['pooled']),
                     pct(cal[setname]['two_sided_c_star']['pooled'])])
        rows.append('\\midrule')
    unc, calb = xc['uncalibrated_two_sided']['per_bearing'], xc['two_sided_c_star']['per_bearing']
    for k in unc:
        r = xres[k]
        rows.append(['XJTU-SY', bname(k), f4(r['raw']), f4(r['ces']), f4(r['pava']), pct(unc[k]), pct(calb[k])])
    rows.append(['', 'pooled', '', '', '', pct(xc['uncalibrated_two_sided']['pooled']),
                 pct(xc['two_sided_c_star']['pooled'])])
    print(f"  check coverage PRONOSTIA all10 {cal['all10']['two_sided_c_star']['pooled']:.2f} (manuscript 88.0); "
          f"XJTU {xc['two_sided_c_star']['pooled']:.2f} (manuscript 86.6)")
    write('s2_ours', table(
        'tab:s2-ours',
        f"Our model per bearing: MAE of the raw, CES and PAVA (offline) predictions and two-sided coverage of the "
        f"90\\% MC-dropout interval without and with the factor fitted on held-out bearings ($c^* = "
        f"{load('calibration_v2.json')['global']['c_star']:.2f}$ on PRONOSTIA, ${xc['c_star']:.2f}$ on XJTU-SY). "
        f"Mean over 20 seeds; pooled: over all windows of the set. The XJTU-SY factor is fitted on its own "
        f"validation runs; Condition-3 bearings of XJTU-SY have no stored MC-dropout arrays for the final runs.",
        'llccccc', 'Dataset & Bearing & Raw MAE & CES MAE & PAVA MAE & Coverage, raw (\\%) & Coverage, calibrated (\\%)',
        rows))


def s2_source():
    di, dx = load('decision_inbearing.json'), load('decision_xjtu_inbearing.json')
    pron_b = list(di['residual_intervals']['SP']['submitted architecture']['per_bearing'].keys())
    rows = [['PRONOSTIA', bname(k), pct(v)] for k, v in zip(pron_b, di['A6_P1_P2']['pooled']['per_bearing'])]
    rows.append(['', 'mean', pct(di['A6_P1_P2']['pooled']['mean'])])
    rows.append('\\midrule')
    rows += [['XJTU-SY', bname(k), pct(v)] for k, v in zip(dx['bearings'], dx['A7_P1_P2']['pooled']['per_bearing'])]
    rows.append(['', 'mean', pct(dx['A7_P1_P2']['pooled']['mean'])])
    write('s2_source', table(
        'tab:s2-source',
        'Per-bearing gain of held-out over in-bearing calibration (manuscript Section 5.6, Table 9): '
        '$|$coverage $-$ 90$|$ under in-bearing calibration minus the same under held-out calibration, in points, '
        'averaged over the eight (PRONOSTIA) and four (XJTU-SY) architectures; residual interval, five seeds.',
        'llc', 'Dataset & Bearing & Gain (points)', rows, resize=False))


# -------------------------------------------------------------------------------------------- S3: decisions
def segment_text(s):
    lo = 'below' if s['r_from'] <= 0.5 else f"{s['r_from']:.2f} to"
    if s['r_from'] <= 0.5:
        return f"{RULE[s['variant']]} below {s['r_to']:.2f}"
    if s['r_to'] >= 100:
        return f"{RULE[s['variant']]} above {s['r_from']:.2f}"
    return f"{RULE[s['variant']]} {lo} {s['r_to']:.2f}"


def s3_timing():
    for tag, name, lab in (('ours', 'decision_final20_all10.json', 'our model'),
                           ('variant', 'decision_temporal_B5_all10.json', 'the temporal-context variant')):
        d = load(name)
        rows = []
        for tau in ('0.30', '0.20', '0.10'):
            st = d['timing'][tau]['stats']
            late_b = d['timing'][tau]['late_bearings_majority']
            for rule in ('raw', 'ces', 'pava', 'lb_raw', 'lb_ces'):
                s = st[rule]
                L = [r * s['A'] + s['B'] for r in (2, 5, 10)]
                rows.append([f'{tau}' if rule == 'raw' else '', RULE[rule], f"{s['dt']:+.1f}", pct(100 * s['late']),
                             pct(100 * s['severe']), f"{len(late_b.get(rule, []))}/10", f"{s['A']:.1f}",
                             f"{s['B']:.1f}"] + [f'{x:.1f}' for x in L])
            rows.append('\\midrule' if tau != '0.10' else '')
        rows = [r for r in rows if r != '']
        write(f's3_timing_{tag}', table(
            f'tab:s3-timing-{tag}',
            f"Alarm timing of {lab} on the ten PRONOSTIA test bearings (20 seeds; windows of 10 s). "
            f"$\\overline{{\\Delta t}}$: mean alarm-time error; late and severe late: share of bearing-seed "
            f"trajectories; late bearings: bearings on which most seeds alarm late; $A$, $B$: mean lateness and "
            f"earliness; $L(r) = rA + B$. RTRM alarms exactly when the raw prediction does. LB: lower bound of the "
            f"calibrated MC-dropout interval.",
            'llccccccccc', '$\\tau$ & Rule & $\\overline{\\Delta t}$ & Late (\\%) & Severe (\\%) & Late bearings & '
            '$A$ & $B$ & $L(2)$ & $L(5)$ & $L(10)$', rows))
        opt = []
        for tau in ('0.30', '0.20', '0.10'):
            opt.append([tau, '; '.join(segment_text(s) for s in d['timing'][tau]['optimal_by_ratio'])])
        write(f's3_optimal_{tag}', table(
            f'tab:s3-optimal-{tag}',
            f"Decision-optimal rule of {lab} by range of the cost ratio $r$ (lower envelope of $L(r) = rA + B$ over "
            f"the rules of Table~\\ref{{tab:s3-timing-{tag}}}, evaluated for $0.5 \\le r \\le 100$).",
            'lp{0.82\\linewidth}', '$\\tau$ & Optimal rule (range of $r$)', opt, resize=False))


def s3_cmq():
    dm = load('decision_models_mc30.json')
    models = ['submitted architecture', 'adopted variant', '1D CNN', 'BiLSTM', 'CNN-LSTM', '1D ResNet',
              'Transformer', 'TSMC-CNN [21]', 'MSCNN [7]']
    rows = []
    for m in models:
        p = dm['P1a'][m]
        rows.append([SHORT.get(m, m)] + [f"{p[r]['cmq']:.0f} ({p[r]['excess_pct']:+.0f}\\%, {RULE[p[r]['best_other']]})"
                                         for r in ('2', '5', '10', '20', '100')])
    write('s3_cmq', table(
        'tab:s3-cmq',
        'Expected loss of the cost-matched trigger of each model ($\\tau = 0.20$, ten PRONOSTIA test bearings) and, '
        'in parentheses, its excess over the cheapest other deployable rule (negative: the cost-matched trigger is '
        'cheaper) and that rule. Residual 0.9 bound: raw prediction minus the 0.9-quantile of the held-out errors.',
        'lccccc', 'Model & $r = 2$ & $r = 5$ & $r = 10$ & $r = 20$ & $r = 100$', rows))


def s3_source_costs():
    di, dx = load('decision_inbearing.json'), load('decision_xjtu_inbearing.json')
    rows = []
    for ds, d, archs in (('PRONOSTIA', di, ARCH), ('XJTU-SY', dx, ARCH[:4])):
        for i, m in enumerate(archs):
            cells = [ds if i == 0 else '', SHORT.get(m, m)]
            for r in ('5', '10'):
                L = d['secondary']['decisions'][r]['L_mean'][m]
                if ds == 'PRONOSTIA':
                    ls = d['secondary']['decisions'][r]['late_share'][m]
                else:
                    ls = d['A7_P3'][r]['late_share'][m]
                cells += [f"{L['SP']:.0f} / {L['PR']:.0f}", f"{100 * ls['SP']:.0f} / {100 * ls['PR']:.0f}"]
            rows.append(cells)
        rows.append('\\midrule')
    rows = rows[:-1]
    write('s3_source_costs', table(
        'tab:s3-source-costs',
        'Cost-matched trigger with each pipeline\'s own quantiles ($\\tau = 0.20$): expected loss $L$ in windows and '
        'share of late alarms (\\%), in-bearing calibration (SP) / held-out calibration (PR), five seeds. '
        'PRONOSTIA: ten test bearings, one window per 10 s; XJTU-SY: six test and five Condition-3 bearings, twelve '
        'windows per minute of operation. Descriptive; '
        'the bearing-level tests are in Section 5.7 of the manuscript.',
        'llcccc', 'Dataset & Architecture & $L(5)$ SP / PR & Late, $r = 5$ & $L(10)$ SP / PR & Late, $r = 10$', rows))


# -------------------------------------------------------------------------------------------- S5, S6, S7
def s5_candidates():
    cv = load('calibration_v2.json')
    b1, b3, b4 = cv['B1']['test']['all10'], cv['B3'], cv['B4']
    rows = [['Single model (reference)', f4(b1['single_model_ces_mae_bearing_avg']), '88.0', '58.9'],
            ['Seed ensemble, 5 members (mean of 4 groups)', f4(b1['5-member_mean_of_4_groups']['ces_mae_bearing_avg']),
             pct(b1['5-member_mean_of_4_groups']['two_sided_pooled']), pct(b1['5-member_mean_of_4_groups']['two_sided_late'])],
            ['Seed ensemble, 20 members', f4(b1['20-member']['ces_mae_bearing_avg']),
             pct(b1['20-member']['two_sided_pooled']), pct(b1['20-member']['two_sided_late'])]]
    write('s5_ensembles', table(
        'tab:s5-ensembles',
        f"Seed ensembles of our model (ten PRONOSTIA test bearings). Ensembles are calibrated with a factor fitted "
        f"on five-member validation ensembles ($c^* = {cv['B1']['c_star_ensemble']:.2f}$).",
        'lccc', 'Model & CES MAE, bearing avg. & Coverage (\\%) & Late-life coverage (\\%)', rows, resize=False))
    rows = []
    for fold, v in b3['leave_one_fold_out'].items():
        rows.append([fold, pct(v['global']['pooled']), pct(v['stage_aware']['pooled']),
                     pct(v['global']['late [0, 0.2)']), pct(v['stage_aware']['late [0, 0.2)'])])
    write('s5_stage', table(
        'tab:s5-stage',
        'Stage-aware calibration factors (%.2f, %.2f and %.2f for predicted RUL in [0, 0.3), [0.3, 0.6) and [0.6, 1]): '
        'leave-one-fold-out coverage on the held-out validation bearing of each fold, global factor against '
        'stage-aware factors, overall and in late life ($y < 0.2$). On the test bearings: %.1f\\%% overall.'
        % (tuple(b3['band_factors']) + (b3['test']['all10']['pooled'],)),
        'lcccc', 'Fold & Global (\\%) & Stage-aware (\\%) & Global, late (\\%) & Stage-aware, late (\\%)', rows,
        resize=False))
    rows = [[a, f4(v['mean']), f4(v['se']), 'selected' if float(a) == b4['selected_alpha'] else ''] for a, v in
            b4['scores'].items()]
    write('s5_ces', table(
        'tab:s5-ces',
        'Smoothing constant $\\alpha$ of CES: four-fold validation MAE (mean and standard error over the 20 runs). '
        'The one-standard-error rule keeps the default 0.2, within one standard error of the minimum.',
        'lccc', '$\\alpha$ & Validation CES MAE & SE & ', rows, resize=False))


def s6_noise():
    nz = load('noise_B10_all10.json')
    rows = []
    for snr in nz['snr_levels_db']:
        k = 'clean' if snr == 'clean' else f'{snr}dB'
        a, s = nz['adaptive_gate'][k], nz['static_gate'][k]
        rows.append([str(snr) if snr == 'clean' else f'{snr} dB', f4(a['mae_mean']), f4(s['mae_mean']),
                     f"{a['alpha_mean']:.3f}"])
    write('s6_noise', table(
        'tab:s6-noise',
        'Additive white noise on the test windows (our model, ten PRONOSTIA test bearings, 20 seeds, identical '
        'checkpoints): raw MAE with the adaptive gate and with static fusion, and the mean gate weight.',
        'lccc', 'SNR & Adaptive gate MAE & Static fusion MAE & Mean gate weight', rows, resize=False))


def s8_randsplit():
    f = AN / 'tsmc_randsplit.json'
    if not f.exists():
        print('  tsmc_randsplit.json not yet available; table s8 skipped')
        return
    r = load('tsmc_randsplit.json')
    bl_r2 = load('tables_R1.json')['sets']['all10']['rows']['TSMC-CNN [21]']['pooled']['raw_r2'][0]
    rows = [["R8-a: [21]'s bearings (Bearing2\\_1 to 2\\_4), random 70/30", f"{r['a']['n_windows']:,}",
             f"{r['a']['mae_mean']:.4f} ({r['a']['mae_sd']:.4f})", f"{r['a']['r2_mean']:.3f}"],
            ['R8-b: the 14 Condition 1--2 bearings, random 70/30', f"{r['b']['n_windows']:,}",
             f"{r['b']['mae_mean']:.4f} ({r['b']['mae_sd']:.4f})", f"{r['b']['r2_mean']:.3f}"],
            ['Bearing level: trained on 4 bearings, tested on 10', '5,382 / 16,921',
             f"{r['bearing_level_pooled_raw_mae']['mean']:.4f} ({r['bearing_level_pooled_raw_mae']['sd']:.4f})",
             f'{bl_r2:.3f}'],
            ['Published value of [21] (TSMC-CNN, $K = 4$, 70/30)', '', '0.0214', '']]
    write('s9_randsplit', table(
        'tab:s9-randsplit',
        'TSMC-CNN under random 70/30 splits of windows (20 splits; raw MAE on the held-out windows, mean and SD) and '
        'under bearing-level evaluation (20 seeds; pooled raw MAE on the ten test bearings). Analysis plan '
        'Amendment 8.', 'lccc', 'Evaluation & Windows & MAE (SD) & $R^2$', rows, resize=False))


def s4_data():
    mp = json.loads((ROOT / 'Processed_PRONOSTIA_v2' / 'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for key, v in mp.items():
        role = {'train': 'training', 'test': 'test', 'zs': 'zero-shot'}[key.split('_')[0]]
        rows.append([bname(key), role, v['raw'].replace('_', '\\_'), f"{v['acc_files']:,}",
                     str(v['temp_files_ignored']), f"{v['windows']:,}", f"{v['duration_h']:.2f}"])
    write('s4_pronostia', table(
        'tab:s4-pronostia',
        'PRONOSTIA file inventory: vibration (acc) records read, temperature (temp) records in the same folder '
        'and not read, windows (one per vibration record) and recorded duration.',
        'lllcccc', 'Bearing & Set & Raw folder & Vibration records & Temperature records (not read) & Windows & Hours',
        rows))
    mx = json.loads((ROOT / 'Processed_XJTU' / 'manifest.json').read_text(encoding='utf-8'))
    rows = [[k.replace('_', '\\_'), {'train': 'training', 'test': 'test', 'zeroshot': 'zero-shot'}[v['role']],
             f"{v['files']:,}", f"{v['windows']:,}"] for k, v in mx.items()]
    write('s4_xjtu', table(
        'tab:s4-xjtu',
        'XJTU-SY: one-minute recordings per bearing (= lifetime in minutes) and windows (twelve per recording).',
        'llcc', 'Bearing & Set & Recordings & Windows', rows, resize=False))


def s10_mechanism():
    from scipy.stats import kendalltau
    d = load('mechanism.json')
    res = d['results']
    assert not d['dry_run'] and res['A9_P1']['T'] == -43 and abs(res['A9_P1']['p_two_sided'] - 0.0025) < 1e-4
    names = {'submitted architecture': 'Ours'}

    def tau(x, y):
        t = kendalltau(x, y).statistic
        return '--' if not np.isfinite(t) else f'${t:+.2f}$'

    def kendall_s(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return int(sum(np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
                       for i in range(len(x)) for j in range(i + 1, len(x))))

    rows = []
    for c in d['combinations']:
        rho = [a / b for a, b in zip(c['q_cal'], c['q_test'])]
        epochs = ['$>$90' if cen else str(x) for x, cen in zip(c['e'], c['censored'])]
        rows.append([c['dataset'], names.get(c['model'], c['model']), ', '.join(epochs),
                     ', '.join(f'{v:.2f}' for v in rho), f"${kendall_s(c['e'], rho):+d}$", tau(c['e'], rho),
                     tau(c['e'], c['q_cal']), tau(c['e'], c['q_test'])])
    write('s10_runs', table(
        'tab:s10-runs',
        'Runs of Amendment 9 (seeds 42, 123, 777, 999 and 7, in this order): epoch $e$ of the checkpoint selected by '
        'in-bearing early stopping and ratio $\\rho$ of the 0.9-quantile of the absolute error on the in-bearing '
        'validation windows to that on the test bearings. Per combination: Kendall\'s $S$ and $\\tau_b$ of $e$ against '
        '$\\rho$, against the in-bearing quantile and against the test quantile. $>$90: trained 120 epochs without '
        'stopping.',
        'llllcccc', 'Dataset & Model & $e$ & $\\rho$ & $S$ & $\\tau_b$, $\\rho$ & $\\tau_b$, in-bearing & $\\tau_b$, test',
        rows))

    def row(label, r, expectation):
        return [label, expectation, f"${r['T']:+d}$ ($\\pm${r['T_min_max'][1]})", f"{r['p_two_sided']:.4f}",
                f"${r['mean_tau_b']:+.2f}$", f"{r['n_tau_negative']} of {r['n_tau_defined']}"]
    rows = [row('$\\rho$, twelve combinations (A9-P1, primary)', res['A9_P1'], '$T < 0$'),
            row('$\\rho$, without our XJTU-SY network (A9-S1)', res['A9_S1'], '$T < 0$'),
            row('In-bearing 0.9-quantile (A9-S2)', res['A9_S2_q_cal'], '$T < 0$'),
            row('Test 0.9-quantile (A9-S2)', res['A9_S2_q_test'], 'none')]
    write('s10_tests', table(
        'tab:s10-tests',
        'Tests of Amendment 9. $T$: sum over combinations of Kendall\'s $S$ between the selected epoch and the outcome, '
        'with its attainable range; $p$: exact two-sided, from the within-combination permutation distribution. '
        'Between combinations (descriptive), Kendall\'s $\\tau_b$ of the median epoch and the architecture-level '
        f"ratio is ${res['A9_S3_between']['tau_b']:+.2f}$.",
        'llcccc', 'Outcome & Expectation & $T$ (range) & $p$ & Mean $\\tau_b$ & $\\tau_b < 0$', rows, resize=False))


# -------------------------------------------------------------------------------------------- S11: rebuilt components
def s11_components():
    d = load('components.json')
    assert d['reproduction_check']['max_abs_diff'] <= 1e-9 and d['selected'] == {'budget': 5, 'lambda': 0.08}
    sel = {}
    for tag, root in (('pron', 'outputs/r1_components'), ('xjtu', 'outputs/r1_components_xjtu')):
        budget = json.loads((ROOT / root / 'budget_selection.json').read_text(encoding='utf-8'))
        lam = json.loads((ROOT / root / 'selection.json').read_text(encoding='utf-8'))
        assert lam['budget'] == budget['selected']
        sel[tag] = (budget, lam)
    assert list(sel['pron'][0]['scores']) == list(sel['xjtu'][0]['scores'])
    assert list(sel['pron'][1]['scores']) == list(sel['xjtu'][1]['scores'])

    def ms(s):
        return f"{s['mean']:.4f} ({s['se']:.4f})"

    rows = ['\\multicolumn{3}{l}{\\textit{Budget, at $\\lambda = 0.08$}} \\\\']
    rows += [[f'$B = {k}$', ms(sel['pron'][0]['scores'][k]), ms(sel['xjtu'][0]['scores'][k])]
             for k in sel['pron'][0]['scores']]
    rows.append('\\multicolumn{3}{l}{\\textit{Weight $\\lambda$ of PGMC, at the selected budget}} \\\\')
    rows += [[f'$\\lambda = {k}$', ms(sel['pron'][1]['scores'][k]), ms(sel['xjtu'][1]['scores'][k])]
             for k in sel['pron'][1]['scores']]
    rows.append('\\midrule')
    rows.append(['One-standard-error threshold'] + [f"{sel[t][1]['threshold']:.4f} (minimum at $\\lambda = "
                                                    f"{float(sel[t][1]['argmin'])}$)" for t in ('pron', 'xjtu')])
    rows.append(['Selected'] + [f"$B^* = {sel[t][0]['selected']}$, $\\lambda^* = {float(sel[t][1]['selected'])}$"
                                for t in ('pron', 'xjtu')])
    cur = sel['pron'][0]['current_model']
    write('s11_selection', table(
        'tab:s11-selection',
        'Selection of the rebuilt network R (Amendment 10): four-fold validation MAE, mean over the 20 fold-seed runs '
        '(standard error); test bearings not loaded. The budget is selected at $\\lambda = 0.08$, then $\\lambda$ at the '
        'selected budget by the one-standard-error rule, which keeps 0.08 when its mean is below the minimum plus the '
        'standard error of the minimizing weight. On PRONOSTIA, R would have replaced the submitted network only below '
        f"{cur['mean'] - cur['se']:.4f} (the submitted network's {cur['mean']:.4f} minus its standard error "
        f"{cur['se']:.4f}); XJTU-SY has no adoption rule.",
        'lcc', 'Setting & PRONOSTIA & XJTU-SY', rows, resize=False))

    labels = {'A10-E': 'E, band envelope (against raw-vibration stream only)',
              'A10-G': 'G, gate trained with noise (against static fusion, under noise)',
              'A10-M': 'M, PGMC on window pairs (against without PGMC)',
              'old_env': 'submitted envelope extractor', 'no_aug': 'no input noise in training',
              'adjacent': 'PGMC on adjacent windows', 'no_dann': 'without DANN', 'spectral': 'envelope stream only',
              'submitted network': 'submitted network', '1D CNN': '1D CNN'}

    def trow(key, r):
        rel = r['rel_pct']
        return [labels[key], f"{r['R']:.4f}", f"{r['other']:.4f}",
                f'+{rel:.1f}\\%' if rel >= 0 else f'$-${-rel:.1f}\\%', f"{r['R_better_on']}/{r['n']}",
                f"{r['p_flip']:.3f}", f"{r['p_holm']:.3f}"]
    families = [('Primary family: PRONOSTIA, ten test bearings, Holm over three', d['primary']),
                ('Replication family: XJTU-SY, six test and five Condition-3 bearings, Holm over three',
                 d['replication']),
                ('Secondary (i): R against one change at a time, Holm over five', d['secondary_i']),
                ('Secondary (ii): R against other models, Holm over two', d['secondary_ii'])]
    assert all(r['p_holm'] >= 0.05 for _, fam in families for r in fam.values())
    rows = []
    for title, fam in families:
        rows.append(f'\\multicolumn{{7}}{{l}}{{\\textit{{{title}}}}} \\\\')
        rows += [trow(k, r) for k, r in fam.items()]
    write('s11_tests', table(
        'tab:s11-tests',
        'Tests of Amendment 10: bearing-averaged CES MAE of the rebuilt network R and of the comparator (per bearing, '
        'the mean over the 20 final seeds; for G, also over the noise levels 20, 10, 5, 0 and $-5$~dB), relative '
        'difference (positive when R is lower), bearings on which R is lower, and the exact two-sided sign-flip $p$ '
        'over bearings, unadjusted and Holm-adjusted within each family. No difference is significant after Holm '
        'adjustment.',
        'lcccccc', 'Test & R & Comparator & R lower by & R lower on & $p$ & $p_{\\mathrm{Holm}}$', rows))

    names = {'full': 'R', 'rawonly': 'Raw-vibration stream only', 'static': 'Static fusion',
             'no_pgmc': 'Without PGMC'}
    rows = []
    for arm, r in d['descriptive'].items():
        def p(k):
            return '' if arm == 'full' else f' ({r[k]:.3f})'
        rows.append([names[arm], f"{r['c_star']:.2f}", f"{r['coverage']:.1f}\\%", f"{r['late_coverage']:.1f}\\%",
                     f"{r['interval_score']:.3f}{p('p_interval_score')}", f"{r['tail_ratio']:.2f}",
                     f"{r['cost_r5']:.1f}{p('p_cost_r5')}", f"{r['cost_r10']:.1f}{p('p_cost_r10')}",
                     f"{r['ces_mae']:.4f}"])
    sub = d['reproduction_check']['recomputed']
    write('s11_descriptive', table(
        'tab:s11-descriptive',
        'Calibration and maintenance decisions of R and its primary ablations on the ten PRONOSTIA test bearings, each '
        'calibrated on its own four-fold validation runs by the rule of the manuscript (descriptive; in parentheses, '
        'the unadjusted sign-flip $p$ against R). Coverage of the two-sided 90\\% MC-dropout interval, overall and in '
        'late life; tail ratio: 0.9-quantile of the absolute error on the held-out validation bearings over that on '
        'the test bearings; $L$: expected cost of the cost-matched trigger at $\\tau = 0.20$. Submitted network, with '
        f"the same functions: $c^* = {sub['c_star']:.2f}$, CES MAE {sub['ces_mae']:.4f}, $L$ = {sub['cost_r5']:.1f} "
        f"and {sub['cost_r10']:.1f}.",
        'lcccccccc', 'Model & $c^*$ & Coverage & Late life & Interval score & Tail ratio & $L$, $r = 5$ & $L$, '
        '$r = 10$ & CES MAE', rows))


def s12_cmapss():
    d = load('cmapss.json')
    fleets = ['4', '10', '25', '80']
    tests = ['P1_n4', 'P1_n10', 'P1_n25', 'P1_n80', 'P2', 'P3']
    names = {'P1_n4': 'A11-P1, $n = 4$', 'P1_n10': 'A11-P1, $n = 10$', 'P1_n25': 'A11-P1, $n = 25$',
             'P1_n80': 'A11-P1, $n = 80$', 'P2': 'A11-P2, $n = 80$', 'P3': 'A11-P3, trend over $n$'}

    def pfmt(p):
        if p >= 0.001:
            return f'{p:.3f}'
        m, e = f'{p:.1e}'.split('e')
        return f'${m} \\times 10^{{{int(e)}}}$'

    def cell(t, k):
        if k.startswith('P1'):
            cnt = f"{t['n_positive']}/{t['n']}"
            val = f"${t['mean']:+.1f}$"
        elif k == 'P2':
            cnt = f"{t['n_negative']}/{t['n']}"
            val = f"{t['geometric_mean_rho']:.2f}"
        else:
            cnt = f"$\\tau_b$ {t['mean_tau_b']:.2f}"
            val = f"$T = {t['mean']:+.0f}$"
        p = pfmt(t['p_holm'])
        if t['significant']:
            p = '\\textbf{\\boldmath ' + p + '}'
        elif t['p_holm'] < 0.05:                      # significant against the stated direction
            val = val + '$^{\\dagger}$'
        return f"{val} ({cnt}; {p})"

    rows = []
    for block, key in (('Size-matched (primary; correction 1 and 2)', 'tests'),
                       ('As first written (secondary)', 'tests_as_first_written')):
        rows.append(f'\\multicolumn{{3}}{{l}}{{\\textit{{{block}}}}} \\\\')
        rows += [[names[k], cell(d['FD001'][key][k], k), cell(d['FD003'][key][k], k)] for k in tests]
    write('s12_tests', table(
        'tab:s12-tests',
        'C-MAPSS check (Amendment 11): primary tests, 20 repeats per fleet size, Holm over the six tests of each subset '
        '(FD001 primary, FD003 replication); bold: significant after Holm in the stated direction; '
        '$^{\\dagger}$significant after Holm against it. A11-P1: coverage error '
        'of in-unit calibration minus that of calibration on held-out engines, in points (repeats with a positive '
        'difference; Holm $p$). A11-P2: geometric mean of the ratio of the 0.9-quantile of the absolute error on the '
        'in-unit validation windows to that on the test windows at $n = 80$ (repeats below 1). A11-P3: sum over repeats '
        'of Kendall\'s $S$ between fleet size and that ratio (mean $\\tau_b$). Size-matched: the held-out coverage error '
        'and the test quantile are computed at the size of the in-unit calibration set (Section S12).',
        'lcc', 'Test & FD001 & FD003', rows, resize=False))

    rows = []
    for ds in ('FD001', 'FD003'):
        rows.append(f'\\multicolumn{{9}}{{l}}{{\\textit{{{ds}}}}} \\\\')
        for n in fleets:
            s = d[ds]['secondary'][n]

            def rng_(key, f='.1f'):
                v = s[key]
                return f"{v['mean']:{f}} ({v['min']:{f}}--{v['max']:{f}})"
            rows.append([
                n, f"{s['SP_n_cal']['mean']:.0f} / {s['PR_n_cal']['mean']:.0f}",
                rng_('SP_coverage'), rng_('PR_coverage'),
                f"${s['d_matched']['mean']:+.1f}$ (${s['d']['mean']:+.1f}$)",
                f"{s['SP_rho_matched']['mean']:.2f} ({s['SP_rho']['mean']:.2f})", f"{s['PR_rho']['mean']:.2f}",
                f"{s['SP_decision_cov_r5']['mean']:.1f} / {s['PR_decision_cov_r5']['mean']:.1f}",
                f"{s['SP_rmse_last']['mean']:.1f} / {s['PR_rmse_last']['mean']:.1f}"])
    write('s12_results', table(
        'tab:s12-results',
        'C-MAPSS check: means over the 20 repeats (range). $n$: training engines. Calibration errors: number of '
        'calibration errors, in-unit / held-out engines. Coverage of the 90\\% residual interval on all test windows. '
        'Gain: in-unit coverage error minus held-out coverage error, size-matched (as first written). Ratio: 0.9-quantile '
        'of the absolute error on the calibration data over that on the test windows, in-unit size-matched (plain), and '
        'held-out (plain). Decision quantile: share of test windows whose error lies below the calibrated '
        '$r/(1+r)$-quantile for $r = 5$ (target 83.3\\%). RMSE: last window of each test engine, cycles, in-unit / '
        'held-out deployed model.',
        'rcccccccc', '$n$ & Calibration errors & Coverage, in-unit & Coverage, held-out & Gain & Ratio, in-unit & '
        'Ratio, held-out & Decision quantile & RMSE', rows))

    rows = []
    for ds in ('FD001', 'FD003'):
        for n in fleets:
            s = d[ds]['secondary'][n]
            st = s['SP_kept_step']
            counts = ', '.join(str(v) for v in s['PR_s_star_counts'].values())
            tau = s['tau_kept_step_rho_matched_SP']
            rows.append([ds, n, f"{st['median']:,.0f} ({st['min']:,}--{st['max']:,})", counts,
                         f"{s['SP_decision_cov_r10']['mean']:.1f} / {s['PR_decision_cov_r10']['mean']:.1f}",
                         f"{s['SP_width']['mean']:.1f} / {s['PR_width']['mean']:.1f}",
                         'undefined' if tau is None else f'${tau:.2f}$'])
    write('s12_training', table(
        'tab:s12-training',
        'C-MAPSS check: training and further secondary quantities. Kept step: step of the state kept by in-unit early '
        'stopping, median (range) over 20 repeats (at most 8,000). Budget: number of repeats in which the engine folds '
        'selected 500, 1,000, 2,000, 4,000 and 8,000 steps. Decision quantile for $r = 10$ (target 90.9\\%) and interval '
        'width (cycles), in-unit / held-out. $\\tau$: Kendall\'s $\\tau_b$ between the kept step and the size-matched '
        'in-unit ratio within each fleet size (descriptive, as in S10).',
        'llccccc', 'Subset & $n$ & Kept step & Budget counts & Decision quantile, $r = 10$ & Width & $\\tau$',
        rows, resize=False))


def s12_blocked():
    d = load('cmapss_blocked.json')
    fleets = ['4', '10', '25']
    tests = ['P1_n4', 'P1_n10', 'P1_n25', 'P2_n4', 'P2_n10', 'P2_n25']

    def pfmt(p):
        if p >= 0.001:
            return f'{p:.3f}'
        m, e = f'{p:.1e}'.split('e')
        return f'${m} \\times 10^{{{int(e)}}}$'

    def cell(t, k):
        val = f"${t['mean']:+.1f}$" if k.startswith('P1') else f"{t['geometric_mean_ratio']:.2f}"
        p = pfmt(t['p_holm'])
        if t['significant']:
            p = '\\textbf{\\boldmath ' + p + '}'
        elif t['significant_opposite']:
            val = val + '$^{\\dagger}$'
        return f"{val} ({t['n_predicted']}/{t['n']}; {p})"

    names = {k: (f"A12-P1, $n = {k[4:]}$" if k.startswith('P1') else f"A12-P2, $n = {k[4:]}$") for k in tests}
    rows = [[names[k], cell(d['FD001']['tests'][k], k), cell(d['FD003']['tests'][k], k)] for k in tests]
    write('s12_blocked_tests', table(
        'tab:s12-blocked-tests',
        'Blocked in-unit calibration (Amendment 12): primary tests, 20 repeats per fleet size, Holm over the six tests '
        'of each subset (FD001 primary, FD003 replication); bold: significant after Holm in the stated direction; '
        '$^{\\dagger}$significant after Holm against it. A12-P1: coverage with held-out engines minus coverage with '
        'time blocks of the training engines, in points (repeats with a positive difference; Holm $p$). A12-P2: '
        'geometric mean over repeats of the tail ratio with blocks divided by that with held-out engines (repeats '
        'below 1). Both schemes have one calibration error per fleet window.',
        'lcc', 'Test & FD001 & FD003', rows, resize=False))

    rows = []
    for ds in ('FD001', 'FD003'):
        rows.append(f'\\multicolumn{{8}}{{l}}{{\\textit{{{ds}}}}} \\\\')
        for n in fleets:
            s = d[ds]['secondary'][n]
            g = s['g']
            rows.append([
                n,
                f"{s['SP_coverage']['mean']:.1f} / {s['SB_coverage']['mean']:.1f} "
                f"({s['SB_coverage']['min']:.1f}--{s['SB_coverage']['max']:.1f}) / {s['PR_coverage']['mean']:.1f}",
                f"{s['SP_rho_matched']['mean']:.2f} / {s['SB_rho']['mean']:.2f} / {s['PR_rho']['mean']:.2f}",
                'n/a' if g['median'] is None else f"{g['median']:.2f}",
                f"{s['SB_width']['mean']:.1f} / {s['PR_width']['mean']:.1f}",
                f"{s['SB_decision_cov_r5']['mean']:.1f} / {s['PR_decision_cov_r5']['mean']:.1f}",
                f"{s['SB_rmse_last']['mean']:.1f} / {s['PR_rmse_last']['mean']:.1f}",
                ', '.join(str(v) for v in s['SB_s_star_counts'].values()) + f" ({s['same_s_star']})"])
    write('s12_blocked_results', table(
        'tab:s12-blocked-results',
        'Blocked in-unit calibration: means over the 20 repeats. $n$: training engines. Coverage of the 90\\% residual '
        'interval on all test windows with calibration on every 10th window of the training engines (in-unit, '
        'Amendment 11) / on time blocks of the training engines (blocked; range) / on held-out engines. Ratio: '
        '0.9-quantile of the absolute error on the calibration data over that on the test windows (in-unit: '
        'size-matched). Share removed: median over repeats of the share of the in-unit understatement of the tail that '
        'blocking removes, $g$ (0: none, 1: all). Width (cycles), decision quantile for $r = 5$ (target 83.3\\%) and '
        'last-window RMSE (cycles), blocked / held-out. Budget: repeats in which the blocks selected 500, 1,000, 2,000, '
        '4,000 and 8,000 steps (repeats with the same budget as the engine folds, in which the two deployed models '
        'coincide).',
        'rccccccc', '$n$ & Coverage, in-unit / blocked / held-out & Ratio & Share removed & Width & Decision quantile & '
        'RMSE & Budget', rows))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    refnums()
    s4_data()
    s10_mechanism()
    s11_components()
    if (ROOT / 'outputs' / 'r1_analysis' / 'cmapss.json').exists():
        s12_cmapss()
    if (ROOT / 'outputs' / 'r1_analysis' / 'cmapss_blocked.json').exists():
        s12_blocked()
    res = s2_pronostia()
    s2_xjtu()
    s2_ours(res)
    s2_source()
    s3_timing()
    s3_cmq()
    s3_source_costs()
    s5_candidates()
    s6_noise()
    s8_randsplit()


if __name__ == '__main__':
    main()
