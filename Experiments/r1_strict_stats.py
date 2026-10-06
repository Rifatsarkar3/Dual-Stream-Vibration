"""
r1_strict_stats.py — Revision R1: strict bearing-level inference on saved predictions.

The bearing, not the seed, is the independent unit (reviewer comment 4). For every
comparison against the reference model:
  per-bearing seed-mean MAE (raw and CES) over the seeds both variants share,
  difference d_b = comparator - reference (> 0: reference better on bearing b),
  exact two-sided sign-flip permutation test on mean(d) over all 2^n sign patterns,
  exact two-sided sign test, percentile bootstrap interval over bearings (reported
  for continuity only; it is anti-conservative for n <= 10),
  Holm adjustment across the baseline family. Protocol and ablation comparisons are
  separate families and are reported unadjusted.
With n bearings the smallest attainable two-sided permutation p-value is 2 / 2^n
(0.125 for n = 4, 0.031 for n = 6, 0.002 for n = 10).

Inference only on predictions already written by r1_final_B10.sh and r1_reeval.py;
no model is trained or evaluated here.

USAGE:
  python Experiments/r1_strict_stats.py [--out outputs/r1_analysis/strict_B10.md]
"""

import argparse
import json
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
from r1_analysis import SETS, load_variant, variants_of  # noqa: E402
from r1_provenance import provenance  # noqa: E402

SETS = dict(SETS)
SETS['new6'] = [b for b in SETS['all10'] if b not in SETS['orig4']]   # never used before R1

REF = 'outputs/r1_final_B10/final20'
# Primary family (analysis plan Section 8): the seven baselines. A baseline retrained under
# the 5-seed selection (outputs/r1_baselines_final5/<name>, 20 seeds) replaces its old runs.
BASELINES = {
    '1D CNN': ('cnn', 'outputs/r1_reeval/baseline_cnn'),
    'BiLSTM': ('bilstm', 'outputs/r1_reeval/baseline_bilstm'),
    'CNN-LSTM': ('cnnlstm', 'outputs/r1_reeval/baseline4f_cnnlstm'),
    '1D ResNet': ('resnet', 'outputs/r1_reeval/baseline_resnet'),
    'Transformer': ('transformer', 'outputs/r1_reeval/baseline4f_transformer'),
    'TSMC-CNN [21]': ('tsmccnn', 'outputs/r1_reeval/published_tsmccnn'),
    'MSCNN [7]': ('mscnn', 'outputs/r1_mscnn/final'),
}
ABLATION_ROOT = 'outputs/r1_final_B10'
ABLATIONS = {   # folder names under the ablation root (--ablation-root)
    'static fusion': 'abl20_static',
    'temporal stream only': 'abl20_temporal',
    'spectral stream only': 'abl20_spectral',
    'without PGMC': 'abl20_no_pgmc',
    'without DANN': 'abl20_no_dann',
    'without PGMC and DANN': 'abl20_neither',
    'PGMC batch ordering': 'abl20_pgmc_soft',
}
PROTOCOL = {
    'in-bearing early stopping': 'outputs/r1_reeval/full_inbearing_es',
    'lambda by validation argmin': 'outputs/r1_reeval/final21_lambda2',
}


def n_preds(path):
    return len(list((ROOT / path).glob('preds_seed*.npz')))


def baseline_paths():
    """A baseline whose budget changed under the 5-seed selection must use its retrained runs
    (plan Section 5); while those are incomplete it counts as missing, never as the old runs."""
    out = {}
    for label, (short, path) in BASELINES.items():
        sel = ROOT / 'outputs' / 'r1_baseline_sel5' / f'{short}.json'
        changed = sel.exists() and json.loads(sel.read_text(encoding='utf-8')).get('budget_changed')
        out[label] = f'outputs/r1_baselines_final5/{short}' if changed else path
    return out


def per_bearing_mae(variant, bearings, seeds, method):
    return np.array([np.mean([np.mean(np.abs(variants_of(variant[s][b]['raw'])[method]
                                             - variant[s][b]['y'])) for s in seeds])
                     for b in bearings])


def signflip_p(d):
    obs = abs(d.mean())
    hits = sum(abs((d * np.array(signs)).mean()) >= obs - 1e-15
               for signs in itertools.product((1, -1), repeat=len(d)))
    return hits / 2 ** len(d)


def holm(pvals):
    order, m = np.argsort(pvals), len(pvals)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def compare(ref, other, bearings, method):
    seeds = sorted(set(ref) & set(other))
    a = per_bearing_mae(ref, bearings, seeds, method)
    b = per_bearing_mae(other, bearings, seeds, method)
    d = b - a
    boot = np.random.default_rng(0).choice(d, size=(20000, len(d)), replace=True).mean(axis=1)
    k = int((d > 0).sum())
    return {'n_seeds': len(seeds), 'ref': a.mean(), 'other': b.mean(),
            'rel': 100 * (b.mean() - a.mean()) / b.mean(), 'k': k, 'n': len(d),
            'p_flip': signflip_p(d), 'p_sign': stats.binomtest(k, len(d)).pvalue,
            'ci': (np.quantile(boot, 0.025), np.quantile(boot, 0.975))}


def row(name, r, p_adj):
    adj = f'{p_adj:.4f}' if p_adj is not None else '–'
    return (f"| {name} | {r['ref']:.4f} | {r['other']:.4f} | {r['rel']:+.1f} | {r['k']}/{r['n']} | "
            f"{r['p_flip']:.4f} | {adj} | {r['p_sign']:.4f} | "
            f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | {r['n_seeds']} |")


def family_table(ref, members, bearings, method, holm_adjust=True):
    res = {k: compare(ref, v, bearings, method) for k, v in members.items()}
    adj = holm(np.array([r['p_flip'] for r in res.values()])) if holm_adjust and res else [None] * len(res)
    return [row(k, r, pa) for (k, r), pa in zip(res.items(), adj)], {
        k: {**{kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in r.items()},
            'p_holm': (None if pa is None else float(pa))} for (k, r), pa in zip(res.items(), adj)}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'strict_B10.md'))
    ap.add_argument('--provisional', action='store_true',
                    help='allow an incomplete baseline family (labelled provisional)')
    ap.add_argument('--ref', default=REF, help='reference model predictions (20 seeds)')
    ap.add_argument('--ablation-root', default=ABLATION_ROOT,
                    help="folder holding the reference model's abl20_* ablation runs")
    ap.add_argument('--skip-protocol', action='store_true',
                    help='omit the protocol family (its runs use the submitted architecture)')
    ap.add_argument('--protocol', action='append', default=None, metavar='LABEL=PATH',
                    help='protocol family members for this reference (repeatable; replaces the default family)')
    args = ap.parse_args()
    if args.skip_protocol:
        protocol_paths = {}
    elif args.protocol:
        protocol_paths = dict(item.split('=', 1) for item in args.protocol)
    else:
        protocol_paths = PROTOCOL

    paths = baseline_paths()
    ready = {k: v for k, v in paths.items() if n_preds(v) == 20}
    missing = sorted(set(paths) - set(ready))
    if missing and not args.provisional:
        raise SystemExit(f'baseline family incomplete (missing: {missing}); use --provisional')
    ref = load_variant(ROOT / args.ref)
    base = {k: load_variant(ROOT / v) for k, v in ready.items()}
    abl = {k: load_variant(ROOT / args.ablation_root / v) for k, v in ABLATIONS.items()}
    prot = {k: load_variant(ROOT / v) for k, v in protocol_paths.items()}
    status = 'PROVISIONAL: baseline family incomplete, missing ' + ', '.join(missing) if missing else 'complete'
    lines = [f'# Strict bearing-level inference (reference: {args.ref})', '', f'Status: {status}.', '',
             'Baselines used: ' + '; '.join(f'{k} = {v}' for k, v in ready.items()), '',
             'Δ% = (comparator − reference) / comparator on bearing-averaged MAE; positive: reference '
             'better. flip p: exact two-sided sign-flip permutation test over bearings. Holm: adjusted '
             'within each family (baselines, ablations, protocol). Bootstrap interval shown for continuity only.']
    header = ('| Comparator | reference | comparator | Δ% | reference better on | flip p | Holm p | '
              'sign p | bootstrap 95% | seeds |\n|---|---|---|---|---|---|---|---|---|---|')
    report = {'reference': args.ref, 'ablation_root': args.ablation_root, 'status': status,
              'baselines': ready, 'protocol': protocol_paths, 'sets': {}}
    for set_name in ('all10', 'new6', 'orig4', 'zs3'):
        bearings = SETS[set_name]
        report['sets'][set_name] = {}
        for method in ('ces', 'raw'):
            entry = {}
            lines += ['', f'## {set_name} ({len(bearings)} bearings), {method.upper()} MAE '
                          f'(smallest attainable flip p = {2 / 2 ** len(bearings):.4f})']
            for fam_name, fam in (('baselines', base), ('ablations', abl), ('protocol', prot)):
                if not fam:
                    continue
                rows, data = family_table(ref, fam, bearings, method)
                lines += ['', f'### {fam_name}', '', header] + rows
                entry[fam_name] = data
            report['sets'][set_name][method] = entry
    report['provenance'] = provenance(('Experiments/r1_strict_stats.py', 'Experiments/r1_analysis.py'))
    text = '\n'.join(lines) + '\n'
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding='utf-8')
    Path(args.out).with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'written {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
