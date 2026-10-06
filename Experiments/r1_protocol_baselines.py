"""
r1_protocol_baselines.py — Revision R1, analysis plan Amendment 1, E3: the selection-channel
demonstration (in-bearing early stopping) for every PRONOSTIA baseline.

Each baseline is trained exactly as in our model's protocol demonstration: every 10th window of
each training bearing is held out for validation (the exp16 'window' split, val_every_k = 10),
at most 120 epochs, patience 30, best-validation checkpoint, seeds 42, 123, 777, 999 and 7,
deterministic kernels, 10 MC-dropout passes, the 13 evaluation bearings. All other settings are
the baseline's own: exp19's training harness for the six exp19 models (including TSMC-CNN's
published loss, optimizer and two channels), and for MSCNN [7] the published optimizer, learning
rate, batch size and loss of r1_mscnn.py with the input normalization fixed by its smoke test.

  --model <name>|all [--smoke]   training (smoke: seed 42, 2 epochs, separate folder)
  --stats                        comparison with each baseline's final runs (Sections 5 and 6):
                                 five shared seeds, all10 bearing-averaged CES MAE, exact sign-flip
                                 test, Holm over the seven baselines; raw co-reported; new6, orig4,
                                 zs3 descriptive

USAGE:
  python Experiments/r1_protocol_baselines.py --model cnn --smoke
  python Experiments/r1_protocol_baselines.py --model all
  python Experiments/r1_protocol_baselines.py --stats
"""

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
import Experiments.r1_mscnn as rm  # noqa: E402
from Experiments.exp16_ress import CONFIG, WindowDataset, evaluate_on_test, load_bearings  # noqa: E402
from Experiments.exp19_baselines import BaselineAdapter, train_baseline  # noqa: E402
from Experiments.r1_provenance import provenance  # noqa: E402
from src.mscnn_li2019 import MSCNN  # noqa: E402

SEEDS = [42, 123, 777, 999, 7]
EVAL_BEARINGS = rm.EVAL_BEARINGS
MAX_EPOCHS, PATIENCE, VAL_EVERY, MC_PASSES = 120, 30, 10, 10
MODELS = ['cnn', 'bilstm', 'cnnlstm', 'resnet', 'transformer', 'tsmccnn', 'mscnn']
LABELS = {'cnn': '1D CNN', 'bilstm': 'BiLSTM', 'cnnlstm': 'CNN-LSTM', 'resnet': '1D ResNet',
          'transformer': 'Transformer', 'tsmccnn': 'TSMC-CNN [21]', 'mscnn': 'MSCNN [7]'}
OUT = ROOT / 'outputs' / 'r1_protocol_baselines'
EXTRA_CODE = ('Experiments/r1_protocol_baselines.py', 'Experiments/r1_mscnn.py', 'src/mscnn_li2019.py')


def in_bearing_split(bearings, k=VAL_EVERY):
    """the exp16 'window' split: window i of a bearing is validation when i % k == k - 1"""
    tr, va = [], []
    for b in bearings:
        idx = np.arange(b.n)
        mask = (idx % k) == (k - 1)
        tr.append(idx[~mask])
        va.append(idx[mask])
    return tr, va


def train_mscnn_es(seed, epochs, device, input_norm):
    """MSCNN with its published training settings (r1_mscnn.train) and in-bearing early stopping"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    bearings = rm.with_history(load_bearings(str(rm.DATA), CONFIG['train_bearings']))
    tr, va = in_bearing_split(bearings)
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(WindowDataset(bearings, tr), batch_size=rm.BATCH, shuffle=True,
                                               generator=gen, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(WindowDataset(bearings, va), batch_size=256, shuffle=False,
                                             num_workers=0)
    model = BaselineAdapter(MSCNN(n_hist=rm.N_HIST, input_norm=input_norm)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=rm.LR, weight_decay=0.0)
    l1 = nn.L1Loss()
    best, best_state, best_epoch, wait, val_hist = float('inf'), None, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y, *_ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, *_r = model(x)
            loss = l1(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        vp, vt = rm.predict(model, val_loader, device)
        val_hist.append(round(float(np.mean(np.abs(vp - vt))), 5))
        if val_hist[-1] < best:
            best, best_epoch, wait = val_hist[-1], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, {'best_epoch': best_epoch, 'best_val_mae': best, 'epochs_trained': len(val_hist),
                   'val_history': val_hist}


def train_one(name, seed, epochs, device, test_cache):
    cfg = dict(CONFIG, mc_dropout_passes=MC_PASSES)
    if name == 'mscnn':
        sel = json.loads((ROOT / 'outputs' / 'r1_mscnn' / 'selection.json').read_text(encoding='utf-8'))
        model, info = train_mscnn_es(seed, epochs, device, sel['input_norm'])
        if 'mscnn' not in test_cache:
            test_cache['mscnn'] = rm.with_history(load_bearings(str(rm.DATA), EVAL_BEARINGS))
        res = evaluate_on_test(model, test_cache['mscnn'], cfg, device)
        return res, model, info
    cfg.update({'data_dir': str(ROOT / ('Processed_PRONOSTIA_v2hv' if name == 'tsmccnn' else 'Processed_PRONOSTIA_v2')),
                'val_mode': 'window', 'val_every_k': VAL_EVERY, 'epochs': epochs, 'patience': PATIENCE,
                'test_bearings': EVAL_BEARINGS})
    res, model = train_baseline(name, seed, cfg, device, verbose=False)
    info = {'best_epoch': res['best_epoch'], 'best_val_mae': res['best_val_mae'],
            'epochs_trained': len(res['val_history']), 'val_history': res['val_history']}
    return res, model, info


def run(names, smoke):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds, epochs = ([42], 2) if smoke else (SEEDS, MAX_EPOCHS)
    root = ROOT / 'outputs' / ('r1_protocol_baselines_smoke' if smoke else 'r1_protocol_baselines')
    test_cache = {}
    for name in names:
        out_dir = root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_f = out_dir / 'per_seed.json'
        per_seed = json.loads(summary_f.read_text()) if summary_f.exists() else []
        for seed in seeds:
            if (out_dir / f'preds_seed{seed}.npz').exists():
                continue
            t0 = time.time()
            res, model, info = train_one(name, seed, epochs, device, test_cache)
            a = res.pop('_arrays')
            np.savez_compressed(out_dir / f'preds_seed{seed}.npz',
                                **{f'raw_{n}': v for n, v in a['raw_p'].items()},
                                **{f'tgt_{n}': v for n, v in a['tgt'].items()},
                                **{f'alpha_{n}': v for n, v in a['alphas'].items()},
                                **{f'mcmean_{n}': v for n, v in a['mc_mean'].items()},
                                **{f'mcstd_{n}': v for n, v in a['mc_std'].items()})
            torch.save(model.state_dict(), out_dir / f'model_seed{seed}.pth')
            per_seed.append({'seed': seed, **info, 'raw_mae_13': res['raw']['mae'], 'ces_mae_13': res['ces']['mae'],
                             'minutes': round((time.time() - t0) / 60, 1)})
            summary_f.write_text(json.dumps(per_seed, indent=1))
            print(f"{name} seed {seed}: best epoch {info['best_epoch']} of {info['epochs_trained']}, "
                  f"pooled raw MAE (13 bearings) {res['raw']['mae']:.4f} ({per_seed[-1]['minutes']} min)", flush=True)
        (out_dir / 'results.json').write_text(json.dumps(
            {'model': name, 'protocol': 'in-bearing early stopping', 'max_epochs': epochs, 'patience': PATIENCE,
             'val_every_k': VAL_EVERY, 'mc_dropout_passes': MC_PASSES, 'per_seed': per_seed,
             'provenance': provenance(EXTRA_CODE)}, indent=1), encoding='utf-8')
        print(f'{name}: in-bearing early stopping runs complete', flush=True)
    return 0


def stats_report(out_path):
    from r1_analysis import load_variant
    from r1_strict_stats import BASELINES, SETS, baseline_paths, compare, holm
    finals = baseline_paths()
    label_of = {short: label for label, (short, _) in BASELINES.items()}
    ours = {k: json.loads((ROOT / f).read_text(encoding='utf-8'))
            for k, f in (('submitted', 'outputs/r1_analysis/strict_B10.json'),
                         ('adopted', 'outputs/r1_analysis/strict_temporal_B5.json'))}
    report = {'sets': {}, 'finals': finals}
    lines = ['# In-bearing early stopping for every baseline (revision R1, Amendment 1, E3)', '',
             'Reference: each baseline\'s final runs (fixed budget selected on validation bearings). Comparator: '
             'in-bearing early stopping (every 10th window of each training bearing as validation, patience 30, '
             'at most 120 epochs, best-validation checkpoint). Five shared seeds. Δ = (early stopping − final) / '
             'early stopping on bearing-averaged MAE; positive: early stopping worse. p: exact sign-flip test over '
             'bearings; Holm over the seven baselines (all10).']
    for set_name in ('all10', 'new6', 'orig4', 'zs3'):
        bearings = SETS[set_name]
        res = {}
        for short in MODELS:
            label = label_of[short]
            es = load_variant(OUT / short)
            if len(es) != len(SEEDS):
                raise SystemExit(f'{short}: expected {len(SEEDS)} early-stopping runs, found {len(es)}')
            fin = load_variant(ROOT / finals[label])
            res[label] = {m: compare(fin, es, bearings, m) for m in ('ces', 'raw')}
        for m in ('ces', 'raw'):
            adj = holm(np.array([r[m]['p_flip'] for r in res.values()]))
            for r, pa in zip(res.values(), adj):
                r[m]['p_holm'] = float(pa)
        report['sets'][set_name] = {k: {m: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                           for kk, vv in r[m].items()} for m in r} for k, r in res.items()}
        lines += ['', f'## {set_name} ({len(bearings)} bearings; smallest attainable p = {2 / 2 ** len(bearings):.4f})', '',
                  '| Model | final, CES bearing avg | early stopping, CES | Δ CES | early stopping worse on | p | Holm | '
                  'Δ raw | p raw | Holm raw |', '|---|---|---|---|---|---|---|---|---|---|']
        for label, r in res.items():
            c, w = r['ces'], r['raw']
            lines.append(f"| {label} | {c['ref']:.4f} | {c['other']:.4f} | {c['rel']:+.1f} % | {c['k']}/{c['n']} | "
                         f"{c['p_flip']:.3f} | {c['p_holm']:.3f} | {w['rel']:+.1f} % | {w['p_flip']:.3f} | {w['p_holm']:.3f} |")
        for key, label in (('submitted', 'DSAFLite+PGMC, submitted architecture'),
                           ('adopted', 'DSAFLite+PGMC with temporal context (adopted)')):
            p = ours[key]['sets'][set_name]['ces'].get('protocol', {}).get('in-bearing early stopping')
            if p:
                lines.append(f"| {label} (plan Section 8) | {p['ref']:.4f} | {p['other']:.4f} | {p['rel']:+.1f} % | "
                             f"{p['k']}/{p['n']} | {p['p_flip']:.3f} | – | | | |")
        if set_name == 'all10':
            worse = int(sum(r['ces']['rel'] > 0 for r in res.values()))
            sub = ours['submitted']['sets']['all10']['ces']['protocol']['in-bearing early stopping']['rel'] > 0
            report['models_worse_with_es_all10_ces'] = {'baselines': worse, 'submitted_architecture': bool(sub),
                                                         'of_eight': worse + int(sub)}
            lines += ['', f'In-bearing early stopping gives the higher all10 CES error for {worse} of the 7 baselines '
                          f'and for the submitted architecture: {worse + int(sub)} of 8 models.']
    report['provenance'] = provenance(EXTRA_CODE + ('Experiments/r1_strict_stats.py',))
    text = '\n'.join(lines) + '\n'
    print(text)
    Path(out_path).write_text(text, encoding='utf-8')
    Path(out_path).with_suffix('.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'written {out_path}')
    return 0


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None, help='one of ' + ', '.join(MODELS) + ', or all')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'r1_analysis' / 'protocol_baselines.md'))
    args = ap.parse_args()
    if args.stats:
        return stats_report(args.out)
    if not args.model:
        raise SystemExit('--model or --stats required')
    return run(MODELS if args.model == 'all' else [args.model], args.smoke)


if __name__ == '__main__':
    sys.exit(main())
