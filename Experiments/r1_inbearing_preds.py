"""
r1_inbearing_preds.py — Revision R1, analysis plan Amendment 6: predictions needed to measure what
standard-practice calibration does. Inference only; no training, no selection.

  T6a  each of the eight architectures' five in-bearing early-stopping checkpoints (seeds 42, 123, 777,
       999, 7) on its own in-bearing validation windows (every 10th window of each training bearing,
       the exp16 'window' split), deterministic and with 30 MC-dropout passes. Check: the pooled
       validation MAE reproduces the run's recorded best validation MAE within 1e-4.
  T6b  the six dropout baselines' early-stopping test predictions re-scored with 30 passes (stored: 10).
       Check: deterministic predictions reproduce the stored ones within 1e-5, targets exactly.
  T6c  our model's 20 best-validation checkpoints of the four-fold selection runs on their own held-out
       calibration bearing, deterministic and with 30 passes (no test bearing loaded). Check: pooled MAE
       reproduces the run's recorded best validation MAE within 1e-4.

USAGE:
  python Experiments/r1_inbearing_preds.py --stage t6a --model cnn     # ours, cnn, bilstm, cnnlstm, resnet, transformer, tsmccnn, mscnn
  python Experiments/r1_inbearing_preds.py --stage t6b --model cnn     # cnn, bilstm, cnnlstm, resnet, transformer, mscnn
  python Experiments/r1_inbearing_preds.py --stage t6c
  python Experiments/r1_inbearing_preds.py --smoke                     # Amendment 6 smoke tests, 2 passes, separate folder
"""

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # before the first cuBLAS call

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
from Experiments.exp16_ress import CONFIG, BearingArrays, evaluate_on_test, load_bearings, make_model  # noqa: E402
from Experiments.exp19_baselines import MODELS, BaselineAdapter  # noqa: E402
from Experiments.r1_reeval import EVAL_BEARINGS  # noqa: E402
import r1_mscnn as rm  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from src.mscnn_li2019 import MSCNN  # noqa: E402

ES_SEEDS = [42, 123, 777, 999, 7]
ES_DIRS = {'ours': 'outputs/exp16_ress/full',
           **{m: f'outputs/r1_protocol_baselines/{m}'
              for m in ('cnn', 'bilstm', 'cnnlstm', 'resnet', 'transformer', 'tsmccnn', 'mscnn')}}
DROPOUT_BASELINES = ['cnn', 'bilstm', 'cnnlstm', 'resnet', 'transformer', 'mscnn']
FOLDS = {'A': 'train_02_Bearing1_2', 'B': 'train_04_Bearing2_2', 'C': 'train_01_Bearing1_1', 'D': 'train_03_Bearing2_1'}
VAL_EVERY = 10
MC_PASSES = 30
TOL_MAE, TOL_RAW = 1e-4, 1e-5
OUT = ROOT / 'outputs' / 'r1_inbearing'


def data_dir(model):
    return ROOT / ('Processed_PRONOSTIA_v2hv' if model == 'tsmccnn' else 'Processed_PRONOSTIA_v2')


def channels(model):
    return 'hv' if model == 'tsmccnn' else 'h'


def build(model, device):
    if model == 'ours':
        return make_model(CONFIG, device)
    if model == 'mscnn':
        sel = json.loads((ROOT / 'outputs' / 'r1_mscnn' / 'selection.json').read_text(encoding='utf-8'))
        return BaselineAdapter(MSCNN(n_hist=rm.N_HIST, input_norm=sel['input_norm'])).to(device)
    return BaselineAdapter(MODELS[model](), needs_spectrogram=(model == 'transformer')).to(device)


def bearings_for(model, names):
    bs = load_bearings(str(data_dir(model)), names, channels=channels(model))
    return rm.with_history(bs) if model == 'mscnn' else bs


def in_bearing_windows(bearings, k=VAL_EVERY):
    """the exp16 'window' split: window i of a bearing is validation when i % k == k - 1"""
    out = []
    for b in bearings:
        mask = (np.arange(b.n) % k) == (k - 1)
        out.append(BearingArrays(b.name, b.vib[mask], b.rul[mask], b.condition))
    return out


def recorded_best(es_dir):
    res = json.loads((ROOT / es_dir / 'results.json').read_text(encoding='utf-8'))
    return {int(r['seed']): float(r['best_val_mae']) for r in res['per_seed']}


def predict(model, ckpt, seed, bearings, passes, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build(model, device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    return evaluate_on_test(net, bearings, dict(CONFIG, mc_dropout_passes=passes), device)['_arrays']


def save(path, a):
    np.savez_compressed(path,
                        **{f'raw_{n}': v for n, v in a['raw_p'].items()},
                        **{f'tgt_{n}': v for n, v in a['tgt'].items()},
                        **{f'mcmean_{n}': v for n, v in a['mc_mean'].items()},
                        **{f'mcstd_{n}': v for n, v in a['mc_std'].items()})


def pooled_mae(path):
    d = np.load(path)
    names = [k[4:] for k in d.files if k.startswith('tgt_')]
    return float(np.mean(np.abs(np.concatenate([d['raw_' + n].astype(float) for n in names])
                                - np.concatenate([d['tgt_' + n].astype(float) for n in names]))))


def write_meta(out_dir, meta):
    meta.update({'date': time.strftime('%Y-%m-%d %H:%M'),
                 'provenance': provenance(('Experiments/r1_inbearing_preds.py', 'Experiments/r1_mscnn.py',
                                           'src/mscnn_li2019.py'))})
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')


def t6a(model, device, out_root, passes, seeds):
    es_dir, best = ES_DIRS[model], recorded_best(ES_DIRS[model])
    val = in_bearing_windows(bearings_for(model, CONFIG['train_bearings']))
    out_dir = out_root / 't6a' / model
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = {}
    for seed in seeds:
        f = out_dir / f'inval_seed{seed}.npz'
        if not f.exists():
            save(f, predict(model, ROOT / es_dir / f'model_seed{seed}.pth', seed, val, passes, device))
        mae = pooled_mae(f)
        checks[seed] = {'pooled_val_mae': mae, 'recorded_best_val_mae': best[seed], 'abs_diff': abs(mae - best[seed])}
        if checks[seed]['abs_diff'] > TOL_MAE:
            raise SystemExit(f'T6a {model} seed {seed}: in-bearing validation MAE {mae:.6f} differs from the recorded '
                             f'{best[seed]:.6f} by more than {TOL_MAE:g}; stopping (Amendment 6)')
    write_meta(out_dir, {'stage': 'T6a', 'model': model, 'checkpoints': es_dir, 'mc_dropout_passes': passes,
                         'val_every_k': VAL_EVERY, 'n_runs': len(seeds), 'checks': checks,
                         'max_abs_diff': max(c['abs_diff'] for c in checks.values())})
    print(f"{model}: T6a in-bearing validation predictions complete ({len(seeds)} checkpoints, validation MAE within "
          f"{max(c['abs_diff'] for c in checks.values()):.1e} of the recorded values)", flush=True)


def t6b(model, device, out_root, passes, seeds):
    es_dir = ES_DIRS[model]
    test = bearings_for(model, EVAL_BEARINGS)
    out_dir = out_root / 't6b' / model
    out_dir.mkdir(parents=True, exist_ok=True)
    max_diff = 0.0
    for seed in seeds:
        f = out_dir / f'preds_seed{seed}.npz'
        if not f.exists():
            save(f, predict(model, ROOT / es_dir / f'model_seed{seed}.pth', seed, test, passes, device))
        new, old = np.load(f), np.load(ROOT / es_dir / f'preds_seed{seed}.npz')
        for b in EVAL_BEARINGS:
            if not np.array_equal(new[f'tgt_{b}'], old[f'tgt_{b}']):
                raise SystemExit(f'T6b {model} seed {seed} {b}: targets differ from the stored predictions')
            max_diff = max(max_diff, float(np.max(np.abs(new[f'raw_{b}'].astype(float) - old[f'raw_{b}'].astype(float)))))
        if max_diff > TOL_RAW:
            raise SystemExit(f'T6b {model} seed {seed}: deterministic predictions differ from the stored ones by '
                             f'{max_diff:.2e}; stopping (Amendment 6)')
    write_meta(out_dir, {'stage': 'T6b', 'model': model, 'checkpoints': es_dir, 'mc_dropout_passes': passes,
                         'n_checkpoints': len(seeds), 'raw_check_max_abs_diff': max_diff})
    print(f'{model}: T6b {passes}-pass test predictions complete ({len(seeds)} checkpoints, deterministic predictions '
          f'within {max_diff:.1e} of the stored ones)', flush=True)


def t6c(device, out_root, passes, runs):
    out_dir = out_root / 't6c' / 'ours'
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = {}
    for fold, seed in runs:
        run_dir = ROOT / 'outputs' / 'r1_lambda' / f'lam0.08_B10_fold{fold}'
        best = recorded_best(run_dir.relative_to(ROOT))[seed]
        f = out_dir / f'fold{fold}_seed{seed}.npz'
        if not f.exists():
            val = bearings_for('ours', [FOLDS[fold]])
            save(f, predict('ours', run_dir / f'model_seed{seed}.pth', seed, val, passes, device))
        mae = pooled_mae(f)
        checks[f'{fold}_{seed}'] = {'val_mae': mae, 'recorded_best_val_mae': best, 'abs_diff': abs(mae - best)}
        if abs(mae - best) > TOL_MAE:
            raise SystemExit(f'T6c fold {fold} seed {seed}: validation MAE {mae:.6f} differs from the recorded best '
                             f'{best:.6f}; stopping (Amendment 6)')
    write_meta(out_dir, {'stage': 'T6c', 'model': 'ours', 'checkpoints': 'outputs/r1_lambda/lam0.08_B10_fold*/model_seed*.pth',
                         'mc_dropout_passes': passes, 'n_runs': len(runs), 'checks': checks,
                         'max_abs_diff': max(c['abs_diff'] for c in checks.values())})
    print(f"ours: T6c best-validation predictions complete ({len(runs)} runs, validation MAE within "
          f"{max(c['abs_diff'] for c in checks.values()):.1e} of the recorded best values)", flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['t6a', 't6b', 't6c'])
    ap.add_argument('--model')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        smoke = ROOT / 'outputs' / 'r1_inbearing_smoke'
        for m in ('ours', 'cnn', 'tsmccnn', 'mscnn'):
            t6a(m, device, smoke, 2, [42])
        t6b('cnn', device, smoke, 2, [42])
        t6c(device, smoke, 2, [('A', 42)])
        return 0
    if args.stage == 't6a':
        if args.model not in ES_DIRS:
            raise SystemExit(f'--model must be one of {sorted(ES_DIRS)}')
        t6a(args.model, device, OUT, MC_PASSES, ES_SEEDS)
    elif args.stage == 't6b':
        if args.model not in DROPOUT_BASELINES:
            raise SystemExit(f'--model must be one of {DROPOUT_BASELINES}')
        t6b(args.model, device, OUT, MC_PASSES, ES_SEEDS)
    elif args.stage == 't6c':
        t6c(device, OUT, MC_PASSES, [(f, s) for f in 'ABCD' for s in ES_SEEDS])
    else:
        raise SystemExit('give --stage or --smoke')
    return 0


if __name__ == '__main__':
    sys.exit(main())
