"""
r1_xjtu_inbearing.py — Revision R1, analysis plan Amendment 7, part A: the XJTU-SY predictions needed to
compare in-bearing calibration with held-out calibration, for the submitted architecture ('ours') and the
three XJTU-SY baselines (1D CNN, BiLSTM, CNN-LSTM). Seeds 42, 123, 777, 999, 7; eleven bearings (six test,
five Condition 3).

  es   in-bearing early stopping for the three baselines with exp19's harness: every 10th window of each
       training bearing as validation, at most 120 epochs, patience 30, best-validation checkpoint,
       deterministic kernels, test bearings not loaded. Our model's early-stopping runs come from
       exp17_xjtu.py with flags only (chain script).
  t7a  each early-stopping checkpoint on its own in-bearing validation windows, deterministic and 30
       MC-dropout passes. Check: pooled MAE reproduces the recorded best validation MAE within 1e-4.
  t7b  the early-stopping checkpoints on the eleven bearings, deterministic and 30 passes.
  t7c  the final runs on the same five seeds re-scored on the eleven bearings with 30 passes. Check:
       deterministic predictions reproduce the stored ones within 1e-5, targets exactly.
  v7   the baselines' four-fold selection runs (E6) at the selected budget repeated with deterministic
       kernels, final-epoch model kept (r1_baseline_valpreds' loop), predicting its held-out training
       bearing (deterministic and 30 passes). Check: final-epoch validation MAE reproduces E6's selection
       file within 1e-9.

USAGE:
  python Experiments/r1_xjtu_inbearing.py --stage es --model cnn        # cnn, bilstm, cnnlstm
  python Experiments/r1_xjtu_inbearing.py --stage t7a --model ours     # ours, cnn, bilstm, cnnlstm
  python Experiments/r1_xjtu_inbearing.py --stage t7b --model ours
  python Experiments/r1_xjtu_inbearing.py --stage t7c --model ours
  python Experiments/r1_xjtu_inbearing.py --stage v7 --model cnnlstm   # cnn, bilstm, cnnlstm
  python Experiments/r1_xjtu_inbearing.py --smoke                      # Amendment 7 smoke tests, separate folder
"""

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # before the first cuBLAS call

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
from Experiments.exp16_ress import evaluate_on_test, load_bearings, make_model  # noqa: E402
from Experiments.exp17_xjtu import XJTU_CONFIG  # noqa: E402
from Experiments.exp19_baselines import MODELS as EXP19_MODELS  # noqa: E402
from Experiments.exp19_baselines import BaselineAdapter, train_baseline  # noqa: E402
from Experiments.r1_baseline_valpreds import train_exp19_final  # noqa: E402
from Experiments.r1_inbearing_preds import in_bearing_windows, pooled_mae, save  # noqa: E402
from Experiments.r1_provenance import provenance  # noqa: E402

SEEDS = [42, 123, 777, 999, 7]
MODELS = ['ours', 'cnn', 'bilstm', 'cnnlstm']
BASELINES = ['cnn', 'bilstm', 'cnnlstm']
MAX_EPOCHS, PATIENCE, VAL_EVERY, MC_PASSES = 120, 30, 10, 30
TOL_MAE, TOL_RAW, TOL_REPLAY = 1e-4, 1e-5, 1e-9
TRAIN = list(XJTU_CONFIG['train_bearings'])
EVAL = list(XJTU_CONFIG['test_bearings']) + list(XJTU_CONFIG['zeroshot_bearings'])
FINAL_DIRS = {'ours': 'outputs/exp17_xjtu/final20',
              **{m: f'outputs/r1_xjtu_baselines/final/{m}' for m in BASELINES}}
SELECT = 'outputs/r1_xjtu_baselines/select/{m}.json'
PLAN_K = {'cnn': 20, 'bilstm': 20, 'cnnlstm': 5}
FOLDS = {'A': 'train_02_Bearing1_2', 'B': 'train_04_Bearing2_2', 'C': 'train_01_Bearing1_1', 'D': 'train_03_Bearing2_1'}
SELECT_SEEDS = [7, 42, 123, 777, 999]
OUT = ROOT / 'outputs' / 'r1_xjtu_inbearing'
SMOKE_OUT = ROOT / 'outputs' / 'r1_xjtu_inbearing_smoke'
EXTRA_CODE = ('Experiments/r1_xjtu_inbearing.py', 'Experiments/exp17_xjtu.py', 'Experiments/r1_baseline_valpreds.py',
              'Experiments/r1_inbearing_preds.py')


def build(model, device):
    if model == 'ours':
        return make_model(XJTU_CONFIG, device)
    return BaselineAdapter(EXP19_MODELS[model](), needs_spectrogram=(model == 'transformer')).to(device)


def xjtu(names):
    return load_bearings(XJTU_CONFIG['data_dir'], names)


def predict(model, ckpt, seed, bearings, passes, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build(model, device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    return evaluate_on_test(net, bearings, dict(XJTU_CONFIG, mc_dropout_passes=passes), device)['_arrays']


def recorded_best(es_dir):
    res = json.loads((es_dir / 'results.json').read_text(encoding='utf-8'))
    return {int(r['seed']): float(r['best_val_mae']) for r in res['per_seed']}


def write_meta(out_dir, meta):
    meta.update({'date': time.strftime('%Y-%m-%d %H:%M'), 'provenance': provenance(EXTRA_CODE)})
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')


# ---------------------------------------------------------------- stages ------------------------
def es(model, device, out_root, seeds, epochs):
    out_dir = out_root / 'es' / model
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_f = out_dir / 'per_seed.json'
    per_seed = json.loads(summary_f.read_text()) if summary_f.exists() else []
    for seed in seeds:
        if (out_dir / f'model_seed{seed}.pth').exists() and any(r['seed'] == seed for r in per_seed):
            continue
        t0 = time.time()
        cfg = dict(XJTU_CONFIG)
        cfg.update({'val_mode': 'window', 'val_every_k': VAL_EVERY, 'epochs': epochs, 'patience': PATIENCE,
                    'skip_test': True, 'test_bearings': [], 'mc_dropout_passes': MC_PASSES})
        res, net = train_baseline(model, seed, cfg, device, verbose=False)
        torch.save(net.state_dict(), out_dir / f'model_seed{seed}.pth')
        per_seed = [r for r in per_seed if r['seed'] != seed]
        per_seed.append({'seed': seed, 'best_epoch': res['best_epoch'], 'best_val_mae': res['best_val_mae'],
                         'epochs_trained': len(res['val_history']), 'val_history': res['val_history'],
                         'minutes': round((time.time() - t0) / 60, 1)})
        summary_f.write_text(json.dumps(per_seed, indent=1))
        print(f"{model} seed {seed}: best epoch {res['best_epoch']} of {len(res['val_history'])}, best in-bearing "
              f"validation MAE {res['best_val_mae']:.5f} ({per_seed[-1]['minutes']} min)", flush=True)
    (out_dir / 'results.json').write_text(json.dumps(
        {'model': model, 'protocol': 'in-bearing early stopping (XJTU-SY)', 'max_epochs': epochs, 'patience': PATIENCE,
         'val_every_k': VAL_EVERY, 'per_seed': per_seed, 'provenance': provenance(EXTRA_CODE)}, indent=1), encoding='utf-8')
    print(f'{model}: XJTU-SY in-bearing early-stopping runs complete ({len(per_seed)} seeds)', flush=True)


def t7a(model, device, out_root, passes, seeds):
    es_dir = out_root / 'es' / model
    best = recorded_best(es_dir)
    val = in_bearing_windows(xjtu(TRAIN), VAL_EVERY)
    out_dir = out_root / 't7a' / model
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = {}
    for seed in seeds:
        f = out_dir / f'inval_seed{seed}.npz'
        if not f.exists():
            save(f, predict(model, es_dir / f'model_seed{seed}.pth', seed, val, passes, device))
        mae = pooled_mae(f)
        checks[seed] = {'pooled_val_mae': mae, 'recorded_best_val_mae': best[seed], 'abs_diff': abs(mae - best[seed])}
        if checks[seed]['abs_diff'] > TOL_MAE:
            raise SystemExit(f'T7a {model} seed {seed}: in-bearing validation MAE {mae:.6f} differs from the recorded '
                             f'{best[seed]:.6f} by more than {TOL_MAE:g}; stopping (Amendment 7)')
    write_meta(out_dir, {'stage': 'T7a', 'model': model, 'checkpoints': str(es_dir.relative_to(ROOT)),
                         'mc_dropout_passes': passes, 'val_every_k': VAL_EVERY, 'n_runs': len(seeds), 'checks': checks,
                         'max_abs_diff': max(c['abs_diff'] for c in checks.values())})
    print(f"{model}: T7a in-bearing validation predictions complete ({len(seeds)} checkpoints, validation MAE within "
          f"{max(c['abs_diff'] for c in checks.values()):.1e} of the recorded values)", flush=True)


def t7b(model, device, out_root, passes, seeds):
    es_dir = out_root / 'es' / model
    bearings = xjtu(EVAL)
    out_dir = out_root / 't7b' / model
    out_dir.mkdir(parents=True, exist_ok=True)
    mae = {}
    for seed in seeds:
        f = out_dir / f'preds_seed{seed}.npz'
        if not f.exists():
            save(f, predict(model, es_dir / f'model_seed{seed}.pth', seed, bearings, passes, device))
        d = np.load(f)
        if sorted(k[4:] for k in d.files if k.startswith('tgt_')) != sorted(EVAL):
            raise SystemExit(f'T7b {model} seed {seed}: bearings missing')
        mae[seed] = pooled_mae(f)
    write_meta(out_dir, {'stage': 'T7b', 'model': model, 'checkpoints': str(es_dir.relative_to(ROOT)),
                         'mc_dropout_passes': passes, 'n_checkpoints': len(seeds), 'bearings': EVAL,
                         'pooled_raw_mae_11': mae})
    print(f'{model}: T7b {passes}-pass predictions of the early-stopping checkpoints complete ({len(seeds)} checkpoints)',
          flush=True)


def stored_raw(model, seed):
    """{bearing: (raw, tgt)} of the stored final-run predictions on the eleven bearings"""
    d = ROOT / FINAL_DIRS[model]
    files = [d / f'preds_seed{seed}.npz'] + ([d / f'zeroshot_preds_seed{seed}.npz'] if model == 'ours' else [])
    out = {}
    for f in files:
        z = np.load(f)
        out.update({k[4:]: (z['raw_' + k[4:]], z[k]) for k in z.files if k.startswith('tgt_')})
    return out


def t7c(model, device, out_root, passes, seeds):
    bearings = xjtu(EVAL)
    out_dir = out_root / 't7c' / model
    out_dir.mkdir(parents=True, exist_ok=True)
    max_diff = 0.0
    for seed in seeds:
        f = out_dir / f'preds_seed{seed}.npz'
        if not f.exists():
            save(f, predict(model, ROOT / FINAL_DIRS[model] / f'model_seed{seed}.pth', seed, bearings, passes, device))
        new, old = np.load(f), stored_raw(model, seed)
        for b in EVAL:
            if not np.array_equal(new[f'tgt_{b}'], old[b][1]):
                raise SystemExit(f'T7c {model} seed {seed} {b}: targets differ from the stored predictions')
            max_diff = max(max_diff, float(np.max(np.abs(new[f'raw_{b}'].astype(float) - old[b][0].astype(float)))))
        if max_diff > TOL_RAW:
            raise SystemExit(f'T7c {model} seed {seed}: deterministic predictions differ from the stored ones by '
                             f'{max_diff:.2e}; stopping (Amendment 7)')
    write_meta(out_dir, {'stage': 'T7c', 'model': model, 'checkpoints': FINAL_DIRS[model], 'mc_dropout_passes': passes,
                         'n_checkpoints': len(seeds), 'bearings': EVAL, 'raw_check_max_abs_diff': max_diff})
    print(f'{model}: T7c {passes}-pass test predictions of the final runs complete ({len(seeds)} checkpoints, '
          f'deterministic predictions within {max_diff:.1e} of the stored ones)', flush=True)


def v7(model, device, out_root, passes, runs):
    sel = json.loads((ROOT / SELECT.format(m=model)).read_text())
    k = int(sel['K'])
    if k != PLAN_K[model]:
        raise SystemExit(f'{model}: selection file gives K = {k}, plan Amendment 7 says {PLAN_K[model]}')
    ref = {(r['fold'][-1], r['seed']): r for r in sel['runs'] if r['budget'] == k}
    out_dir = out_root / 'v7' / model
    out_dir.mkdir(parents=True, exist_ok=True)
    state_f = out_dir / 'runs.json'
    state = json.loads(state_f.read_text()) if state_f.exists() else []
    for fold, seed in runs:
        f = out_dir / f'fold{fold}_seed{seed}.npz'
        if f.exists() and any((r['fold'], r['seed']) == (fold, seed) for r in state):
            continue
        t0 = time.time()
        val_bearing = FOLDS[fold]
        cfg = dict(XJTU_CONFIG)
        cfg.update({'val_mode': 'bearing', 'val_bearings': [val_bearing], 'epochs': k, 'patience': 10 ** 6,
                    'mc_dropout_passes': passes, 'skip_test': True, 'test_bearings': []})
        net, val_history, cfg = train_exp19_final(model, seed, cfg, device)
        a = evaluate_on_test(net, xjtu([val_bearing]), cfg, device)['_arrays']
        np.savez_compressed(f, **{f'{tag}_{val_bearing}': a[key][val_bearing]
                                  for key, tag in (('raw_p', 'raw'), ('tgt', 'tgt'), ('mc_mean', 'mcmean'), ('mc_std', 'mcstd'))})
        orig = ref[(fold, seed)]
        state = [r for r in state if (r['fold'], r['seed']) != (fold, seed)]
        state.append({'fold': fold, 'seed': seed, 'final_val_mae': val_history[-1], 'val_history': val_history,
                      'original_final_val_mae': orig['final_val_mae'],
                      'replay_abs_diff': abs(val_history[-1] - orig['final_val_mae']),
                      'history_identical': val_history == orig['val_history'],
                      'saved_prediction_mae': pooled_mae(f), 'minutes': round((time.time() - t0) / 60, 2)})
        state_f.write_text(json.dumps(state, indent=1))
        r = state[-1]
        print(f"{model} fold {fold} seed {seed}: final val MAE {r['final_val_mae']:.5f} (E6 {r['original_final_val_mae']:.5f}), "
              f"saved predictions {r['saved_prediction_mae']:.5f} ({r['minutes']} min)", flush=True)
        if r['replay_abs_diff'] > TOL_REPLAY or not r['history_identical']:
            raise SystemExit(f'V7 {model} fold {fold} seed {seed}: the repeated run does not reproduce E6; stopping (Amendment 7)')
    consistency = max(abs(r['saved_prediction_mae'] - r['final_val_mae']) for r in state)
    if consistency > 1e-4:
        raise SystemExit(f'V7 {model}: saved validation predictions do not match the final-epoch validation MAE')
    write_meta(out_dir, {'stage': 'V7', 'model': model, 'K': k, 'mc_dropout_passes': passes, 'n_runs': len(state),
                         'replay_max_abs_diff': max(r['replay_abs_diff'] for r in state),
                         'all_histories_identical': all(r['history_identical'] for r in state),
                         'max_abs_diff_saved_predictions_vs_val_history': consistency})
    print(f"{model}: V7 validation predictions complete ({len(state)} runs, K = {k}, replay max |diff| "
          f"{max(r['replay_abs_diff'] for r in state):.1e})", flush=True)


# ---------------------------------------------------------------- smoke -------------------------
def smoke(device):
    """Amendment 7 smoke tests: early stopping one seed, two epochs, each architecture; T7a to T7c one run each,
    two passes; V7 one full-length run (CNN-LSTM, fold A, seed 7) to test the replay. Separate folder."""
    out = SMOKE_OUT
    cmd = [sys.executable, str(ROOT / 'Experiments' / 'exp17_xjtu.py'), '--seed', '42', '--epochs', '2',
           '--val-mode', 'window', '--patience', str(PATIENCE), '--deterministic', '--skip-test', '--skip-zeroshot',
           '--output-root', str((out / 'es').relative_to(ROOT)), '--variant-name', 'ours']
    if not (out / 'es' / 'ours' / 'results.json').exists():
        subprocess.run(cmd, check=True, cwd=str(ROOT))
    for m in BASELINES:
        es(m, device, out, [42], 2)
    for m in MODELS:
        t7a(m, device, out, 2, [42])
        t7b(m, device, out, 2, [42])
        t7c(m, device, out, 2, [42])
    v7('cnnlstm', device, out, 2, [('A', 7)])
    print('Amendment 7 part A smoke tests passed', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['es', 't7a', 't7b', 't7c', 'v7'])
    ap.add_argument('--model')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        smoke(device)
        return 0
    if args.stage in ('es', 'v7') and args.model not in BASELINES:
        raise SystemExit(f'--model must be one of {BASELINES} for --stage {args.stage}')
    if args.stage in ('t7a', 't7b', 't7c') and args.model not in MODELS:
        raise SystemExit(f'--model must be one of {MODELS}')
    if args.stage == 'es':
        es(args.model, device, OUT, SEEDS, MAX_EPOCHS)
    elif args.stage == 't7a':
        t7a(args.model, device, OUT, MC_PASSES, SEEDS)
    elif args.stage == 't7b':
        t7b(args.model, device, OUT, MC_PASSES, SEEDS)
    elif args.stage == 't7c':
        t7c(args.model, device, OUT, MC_PASSES, SEEDS)
    elif args.stage == 'v7':
        v7(args.model, device, OUT, MC_PASSES, [(f, s) for f in FOLDS for s in SELECT_SEEDS])
    else:
        raise SystemExit('give --stage or --smoke')
    return 0


if __name__ == '__main__':
    sys.exit(main())
