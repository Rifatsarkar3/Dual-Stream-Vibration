"""
r1_baseline_mc30.py — Revision R1, analysis plan Amendment 4, T4: test predictions of the six
PRONOSTIA baselines with dropout at 30 MC-dropout passes, the number used by our model's final
runs (the baselines' final runs used 10). Inference only: each baseline's 20 final checkpoints are
re-evaluated on the thirteen bearings with exp16_ress.evaluate_on_test (the evaluation function of
every final run), seed set before each evaluation, deterministic kernels. MSCNN gets its
five-recording history and its selected input normalisation, as in r1_mscnn.py.
Check: the deterministic predictions must reproduce the stored ones used in Amendment 2
(r1_strict_stats.baseline_paths) within 1e-5 and the targets exactly; otherwise the run stops.

USAGE:
  python Experiments/r1_baseline_mc30.py --model cnn     # cnn, bilstm, cnnlstm, resnet, transformer or mscnn
  python Experiments/r1_baseline_mc30.py --smoke         # one checkpoint each of cnn and mscnn, 2 passes, separate folder
"""

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # before the first cuBLAS call

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
from Experiments.exp16_ress import CONFIG, evaluate_on_test, load_bearings  # noqa: E402
from Experiments.exp19_baselines import MODELS, BaselineAdapter  # noqa: E402
from Experiments.r1_reeval import EVAL_BEARINGS  # noqa: E402
import r1_mscnn as rm  # noqa: E402
from r1_provenance import provenance  # noqa: E402
from r1_strict_stats import baseline_paths  # noqa: E402
from src.mscnn_li2019 import MSCNN  # noqa: E402

DATA = ROOT / 'Processed_PRONOSTIA_v2'
# short name -> (checkpoint folder, label in r1_strict_stats.baseline_paths)
SOURCES = {'cnn': ('outputs/exp19_baselines20/cnn', '1D CNN'),
           'bilstm': ('outputs/r1_baselines_final5/bilstm', 'BiLSTM'),
           'cnnlstm': ('outputs/r1_baselines_final5/cnnlstm', 'CNN-LSTM'),
           'resnet': ('outputs/exp19_baselines20/resnet', '1D ResNet'),
           'transformer': ('outputs/r1_baselines_final5/transformer', 'Transformer'),
           'mscnn': ('outputs/r1_mscnn/final', 'MSCNN [7]')}
MC_PASSES = 30
N_SEEDS = 20
TOL = 1e-5
SEED_RE = re.compile(r'model_seed(\d+)\.pth$')


def build(short, device):
    if short == 'mscnn':
        sel = json.loads((ROOT / 'outputs' / 'r1_mscnn' / 'selection.json').read_text(encoding='utf-8'))
        return BaselineAdapter(MSCNN(n_hist=rm.N_HIST, input_norm=sel['input_norm'])).to(device)
    return BaselineAdapter(MODELS[short](), needs_spectrogram=(short == 'transformer')).to(device)


def rescore(short, out_dir, passes, device, limit=None):
    ckpt_dir, label = SOURCES[short]
    stored = ROOT / baseline_paths()[label]
    ckpts = sorted((ROOT / ckpt_dir).glob('model_seed*.pth'), key=lambda p: int(SEED_RE.search(p.name).group(1)))
    if limit is None and len(ckpts) != N_SEEDS:
        raise SystemExit(f'{short}: expected {N_SEEDS} checkpoints in {ckpt_dir}, found {len(ckpts)}')
    ckpts = ckpts[:limit] if limit else ckpts
    bearings = load_bearings(str(DATA), EVAL_BEARINGS)
    if short == 'mscnn':
        bearings = rm.with_history(bearings)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(CONFIG, mc_dropout_passes=passes)
    max_diff, t0 = 0.0, time.time()
    for ck in ckpts:
        seed = int(SEED_RE.search(ck.name).group(1))
        out_f = out_dir / f'preds_seed{seed}.npz'
        if not out_f.exists():
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build(short, device)
            model.load_state_dict(torch.load(ck, map_location=device))
            a = evaluate_on_test(model, bearings, cfg, device)['_arrays']
            np.savez_compressed(
                out_f,
                **{f'raw_{n}': v for n, v in a['raw_p'].items()},
                **{f'tgt_{n}': v for n, v in a['tgt'].items()},
                **{f'alpha_{n}': v for n, v in a['alphas'].items()},
                **{f'mcmean_{n}': v for n, v in a['mc_mean'].items()},
                **{f'mcstd_{n}': v for n, v in a['mc_std'].items()})
        new, old = np.load(out_f), np.load(stored / f'preds_seed{seed}.npz')
        for b in EVAL_BEARINGS:
            if not np.array_equal(new[f'tgt_{b}'], old[f'tgt_{b}']):
                raise SystemExit(f'{short} seed {seed} {b}: targets differ from the stored predictions')
            max_diff = max(max_diff, float(np.max(np.abs(new[f'raw_{b}'].astype(float) - old[f'raw_{b}'].astype(float)))))
        if max_diff > TOL:
            raise SystemExit(f'{short} seed {seed}: deterministic predictions differ from the stored ones by '
                             f'{max_diff:.2e} > {TOL:g}; stopping (Amendment 4, T4)')
    meta = {'model': short, 'source_checkpoints': ckpt_dir, 'stored_predictions': str(stored.relative_to(ROOT)),
            'n_checkpoints': len(ckpts), 'mc_dropout_passes': passes, 'raw_check_max_abs_diff': max_diff,
            'tolerance': TOL, 'bearings': EVAL_BEARINGS, 'date': time.strftime('%Y-%m-%d %H:%M'),
            'provenance': provenance(('Experiments/r1_baseline_mc30.py', 'Experiments/r1_mscnn.py',
                                      'src/mscnn_li2019.py'))}
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(f'{short}: {passes}-pass test predictions complete ({len(ckpts)} checkpoints, deterministic predictions '
          f'within {max_diff:.2e} of the stored ones, {time.time() - t0:.0f} s)', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=sorted(SOURCES))
    ap.add_argument('--smoke', action='store_true', help='one checkpoint each of cnn and mscnn, 2 passes, separate folder')
    ap.add_argument('--out-root', default=str(ROOT / 'outputs' / 'r1_baseline_mc30'))
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        for short in ('cnn', 'mscnn'):
            rescore(short, ROOT / 'outputs' / 'r1_baseline_mc30_smoke' / short, 2, device, limit=1)
        return 0
    if not args.model:
        raise SystemExit('give --model or --smoke')
    rescore(args.model, Path(args.out_root) / args.model, MC_PASSES, device)
    return 0


if __name__ == '__main__':
    sys.exit(main())
