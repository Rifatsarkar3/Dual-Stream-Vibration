"""
r1_ablation_mc30.py — Revision R1, analysis plan Amendment 3, T3: test predictions of the seven
ablations of the submitted architecture with 30 MC-dropout passes, the number used by the full
model's final runs (the ablations' final runs used 10). Inference only: the saved checkpoints of
outputs/r1_final_B10/abl20_* (20 seeds each) are re-evaluated on the thirteen bearings as
Experiments/r1_reeval.py does (seed set before each evaluation), with deterministic kernels.
Check: the deterministic predictions must reproduce the stored ones within 1e-5 (absolute) and
the targets exactly; otherwise the run stops.
--which final20 re-evaluates the full model's checkpoints the same way (sensitivity of T3).

USAGE:
  python Experiments/r1_ablation_mc30.py --which all          # the seven ablations
  python Experiments/r1_ablation_mc30.py --which final20      # sensitivity, full model
  python Experiments/r1_ablation_mc30.py --smoke              # one checkpoint, 2 passes, separate folder
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
from Experiments.exp16_ress import CONFIG, evaluate_on_test, load_bearings, make_model  # noqa: E402
from Experiments.r1_reeval import EVAL_BEARINGS  # noqa: E402
from r1_provenance import provenance  # noqa: E402

SRC = 'outputs/r1_final_B10'
# variant -> make_model kwargs (the other ablations differ in training only), as r1_reeval.py
VARIANTS = {'abl20_static': {'static_gate': True}, 'abl20_temporal': {'stream': 'temporal'},
            'abl20_spectral': {'stream': 'spectral'}, 'abl20_no_pgmc': {}, 'abl20_no_dann': {},
            'abl20_neither': {}, 'abl20_pgmc_soft': {}, 'final20': {}}
ABLATIONS = [v for v in VARIANTS if v != 'final20']
MC_PASSES = 30
N_SEEDS = 20
TOL = 1e-5
SEED_RE = re.compile(r'model_seed(\d+)\.pth$')


def reevaluate(name, bearings, out_dir, passes, device, limit=None):
    src = ROOT / SRC / name
    ckpts = sorted(src.glob('model_seed*.pth'), key=lambda p: int(SEED_RE.search(p.name).group(1)))
    if limit is None and len(ckpts) != N_SEEDS:
        raise SystemExit(f'{name}: expected {N_SEEDS} checkpoints in {src}, found {len(ckpts)}')
    ckpts = ckpts[:limit] if limit else ckpts
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(CONFIG, mc_dropout_passes=passes)
    max_diff, t0 = 0.0, time.time()
    for ck in ckpts:
        seed = int(SEED_RE.search(ck.name).group(1))
        out_f = out_dir / f'preds_seed{seed}.npz'
        if not out_f.exists():
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = make_model(CONFIG, device, **VARIANTS[name])
            model.load_state_dict(torch.load(ck, map_location=device))
            a = evaluate_on_test(model, bearings, cfg, device)['_arrays']
            np.savez_compressed(
                out_f,
                **{f'raw_{n}': v for n, v in a['raw_p'].items()},
                **{f'tgt_{n}': v for n, v in a['tgt'].items()},
                **{f'alpha_{n}': v for n, v in a['alphas'].items()},
                **{f'mcmean_{n}': v for n, v in a['mc_mean'].items()},
                **{f'mcstd_{n}': v for n, v in a['mc_std'].items()})
        new, old = np.load(out_f), np.load(src / f'preds_seed{seed}.npz')
        for b in EVAL_BEARINGS:
            if not np.array_equal(new[f'tgt_{b}'], old[f'tgt_{b}']):
                raise SystemExit(f'{name} seed {seed} {b}: targets differ from the stored predictions')
            max_diff = max(max_diff, float(np.max(np.abs(new[f'raw_{b}'].astype(float) - old[f'raw_{b}'].astype(float)))))
        if max_diff > TOL:
            raise SystemExit(f'{name} seed {seed}: deterministic predictions differ from the stored ones by '
                             f'{max_diff:.2e} > {TOL:g}; stopping (Amendment 3, T3)')
    meta = {'source_checkpoints': f'{SRC}/{name}', 'n_checkpoints': len(ckpts), 'mc_dropout_passes': passes,
            'raw_check_max_abs_diff': max_diff, 'tolerance': TOL, 'bearings': EVAL_BEARINGS,
            'date': time.strftime('%Y-%m-%d %H:%M'),
            'provenance': provenance(('Experiments/r1_ablation_mc30.py', 'Experiments/r1_reeval.py'))}
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(f'{name}: {passes}-pass test predictions complete ({len(ckpts)} checkpoints, deterministic predictions '
          f'within {max_diff:.2e} of the stored ones, {time.time() - t0:.0f} s)', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', default='all', help="'all' (the seven ablations), 'final20', or a comma list")
    ap.add_argument('--smoke', action='store_true', help='one checkpoint of abl20_static, 2 passes, separate folder')
    ap.add_argument('--data-dir', default=str(ROOT / 'Processed_PRONOSTIA_v2'))
    ap.add_argument('--out-root', default=str(ROOT / 'outputs' / 'r1_ablation_mc30'))
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bearings = load_bearings(args.data_dir, EVAL_BEARINGS)
    if args.smoke:
        reevaluate('abl20_static', bearings, ROOT / 'outputs' / 'r1_ablation_mc30_smoke' / 'abl20_static',
                   2, device, limit=1)
        return 0
    names = ABLATIONS if args.which == 'all' else [s.strip() for s in args.which.split(',')]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f'unknown variants: {unknown}')
    for name in names:
        reevaluate(name, bearings, Path(args.out_root) / name, MC_PASSES, device)
    return 0


if __name__ == '__main__':
    sys.exit(main())
