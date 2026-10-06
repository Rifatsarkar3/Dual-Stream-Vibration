"""
r1_reeval_xjtu.py — Revision R1: re-evaluate the saved XJTU-SY baseline checkpoints
(outputs/exp19_xjtu/{cnn,bilstm,cnnlstm}, 5 seeds each, as reported in the submission).
Inference only, no training.

The submission saved only per-bearing raw and PAVA metrics for these baselines. The plan's
endpoint is the bearing-averaged CES MAE, which needs the prediction sequences, so the
checkpoints are evaluated again with the settings of their final stage in exp19
(XJTU_CONFIG, 10 MC-dropout passes, model seeded with its own seed) on
  the six XJTU-SY test bearings (as in the submission) and
  the five Condition-3 bearings (zero-shot; not evaluated for the baselines in the submission).
Output layout: as exp16/exp17 preds_seed*.npz (raw_, tgt_, alpha_, mcmean_, mcstd_ per bearing),
one file per seed holding both bearing sets. Check: the re-evaluated raw MAE on each test
bearing must match the per-bearing raw MAE stored in the baseline's results.json.

USAGE:
  python Experiments/r1_reeval_xjtu.py [--which cnn,bilstm,cnnlstm]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Experiments'))
from Experiments.exp16_ress import evaluate_on_test, load_bearings  # noqa: E402
from Experiments.exp17_xjtu import XJTU_CONFIG  # noqa: E402
from Experiments.exp19_baselines import MODELS, BaselineAdapter  # noqa: E402
from r1_provenance import provenance  # noqa: E402

SOURCE = 'outputs/exp19_xjtu'
MC_PASSES = 10                      # exp19 final stage
SEED_RE = re.compile(r'model_seed(\d+)\.pth$')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', default='cnn,bilstm,cnnlstm')
    ap.add_argument('--out-root', default=str(ROOT / 'outputs' / 'r1_reeval_xjtu'))
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = dict(XJTU_CONFIG, mc_dropout_passes=MC_PASSES)
    test = load_bearings(cfg['data_dir'], cfg['test_bearings'])
    zeroshot = load_bearings(cfg['data_dir'], cfg['zeroshot_bearings'])

    for name in [s.strip() for s in args.which.split(',')]:
        src = ROOT / SOURCE / name
        stored = {r['seed']: r for r in json.loads((src / 'results.json').read_text(encoding='utf-8'))['per_seed']}
        out_dir = Path(args.out_root) / f'baseline_{name}'
        out_dir.mkdir(parents=True, exist_ok=True)
        t0, max_diff = time.time(), 0.0
        ckpts = sorted(src.glob('model_seed*.pth'))
        for ck in ckpts:
            seed = int(SEED_RE.search(ck.name).group(1))
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = BaselineAdapter(MODELS[name](), needs_spectrogram=(name == 'transformer')).to(device)
            model.load_state_dict(torch.load(ck, map_location=device))
            arrays = {}
            for bearings in (test, zeroshot):
                a = evaluate_on_test(model, bearings, cfg, device)['_arrays']
                for key, tag in (('raw_p', 'raw'), ('tgt', 'tgt'), ('alphas', 'alpha'),
                                 ('mc_mean', 'mcmean'), ('mc_std', 'mcstd')):
                    arrays.update({f'{tag}_{b}': v for b, v in a[key].items()})
            for b in cfg['test_bearings']:
                mae = float(np.mean(np.abs(arrays[f'raw_{b}'] - arrays[f'tgt_{b}'])))
                max_diff = max(max_diff, abs(mae - stored[seed]['per_bearing'][b]['raw']['mae']))
            np.savez_compressed(out_dir / f'preds_seed{seed}.npz', **arrays)
        meta = {'source_checkpoints': f'{SOURCE}/{name}', 'n_checkpoints': len(ckpts),
                'mc_dropout_passes': MC_PASSES, 'data_dir': cfg['data_dir'],
                'test_bearings': cfg['test_bearings'], 'zeroshot_bearings': cfg['zeroshot_bearings'],
                'check_max_abs_diff_test_raw_mae_vs_results_json': max_diff,
                'provenance': provenance(('Experiments/r1_reeval_xjtu.py', 'Experiments/exp17_xjtu.py'))}
        (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
        print(f'{name:8s} {len(ckpts)} checkpoints  {time.time() - t0:6.1f}s  '
              f'max |raw MAE - stored| on test bearings = {max_diff:.2e}', flush=True)
        if max_diff > 1e-4:
            raise SystemExit(f'{name}: re-evaluation does not reproduce the stored results')
    return 0


if __name__ == '__main__':
    sys.exit(main())
