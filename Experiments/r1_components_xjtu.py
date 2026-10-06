"""
r1_components_xjtu.py — Revision R1, analysis plan Amendment 10, XJTU-SY part: the rebuilt
components (Experiments/r1_components.py) run through Experiments/exp17_xjtu.py, with the band
selected on the XJTU-SY training bearings (outputs/r1_components/band_xjtu.json).

exp17 always trains the full model, so this wrapper adds --ablation (none, temporal, spectral,
static, no_pgmc, no_dann), mapped to exp16's ablation arguments. Every other option is exp17's.

USAGE:
  python Experiments/r1_components_xjtu.py --all-seeds --pgmc-mode aux-seq --val-mode bearing \
      --val-bearings train_02_Bearing1_2 --epochs 5 --proportional-warmup --mc-passes 5 \
      --patience 999 --skip-zeroshot --skip-test --save-val-preds --deterministic \
      --output-root outputs/r1_components_xjtu --variant-name lam0.08_B5_foldA
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import Experiments.r1_components as rc  # noqa: E402
import Experiments.exp16_ress as e16  # noqa: E402
import Experiments.exp17_xjtu as e17  # noqa: E402

STREAM = {'temporal': 'temporal', 'spectral': 'spectral'}


def main():
    comp, rest = rc.parse_component_args(sys.argv[1:], ablation=True)
    sys.argv = [sys.argv[0]] + rest
    rc.configure(comp.envelope, comp.augment, comp.pgmc_pairs, 'xjtu', comp.band_json)
    abl = comp.ablation

    def train_one_seed(seed, config, use_dann=True, use_pgmc=True, static_gate=False,
                       epochs_override=None):
        return e16.train_one_seed(
            seed, config, use_dann=use_dann and abl != 'no_dann',
            use_pgmc=use_pgmc and abl != 'no_pgmc', static_gate=static_gate or abl == 'static',
            epochs_override=epochs_override, stream=STREAM.get(abl, 'dual'))

    e17.train_one_seed = train_one_seed
    e17.EXTRA_CODE = rc.CODE + ('Experiments/r1_components_xjtu.py',)
    e17.PROVENANCE_EXTRA = {'components_r2': rc.describe(), 'ablation': abl}
    print(f'[r1_components_xjtu] {rc.describe()} ablation={abl}', flush=True)
    return e17.main()


if __name__ == '__main__':
    sys.exit(main())
