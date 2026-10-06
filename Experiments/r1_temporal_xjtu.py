"""
r1_temporal_xjtu.py — Revision R1, analysis plan Section 7 (B2), XJTU-SY part: the
temporal-context variant (src/temporal_context.py, K = 5) run through
Experiments/exp17_xjtu.py.

Inside this process only, the exp16 factories are replaced exactly as in
Experiments/r1_temporal.py (every window is stacked with its K-1 causal predecessors; the
model is TemporalContext(exp16 model, K)), and exp17's zero-shot loader uses the same
history. Every command-line option is that of exp17_xjtu.py.

USAGE (same options as exp17_xjtu.py):
  python Experiments/r1_temporal_xjtu.py --all-seeds --pgmc-mode aux-seq --val-mode bearing \
      --val-bearings train_02_Bearing1_2 --epochs 5 --proportional-warmup --mc-passes 5 \
      --patience 999 --skip-zeroshot --skip-test --deterministic \
      --output-root outputs/r1_temporal_xjtu --variant-name lam0.08_B5_foldA
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import Experiments.r1_temporal as rt  # noqa: E402  (replaces the exp16 factories)
import Experiments.exp17_xjtu as e17  # noqa: E402

e17.load_bearings = rt.load_bearings_with_history
e17.EXTRA_CODE = ('Experiments/r1_temporal.py', 'src/temporal_context.py',
                  'Experiments/r1_temporal_xjtu.py')
e17.PROVENANCE_EXTRA = {'temporal_context_k': rt.K}

if __name__ == '__main__':
    sys.exit(e17.main())
