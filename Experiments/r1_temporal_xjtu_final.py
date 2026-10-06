"""
r1_temporal_xjtu_final.py — Revision R1, analysis plan Section 7 (B2 adopted), XJTU-SY final
runs of the temporal-context variant (K = 5) through Experiments/exp17_xjtu.py.

Same model and the same windows as Experiments/r1_temporal_xjtu.py; only the memory layout
of the window history differs. r1_temporal.py materializes each window with its K-1
predecessors (K copies of every window, and BearingArrays copies once more). For the five
Condition-3 zero-shot bearings (84,408 windows) that needs about 4.3 GB per copy. Here the
history is a read-only strided view of the bearing padded with K-1 copies of its first
window: view[i, j] = window max(i + j - (K - 1), 0), identical to r1_temporal.py's
idx[i, j] = max(i - (K - 1 - j), 0). tests/test_temporal_history_view.py checks the
equivalence.

USAGE (same options as exp17_xjtu.py):
  python Experiments/r1_temporal_xjtu_final.py --all-seeds --seeds 42,...,10946 --pgmc-mode aux-seq \
      --val-mode none --epochs <B*> --lambda-pgmc <lambda*> --proportional-warmup --mc-passes 10 \
      --deterministic --output-root outputs/r1_final_temporal_xjtu --variant-name final20
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import Experiments.r1_temporal as rt  # noqa: E402  (replaces the exp16 model factory)
import Experiments.exp16_ress as e16  # noqa: E402
import Experiments.exp17_xjtu as e17  # noqa: E402


def history_view(vib: np.ndarray, k: int) -> np.ndarray:
    """(N, L) windows -> read-only (N, k, L) view; row i holds windows i-k+1 .. i (causal),
    the first window repeated before the start of the bearing."""
    pad = np.ascontiguousarray(np.concatenate([np.repeat(vib[:1], k - 1, axis=0), vib]))
    s_row, s_col = pad.strides
    return np.lib.stride_tricks.as_strided(pad, shape=(len(vib), k, vib.shape[1]),
                                           strides=(s_row, s_row, s_col), writeable=False)


def load_bearings_with_history_view(data_dir, patterns, channels='h'):
    out = []
    for b in rt._load_bearings(data_dir, patterns, channels):
        arr = e16.BearingArrays.__new__(e16.BearingArrays)   # skip the copying constructor
        arr.name, arr.vib, arr.rul = b.name, history_view(b.vib, rt.K), b.rul
        arr.condition, arr.n = b.condition, b.n
        out.append(arr)
    return out


e16.load_bearings = load_bearings_with_history_view
e17.load_bearings = load_bearings_with_history_view
e17.EXTRA_CODE = ('Experiments/r1_temporal.py', 'src/temporal_context.py',
                  'Experiments/r1_temporal_xjtu_final.py')
e17.PROVENANCE_EXTRA = {'temporal_context_k': rt.K, 'window_history': 'strided view'}

if __name__ == '__main__':
    sys.exit(e17.main())
