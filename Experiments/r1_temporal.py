"""
r1_temporal.py — Revision R1, analysis plan Section 7, candidate B2: DSAFLite with temporal
context (src/temporal_context.py; the K = 5 most recent windows, fused features averaged).

Training, validation, PGMC, DANN, evaluation, outputs and every command-line option are
exactly those of Experiments/exp16_ress.py. Inside this process only, two exp16 factories
are replaced:
  load_bearings -> each window stacked with its K-1 predecessors (causal; the first
                   window of a bearing is repeated), shape (N, K, 2560)
  make_model    -> TemporalContext(exp16 model, K)   (no new parameters)
and the provenance block also fingerprints this file and the wrapper. exp16 itself is
not modified, so runs in progress elsewhere are unaffected.

USAGE (same options as exp16_ress.py):
  python Experiments/r1_temporal.py --all-seeds --val-mode bearing --val-bearings ... \
      --epochs 10 --deterministic --skip-test --save-val-preds --output-root ... --variant-name ...
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import Experiments.exp16_ress as e16  # noqa: E402
from Experiments.r1_provenance import provenance as _provenance  # noqa: E402
from src.temporal_context import TemporalContext  # noqa: E402

K = 5
_load_bearings, _make_model = e16.load_bearings, e16.make_model


def load_bearings_with_history(data_dir, patterns, channels='h'):
    out = []
    for b in _load_bearings(data_dir, patterns, channels):
        idx = np.clip(np.arange(b.n)[:, None] - np.arange(K - 1, -1, -1)[None, :], 0, None)
        out.append(e16.BearingArrays(b.name, b.vib[idx], b.rul, b.condition))
    return out


def make_temporal_model(config, device, static_gate=False, stream='dual'):
    base = _make_model(config, device, static_gate=static_gate, stream=stream)
    return TemporalContext(base, K).to(device)


e16.load_bearings = load_bearings_with_history
e16.make_model = make_temporal_model
e16.provenance = lambda: {**_provenance(extra_code=('Experiments/r1_temporal.py',
                                                    'src/temporal_context.py')),
                          'temporal_context_k': K}

if __name__ == '__main__':
    sys.exit(e16.main())
