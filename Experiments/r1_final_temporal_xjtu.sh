#!/usr/bin/env bash
# Revision R1, analysis plan Section 7 (B2 adopted): XJTU-SY final runs of the temporal-context
# variant at the setting selected on the XJTU validation folds (read from budget_selection.json
# and selection.json under outputs/r1_temporal_xjtu, never typed). Settings of the submitted XJTU
# final (outputs/exp17_xjtu/final20): 20 seeds, aux-seq PGMC, proportional warm-ups, no validation,
# 10 MC passes, 6 in-domain test and 5 Condition-3 zero-shot bearings; plus temporal context
# (strided window history, Experiments/r1_temporal_xjtu_final.py) and deterministic kernels.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
SEL=outputs/r1_temporal_xjtu
OUT=outputs/r1_final_temporal_xjtu
B=$($PY -c "import json; print(json.load(open('$SEL/budget_selection.json'))['selected'])")
LAM=$($PY -c "import json; print(json.load(open('$SEL/selection.json'))['selected'])")
SEEDS=42,123,777,999,7,8,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946
if [ -f "$OUT/final20/results.json" ]; then echo "skip final20"; exit 0; fi
echo "=== XJTU temporal final: B=$B lambda=$LAM ==="
$PY Experiments/r1_temporal_xjtu_final.py --all-seeds --seeds "$SEEDS" --pgmc-mode aux-seq \
  --val-mode none --epochs "$B" --lambda-pgmc "$LAM" --proportional-warmup --mc-passes 10 \
  --deterministic --output-root "$OUT" --variant-name final20
echo "=== r1_final_temporal_xjtu done ==="
