#!/usr/bin/env bash
# Revision R1, analysis plan Section 7, candidate B2 (temporal context, K = 5): the lambda
# rule of Section 3.2 at the variant's selected budget B* (budget_selection.json written by
# r1_select_budget.py). The lambda = 0.08 runs at B* come from r1_temporal_sweep.sh; this adds
# lambda in {0.2, 0.5, 1.0, 2.0, 5.0} with identical settings.
# Usage: bash Experiments/r1_temporal_lambda.sh <B*>
# Selection: python Experiments/r1_select_lambda.py --root outputs/r1_temporal --budget <B*> --no-repro-check
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
B=${1:?usage: r1_temporal_lambda.sh <B*>}

for LAM in 0.2 0.5 1.0 2.0 5.0; do
  for FOLD in A B C D; do
    case $FOLD in
      A) VAL=train_02_Bearing1_2 ;;
      B) VAL=train_04_Bearing2_2 ;;
      C) VAL=train_01_Bearing1_1 ;;
      D) VAL=train_03_Bearing2_1 ;;
    esac
    NAME="lam${LAM}_B${B}_fold${FOLD}"
    if [ -f "outputs/r1_temporal/${NAME}/results.json" ]; then continue; fi
    echo "=== temporal K=5 lambda $LAM budget $B fold $FOLD (val $VAL) ==="
    $PY Experiments/r1_temporal.py --all-seeds --pgmc-mode aux-seq --val-mode bearing \
      --val-bearings "$VAL" --epochs "$B" --proportional-warmup --mc-passes 5 \
      --patience 999 --lambda-pgmc "$LAM" --skip-test --save-val-preds --deterministic \
      --data-dir Processed_PRONOSTIA_v2 --output-root outputs/r1_temporal --variant-name "$NAME"
  done
done
echo "=== r1_temporal_lambda done ==="
