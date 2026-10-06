#!/usr/bin/env bash
# Revision R1, analysis plan Section 7, candidate B2 (temporal context, K = 5): budget
# sweep B in {5, 10, 20, 40} at lambda = 0.08 on the four leave-one-bearing-out folds,
# five selection seeds each; identical settings to the Section 3 selection runs.
# Run names lam0.08_B{B}_fold{F} let r1_select_lambda.py reuse the selected budget's runs.
# Budget: python Experiments/r1_select_budget.py --root outputs/r1_temporal
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
LAM=0.08

for B in 10 5 20 40; do
  for FOLD in A B C D; do
    case $FOLD in
      A) VAL=train_02_Bearing1_2 ;;
      B) VAL=train_04_Bearing2_2 ;;
      C) VAL=train_01_Bearing1_1 ;;
      D) VAL=train_03_Bearing2_1 ;;
    esac
    NAME="lam${LAM}_B${B}_fold${FOLD}"
    if [ -f "outputs/r1_temporal/${NAME}/results.json" ]; then continue; fi
    echo "=== temporal K=5 budget $B fold $FOLD (val $VAL) ==="
    $PY Experiments/r1_temporal.py --all-seeds --pgmc-mode aux-seq --val-mode bearing \
      --val-bearings "$VAL" --epochs "$B" --proportional-warmup --mc-passes 5 \
      --patience 999 --lambda-pgmc "$LAM" --skip-test --save-val-preds --deterministic \
      --data-dir Processed_PRONOSTIA_v2 --output-root outputs/r1_temporal --variant-name "$NAME"
  done
done
echo "=== r1_temporal_sweep done ==="
