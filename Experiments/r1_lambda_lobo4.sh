#!/usr/bin/env bash
# Revision R1, analysis plan Section 3.2: lambda_PGMC sensitivity on the four
# leave-one-bearing-out folds at the selected budget B = 10, five selection seeds per fold.
# Selection runs never load test bearings (--skip-test), save final-epoch validation
# predictions for calibration and Track B (--save-val-preds), and use deterministic CUDA
# kernels (--deterministic). Resumable: finished runs are skipped.
# Selection: python Experiments/r1_select_lambda.py
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
B=10

for LAM in 0.08 0.2 0.5 1.0 2.0 5.0; do
  for FOLD in A B C D; do
    case $FOLD in
      A) VAL=train_02_Bearing1_2 ;;
      B) VAL=train_04_Bearing2_2 ;;
      C) VAL=train_01_Bearing1_1 ;;
      D) VAL=train_03_Bearing2_1 ;;
    esac
    NAME="lam${LAM}_B${B}_fold${FOLD}"
    if [ -f "outputs/r1_lambda/${NAME}/results.json" ]; then continue; fi
    echo "=== lambda $LAM fold $FOLD (val $VAL) ==="
    $PY Experiments/exp16_ress.py --all-seeds --pgmc-mode aux-seq --val-mode bearing \
      --val-bearings "$VAL" --epochs "$B" --proportional-warmup --mc-passes 5 \
      --patience 999 --lambda-pgmc "$LAM" --skip-test --save-val-preds --deterministic \
      --data-dir Processed_PRONOSTIA_v2 --output-root outputs/r1_lambda --variant-name "$NAME"
  done
done
echo "=== r1_lambda_lobo4 done ==="
