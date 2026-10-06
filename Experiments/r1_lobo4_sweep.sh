#!/usr/bin/env bash
# Revision R1, comment 2: complete the 4-fold leave-one-bearing-out budget sweep.
# Folds A (val Bearing1_2) and B (val Bearing2_2) are the published runs in
# outputs/exp16_ress/sweepseq_B*_fold{A,B}; their validation data are bitwise
# identical in Processed_PRONOSTIA_v2. This script adds folds C and D with the
# same settings (aux-seq PGMC, lambda 0.08, proportional warm-ups, 5 seeds,
# no early stopping, 5 MC passes).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

for FOLD in C D; do
  case $FOLD in
    C) VAL=train_01_Bearing1_1 ;;
    D) VAL=train_03_Bearing2_1 ;;
  esac
  for B in 5 10 20 40; do
    echo "=== fold $FOLD (val $VAL) budget $B ==="
    $PY Experiments/exp16_ress.py --all-seeds --val-mode bearing --val-bearings "$VAL" \
      --epochs "$B" --proportional-warmup --mc-passes 5 --patience 999 \
      --data-dir Processed_PRONOSTIA_v2 --output-root outputs/r1_lobo \
      --variant-name "sweepseq_B${B}_fold${FOLD}"
  done
done
echo "=== r1_lobo4_sweep done ==="
