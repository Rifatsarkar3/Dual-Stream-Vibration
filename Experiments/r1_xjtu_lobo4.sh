#!/usr/bin/env bash
# Revision R1, comment 2 (XJTU-SY): 4-fold leave-one-bearing-out budget sweep with the
# model actually used for the XJTU-SY results (aux-seq PGMC). The published 2-fold sweep
# (outputs/exp17_xjtu/sweep_B*_fold{A,B}) used the soft PGMC variant, so all four folds
# are run here with identical settings: lambda 0.08, proportional warm-ups, 5 seeds,
# no early stopping, 5 MC passes. Validation bearings only; zero-shot bearings skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe

for FOLD in A B C D; do
  case $FOLD in
    A) VAL=train_02_Bearing1_2 ;;
    B) VAL=train_04_Bearing2_2 ;;
    C) VAL=train_01_Bearing1_1 ;;
    D) VAL=train_03_Bearing2_1 ;;
  esac
  for B in 5 10 20 40; do
    if [ -f "outputs/r1_xjtu_lobo/sweepseq_B${B}_fold${FOLD}/results.json" ]; then continue; fi
    echo "=== XJTU fold $FOLD (val $VAL) budget $B ==="
    $PY Experiments/exp17_xjtu.py --all-seeds --pgmc-mode aux-seq --val-mode bearing \
      --val-bearings "$VAL" --epochs "$B" --proportional-warmup --mc-passes 5 \
      --patience 999 --skip-zeroshot --output-root outputs/r1_xjtu_lobo \
      --variant-name "sweepseq_B${B}_fold${FOLD}"
  done
done
echo "=== r1_xjtu_lobo4 done ==="
