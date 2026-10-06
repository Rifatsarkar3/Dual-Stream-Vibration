#!/usr/bin/env bash
# Revision R1, analysis plan Section 7, candidate B2 adopted: the Section 3 procedure for the
# temporal-context variant on the XJTU-SY folds (validation only; test and zero-shot bearings
# are not loaded). Settings as in r1_xjtu_lobo4.sh plus --skip-test and --deterministic.
#   stage "budget":       B in {5, 10, 20, 40} at lambda = 0.08, 4 folds x 5 selection seeds
#   stage "lambda <B*>":  lambda in {0.2, 0.5, 1.0, 2.0, 5.0} at the selected budget
# Budget: python Experiments/r1_select_budget.py --root outputs/r1_temporal_xjtu --no-adoption
# Lambda: python Experiments/r1_select_lambda.py --root outputs/r1_temporal_xjtu --budget <B*> --no-repro-check
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
OUT=outputs/r1_temporal_xjtu
STAGE=${1:?usage: r1_temporal_xjtu_sweep.sh budget | lambda <B*>}

run() {  # lambda, budget, fold
  local LAM=$1 B=$2 FOLD=$3 VAL
  case $FOLD in
    A) VAL=train_02_Bearing1_2 ;;
    B) VAL=train_04_Bearing2_2 ;;
    C) VAL=train_01_Bearing1_1 ;;
    D) VAL=train_03_Bearing2_1 ;;
  esac
  local NAME="lam${LAM}_B${B}_fold${FOLD}"
  if [ -f "$OUT/${NAME}/results.json" ]; then return; fi
  echo "=== XJTU temporal K=5 lambda $LAM budget $B fold $FOLD (val $VAL) ==="
  $PY Experiments/r1_temporal_xjtu.py --all-seeds --pgmc-mode aux-seq --val-mode bearing \
    --val-bearings "$VAL" --epochs "$B" --proportional-warmup --mc-passes 5 \
    --patience 999 --lambda-pgmc "$LAM" --skip-zeroshot --skip-test --deterministic \
    --output-root "$OUT" --variant-name "$NAME"
}

case $STAGE in
  budget)
    for B in 5 10 20 40; do for FOLD in A B C D; do run 0.08 "$B" "$FOLD"; done; done ;;
  lambda)
    BSTAR=${2:?usage: r1_temporal_xjtu_sweep.sh lambda <B*>}
    for LAM in 0.2 0.5 1.0 2.0 5.0; do for FOLD in A B C D; do run "$LAM" "$BSTAR" "$FOLD"; done; done ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "=== r1_temporal_xjtu_sweep $STAGE done ==="
