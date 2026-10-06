#!/usr/bin/env bash
# Revision R1, analysis plan Amendment 1, E5: the four-fold XJTU-SY selection runs at the selected
# setting (B 5, lambda 0.08, seeds 42, 123, 777, 999, 7; settings of the original sweeps) repeated
# with deterministic kernels and final-epoch MC-dropout validation predictions, for both versions
# of our model. Test bearings are never loaded.
#   bash Experiments/r1_xjtu_calib_runs.sh smoke    # fold C, seed 42, 1 epoch, each version
#   bash Experiments/r1_xjtu_calib_runs.sh full
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
STAGE=${1:?usage: r1_xjtu_calib_runs.sh smoke|full}

run() {  # version, fold, out root, extra args
  local VERSION=$1 FOLD=$2 OUT=$3; shift 3
  local VAL SCRIPT
  case $FOLD in
    A) VAL=train_02_Bearing1_2 ;;
    B) VAL=train_04_Bearing2_2 ;;
    C) VAL=train_01_Bearing1_1 ;;
    D) VAL=train_03_Bearing2_1 ;;
  esac
  case $VERSION in
    temporal) SCRIPT=Experiments/r1_temporal_xjtu.py ;;
    submitted) SCRIPT=Experiments/exp17_xjtu.py ;;
  esac
  local NAME="lam0.08_B5_fold${FOLD}"
  if [ -f "$OUT/$VERSION/${NAME}/results.json" ]; then return; fi
  echo "=== E5 $VERSION fold $FOLD (val $VAL) ==="
  $PY "$SCRIPT" --pgmc-mode aux-seq --val-mode bearing --val-bearings "$VAL" --proportional-warmup \
    --mc-passes 5 --patience 999 --lambda-pgmc 0.08 --skip-zeroshot --skip-test --deterministic \
    --save-val-preds --output-root "$OUT/$VERSION" --variant-name "$NAME" "$@"
}

case $STAGE in
  smoke)
    for V in temporal submitted; do run "$V" C outputs/r1_xjtu_calib_smoke --seed 42 --epochs 1; done ;;
  full)
    for V in temporal submitted; do for F in A B C D; do run "$V" "$F" outputs/r1_xjtu_calib --all-seeds --epochs 5; done; done ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "=== r1_xjtu_calib_runs $STAGE done ==="
