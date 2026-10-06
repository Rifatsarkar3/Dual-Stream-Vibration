#!/usr/bin/env bash
# Revision R1, analysis plan Amendment 3, V3: final-epoch validation predictions of the seven
# ablations of the submitted architecture, from the four-fold selection runs of Section 3.2 at
# the selected setting (B = 10, lambda 0.08). Settings of r1_lambda_lobo4.sh plus the ablation
# flags of r1_final_B10.sh. Selection runs never load test bearings (--skip-test), save
# final-epoch validation predictions (30 MC-dropout passes, exp16's default for validation)
# and use deterministic CUDA kernels. Resumable: finished runs are skipped.
#   bash Experiments/r1_ablation_valpreds.sh smoke    # one ablation, fold A, seed 42, 1 epoch
#   bash Experiments/r1_ablation_valpreds.sh replay   # full model, fold A, five seeds (replay check)
#   bash Experiments/r1_ablation_valpreds.sh all      # seven ablations x four folds x five seeds
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
MODE=${1:-all}
B=10
LAM=0.08
ABLATIONS="abl20_static abl20_temporal abl20_spectral abl20_no_pgmc abl20_no_dann abl20_neither abl20_pgmc_soft"

val_of() {
  case $1 in
    A) echo train_02_Bearing1_2 ;;
    B) echo train_04_Bearing2_2 ;;
    C) echo train_01_Bearing1_1 ;;
    D) echo train_03_Bearing2_1 ;;
  esac
}

flags_of() {  # as r1_final_B10.sh; aux-seq is passed explicitly, as in r1_lambda_lobo4.sh
  case $1 in
    full)            echo "--pgmc-mode aux-seq" ;;
    abl20_static)    echo "--pgmc-mode aux-seq --static-gate" ;;
    abl20_temporal)  echo "--pgmc-mode aux-seq --stream temporal" ;;
    abl20_spectral)  echo "--pgmc-mode aux-seq --stream spectral" ;;
    abl20_no_pgmc)   echo "--pgmc-mode aux-seq --no-pgmc" ;;
    abl20_no_dann)   echo "--pgmc-mode aux-seq --no-dann" ;;
    abl20_neither)   echo "--pgmc-mode aux-seq --no-pgmc --no-dann" ;;
    abl20_pgmc_soft) echo "--pgmc-mode soft" ;;
  esac
}

run() {  # output root, run name, fold, extra flags (later flags override earlier ones)
  local root=$1 name=$2 fold=$3; shift 3
  if [ -f "$root/$name/results.json" ]; then echo "skip $name"; return; fi
  echo "=== $name (val $(val_of "$fold")) ==="
  $PY Experiments/exp16_ress.py --all-seeds --val-mode bearing --val-bearings "$(val_of "$fold")" \
    --epochs "$B" --proportional-warmup --mc-passes 5 --patience 999 --lambda-pgmc "$LAM" \
    --skip-test --save-val-preds --deterministic --data-dir Processed_PRONOSTIA_v2 \
    --output-root "$root" --variant-name "$name" "$@"
}

case $MODE in
  smoke)
    # shellcheck disable=SC2046
    run outputs/r1_ablation_val_smoke abl20_static_foldA A $(flags_of abl20_static) --epochs 1 --seeds 42 ;;
  replay)
    # shellcheck disable=SC2046
    run outputs/r1_ablation_val_replay full_foldA A $(flags_of full) ;;
  all)
    for ABL in $ABLATIONS; do
      for FOLD in A B C D; do
        # shellcheck disable=SC2046
        run outputs/r1_ablation_val "${ABL}_fold${FOLD}" "$FOLD" $(flags_of "$ABL")
      done
    done ;;
  *) echo "usage: $0 smoke|replay|all"; exit 2 ;;
esac
echo "=== r1_ablation_valpreds $MODE done ==="
