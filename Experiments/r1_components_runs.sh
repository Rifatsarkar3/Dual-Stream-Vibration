#!/usr/bin/env bash
# Revision R1, analysis plan Amendment 10: runs of the rebuilt dual-stream network (R:
# Experiments/r1_components.py, all three fixes on) and its ablations. Stages, in the order of
# the amendment; every stage is resumable (finished runs are skipped). The selected budget and
# lambda are read from the selection files, never typed.
#   band           demodulation band per dataset, training bearings only
#   smoke          R, fold A, seed 42, 10 epochs, validation only
#   budget         R, B in {5, 10, 20, 40} at lambda 0.08, four folds x five selection seeds
#   lambda         R, lambda in {0.2, 0.5, 1.0, 2.0, 5.0} at B* (budget_selection.json)
#   valpreds       primary ablations of R at (B*, lambda*), four folds x five selection seeds
#                  (final-epoch validation predictions for their calibration)
#   final          R and its eight ablations, 20 final seeds, 13 PRONOSTIA bearings, 30 MC passes
#   xjtu-budget    R on the XJTU-SY folds, B in {5, 10, 20, 40} at lambda 0.08
#   xjtu-lambda    R on the XJTU-SY folds, lambda in {0.2, ..., 5.0} at the XJTU-SY B*
#   xjtu-valpreds  primary ablations at the XJTU-SY (B*, lambda*)
#   xjtu-final     R and its primary ablations, 20 final seeds, 6 test and 5 Condition-3
#                  bearings, 10 MC passes (as the submitted XJTU-SY final)
# Selection between stages:
#   python Experiments/r1_select_budget.py --root outputs/r1_components            (adoption rule)
#   python Experiments/r1_select_lambda.py --root outputs/r1_components --budget <B*> --no-repro-check
#   python Experiments/r1_select_budget.py --root outputs/r1_components_xjtu --no-adoption
#   python Experiments/r1_select_lambda.py --root outputs/r1_components_xjtu --budget <B*> --no-repro-check
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
STAGE=${1:?usage: r1_components_runs.sh band|smoke|budget|lambda|valpreds|final|xjtu-budget|xjtu-lambda|xjtu-valpreds|xjtu-final}
SEL=outputs/r1_components
SELX=outputs/r1_components_xjtu
SEEDS=42,123,777,999,7,8,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946
TEST=test_01_Bearing1_3,test_02_Bearing1_4,test_03_Bearing1_5,test_04_Bearing2_3,test_05_Bearing1_6,test_06_Bearing1_7,test_07_Bearing2_4,test_08_Bearing2_5,test_09_Bearing2_6,test_10_Bearing2_7,zs_01_Bearing3_1,zs_02_Bearing3_2,zs_03_Bearing3_3
PRIMARY="rawonly static no_pgmc"
ALL="full rawonly static no_pgmc old_env no_aug adjacent no_dann spectral"
SELECT_FLAGS="--all-seeds --pgmc-mode aux-seq --val-mode bearing --proportional-warmup --mc-passes 5 --patience 999 --skip-test --save-val-preds --deterministic"

val_of() {
  case $1 in
    A) echo train_02_Bearing1_2 ;;
    B) echo train_04_Bearing2_2 ;;
    C) echo train_01_Bearing1_1 ;;
    D) echo train_03_Bearing2_1 ;;
  esac
}

flags_of() {  # PRONOSTIA: exp16 ablation flags and the options that undo one fix
  case $1 in
    full)     echo "" ;;
    rawonly)  echo "--stream temporal" ;;
    static)   echo "--static-gate" ;;
    no_pgmc)  echo "--no-pgmc" ;;
    old_env)  echo "--envelope submitted" ;;
    no_aug)   echo "--augment none" ;;
    adjacent) echo "--pgmc-pairs adjacent" ;;
    no_dann)  echo "--no-dann" ;;
    spectral) echo "--stream spectral" ;;
  esac
}

xflags_of() {  # XJTU-SY: exp17 has no ablation flags; Experiments/r1_components_xjtu.py maps these
  case $1 in
    full)    echo "" ;;
    rawonly) echo "--ablation temporal" ;;
    static)  echo "--ablation static" ;;
    no_pgmc) echo "--ablation no_pgmc" ;;
  esac
}

selected() {  # file, key
  $PY -c "import json; print(json.load(open('$1'))['selected'])"
}

prun() {  # PRONOSTIA run: output root, run name, flags
  local root=$1 name=$2; shift 2
  if [ -f "$root/$name/results.json" ]; then echo "skip $root/$name"; return; fi
  echo "=== $root/$name ==="
  # shellcheck disable=SC2068
  $PY Experiments/r1_components.py --data-dir Processed_PRONOSTIA_v2 --output-root "$root" \
    --variant-name "$name" $@
}

xrun() {  # XJTU-SY run: output root, run name, flags
  local root=$1 name=$2; shift 2
  if [ -f "$root/$name/results.json" ]; then echo "skip $root/$name"; return; fi
  echo "=== $root/$name ==="
  # shellcheck disable=SC2068
  $PY Experiments/r1_components_xjtu.py --output-root "$root" --variant-name "$name" $@
}

case $STAGE in
  band)
    $PY Experiments/r1_band_select.py --dataset pronostia
    $PY Experiments/r1_band_select.py --dataset xjtu ;;
  smoke)
    prun outputs/r1_components_smoke full_foldA $SELECT_FLAGS --val-bearings "$(val_of A)" \
      --epochs 10 --lambda-pgmc 0.08 --seeds 42 ;;
  budget)
    for B in 5 10 20 40; do for F in A B C D; do
      prun "$SEL" "lam0.08_B${B}_fold${F}" $SELECT_FLAGS --val-bearings "$(val_of "$F")" \
        --epochs "$B" --lambda-pgmc 0.08
    done; done ;;
  lambda)
    B=$(selected "$SEL/budget_selection.json")
    for LAM in 0.2 0.5 1.0 2.0 5.0; do for F in A B C D; do
      prun "$SEL" "lam${LAM}_B${B}_fold${F}" $SELECT_FLAGS --val-bearings "$(val_of "$F")" \
        --epochs "$B" --lambda-pgmc "$LAM"
    done; done ;;
  valpreds)
    B=$(selected "$SEL/budget_selection.json"); LAM=$(selected "$SEL/selection.json")
    for ABL in $PRIMARY; do for F in A B C D; do
      prun outputs/r1_components_val "${ABL}_fold${F}" $SELECT_FLAGS --val-bearings "$(val_of "$F")" \
        --epochs "$B" --lambda-pgmc "$LAM" $(flags_of "$ABL")
    done; done ;;
  final)
    B=$(selected "$SEL/budget_selection.json"); LAM=$(selected "$SEL/selection.json")
    for ABL in $ALL; do
      prun outputs/r1_components_final "$ABL" --all-seeds --seeds "$SEEDS" --pgmc-mode aux-seq \
        --val-mode none --epochs "$B" --lambda-pgmc "$LAM" --proportional-warmup --mc-passes 30 \
        --deterministic --test-bearings "$TEST" $(flags_of "$ABL")
    done ;;
  xjtu-budget)
    for B in 5 10 20 40; do for F in A B C D; do
      xrun "$SELX" "lam0.08_B${B}_fold${F}" $SELECT_FLAGS --skip-zeroshot --val-bearings "$(val_of "$F")" \
        --epochs "$B" --lambda-pgmc 0.08
    done; done ;;
  xjtu-lambda)
    B=$(selected "$SELX/budget_selection.json")
    for LAM in 0.2 0.5 1.0 2.0 5.0; do for F in A B C D; do
      xrun "$SELX" "lam${LAM}_B${B}_fold${F}" $SELECT_FLAGS --skip-zeroshot --val-bearings "$(val_of "$F")" \
        --epochs "$B" --lambda-pgmc "$LAM"
    done; done ;;
  xjtu-valpreds)
    B=$(selected "$SELX/budget_selection.json"); LAM=$(selected "$SELX/selection.json")
    for ABL in $PRIMARY; do for F in A B C D; do
      xrun outputs/r1_components_xjtu_val "${ABL}_fold${F}" $SELECT_FLAGS --skip-zeroshot \
        --val-bearings "$(val_of "$F")" --epochs "$B" --lambda-pgmc "$LAM" $(xflags_of "$ABL")
    done; done ;;
  xjtu-final)
    B=$(selected "$SELX/budget_selection.json"); LAM=$(selected "$SELX/selection.json")
    for ABL in full $PRIMARY; do
      xrun outputs/r1_components_xjtu_final "$ABL" --all-seeds --seeds "$SEEDS" --pgmc-mode aux-seq \
        --val-mode none --epochs "$B" --lambda-pgmc "$LAM" --proportional-warmup --mc-passes 10 \
        --deterministic $(xflags_of "$ABL")
    done ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "=== r1_components_runs $STAGE done ==="
