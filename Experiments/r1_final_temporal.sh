#!/usr/bin/env bash
# Revision R1, analysis plan Section 7, candidate B2 (temporal context, K = 5). Selected
# setting from the validation folds only: B* = 5 (outputs/r1_temporal/budget_selection.txt),
# lambda = 0.08 (outputs/r1_temporal/selection.txt). The variant met the pre-registered
# adoption rule (4-fold 0.1424 < 0.1510 - 0.0078), so the final model and all seven
# ablations are trained here with the settings of r1_final_B10.sh (20 seeds, aux-seq PGMC,
# lambda 0.08, proportional warm-ups, no validation, all 13 corrected PRONOSTIA bearings);
# differences: temporal context (Experiments/r1_temporal.py), B = 5, deterministic kernels.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
OUT=outputs/r1_final_temporal_B5
SEEDS=42,123,777,999,7,8,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946
TEST=test_01_Bearing1_3,test_02_Bearing1_4,test_03_Bearing1_5,test_04_Bearing2_3,test_05_Bearing1_6,test_06_Bearing1_7,test_07_Bearing2_4,test_08_Bearing2_5,test_09_Bearing2_6,test_10_Bearing2_7,zs_01_Bearing3_1,zs_02_Bearing3_2,zs_03_Bearing3_3
COMMON="--all-seeds --seeds $SEEDS --val-mode none --epochs 5 --proportional-warmup --deterministic --data-dir Processed_PRONOSTIA_v2 --output-root $OUT --test-bearings $TEST"

run() {  # name, mc passes, extra flags
  local name=$1 mc=$2; shift 2
  if [ -f "$OUT/$name/results.json" ]; then echo "skip $name"; return; fi
  echo "=== $name ==="
  $PY Experiments/r1_temporal.py $COMMON --mc-passes "$mc" --variant-name "$name" "$@"
}

run final20 30
run abl20_static 10 --static-gate
run abl20_temporal 10 --stream temporal
run abl20_spectral 10 --stream spectral
run abl20_no_pgmc 10 --no-pgmc
run abl20_no_dann 10 --no-dann
run abl20_neither 10 --no-pgmc --no-dann
run abl20_pgmc_soft 10 --pgmc-mode soft
echo "=== r1_final_temporal done ==="
