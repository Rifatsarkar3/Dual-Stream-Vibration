#!/usr/bin/env bash
# Revision R1, analysis plan Sections 7 (B2 adopted) and 8: the protocol demonstration "in-bearing
# early stopping" repeated for the temporal-context variant. Settings mirror the submitted run it
# re-creates (outputs/exp16_ress/full, re-evaluated as outputs/r1_reeval/full_inbearing_es):
# validation = every 10th window of each training bearing, early stopping with patience 30,
# at most 120 epochs, fixed (not proportional) warm-ups, seeds 42, 123, 777, 999, 7, best-validation
# checkpoint evaluated once on the 13 corrected bearings with 30 MC passes; plus temporal context
# and deterministic kernels. The second demonstration (lambda by validation argmin) does not exist
# for this variant: its 2-fold and 4-fold validation argmins are both its selected lambda (0.08).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
OUT=outputs/r1_protocol_temporal
TEST=test_01_Bearing1_3,test_02_Bearing1_4,test_03_Bearing1_5,test_04_Bearing2_3,test_05_Bearing1_6,test_06_Bearing1_7,test_07_Bearing2_4,test_08_Bearing2_5,test_09_Bearing2_6,test_10_Bearing2_7,zs_01_Bearing3_1,zs_02_Bearing3_2,zs_03_Bearing3_3
if [ -f "$OUT/full_inbearing_es/results.json" ]; then echo "skip"; exit 0; fi
$PY Experiments/r1_temporal.py --all-seeds --val-mode window --epochs 120 --patience 30 \
  --mc-passes 30 --deterministic --data-dir Processed_PRONOSTIA_v2 --output-root "$OUT" \
  --variant-name full_inbearing_es --test-bearings "$TEST"
echo "=== r1_protocol_temporal done ==="
