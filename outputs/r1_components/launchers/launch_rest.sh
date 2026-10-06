#!/usr/bin/env bash
# Amendment 10, remaining stages in the amendment's order, detached. Every choice is made by
# the selection scripts; the chain stops if lambda changes (Section 3.3 iteration needed) or
# if any step fails. Times are local (+08:00); progress in outputs/r1_components/rest_log.txt.
cd "/e/Yolo-Thermal/Dual-Stream Vibration-Vision" || exit 1
PY=./.venv/Scripts/python.exe
O=outputs/r1_components
L=$O/rest_log.txt
stamp() { echo "$(date '+%H:%M:%S') $*" >> "$L"; }
fail() { stamp "FAILED: $*"; echo "exit 2 $(date +%H:%M:%S) $*" > "$O/rest_done.txt"; exit 2; }
sel() { "$PY" -c "import json; print(json.load(open('$1'))['selected'])"; }

stamp "waiting for the PRONOSTIA final runs"
while [ ! -f "$O/final_done.txt" ]; do sleep 30; done
grep -q "exit 0" "$O/final_done.txt" || fail "PRONOSTIA final did not exit 0"
stamp "PRONOSTIA final finished: $(cat "$O/final_done.txt")"

stamp "noise pronostia start"
"$PY" Experiments/r1_components_noise.py --dataset pronostia > "$O/noise_pronostia_stdout.txt" 2>&1 || fail "noise pronostia"
stamp "noise pronostia done"

stamp "xjtu-budget start"
bash Experiments/r1_components_runs.sh xjtu-budget > "$O/xjtu_budget_stdout.txt" 2>&1 || fail "xjtu-budget"
"$PY" Experiments/r1_select_budget.py --root outputs/r1_components_xjtu --no-adoption > "$O/xjtu_budget_select.txt" 2>&1 || fail "xjtu budget selection"
B=$(sel outputs/r1_components_xjtu/budget_selection.json)
stamp "xjtu-budget done; B* = $B"

stamp "xjtu-lambda start"
bash Experiments/r1_components_runs.sh xjtu-lambda > "$O/xjtu_lambda_stdout.txt" 2>&1 || fail "xjtu-lambda"
"$PY" Experiments/r1_select_lambda.py --root outputs/r1_components_xjtu --budget "$B" --no-repro-check > "$O/xjtu_lambda_select.txt" 2>&1 || fail "xjtu lambda selection"
LAM=$(sel outputs/r1_components_xjtu/selection.json)
stamp "xjtu-lambda done; lambda* = $LAM"
if [ "$LAM" != "0.08" ]; then stamp "lambda changed: Section 3.3 iteration needed, chain stopped"; echo "exit 3 $(date +%H:%M:%S) lambda changed" > "$O/rest_done.txt"; exit 3; fi

stamp "xjtu-valpreds start"
bash Experiments/r1_components_runs.sh xjtu-valpreds > "$O/xjtu_valpreds_stdout.txt" 2>&1 || fail "xjtu-valpreds"
stamp "xjtu-valpreds done"

stamp "xjtu-final start"
bash Experiments/r1_components_runs.sh xjtu-final > "$O/xjtu_final_stdout.txt" 2>&1 || fail "xjtu-final"
stamp "xjtu-final done"

stamp "noise xjtu start"
"$PY" Experiments/r1_components_noise.py --dataset xjtu > "$O/noise_xjtu_stdout.txt" 2>&1 || fail "noise xjtu"
stamp "noise xjtu done; all stages finished"
echo "exit 0 $(date +%H:%M:%S)" > "$O/rest_done.txt"
