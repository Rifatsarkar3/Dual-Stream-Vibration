#!/usr/bin/env bash
# Amendment 12 runs (blocked in-unit scheme SB), detached (survives the tool's time limit). Four
# processes: FD001 and FD003, repeats 0-9 and 10-19 each, fleet sizes 4, 10 and 25. Deterministic
# kernels, so the results do not depend on this split. Times are local (+08:00).
cd "/e/Yolo-Thermal/Dual-Stream Vibration-Vision" || exit 1
PY=./.venv/Scripts/python.exe
O=outputs/r1_cmapss_blocked
mkdir -p "$O/logs"
date "+%H:%M:%S" > "$O/runs_started.txt"
run() {
  "$PY" Experiments/r1_cmapss_blocked.py --dataset "$1" --repeats "$2" > "$O/logs/$1_r$2.txt" 2>&1
  echo "$1 $2 exit $? $(date +%H:%M:%S)" >> "$O/runs_exit.txt"
}
run FD001 0-9 &
run FD001 10-19 &
run FD003 0-9 &
run FD003 10-19 &
wait
echo "done $(date +%H:%M:%S)" > "$O/runs_done.txt"
