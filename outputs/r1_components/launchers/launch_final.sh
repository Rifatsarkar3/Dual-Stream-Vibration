#!/usr/bin/env bash
# Detached launcher for Amendment 10 final runs (survives the tool's time limit).
cd "/e/Yolo-Thermal/Dual-Stream Vibration-Vision" || exit 1
date "+%H:%M:%S" > outputs/r1_components/final_started.txt
bash Experiments/r1_components_runs.sh final > outputs/r1_components/final_stdout.txt 2>&1
echo "exit $? $(date +%H:%M:%S)" > outputs/r1_components/final_done.txt
