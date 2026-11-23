#!/usr/bin/env bash
# The GPU queue on penny (RTX 3070, 2026-10-20). One command per line in
# training/runs/penny_queue.txt, run in order; the queue is re-read after each
# line, so lines appended while a run is in progress are picked up. A line
# reading "exit" ends the queue. Launch detached:
#
#   setsid nohup bash training/penny_queue.sh > training/runs/penny_queue.log 2>&1 &
#   setsid nohup bash training/penny_queue.sh penny_queue2.txt > training/runs/penny_queue2.log 2>&1 &
#
# Two lanes pay on penny (a 2017 Ryzen 1800X): a run there is host-bound at 0.39
# step/s with the GPU 19 % busy, so a second run costs the first little. Lane 2's
# runs are launched with --gpu-mem-mb so both fit the 8 GB card.
#
# Conventions, the same as arm_b.sh: 6 x 500 steps of batch 32 x 4 s from the
# float fine-tune ft-f32; every run scored through training.evaluate against
# dtln-ft-f32-f32 (a pruned run with --shrunk, at its live sizes).
set -uo pipefail
cd "$(dirname "$0")/.."
source gpu_env.sh
export BENCHMARK_DATA_ROOT=${BENCHMARK_DATA_ROOT:-$HOME/benchmark-data}
export PY=.venv/bin/python R=training/runs INIT=training/runs/ft-f32/best.authors.weights.h5
export EPOCHS=${EPOCHS:-6} STEPS=${STEPS:-500}
Q=training/runs/${1:-penny_queue.txt}
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
i=1
while true; do
  line=$(sed -n "${i}p" "$Q" 2>/dev/null || true)
  if [ -z "$line" ]; then sleep 60; continue; fi
  i=$((i + 1))
  case "$line" in \#*) continue ;; exit) echo "[$(stamp)] queue ended"; break ;; esac
  echo "[$(stamp)] start: $line"
  t0=$(date +%s)
  bash -c "$line"; rc=$?
  echo "[$(stamp)] done (rc $rc, $(( ($(date +%s) - t0) / 60 )) min): $line"
done
