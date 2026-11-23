#!/usr/bin/env bash
# The rest of the Arm B chain, run in PARALLEL where the dependencies allow
# (2026-10-19, after the user noticed the GPU half idle):
#
#   prune-learned  (already running, started by arm_b.sh)   ---> evaluate ---> prune-fixed ---> evaluate
#   qat            (started here, alongside, memory-capped) ---> evaluate
#
# Each training process is capped at GPU_MB so two fit in the 8 GB card.
set -uo pipefail
cd "$(dirname "$0")/.."
source gpu_env.sh
PY=.venv/bin/python
EPOCHS=${EPOCHS:-6}; STEPS=${STEPS:-500}; GPU_MB=${GPU_MB:-3500}
R=training/runs
INIT=$R/ft-f32/best.authors.weights.h5
stamp() { date '+%H:%M:%S'; }

echo "[$(stamp)] qat starts in parallel with prune-learned (cap $GPU_MB MB)"
( $PY -m training.train --out $R/qat --init $INIT --fq --epochs $EPOCHS --steps $STEPS --gpu-mem-mb $GPU_MB > $R/qat.log 2>&1 \
  && $PY -m training.evaluate --run $R/qat --schemes f32,i16,int8 --against dtln-ft-f32-f32 > $R/qat.eval.out 2>&1 \
  && echo "[$(stamp)] qat done + scored" ) &
QAT_PID=$!

echo "[$(stamp)] waiting for prune-learned to finish"
until [ -f $R/prune-learned/summary.json ]; do sleep 30; done
echo "[$(stamp)] prune-learned done; scoring"
$PY -m training.evaluate --run $R/prune-learned --schemes f32,i16 --against dtln-ft-f32-f32 > $R/prune-learned.eval.out 2>&1
UNITS=$($PY -c "import json; print(','.join(map(str, json.load(open('$R/prune-learned/summary.json'))['live_units'])))")

echo "[$(stamp)] prune-fixed to $UNITS"
$PY -m training.train --out $R/prune-fixed --init $INIT --fixed-prune "$UNITS" --epochs $EPOCHS --steps $STEPS --gpu-mem-mb $GPU_MB > $R/prune-fixed.log 2>&1
$PY -m training.evaluate --run $R/prune-fixed --schemes f32,i16 --against dtln-ft-f32-f32 > $R/prune-fixed.eval.out 2>&1
echo "[$(stamp)] prune-fixed done + scored"

wait $QAT_PID
echo "[$(stamp)] chain done"
