#!/usr/bin/env bash
# Arm B, first operating point -- the chain, in order, each run scored through the
# same oracle and harness as the baseline. Launch detached:
#
#   nohup bash training/arm_b.sh > training/runs/arm_b.log 2>&1 &
#   tail -f training/runs/<run>.log            # live tqdm bars of the run in progress
#
# 1. ft-f32       float fine-tune on VoiceBank train: the baseline every delta is against
# 2. prune-learned learned-threshold structured pruning from ft-f32 (TinyLSTMs eq. 3)
# 3. prune-fixed   fixed magnitude thresholds to the SAME live units, then fine-tune (ablation)
# 4. qat           fake-quantized fine-tune from ft-f32; scored int8 as well as 16x8
# Each: EPOCHS x STEPS steps of batch 32 x 4 s (~5 min per 500 steps on the 4060).
set -euo pipefail
cd "$(dirname "$0")/.."
source gpu_env.sh
PY=.venv/bin/python
EPOCHS=${EPOCHS:-6}; STEPS=${STEPS:-500}; LAMBDA=${LAMBDA:-0.02}
R=training/runs
stamp() { date '+%H:%M:%S'; }

echo "[$(stamp)] 1/4 ft-f32"
$PY -m training.train --out $R/ft-f32 --epochs $EPOCHS --steps $STEPS > $R/ft-f32.log 2>&1
$PY -m training.evaluate --run $R/ft-f32 --schemes f32,i16 --against dtln-f32 > $R/ft-f32.eval.out 2>&1
INIT=$R/ft-f32/best.authors.weights.h5

echo "[$(stamp)] 2/4 prune-learned (lambda $LAMBDA)"
$PY -m training.train --out $R/prune-learned --init $INIT --prune-lambda $LAMBDA --epochs $EPOCHS --steps $STEPS > $R/prune-learned.log 2>&1
$PY -m training.evaluate --run $R/prune-learned --schemes f32,i16 --against dtln-ft-f32-f32 > $R/prune-learned.eval.out 2>&1
UNITS=$($PY -c "import json; print(','.join(map(str, json.load(open('$R/prune-learned/summary.json'))['live_units'])))")

echo "[$(stamp)] 3/4 prune-fixed to $UNITS"
$PY -m training.train --out $R/prune-fixed --init $INIT --fixed-prune "$UNITS" --epochs $EPOCHS --steps $STEPS > $R/prune-fixed.log 2>&1
$PY -m training.evaluate --run $R/prune-fixed --schemes f32,i16 --against dtln-ft-f32-f32 > $R/prune-fixed.eval.out 2>&1

echo "[$(stamp)] 4/4 qat"
$PY -m training.train --out $R/qat --init $INIT --fq --epochs $EPOCHS --steps $STEPS > $R/qat.log 2>&1
$PY -m training.evaluate --run $R/qat --schemes f32,i16,int8 --against dtln-ft-f32-f32 > $R/qat.eval.out 2>&1

echo "[$(stamp)] chain done"
