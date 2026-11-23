#!/bin/bash
# Smoke twin of run_rnnoise_variants.sh: flash each named image and run the HEAD of
# the VoiceBank 48 kHz test set (--limit, default 5: p232_001/002/003/005/006, a
# head-of-manifest cut -- trap 14 -- never a corpus figure), paced, outputs kept
# under rnnoise_f411b/<v>_smoke<N>/. For diagnostic images (fpc0, stamped) whose
# question is answered by a few utterances. Literal command in the unit, no ${v}.
set -u
cd "$(dirname "$(readlink -f "$0")")/.."   # the project root
M=/mnt/data/benchmark-data/manifest.csv
REF=/mnt/data/benchmark-data/enhanced/rnnoise/voicebank48k
R=benchmark_f411/results/rnnoise_f411b
LIMIT=${LIMIT:-5}
for v in "$@"; do
  B=benchmark_f411/results/rnnoise_${v}_image.bin
  echo "=== $(date -Is) flash $v ($B)"
  timeout 120 openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
    -c "init; reset halt; flash write_image erase $B 0x08000000; verify_image $B 0x08000000; reset run; shutdown" 2>&1 \
    | grep -E "Error|error|verified|wrote" || true
  for i in $(seq 1 12); do lsusb | grep -q -i "cafe:4001" && break; sleep 5; done
  lsusb | grep -i "cafe:4001" || { echo "CDC did not enumerate after flashing $v"; exit 1; }
  sleep 3
  echo "=== $(date -Is) smoke $v (limit $LIMIT)"
  .venv/bin/python harness/abf_host.py batch $M $R/${v}_smoke${LIMIT} --corpus voicebank48k --algo rnnoise \
    --frame 960 --hop 480 --paced --keep --limit $LIMIT --every 1 --timeout 120 --check-bin $B --ref-dir $REF
  echo "=== $(date -Is) done $v (exit $?)"
done
