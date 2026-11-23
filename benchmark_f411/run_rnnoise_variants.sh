#!/bin/bash
# Flash each RNNoise variant image onto the F411 and run the whole VoiceBank 48 kHz test set through it,
# paced at the 10 ms hop, outputs kept, against the host oracle. One variant after the other on the one
# board. Run as a systemd --user unit so it outlives the terminal (trap 21).
set -u
cd "$(dirname "$(readlink -f "$0")")/.."   # the project root
M=/mnt/data/benchmark-data/manifest.csv
REF=/mnt/data/benchmark-data/enhanced/rnnoise/voicebank48k
R=benchmark_f411/results/rnnoise_f411b
for v in "$@"; do
  B=benchmark_f411/results/rnnoise_${v}_image.bin
  echo "=== $(date -Is) flash $v ($B)"
  timeout 120 openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
    -c "init; reset halt; flash write_image erase $B 0x08000000; verify_image $B 0x08000000; reset run; shutdown" 2>&1 \
    | grep -E "Error|error|verified|wrote" || true
  for i in $(seq 1 12); do lsusb | grep -q -i "cafe:4001" && break; sleep 5; done
  lsusb | grep -i "cafe:4001" || { echo "CDC did not enumerate after flashing $v"; exit 1; }
  sleep 3
  echo "=== $(date -Is) corpus $v"
  .venv/bin/python harness/abf_host.py batch $M $R/${v}_corpus --corpus voicebank48k --algo rnnoise \
    --frame 960 --hop 480 --paced --keep --every 25 --timeout 120 --check-bin $B --ref-dir $REF
  echo "=== $(date -Is) done $v (exit $?)"
done
