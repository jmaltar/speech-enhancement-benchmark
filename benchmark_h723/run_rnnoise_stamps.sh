#!/bin/bash
# Run the whole VoiceBank 48 kHz test set through the stamped RNNoise image already on the H723 (paced at the
# 10 ms hop, outputs kept, against the host oracle), then read the per-stage DWT sums over SWD BEFORE any reset
# (reset zeroes .bss). The board must have been reset right before launch so the sums start at zero. Run as a
# systemd --user unit so it outlives the terminal (trap 21); no variables reach systemd, this is a script file.
set -u
cd "$(dirname "$(readlink -f "$0")")/.."   # the project root
M=/mnt/data/benchmark-data/manifest.csv
REF=/mnt/data/benchmark-data/enhanced/rnnoise/voicebank48k
B=benchmark_h723/results/rnnoise_stamp_image.bin
E=benchmark_h723/results/rnnoise_stamp_image.elf
OUT=benchmark_h723/results/rnnoise/stamp_corpus
lsusb | grep -i "cafe:4101" || { echo "H723 CDC not enumerated"; exit 1; }
echo "=== $(date -Is) corpus stamp -> $OUT"
.venv/bin/python harness/abf_host.py batch $M $OUT --corpus voicebank48k --algo rnnoise \
  --frame 960 --hop 480 --paced --keep --every 25 --timeout 120 --check-bin $B --ref-dir $REF
echo "=== $(date -Is) done corpus (exit $?)"
bash benchmark_h723/read_rnnoise_stamps.sh $E stamp_corpus
echo "=== $(date -Is) stamps read"
