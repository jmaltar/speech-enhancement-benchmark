#!/bin/bash
# RPi5 DTLN rows (task 10 -> 16, 2026-10-27): the pruned 16x8 model the H723 and S3 run (p020) and
# the full-size 16x8 export no MCU tier can hold (full), each paced at DTLN's native 8 ms hop
# (512/128 at 16 kHz) over VoiceBank test and the DNS no_reverb set with kept WAVs, then one
# unpaced VoiceBank pass. Launched as a systemd --user unit, CSVs/log copied to the SD card at the
# end (CLAUDE.md trap 21: tmpfs dies with the login session).
set -u
OUT=/dev/shm/results_dtln; mkdir -p $OUT
BUILD="Release gcc14.2 $(date +%Y%m%dT%H%M)"
LOG=$OUT/batch_dtln_$(date +%F).log
B=~/benchmark/benchmark_rpi5/dtln_run/build
run() {  # runner oracle corpus mode
  local R=$1 ORACLE=$2 CORPUS=$3 MODE=$4 PACED="" KEEP=""
  [ "$MODE" = paced ] && PACED="--paced" && KEEP="--keep"
  echo "=== dtln $R $CORPUS $MODE $(date +%T)" | tee -a $LOG
  python3 ~/benchmark/harness/abf_batch_file.py /dev/shm/bench/manifest.csv $OUT/$R \
      --runner $B/dtln_run_$R --corpus $CORPUS --algo dtln --frame 512 --hop 128 $PACED $KEEP \
      --ref-dir /dev/shm/bench/enhanced/$ORACLE/$CORPUS \
      --remap /mnt/data/benchmark-data=/dev/shm/bench/data --every 100 --build "$BUILD" 2>&1 | tee -a $LOG
}
for R in p020 full; do
  ORACLE=dtln-prune-learned-shrunk-i16pt-tflm; [ $R = full ] && ORACLE=dtln-i16-tflm
  run $R $ORACLE voicebank16k paced
  run $R $ORACLE dns paced
  run $R $ORACLE voicebank16k unpaced
done
echo "=== all done $(date +%T)" | tee -a $LOG
mkdir -p ~/benchmark/results_sd/dtln && cp -r $OUT/p020 $OUT/full $LOG ~/benchmark/results_sd/dtln/ && sync
