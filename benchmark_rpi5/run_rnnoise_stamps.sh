#!/bin/bash
# RPi5 stamped RNNoise run (2026-11-02): the 10-26 row's batch (run_rnnoise_batches.sh) through rnnoise_run_stamp,
# paced at the real 10 ms hop with outputs kept, each utterance's CLOCK_MONOTONIC stage sums appended to a TSV by
# abf_batch_file.py --stages-log. Launched as a systemd --user unit; CSV, log and TSV copied to the SD card at the
# end (CLAUDE.md trap 21: tmpfs dies with the login session).
#   usage: run_rnnoise_stamps.sh <smoke|corpus>
set -u
MODE=$1
case $MODE in smoke) LIM="--limit 3";; corpus) LIM="";; *) echo "mode: smoke|corpus"; exit 2;; esac
OUT=/dev/shm/results_rnnoise_stamp_$MODE; mkdir -p $OUT
BUILD="Release gcc14.2 stamp $(date +%Y%m%dT%H%M)"
LOG=$OUT/batch_rnnoise_stamp_$(date +%F).log
echo "=== rnnoise v0.1.1 stamped paced $MODE $(date +%T)" | tee -a $LOG
python3 ~/benchmark/harness/abf_batch_file.py /dev/shm/bench/manifest.csv $OUT \
    --runner ~/benchmark/rnnoise/build_stamp/rnnoise_run_stamp --corpus voicebank48k --algo rnnoise \
    --frame 960 --hop 480 --paced --keep --ref-dir /dev/shm/bench/enhanced/rnnoise/voicebank48k \
    --remap /mnt/data/benchmark-data=/dev/shm/bench/data --every 100 --build "$BUILD" \
    --stages-log $OUT/stages.tsv $LIM 2>&1 | tee -a $LOG
echo "=== all done $(date +%T)" | tee -a $LOG
mkdir -p ~/benchmark/results_sd/rnnoise_stamp_$MODE && cp $OUT/*.csv $OUT/stages.tsv $LOG ~/benchmark/results_sd/rnnoise_stamp_$MODE/ && sync
