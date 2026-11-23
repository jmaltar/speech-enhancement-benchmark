#!/bin/bash
# One power session on the RPi5 (task 22, the Pi's PMIC method): an idle leg, an unpaced batch and a paced batch of
# RNNoise over the 48 kHz VoiceBank test set through the file binding, `pmic_log.py` sampling the PMIC rails at 1 Hz
# on core 0 alongside each (the batches run on core 2), then power_summary.py. The Pi's tick is 1 ns, so the summary
# is given --clk-mhz 1000: its "cycles" are nanoseconds and its RTF is time over the 10 ms hop, as the batch CSVs are.
# Mirrors power_logger_f411/power_session.sh; outputs under $OUT, copied to the SD card at the end (trap 21).
#   systemd-run --user --unit power-rpi5-rnnoise ~/benchmark/benchmark_rpi5/power_session_rpi5.sh
set -euo pipefail
B=~/benchmark; OUT=/dev/shm/results_power_rnnoise; SD=$B/results_sd/power/rnnoise
RUNNER=${RUNNER:-$B/rnnoise/build/rnnoise_run}
STAGE=/dev/shm/bench; M=$STAGE/manifest.csv; REF=$STAGE/enhanced/rnnoise/voicebank48k
IDLE_S=${IDLE_S:-60}; BUILD="Release gcc14.2 $(date +%Y%m%dT%H%M)"
for f in "$RUNNER" "$M" "$REF" $B/benchmark_rpi5/pmic_log.py $B/power_logger_f411/power_summary.py; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
mkdir -p $OUT; LOG=$OUT/session.log; exec > >(tee -a $LOG) 2>&1
echo "=== $(date -Is) power session rpi5 rnnoise runner $RUNNER ($(md5sum < "$RUNNER" | cut -c1-8)) -> $OUT"
echo "conditions: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c | tr '\n' ' ') $(vcgencmd get_throttled) $(vcgencmd measure_temp)"
logger() { taskset -c 0 python3 $B/benchmark_rpi5/pmic_log.py --quiet "$@"; }
echo "=== $(date -Is) idle ${IDLE_S} s"
logger --out $OUT/power_idle.csv --seconds "$IDLE_S"
for leg in unpaced paced; do
  p=(); [ "$leg" = paced ] && p=(--paced --keep)
  echo "=== $(date -Is) $leg batch"
  logger --out $OUT/power_$leg.csv & lp=$!
  sleep 3; rc=0
  python3 $B/harness/abf_batch_file.py $M $OUT/$leg --runner "$RUNNER" --corpus voicebank48k --algo rnnoise \
      --frame 960 --hop 480 --ref-dir $REF --remap /mnt/data/benchmark-data=$STAGE/data --every 100 \
      --build "$BUILD" "${p[@]}" > $OUT/batch_$leg.log 2>&1 || rc=$?
  sleep 2; kill -TERM $lp; wait $lp || true   # TERM, not INT: a non-interactive bash starts background jobs with SIGINT ignored (2026-11-04)
  tail -2 $OUT/batch_$leg.log; [ $rc = 0 ] || echo "batch $leg exited $rc"
done
echo "=== $(date -Is) summary (tick = 1 ns, so clk 1000 'MHz'; PMIC figure excludes converter loss and 5V-direct loads)"
python3 $B/power_logger_f411/power_summary.py $OUT --clk-mhz 1000 --hop-s 0.010 | tee $OUT/power_summary.md
echo "conditions after: $(vcgencmd get_throttled) $(vcgencmd measure_temp)"
mkdir -p $SD && cp -r $OUT/power_*.csv $OUT/power_summary.* $OUT/batch_*.log $OUT/unpaced $OUT/paced $LOG $SD/ && sync
echo "=== $(date -Is) copied to $SD"
