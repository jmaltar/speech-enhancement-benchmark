#!/bin/bash
# S3 stamped RNNoise run (2026-11-02): flash an archived stamped image (app only, 0x10000, as every restore on this
# tier), log the UART console from before the batch to after it, run VoiceBank 48 kHz paced at the 10 ms hop with
# outputs kept against the host oracle, then decode the last RN_STAGES console line (cumulative CCOUNT sums since
# the boot that preceded the batch). Run as a systemd --user unit so it outlives the terminal (trap 21); this is a
# script file, so no ${var} reaches systemd (memory note: systemd-run expands braced variables).
#   usage: run_rnnoise_stamps.sh <ship|both> <smoke|corpus>
set -u
cd "$(dirname "$(readlink -f "$0")")/.."   # the project root
V=$1; MODE=$2
case $V in ship) IMG=rnnoise_stamp;; both) IMG=rnnoise_both_stamp;; *) echo "variant: ship|both"; exit 2;; esac
case $MODE in smoke) LIM="--limit 2";; corpus) LIM="";; *) echo "mode: smoke|corpus"; exit 2;; esac
M=/mnt/data/benchmark-data/manifest.csv
REF=/mnt/data/benchmark-data/enhanced/rnnoise/voicebank48k
B=benchmark_s3/results/${IMG}_image.bin
OUT=benchmark_s3/results/rnnoise/stamp_${V}_${MODE}_$(date +%F)
CON=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5981000468-if00
IDFPY=${IDFPY:-$HOME/.espressif/python_env/idf5.4_py3.12_env/bin/python}
mkdir -p $OUT
cp $B $OUT/image.bin
echo "=== $(date -Is) flash $B" | tee $OUT/batch.log
esptool --chip esp32s3 -p $CON -b 921600 write-flash 0x10000 $B > $OUT/flash.log 2>&1 || { echo "FLASH FAILED"; tail -5 $OUT/flash.log; exit 1; }
tail -3 $OUT/flash.log | tee -a $OUT/batch.log
$IDFPY benchmark_s3/console_capture.py $CON $OUT/console.txt &
CAP=$!
sleep 3
for i in $(seq 1 20); do lsusb | grep -q "cafe:4201" && break; sleep 1; done
lsusb | grep -q "cafe:4201" || { echo "S3 CDC not enumerated"; kill $CAP; exit 1; }
sleep 2
echo "=== $(date -Is) batch $MODE -> $OUT" | tee -a $OUT/batch.log
.venv/bin/python harness/abf_host.py batch $M $OUT --corpus voicebank48k --algo rnnoise \
  --frame 960 --hop 480 --paced --keep --every 25 --timeout 120 --check-bin $B --ref-dir $REF $LIM 2>&1 | tee -a $OUT/batch.log
echo "=== $(date -Is) batch done; waiting for the console's RN_STAGES line" | tee -a $OUT/batch.log
sleep 5
kill $CAP
.venv/bin/python benchmark_s3/read_rnnoise_stamps.py $OUT/console.txt "stamp_${V}_${MODE}" 2>&1 | tee -a $OUT/batch.log
echo "=== $(date -Is) stamps read" | tee -a $OUT/batch.log
