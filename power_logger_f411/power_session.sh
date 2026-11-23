#!/usr/bin/env bash
# One power session for a board under test on the bench: a fresh-boot reading, a one-utterance priming stream, the idle
# leg (post-stream state), an unpaced batch and a paced batch, the INA226
# logger recording alongside each, then the summary. Outputs under <outdir>: power_{idle,unpaced,paced}.csv (+ .log),
# unpaced/ and paced/ batch directories, batch_{unpaced,paced}.log, power_summary.{md,json}.
#
#   power_logger_f411/power_session.sh h723 benchmark_h723/results/rnnoise_image.bin results/power/h723_rnnoise \
#       --clk-mhz 550 --hop-s 0.010 -- --corpus voicebank48k --algo rnnoise --frame 960 --hop 480 --keep --every 25 \
#       --timeout 120 --ref-dir /mnt/data/benchmark-data/enhanced/rnnoise/voicebank48k
#
# Everything after `--` goes to `abf_host.py batch` (never --paced: the script adds it for the paced leg, and never
# --limit for a figure that counts: trap 14). Env: IDLE_S (default 60), LOGGER_PORT (/dev/ttyACM1), USB_ID (the board
# under test's CDC id: cafe:4001 for the F411, cafe:4101 otherwise).
# Run it as a systemd --user unit (trap 21); it is a script file, so no $ reaches systemd.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/.." && pwd); cd "$root"
tier=$1; bin=$2; out=$3; shift 3
clk=""; hops=""
while [ $# -gt 0 ]; do
  case $1 in
    --clk-mhz) clk=$2; shift 2 ;;
    --hop-s) hops=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option $1 (batch args go after --)" >&2; exit 1 ;;
  esac
done
[ -n "$clk" ] && [ -n "$hops" ] || { echo "--clk-mhz and --hop-s are required" >&2; exit 1; }
M=${MANIFEST:-/mnt/data/benchmark-data/manifest.csv}
IDLE_S=${IDLE_S:-60}; PORT=${LOGGER_PORT:-/dev/ttyACM1}
LEGS=${LEGS:-unpaced paced}; SKIP_PRE=${SKIP_PRE:-0}   # LEGS="paced" SKIP_PRE=1 reruns one batch leg into an existing outdir
case $tier in f411) dflt=cafe:4001 ;; s3) dflt=cafe:42 ;; *) dflt=cafe:4101 ;; esac; USB_ID=${USB_ID:-$dflt}
mkdir -p "$out"
lsusb | grep -qi "$USB_ID" || { echo "board under test not enumerated ($USB_ID): check the D-/D+ path through the fixture" >&2; exit 1; }
crc=$(python3 -c "import zlib,sys;print(format(zlib.crc32(open(sys.argv[1],'rb').read()),'08x'))" "$bin")
echo "=== $(date -Is) session $tier image $(basename "$bin") crc32 $crc clk ${clk} MHz hop ${hops} s -> $out"
logger() { python3 "$here/ina226_log.py" --port "$PORT" --quiet "$@"; }

# The firmware has two idle states (H723, 2026-11-04): fresh after enumeration it spins at a higher current than it does
# once a stream has run, and the second is the state between hops of a paced stream. So: a short fresh-boot reading for
# the record, one utterance pushed through (a smoke run, never a figure), then the idle leg the energy model uses.
if [ "$SKIP_PRE" = 0 ]; then
echo "=== $(date -Is) fresh-boot idle 20 s (record only)"
logger --out "$out/power_idle_boot.csv" --seconds 20
echo "=== $(date -Is) priming stream (1 utterance, smoke)"
.venv/bin/python harness/abf_host.py batch "$M" "$out/prime" "$@" --limit 1 --check-bin "$bin" > "$out/batch_prime.log" 2>&1 || { tail -3 "$out/batch_prime.log"; exit 1; }
sleep 3
echo "=== $(date -Is) idle ${IDLE_S} s (post-stream: the P_idle the model uses)"
logger --out "$out/power_idle.csv" --seconds "$IDLE_S"
fi

for leg in $LEGS; do
  p=(); [ "$leg" = paced ] && p=(--paced)
  echo "=== $(date -Is) $leg batch"
  # python3 directly, NOT the logger function: `f & lp=$!` yields the PID of a subshell running f, so the TERM below
  # killed that subshell and left python reading the port beside the next leg's logger (S3 session, 2026-11-04).
  python3 "$here/ina226_log.py" --port "$PORT" --quiet --out "$out/power_$leg.csv" & lp=$!
  sleep 3
  rc=0
  .venv/bin/python harness/abf_host.py batch "$M" "$out/$leg" "$@" "${p[@]}" --check-bin "$bin" > "$out/batch_$leg.log" 2>&1 || rc=$?
  sleep 2
  kill -TERM "$lp"; wait "$lp" || true   # TERM, not INT: a non-interactive bash starts background jobs with SIGINT ignored (2026-11-04)
  pgrep -f "$here/ina226_log.py" >/dev/null && { echo "a logger survived the TERM; refusing to start the next leg on a shared port" >&2; exit 1; }
  tail -2 "$out/batch_$leg.log"
  [ $rc = 0 ] || echo "batch $leg exited $rc"
done
echo "=== $(date -Is) summary"
python3 "$here/power_summary.py" "$out" --clk-mhz "$clk" --hop-s "$hops" | tee "$out/power_summary.md"
