#!/usr/bin/env bash
# Flash the INA226 logger onto the SPARE F411 (the power-bench fixture).
#
# Both F411 boards are the same part, so the script reads the board before it writes: chip UID, and the first 4 KiB
# of flash compared against every archived F411 benchmark image (the first 8 KiB). A board that holds a measurement image is the
# board under test, not the fixture, and is refused unless FORCE=1.
#   ./flash.sh            build if needed, identify, refuse or flash, verify
#   ./flash.sh --identify identify only, write nothing
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
bin=$here/build/ina226_logger.bin
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT

if [ ! -f "$bin" ] || [ "$here/main.c" -nt "$bin" ] || [ "$here/startup.c" -nt "$bin" ]; then
  cmake -S "$here" -B "$here/build" -G Ninja >/dev/null && cmake --build "$here/build"
fi

# The debugger: the WeAct Mini Debugger enumerates as an ST-Link V2.1 (0483:3752) or, with DAPLink firmware, as
# CMSIS-DAP (0d28:0204). Several may be plugged in at once (each wired to a different board), so try each and use the
# one whose target is an STM32F411. DEBUGGER=stlink|cmsis-dap forces one.
cands=()
if [ -z "${DEBUGGER:-}" ] || [ "${DEBUGGER}" = stlink ]; then
  lsusb 2>/dev/null | grep -qE 'ID 0483:(3748|374b|374e|374f|3752|3753)' && cands+=(stlink)
fi
if [ -z "${DEBUGGER:-}" ] || [ "${DEBUGGER}" = cmsis-dap ]; then
  lsusb -d 0d28:0204 >/dev/null 2>&1 && cands+=(cmsis-dap)
fi
[ ${#cands[@]} -gt 0 ] || { echo "no debugger on USB (ST-Link or CMSIS-DAP)" >&2; exit 1; }
oc() { timeout 90 openocd "${iface[@]}" -f target/stm32f4x.cfg -c "$1" 2>&1; }

idcode=""; uid=""
for c in "${cands[@]}"; do
  case $c in
    stlink)    iface=(-f interface/stlink.cfg) ;;
    cmsis-dap) iface=(-f interface/cmsis-dap.cfg -c "transport select swd") ;;
  esac
  # ---- identify (reads only)
  out=$(oc "init; puts \"IDCODE [capture {mdw 0xE0042000}]\"; puts \"UID [capture {mdw 0x1FFF7A10 3}]\";
            dump_image $work/head.bin 0x08000000 8192; shutdown") || true
  idcode=$(echo "$out" | sed -n 's/^IDCODE 0xe0042000: \([0-9a-f]*\).*/\1/p')
  uid=$(echo "$out" | sed -n 's/^UID 0x1fff7a10: \([0-9a-f]*\) \([0-9a-f]*\) \([0-9a-f]*\).*/\1\2\3/p' | tr a-f A-F)
  case "$idcode" in
    *431) echo "debugger: $c"; break ;;
    *) echo "debugger $c: target is not an STM32F411 (DBGMCU_IDCODE=${idcode:-none}), skipped" ;;
  esac
done
case "$idcode" in *431) ;; *) echo "no debugger is wired to an STM32F411" >&2; exit 1;; esac
echo "target: STM32F411, UID $uid"

holds=$(python3 - "$work/head.bin" "$bin" "$root/benchmark_f411" <<'EOF'
import sys, pathlib
head = open(sys.argv[1], 'rb').read()
new = open(sys.argv[2], 'rb').read()
if head[:len(new)] == new[:len(head)] or b"# ina226-logger" in head:   # this build, or an earlier one of the logger
    print("logger"); sys.exit()
if head[:4] == b'\xff' * 4:
    print("blank"); sys.exit()
for p in sorted(pathlib.Path(sys.argv[3]).rglob('*.bin')):
    try:
        b = p.open('rb').read(len(head))
    except OSError:
        continue
    if len(b) == len(head) and b == head:
        print("measurement:" + str(p.relative_to(sys.argv[3]))); sys.exit()
print("unknown")
EOF
)
echo "flash now holds: $holds"
[ "${1:-}" = "--identify" ] && exit 0
case "$holds" in
  measurement:*)
    if [ "${FORCE:-0}" != 1 ]; then
      echo "REFUSED: this board holds a benchmark measurement image; it is the board under test, not the spare." >&2
      echo "         (FORCE=1 overrides; the image is archived under benchmark_f411/ and can be restored.)" >&2
      exit 2
    fi;;
esac

# ---- an image nobody here wrote is backed up whole before the erase (trap 11: boards arrive with foreign images)
if [ "$holds" = unknown ]; then
  mkdir -p "$here/backups"
  bk="$here/backups/f411_${uid}_$(date +%Y-%m-%d_%H%M%S).bin"
  oc "init; reset halt; dump_image $bk 0x08000000 0x80000; reset run; shutdown" >/dev/null
  [ "$(stat -c %s "$bk" 2>/dev/null || echo 0)" = 524288 ] || { echo "backup failed, nothing written" >&2; exit 1; }
  echo "backed up the previous 512 KiB to ${bk#$root/} (crc32 $(python3 -c "import zlib,sys;print(format(zlib.crc32(open(sys.argv[1],'rb').read()),'08x'))" "$bk"))"
fi

# ---- flash and verify (the init / reset halt / write / verify / reset run form used for every board here)
res=$(oc "init; reset halt; flash write_image erase $bin 0x08000000;
          puts \"VERIFY:[catch {verify_image $bin 0x08000000} msg] \$msg\"; reset run; shutdown" | grep '^VERIFY:' || true)
case "$res" in
  "VERIFY:0 verified"*) echo "${res#VERIFY:0 }" ;;
  *) echo "verify FAILED: ${res:-no result from openocd}" >&2; exit 1 ;;
esac
crc=$(python3 -c "import zlib,sys;print(format(zlib.crc32(open(sys.argv[1],'rb').read()),'08x'))" "$bin")
echo "flashed ina226_logger.bin crc32 $crc onto UID $uid"
