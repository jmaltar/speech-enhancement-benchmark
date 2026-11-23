#!/usr/bin/env bash
# Flash a measurement image onto the STM32 under test on the power bench and leave the debug side as quiet as SWD
# allows: init / reset halt / write_image erase / verify_image / reset run (the form that works on the H723; the
# one-line `program ... verify reset` does not), then DBGMCU_CR zeroed and read back, then OpenOCD quits.
#
#   ./flash_dut.sh h723 benchmark_h723/results/rnnoise_image.bin
#   ./flash_dut.sh f411 benchmark_f411/results/<image>.bin
#
# OpenOCD's examine sets DBGMCU_CR (F4 0xE0042004, H7 0x5C001004: DBG_SLEEP/STOP/STANDBY and, on the H7, the D1/D3
# debug clock enables) and that register survives every reset but power-on. Measured on the H723 2026-11-03
# (RNNoise image, idle): 179.3 mA before any SWD session, 183.5 mA with DBGMCU_CR = 0x0070003f, 182.0 mA after zeroing
# it; clearing C_DEBUGEN and the DP power request as well changed nothing further. So after this script, POWER-CYCLE
# the board under test (replug its cable at the breakout) before a reading that counts; the zeroing is kept because it
# is free and removes the larger part.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/.." && pwd)
tier=${1:-}; bin=${2:-}
case $tier in
  f411) tcfg=target/stm32f4x.cfg; cr=0xE0042004; idaddr=0xE0042000; dev=431 ;;
  h723) tcfg=target/stm32h7x.cfg; cr=0x5C001004; idaddr=0x5C001000; dev=483 ;;
  *) echo "usage: $0 f411|h723 <image.bin>" >&2; exit 1 ;;
esac
[ -f "$bin" ] || { echo "no such image: $bin" >&2; exit 1; }
bin=$(realpath "$bin")
crc=$(python3 -c "import zlib,sys;print(format(zlib.crc32(open(sys.argv[1],'rb').read()),'08x'))" "$bin")
oc() { timeout 180 openocd -f interface/stlink.cfg -f "$tcfg" -c "$1" 2>&1; }

idc=$(oc "init; puts \"IDCODE [capture {mdw $idaddr}]\"; shutdown" | sed -n 's/^IDCODE 0x[0-9a-f]*: \([0-9a-f]*\).*/\1/p')
case "$idc" in
  *$dev) echo "target: $tier (DBGMCU_IDCODE $idc)" ;;
  *) echo "the debugger's target is not an STM32${tier^^} (DBGMCU_IDCODE=${idc:-none})" >&2; exit 1 ;;
esac

out=$(oc "init; reset halt; flash write_image erase $bin 0x08000000;
          puts \"VERIFY:[catch {verify_image $bin 0x08000000} msg] \$msg\"; reset run; sleep 300;
          mww $cr 0; puts \"CR:[capture {mdw $cr}]\"; shutdown")
ver=$(echo "$out" | grep '^VERIFY:' || true)
case "$ver" in
  "VERIFY:0 verified"*) echo "${ver#VERIFY:0 }" ;;
  *) echo "verify FAILED: ${ver:-no result}" >&2; echo "$out" | grep -iE 'error' >&2 || true; exit 1 ;;
esac
crval=$(echo "$out" | sed -n 's/^CR:0x[0-9a-f]*: \([0-9a-f]*\).*/\1/p')
[ "$crval" = 00000000 ] || { echo "DBGMCU_CR read back $crval, not zero" >&2; exit 1; }
echo "flashed $(basename "$bin") crc32 $crc ($(stat -c %s "$bin") B) onto the $tier; DBGMCU_CR = 0 read back; OpenOCD quit."
echo "now power-cycle the $tier (replug its cable at the breakout) before a reading that counts."
