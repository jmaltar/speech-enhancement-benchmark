#!/bin/bash
# Read the throughput probe's results struct (g_bench) off a halted H723 through GDB over OpenOCD's
# server -- the "belt and braces" route of docs/retired/f411-bench-notes.tex (print g_bench). The probe
# runs at boot and leaves its results in DTCM; this halts, prints, resumes.
#   usage: read_probe.sh <image.elf> > out.txt
set -eu
ELF=${1:?elf}
openocd -f interface/stlink.cfg -f target/stm32h7x.cfg -c "init; halt" >/dev/null 2>&1 &
OPID=$!; sleep 2
gdb-multiarch -batch -q -ex "set pagination off" -ex "set print pretty off" -ex "set print elements 0" -ex "set print repeats 0" \
  -ex "target extended-remote :3333" -ex "print g_bench" -ex "monitor resume" -ex "monitor shutdown" "$ELF" 2>/dev/null
wait $OPID 2>/dev/null || true
