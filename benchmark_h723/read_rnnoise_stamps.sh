#!/bin/bash
# Read the RNNoise per-stage DWT sums (rnnoise_stamps.h) off a halted H723 and
# print the decomposition. Run AFTER a batch has finished and BEFORE any reset
# (reset zeroes .bss). Halts the core for the read and resumes it. Addresses
# come from the ELF the image was built from, never typed in.
#   usage: read_rnnoise_stamps.sh <image.elf> [label]
# Piped openocd prints nothing from mdw; puts [capture {...}] is the way out.
set -eu
ELF=${1:?elf}; LABEL=${2:-}
sym() { arm-none-eabi-nm "$ELF" | awk -v s="$1" '$3==s{print "0x"$1}'; }
A_ST=$(sym g_rn_stage); A_N=$(sym g_rn_stage_n); A_S=$(sym g_rn_silent_n)
# placement witnesses (rnnoise_place.c), each read by its own symbol -- they are five
# separate globals and the compiler orders them as it likes
PW=""; for w in g_place_itcm_addr g_place_itcm_bytes g_place_axi_addr g_place_axi_bytes g_place_verified; do
  a=$(sym $w); [ -n "$a" ] && PW="$PW puts \"PLACE_$w [capture {mdw $a 1}]\";"; done
RAW=$(timeout 60 openocd -f interface/stlink.cfg -f target/stm32h7x.cfg \
  -c "init; halt; puts \"STAGES [capture {mdw $A_ST 16}]\"; puts \"N [capture {mdw $A_N 1}]\"; puts \"SIL [capture {mdw $A_S 1}]\"; $PW resume; shutdown" 2>/dev/null)
python3 - "$RAW" "$LABEL" <<'PY'
import sys, re
raw, label = sys.argv[1], sys.argv[2]
def words(tag):
    m = re.search(tag + r" (.*?)(?=\n[A-Z]+ |\Z)", raw, re.S)
    return [int(w, 16) for w in re.findall(r"\b([0-9a-f]{8})\b", m.group(1)) if True][1::1] if m else []
def mem(tag):
    m = re.search(tag + r" ((?:.|\n)*?)(?=\n(?:STAGES|N|SIL|PLACE_\w+) |\Z)", raw)
    toks = re.findall(r"0x[0-9a-f]+: ((?:[0-9a-f]{8} ?)+)", m.group(1))
    return [int(w, 16) for line in toks for w in line.split()]
st = mem("STAGES"); n = mem("N")[0]; sil = mem("SIL")[0]
try:
    pl = [mem("PLACE_g_place_" + k)[0] for k in ("itcm_addr", "itcm_bytes", "axi_addr", "axi_bytes", "verified")]
    print(f"== placement witnesses: itcm_addr=0x{pl[0]:08x} itcm_bytes={pl[1]:,} axi_addr=0x{pl[2]:08x} axi_bytes={pl[3]:,} verified={pl[4]}")
except Exception:
    pass
stages = [st[2*i] | (st[2*i+1] << 32) for i in range(8)]
names = ["0 hp + analysis window + 960-pt fwd transform + band E",
         "1 pitch downsample + pitch search + doubling",
         "2 pitch window + 960-pt fwd transform + band E/corr",
         "3 remaining features (log, DCTs, cepstra, variability)",
         "4 GRU network (compute_rnn)",
         "5 pitch filter + gain smoothing + interp + apply",
         "6 synthesis: 960-pt inv transform + window + OLA",
         "7 (unused)"]
tot = sum(stages)
print(f"== RNNoise stage sums {label}: frames={n} silent={sil} total={tot:,} cycles  per-frame={tot/max(n,1):,.0f}")
for nm, c in zip(names, stages):
    print(f"  {nm:58s} {c:>16,}  {c/max(n,1):>12,.0f}/frame  {100*c/max(tot,1):5.1f} %")
PY
