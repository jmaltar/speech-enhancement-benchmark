#!/usr/bin/env python3
"""Decode the LAST `RN_STAGES` line of an S3 console log into the per-stage
decomposition, in the same layout as the Cortex-M tiers' read_rnnoise_stamps.sh
so the four tiers' tables can be read side by side. The sums are cumulative
since boot (CCOUNT cycles at 240 MHz, read on the core that runs RNNoise).
  usage: read_rnnoise_stamps.py <console.txt> [label]
"""
import re, sys

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else ""
lines = [l for l in open(path, errors="replace") if "RN_STAGES" in l]
if not lines:
    sys.exit(f"no RN_STAGES line in {path}")
m = re.search(r"RN_STAGES n=(\d+) sil=(\d+)((?: s\d=[0-9a-f]{16})+)", lines[-1])
if not m:
    sys.exit(f"malformed RN_STAGES line: {lines[-1].strip()}")
n, sil = int(m.group(1)), int(m.group(2))
stages = [int(v, 16) for v in re.findall(r"s\d=([0-9a-f]{16})", m.group(3))]
names = ["0 hp + analysis window + 960-pt fwd transform + band E",
         "1 pitch downsample + pitch search + doubling",
         "2 pitch window + 960-pt fwd transform + band E/corr",
         "3 remaining features (log, DCTs, cepstra, variability)",
         "4 GRU network (compute_rnn)",
         "5 pitch filter + gain smoothing + interp + apply",
         "6 synthesis: 960-pt inv transform + window + OLA",
         "7 (unused)"]
tot = sum(stages)
print(f"== RNNoise stage sums {label}: frames={n} silent={sil} total={tot:,} cycles  "
      f"per-frame={tot / max(n, 1):,.0f}   ({len(lines)} RN_STAGES line(s) in the log; the last is used)")
for nm, c in zip(names, stages):
    print(f"  {nm:58s} {c:>16,}  {c / max(n, 1):>12,.0f}/frame  {100 * c / max(tot, 1):5.1f} %")
