#!/usr/bin/env python3
"""Aggregate the RPi5's per-utterance RNNoise stage sums (abf_batch_file.py
--stages-log, one `rn_stages` line per utterance, CLOCK_MONOTONIC ns) into the
corpus decomposition, in the layout of the other tiers' readouts. Cycles are
ns x 2.4 at the verified 2.4 GHz clock. With the batch CSV it also reports the
stamps' coverage of the harness hop (sum of stages per frame / proc_mean).
  usage: read_rnnoise_stamps.py <stages.tsv> [batch.csv] [label]
"""
import csv, re, statistics, sys

tsv = sys.argv[1]
csvp = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith(".csv") else None
label = sys.argv[-1] if len(sys.argv) > (3 if csvp else 2) else ""
GHZ = 2.4
tot = [0] * 8; frames = sil = utts = 0
per_utt = {}
for line in open(tsv):
    uid, rest = line.rstrip("\n").split("\t", 1)
    m = re.search(r"frames=(\d+) silent=(\d+) ns((?: s\d=\d+)+)", rest)
    if not m:
        sys.exit(f"malformed line: {line.strip()}")
    st = [int(v) for v in re.findall(r"s\d=(\d+)", m.group(3))]
    n = int(m.group(1))
    frames += n; sil += int(m.group(2)); utts += 1
    tot = [a + b for a, b in zip(tot, st)]
    per_utt[uid] = (n, st)
names = ["0 hp + analysis window + 960-pt fwd transform + band E",
         "1 pitch downsample + pitch search + doubling",
         "2 pitch window + 960-pt fwd transform + band E/corr",
         "3 remaining features (log, DCTs, cepstra, variability)",
         "4 GRU network (compute_rnn)",
         "5 pitch filter + gain smoothing + interp + apply",
         "6 synthesis: 960-pt inv transform + window + OLA",
         "7 (unused)"]
T = sum(tot)
print(f"== RNNoise stage sums {label}: utterances={utts} frames={frames} silent={sil} "
      f"total={T:,} ns  per-frame={T / max(frames, 1):,.0f} ns = {GHZ * T / max(frames, 1):,.0f} cycles")
for nm, c in zip(names, tot):
    print(f"  {nm:58s} {c:>16,} ns  {c / max(frames, 1):>9,.0f} ns/frame  "
          f"{GHZ * c / max(frames, 1):>10,.0f} cyc/frame  {100 * c / max(T, 1):5.1f} %")
gru = [100 * st[4] / max(sum(st), 1) for n, st in per_utt.values()]
print(f"  GRU share per utterance: median {statistics.median(gru):.1f} %, "
      f"min {min(gru):.1f} %, max {max(gru):.1f} %")
if csvp:
    rows = [r for r in csv.DictReader(open(csvp)) if r.get("status") == "ok" and r.get("proc_mean")]
    pm = statistics.median(float(r["proc_mean"]) for r in rows)
    print(f"  harness proc_mean, corpus median of per-utterance means: {pm:,.1f} ns = {GHZ * pm:,.0f} cycles; "
          f"stamps cover {100 * (T / max(frames, 1)) / pm:.1f} % of it")
