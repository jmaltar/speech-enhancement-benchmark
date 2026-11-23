"""Pair two separation-tier score CSVs (eval_harness/separation.py output) track by track, B vs A.

Task 15 for the vocal-isolation tier: the Pi's estimates against the host's, at metric level.
Reports the paired per-track difference per metric -- median of A and B, median and mean of d = B - A
with BCa intervals, the largest |d| and how many tracks exceed 0.5 dB. NO locked TOST margin exists
for BSS Eval SDR (the locked margins are PESQ 0.10, STOI 0.01, SI-SDR 0.5 dB), so the 0.5 dB column
is printed for orientation and labelled as such; adopting it is a pre-registration decision.
    python -m separation.compare_tiers A.csv B.csv [--label-a host --label-b rpi5]
"""
import argparse
import numpy as np, pandas as pd
from eval_harness import stats

METRICS = ["vocals_sdr", "vocals_isr", "vocals_sir", "vocals_sar",
           "accompaniment_sdr", "accompaniment_isr", "accompaniment_sir", "accompaniment_sar"]
ORIENT_MARGIN_DB = 0.5   # the SI-SDR margin, borrowed for orientation only -- NOT locked for BSS Eval

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    o = ap.parse_args(argv)
    A = pd.read_csv(o.a); B = pd.read_csv(o.b)
    A = A[A.get("status", "ok") == "ok"] if "status" in A else A
    B = B[B.get("status", "ok") == "ok"] if "status" in B else B
    m = A.merge(B, on="track", suffixes=("_a", "_b"))
    print(f"paired tracks: {len(m)} ({len(A)} in {o.label_a}, {len(B)} in {o.label_b})")
    print(f"{'metric':<18}{'med ' + o.label_a:>12}{'med ' + o.label_b:>12}{'med d':>9}{'BCa95 on med d':>20}"
          f"{'mean d':>9}{'BCa95 on mean d':>20}{'max|d|':>8}{'|d|>0.5':>8}")
    for k in METRICS:
        a = m[f"{k}_a"].to_numpy(float); b = m[f"{k}_b"].to_numpy(float); d = b - a
        ok = np.isfinite(d); a, b, d = a[ok], b[ok], d[ok]
        lo_m, hi_m = stats.bootstrap_ci(d, statistic=np.median)
        lo, hi = stats.bootstrap_ci(d, statistic=np.mean)
        print(f"{k:<18}{np.median(a):>12.3f}{np.median(b):>12.3f}{np.median(d):>9.3f}"
              f"{'[%.3f, %.3f]' % (lo_m, hi_m):>20}{np.mean(d):>9.3f}{'[%.3f, %.3f]' % (lo, hi):>20}"
              f"{np.abs(d).max():>8.3f}{int((np.abs(d) > ORIENT_MARGIN_DB).sum()):>8}")
    print(f"\n(|d|>0.5 counts tracks beyond the SI-SDR margin of {ORIENT_MARGIN_DB} dB, borrowed for orientation only; "
          "no equivalence margin is locked for BSS Eval SDR.)")
    for k in ("vocals_sdr", "accompaniment_sdr"):
        d = (m[f"{k}_b"] - m[f"{k}_a"]); w = m.assign(d=d).reindex(d.abs().sort_values(ascending=False).index).head(5)
        print(f"\n{k}: largest |d| tracks")
        print(w[["track", f"{k}_a", f"{k}_b", "d"]].to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
