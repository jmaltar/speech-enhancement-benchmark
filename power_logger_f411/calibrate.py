#!/usr/bin/env python3
"""M1 calibration (zero + scale) through the logger, one point at a time. Stdlib only.

  python3 calibrate.py point zero                 # nothing drawing through M1: logs 60 s, prints the offset
  python3 calibrate.py point f411  --meter 22.1   # a load through M1 with the multimeter in series; type its mA
  python3 calibrate.py point h723  --meter 179.4
  python3 calibrate.py point r47   --load-ohm 47.0  # no meter: a known resistor as the load, I = V_board / R
  python3 calibrate.py fit                        # offset from 'zero', gain by least squares through the origin

Each point writes results/power/calibration/<name>.csv (+ .log) via ina226_log.py (raw gain 1.0) and <name>.json with
the mean, sd and the meter reading. 'fit' prints the line to pass to ina226_log.py / power_summary.py.
"""
import argparse, csv, json, os, statistics as st, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(here), "results", "power", "calibration")

def point(a):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.name + ".csv")
    cmd = [sys.executable, os.path.join(here, "ina226_log.py"), "--out", path, "--seconds", str(a.seconds), "--quiet"]
    if a.port:
        cmd += ["--port", a.port]
    subprocess.run(cmd, check=True)
    rows = list(csv.DictReader(open(path)))
    if a.skip:
        t0 = float(rows[0]["host_s"]); rows = [r for r in rows if float(r["host_s"]) - t0 >= a.skip]
    i = [float(r["i_mA"]) for r in rows]; v = [float(r["v_board_V"]) for r in rows]
    meter = a.meter
    if a.load_ohm:                       # the INA226's own bus reading (0.1 % gain error) over a known resistor
        meter = st.mean(v) / a.load_ohm * 1e3
    rec = dict(name=a.name, n=len(i), seconds=a.seconds, i_mA=st.mean(i), i_sd=st.pstdev(i), i_min=min(i), i_max=max(i),
               v_V=st.mean(v), meter_mA=meter, meter_model=a.meter_model, load_ohm=a.load_ohm)
    json.dump(rec, open(os.path.join(OUT, a.name + ".json"), "w"), indent=1)
    print(f"{a.name}: n={rec['n']} I={rec['i_mA']:.3f} ± {rec['i_sd']:.3f} mA (min {rec['i_min']:.3f} max {rec['i_max']:.3f}) "
          f"V={rec['v_V']:.3f}" + (f"  reference={meter:.3f} mA  ratio={meter/rec['i_mA']:.4f}" if meter else ""))

def fit(a):
    recs = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json")] if os.path.isdir(OUT) else []
    zero = [r for r in recs if r["name"] == "zero"]
    off = zero[0]["i_mA"] if zero else 0.0
    pts = [r for r in recs if r.get("meter_mA")]
    if not pts:
        sys.exit("no points with --meter yet")
    x = [r["i_mA"] - off for r in pts]; y = [r["meter_mA"] for r in pts]
    gain = sum(a * b for a, b in zip(x, y)) / sum(a * a for a in x)          # least squares through the origin
    print(f"zero offset: {off:+.4f} mA ({'from zero point' if zero else 'no zero point logged, assumed 0'})")
    for r, xi, yi in zip(pts, x, y):
        print(f"  {r['name']:>8}: ina {xi:8.3f} mA  meter {yi:8.3f} mA  ratio {yi/xi:.4f}  residual {yi - gain*xi:+.3f} mA")
    print(f"gain = {gain:.4f}  ->  ina226_log.py --gain {gain:.4f}   (the zero offset is {off:+.3f} mA, "
          f"{abs(off)/0.025:.1f} LSB; subtract it in the summary if it matters)")
    json.dump(dict(zero_offset_mA=off, gain=gain, points=pts), open(os.path.join(OUT, "fit.json"), "w"), indent=1)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("point"); p.add_argument("name"); p.add_argument("--meter", type=float, help="multimeter reading, mA")
p.add_argument("--meter-model"); p.add_argument("--load-ohm", type=float, help="known load resistor: reference = V_board/R")
p.add_argument("--seconds", type=float, default=60); p.add_argument("--skip", type=float, default=3)
p.add_argument("--port"); p.set_defaults(fn=point)
f = sub.add_parser("fit"); f.set_defaults(fn=fit)
a = ap.parse_args(); a.fn(a)
