#!/usr/bin/env python3
"""PMIC power logger for the Raspberry Pi 5 (task 22, the Pi's method, decided 2026-11-02).

Samples `vcgencmd pmic_read_adc` once a second and writes P = sum over rails of V x I to a CSV with the columns
power_summary.py reads (host_s, seq, t_us, i_mA, v_board_V, p_mW) plus one column per rail. v_board_V is the PMIC's
EXT5V rail (the 5 V input) and i_mA is p_mW / v_board_V: the equivalent input current, not a measured one.

What this figure is NOT: it excludes the PMIC's own conversion losses and every load fed from 5 V directly (USB
ports, fan, HDMI 5 V), all of which the MCU figures, taken at the board's 5 V input, include. The Pi is flattered by
roughly the converter loss; the paper states it beside every Pi energy figure (CLAUDE.md, Power Measurement).

Run it pinned to a core the benchmark does not use (the batches run on core 2):
    taskset -c 0 python3 pmic_log.py --out idle.csv --seconds 60
`--once` prints one parsed sample and exits (the format check)."""
import argparse, re, signal, subprocess, sys, time

LINE = re.compile(r"^\s*(\S+?)_([AV])\s+\w+\(\d+\)=([-0-9.]+)[AV]\s*$")


def read_pmic():
    """{'rail': {'A': amps, 'V': volts}}, plus 'EXT5V'/'BATT' voltage-only rails."""
    txt = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2).stdout
    rails = {}
    for ln in txt.splitlines():
        m = LINE.match(ln)
        if m:
            rails.setdefault(m.group(1), {})[m.group(2)] = float(m.group(3))
    if not rails:
        raise RuntimeError(f"pmic_read_adc gave nothing parseable: {txt!r}")
    return rails


def sample():
    rails = read_pmic()
    per = {r: v["A"] * v["V"] for r, v in rails.items() if "A" in v and "V" in v}
    p_w = sum(per.values())
    v5 = rails.get("EXT5V", {}).get("V", float("nan"))
    return p_w, v5, per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="pmic.csv"); ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--period", type=float, default=1.0); ap.add_argument("--once", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    p_w, v5, per = sample()
    if a.once:
        for r, p in sorted(per.items(), key=lambda kv: -kv[1]):
            print(f"{r:12s} {p*1e3:8.1f} mW")
        print(f"{'TOTAL':12s} {p_w*1e3:8.1f} mW   EXT5V {v5:.3f} V   rails with V and I: {len(per)}")
        return
    names = sorted(per)
    csv = open(a.out, "w")
    csv.write("host_s,seq,t_us,i_mA,v_board_V,p_mW," + ",".join(f"p_{r}_mW" for r in names) + "\n")
    csv.write("# method=vcgencmd pmic_read_adc, P=sum(V*I) over rails; excludes PMIC conversion loss and 5V-direct "
              "loads; v_board_V=EXT5V, i_mA=p_mW/v_board_V (equivalent)\n")
    # SIGTERM ends the run like Ctrl-C; SIGINT is ignored in a bash background job (see ina226_log.py)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    t0 = time.time(); seq = 0; nxt = t0
    try:
        while not a.seconds or time.time() - t0 < a.seconds:
            now = time.time()
            if now < nxt:
                time.sleep(nxt - now); continue
            nxt += a.period
            try:
                p_w, v5, per = sample()
            except Exception as e:        # one bad read is logged, never fatal
                print(f"E,{seq},{e}", file=sys.stderr); seq += 1; continue
            t_us = int((time.time() - t0) * 1e6)
            i_ma = p_w / v5 * 1e3 if v5 == v5 and v5 > 0 else float("nan")
            csv.write(f"{time.time():.4f},{seq},{t_us},{i_ma:.3f},{v5:.4f},{p_w*1e3:.2f},"
                      + ",".join(f"{per.get(r, 0)*1e3:.2f}" for r in names) + "\n")
            csv.flush(); seq += 1
            if not a.quiet and seq % 10 == 0:
                print(f"{time.time()-t0:7.0f} s  P={p_w*1e3:7.1f} mW  EXT5V={v5:.3f} V  n={seq}", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    csv.close()


if __name__ == "__main__":
    main()
