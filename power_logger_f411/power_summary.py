#!/usr/bin/env python3
"""Summarise one power session (power_session.sh): mean power per leg, the board's duty cycle from its own cycle
counts, and the energy per audio-second, both measured (paced) and in the corrected form
E = P_active x RTF + P_idle x (1 - RTF) with P_active solved from the unpaced leg.

  python3 power_summary.py <outdir> --clk-mhz 550 --hop-s 0.010 [--skip-s 3]

The first --skip-s seconds of every logger CSV are dropped (the serial port hands over a stale buffer on open).
Writes <outdir>/power_summary.json beside the markdown it prints."""
import argparse, csv, glob, json, os, statistics as st, sys


def leg(path, skip):
    rows = [r for r in csv.DictReader(open(path)) if not r["host_s"].startswith("#")]   # pmic_log.py writes a method line
    if not rows:
        return None
    t0 = float(rows[0]["host_s"])
    rows = [r for r in rows if float(r["host_s"]) >= t0 + skip]
    i = [float(r["i_mA"]) for r in rows]; v = [float(r["v_board_V"]) for r in rows]; p = [float(r["p_mW"]) for r in rows]
    span = float(rows[-1]["host_s"]) - float(rows[0]["host_s"])
    return dict(n=len(rows), span_s=span, i_mA=st.mean(i), i_sd=st.pstdev(i), v_V=st.mean(v), p_mW=st.mean(p),
                p_min=min(p), p_max=max(p))


def batch(outdir, legname):
    cs = [c for c in glob.glob(os.path.join(outdir, legname, "*.csv"))]
    if not cs:
        return None
    rows = [r for c in cs for r in csv.DictReader(open(c)) if r.get("status") == "ok"]
    hops = sum(int(r["hops"]) for r in rows)
    busy = sum(float(r["proc_mean"]) * int(r["hops"]) for r in rows if r["proc_mean"])
    med = st.median(float(r["proc_median_data"]) for r in rows if r["proc_median_data"])
    return dict(utterances=len(rows), hops=hops, busy_cycles=busy, proc_median=med,
                overruns=sum(int(r["overruns"]) for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir"); ap.add_argument("--clk-mhz", type=float, required=True)
    ap.add_argument("--hop-s", type=float, required=True); ap.add_argument("--skip-s", type=float, default=3.0)
    a = ap.parse_args()
    budget = a.clk_mhz * 1e6 * a.hop_s
    out = {"clk_mhz": a.clk_mhz, "hop_s": a.hop_s, "budget_cycles": budget, "legs": {}}
    for name in ("idle_boot", "idle", "unpaced", "paced"):      # idle_boot is recorded, never used in the model
        f = os.path.join(a.outdir, f"power_{name}.csv")
        if os.path.exists(f):
            L = leg(f, a.skip_s)
            if L:
                b = batch(a.outdir, name) if name != "idle" else None
                if b:
                    b["rtf"] = b["proc_median"] / budget
                    b["audio_s"] = b["hops"] * a.hop_s
                    b["duty_wall"] = b["busy_cycles"] / (a.clk_mhz * 1e6 * L["span_s"])
                    L["E_per_audio_s_mJ"] = L["p_mW"] * L["span_s"] / b["audio_s"]
                    L.update(b)
                out["legs"][name] = L
    legs = out["legs"]
    print(f"# power session: {a.outdir}  (clk {a.clk_mhz:g} MHz, hop {a.hop_s*1e3:g} ms, budget {budget:,.0f} cycles)\n")
    print("| leg | n | span s | I mA | V | P mW | duty (board cycles / wall) | RTF | overruns | E mJ per audio-s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, L in legs.items():
        print(f"| {name} | {L['n']} | {L['span_s']:.0f} | {L['i_mA']:.2f} ± {L['i_sd']:.2f} | {L['v_V']:.3f} | "
              f"{L['p_mW']:.1f} | {L.get('duty_wall', 0):.3f} | {L.get('rtf', 0):.3f} | {L.get('overruns','')} | "
              f"{L.get('E_per_audio_s_mJ', float('nan')):.1f} |")
    if "idle" in legs and "unpaced" in legs and legs["unpaced"].get("duty_wall"):
        Pi = legs["idle"]["p_mW"]; U = legs["unpaced"]
        Pa = Pi + (U["p_mW"] - Pi) / U["duty_wall"]
        out["P_idle_mW"] = Pi; out["P_active_mW"] = Pa
        print(f"\nP_idle = {Pi:.1f} mW, P_active (solved from the unpaced leg at duty {U['duty_wall']:.3f}) = {Pa:.1f} mW")
        if "paced" in legs:
            P = legs["paced"]; rtf = P["rtf"]
            pred = Pa * rtf + Pi * (1 - rtf); pdf = Pa * rtf
            out["E_corrected_mJ_s"] = pred; out["E_pdf_form_mJ_s"] = pdf; out["E_measured_paced_mJ_s"] = P["E_per_audio_s_mJ"]
            print(f"paced, measured: {P['E_per_audio_s_mJ']:.1f} mJ per audio-second at RTF {rtf:.3f}")
            print(f"corrected form P_active*RTF + P_idle*(1-RTF) = {pred:.1f} mJ/s; the PDF's P_active*RTF alone = {pdf:.1f} mJ/s")
    json.dump(out, open(os.path.join(a.outdir, "power_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
