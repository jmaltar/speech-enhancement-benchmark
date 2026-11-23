"""
power.py -- the energy axis (task 22 into task 25).

    .venv/bin/python -m analysis.power          # prints the session table; collect.py calls attach()

Reads the five RNNoise power sessions declared in registry.POWER_SESSIONS (their
power_summary.json plus, for the H723, the post-stream idle leg of the replug A/B)
and writes results/power.csv: one row per session with P_idle (post-stream), P_active
solved from the unpaced leg, the paced leg's measured energy per audio-second, and the
corrected real-time form

    E = P_active x RTF + P_idle x (1 - RTF)        (pareto.energy_mj_per_audio_s)

which is defined only for rows that FIT. attach() then puts an energy on EVERY cost row:

  * a row with its own session carries that session's figures ("measured");
  * every other FIT row is DERIVED from its board's constants (BOARD_POWER) at the row's
    own RTF. Energy integrates over time, so at the RTFs of the classical and int8 rows the
    idle term is most of it; the column `energy_pa20_pct` says how much E would move if the
    board's P_active were wrong by 20 %, so the derived figure's sensitivity is on the row;
  * a NO FIT row has no real-time energy (NaN). Where a session exists for it, the paced
    leg's figure is carried as `energy_measured_mj_s` with the wall/audio ratio beside it:
    the energy to process one audio-second in that many seconds, not a real-time figure.

The PDF's form E = P_active x RTF is carried beside it for every FIT row so the factor
between the two is a column, not an assertion.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_harness import pareto  # noqa: E402

from analysis import registry as R  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
FIT_OVERRUN_FRAC = 1e-4
SKIP_S = 3.0      # power_summary.py's skip: the serial port's stale buffer at open


def leg_mean_p(path: Path, skip: float = SKIP_S) -> tuple[float, int, float]:
    """Mean board power over a logger CSV after the first `skip` seconds (power_summary.leg's rule)."""
    rows = [r for r in csv.DictReader(open(path)) if not r["host_s"].startswith("#")]
    t0 = float(rows[0]["host_s"])
    rows = [r for r in rows if float(r["host_s"]) >= t0 + skip]
    p = [float(r["p_mW"]) for r in rows]
    return st.mean(p), len(rows), float(rows[-1]["host_s"]) - float(rows[0]["host_s"])


def session_row(s: R.PowerSession) -> dict:
    d = R.ROOT / s.dir
    js = json.load(open(d / "power_summary.json"))
    legs = js["legs"]
    U, P = legs["unpaced"], legs["paced"]
    p_idle_boot = legs.get("idle_boot", {}).get("p_mW", np.nan)
    p_idle_session = legs["idle"]["p_mW"]
    if s.idle == "power_idle.csv":
        p_idle, idle_n, idle_span = p_idle_session, legs["idle"]["n"], legs["idle"]["span_s"]
    else:
        p_idle, idle_n, idle_span = leg_mean_p(d / s.idle)
    p_active = (U["p_mW"] - (1.0 - U["duty_wall"]) * p_idle) / U["duty_wall"]
    rtf = P["rtf"]
    fit = bool(rtf < 1.0 and P["overruns"] / P["hops"] < FIT_OVERRUN_FRAC)
    e_rt = float(pareto.energy_mj_per_audio_s(p_active, p_idle, rtf)) if fit else np.nan
    e_pdf = float(pareto.energy_mj_per_audio_s_pdf(p_active, rtf)) if fit else np.nan
    return dict(
        platform=s.platform, algorithm=s.algorithm, variant=s.variant, label=s.label, dir=s.dir,
        method=s.method, source=s.source, note=s.note, fit=fit, rtf=rtf, overruns=P["overruns"], hops=P["hops"],
        p_idle_boot_mw=p_idle_boot, p_idle_session_leg_mw=p_idle_session, p_idle_mw=p_idle, idle_file=s.idle,
        idle_n=idle_n, idle_span_s=idle_span,
        p_unpaced_mw=U["p_mW"], duty_unpaced=U["duty_wall"], p_paced_mw=P["p_mW"], duty_paced=P["duty_wall"],
        p_active_mw=p_active, p_active_over_idle=p_active / p_idle - 1.0,
        wall_over_audio_paced=P["span_s"] / P["audio_s"], wall_over_audio_unpaced=U["span_s"] / U["audio_s"],
        energy_measured_paced_mj_s=P["E_per_audio_s_mJ"], energy_measured_unpaced_mj_s=U["E_per_audio_s_mJ"],
        energy_rt_mj_s=e_rt, energy_pdf_form_mj_s=e_pdf,
        energy_rt_over_pdf=(e_rt / e_pdf if fit else np.nan),
        proc_median_cyc=P["proc_median"], span_paced_s=P["span_s"], audio_s=P["audio_s"],
    )


def power_table() -> pd.DataFrame:
    return pd.DataFrame([session_row(s) for s in R.POWER_SESSIONS])


def attach(cost: pd.DataFrame, power: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add the energy columns to every cost row (see the module docstring)."""
    power = power_table() if power is None else power
    key = lambda r: (r.platform, r.algorithm, r.variant, r.label)  # noqa: E731
    own = {key(r): r for r in power.itertuples()}
    board = {plat: own[(plat, "rnnoise", R.FLOAT_GRU, lab)] for plat, (_, lab) in R.BOARD_POWER.items()}
    cols = {c: [] for c in ("p_idle_mw", "p_active_mw", "energy_mj_per_s", "energy_pdf_form_mj_s",
                            "energy_measured_mj_s", "energy_measured_wall_over_audio", "energy_pa20_pct",
                            "energy_active_share", "energy_source", "power_method")}
    for r in cost.itertuples():
        s = own.get(key(r))
        b = s if s is not None else board[r.platform]
        pi, pa = b.p_idle_mw, b.p_active_mw
        e = float(pareto.energy_mj_per_audio_s(pa, pi, r.rtf)) if r.fit else np.nan
        e_pdf = pa * r.rtf if r.fit else np.nan
        cols["p_idle_mw"].append(pi)
        cols["p_active_mw"].append(pa)
        cols["energy_mj_per_s"].append(e)
        cols["energy_pdf_form_mj_s"].append(e_pdf)
        cols["energy_measured_mj_s"].append(s.energy_measured_paced_mj_s if s is not None else np.nan)
        cols["energy_measured_wall_over_audio"].append(s.wall_over_audio_paced if s is not None else np.nan)
        cols["energy_pa20_pct"].append(100.0 * 0.2 * pa * r.rtf / e if r.fit else np.nan)
        cols["energy_active_share"].append(pa * r.rtf / e if r.fit else np.nan)
        if not r.fit:
            cols["energy_source"].append("n/a: NO FIT (no steady-state duty cycle)" +
                                         ("; paced session carried as energy_measured_mj_s" if s is not None else ""))
        elif s is not None:
            cols["energy_source"].append(f"measured session ({s.dir}); energy_mj_per_s is the corrected form")
        else:
            cols["energy_source"].append(f"derived: board constants from {b.dir} at this row's RTF")
        cols["power_method"].append(b.method)
    out = cost.copy()
    for c, v in cols.items():
        out[c] = v
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pt = power_table()
    pt.to_csv(OUT / "power.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.1f}".format):
        print(pt[["platform", "label", "fit", "rtf", "p_idle_boot_mw", "p_idle_mw", "p_active_mw", "p_paced_mw",
                  "duty_paced", "wall_over_audio_paced", "energy_measured_paced_mj_s", "energy_rt_mj_s",
                  "energy_pdf_form_mj_s"]].to_string(index=False))
    c = pd.read_csv(OUT / "cost.csv")
    ce = attach(c, pt)
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3g}".format):
        print(ce[ce.algorithm != "noisy"][["platform", "algorithm", "label", "rtf", "fit", "p_idle_mw", "p_active_mw",
                                            "energy_mj_per_s", "energy_pdf_form_mj_s", "energy_measured_mj_s",
                                            "energy_active_share", "energy_pa20_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
