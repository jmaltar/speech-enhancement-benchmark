"""
pareto.py -- Pareto frontiers over quality / latency / energy (task 6, serving
task 25).

CLAUDE.md is explicit that results are presented "primarily via Pareto frontier
plots (quality vs latency vs energy), not flat leaderboard tables". This module
is the machinery for that, and it enforces four project rules rather than
trusting the caller, in the same spirit as `manifest.py` refusing to hand out
two corpora at once.

  1. QUALITY IS HOST-SIDE PER (algorithm, variant); PLATFORMS CONTRIBUTE COST.
     That is a locked decision -- without it the matrix is 4 platforms x ~8
     compression variants x 4 SNR x 2+ noise types x N>=20 and is unrunnable.
     So a Pareto point is a JOIN: quality comes from the host table keyed by
     (algorithm, variant), cost comes from the platform table keyed by
     (platform, algorithm, variant). `points()` performs that join and REFUSES
     a cost table that carries its own quality column, because the next thing a
     caller would do is plot it.

  2. NEVER POOL TIERS OR CORPORA. Speech quality is PESQ/STOI/SI-SDR at 16 kHz;
     the separation tier is SDR/SIR/SAR at 44.1 kHz; NOIZEUS is narrowband at
     8 kHz. A frontier mixing them is meaningless, so the quality metric is
     declared once per figure and rows carrying a different one are refused.

  3. END-TO-END LATENCY IS `n/a` ON THE RPi5 and that is a property of the
     measurement, not a missing value to be imputed. A frontier drawn on
     `latency_e2e_ms` DROPS that tier and says so on the figure; a frontier on
     `latency_algo_ms` keeps it. Silently plotting NaN as zero would put the
     ceiling-reference tier at the origin, which is the best possible place to
     be and the exact opposite of the truth.

  4. ENERGY USES THE CORRECTED FORM. The reference PDF's section 3.6 gives
     E = P_avg x RTF, which assumes zero idle power. Duty cycle is logged from
     the first run precisely so the honest form can be used:

         E = P_active x RTF + P_idle x (1 - RTF)

     The PDF form flatters the F411 and the S3, where idle is a real share of
     the budget. `energy_mj_per_audio_s()` implements the corrected form and
     `energy_mj_per_audio_s_pdf()` the naive one, so the difference can be
     shown rather than asserted.

WHAT A "FRONTIER" MEANS HERE. A point is on the frontier if no other point is
at least as good on every objective and strictly better on at least one. Ties
are BOTH on the frontier: two configurations that are identical in every
objective do not dominate each other, and dropping one would silently hide a
real choice. This is the convention the ranking literature uses and it is the
one worth mutation-testing, because ">=" and ">" are trivially swappable.

NO REAL DATA EXISTS YET (no platform run, no power measurement), so this module
validates the way `separation.py` does: against constructed inputs with a
PREDICTED answer, not a plausible-looking one. Run it with `--validate`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
# Direction of every quantity that may appear on a frontier. Explicit rather
# than inferred from the column name: getting a direction wrong inverts the
# frontier and the plot still looks entirely reasonable.
DIRECTION = {
    # quality -- higher is better
    "pesq_mos_lqo": "max",
    "pesq_raw": "max",
    "stoi": "max",
    "estoi": "max",
    "si_sdr": "max",
    "sdr": "max", "sir": "max", "sar": "max",     # separation tier only
    # cost -- lower is better
    "latency_algo_ms": "min",
    "latency_e2e_ms": "min",
    "jitter_ms": "min",
    "rtf": "min",
    "energy_mj_per_s": "min",
    "power_mw": "min",
    "flash_bytes": "min",
    "ram_bytes": "min",
    "cycles_per_frame": "min",
}

# Which quality metrics belong to which tier. Rule 2 above: these never mix.
SPEECH_WB = ("pesq_mos_lqo", "stoi", "estoi", "si_sdr")
SPEECH_NB = ("pesq_raw",)
SEPARATION = ("sdr", "sir", "sar")
TIER_OF = ({m: "speech_wb" for m in SPEECH_WB} | {m: "speech_nb" for m in SPEECH_NB}
           | {m: "separation" for m in SEPARATION})

# Metrics that do not exist on every tier, and why. Reported on the figure.
NOT_ON_TIER = {
    "latency_e2e_ms": (
        ("rpi5",),
        "the Pi is fed files over SCP and streams no USB, so end-to-end "
        "latency has no comparable value on that tier",
    ),
}


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
def energy_mj_per_audio_s(p_active_mw, p_idle_mw, rtf):
    """Energy per second of audio, mJ/s. The corrected form.

    E = P_active x RTF + P_idle x (1 - RTF), i.e. the part of each audio second
    spent computing is billed at active power and the rest at idle.
    """
    p_active_mw = np.asarray(p_active_mw, dtype=float)
    p_idle_mw = np.asarray(p_idle_mw, dtype=float)
    rtf = np.asarray(rtf, dtype=float)
    if np.any(rtf < 0) or np.any(rtf > 1):
        raise ValueError(
            "RTF outside [0, 1]: a platform that misses real time has no "
            "steady-state duty cycle and its energy per audio-second is not "
            "defined by this formula. Report it as NO FIT instead."
        )
    return p_active_mw * rtf + p_idle_mw * (1.0 - rtf)


def energy_mj_per_audio_s_pdf(p_avg_mw, rtf):
    """The reference PDF's section 3.6 form, E = P_avg x RTF.

    Kept so the two can be compared on the same data rather than the difference
    being asserted. It assumes idle power is zero, which flatters every tier
    with a low duty cycle.
    """
    return np.asarray(p_avg_mw, dtype=float) * np.asarray(rtf, dtype=float)


def composite_efficiency(si_sdr_db, energy_mj_per_s, latency_ms):
    """Quality per unit energy x latency -- the reference PDF's section 3.7.

    SI-SDR is the numerator by decision, not PESQ: it is what the paired t-test
    applies to, and it is unbounded, which makes the ratio better behaved than a
    bounded MOS-like score compressed against its ceiling.
    """
    denom = np.asarray(energy_mj_per_s, dtype=float) * np.asarray(latency_ms, dtype=float)
    if np.any(denom <= 0):
        raise ValueError("energy x latency must be positive to form a ratio")
    return np.asarray(si_sdr_db, dtype=float) / denom


# ---------------------------------------------------------------------------
# The frontier
# ---------------------------------------------------------------------------
def _oriented(df: pd.DataFrame, objectives) -> np.ndarray:
    """Objective columns re-signed so that larger is better for all of them."""
    cols = []
    for name in objectives:
        if name not in DIRECTION:
            raise ValueError(
                f"no direction is declared for {name!r}. Add it to DIRECTION "
                f"rather than letting the frontier guess: a wrong direction "
                f"inverts the answer and the plot still looks reasonable."
            )
        v = df[name].to_numpy(dtype=float)
        cols.append(v if DIRECTION[name] == "max" else -v)
    return np.column_stack(cols)


def frontier_mask(df: pd.DataFrame, objectives) -> np.ndarray:
    """Boolean mask of the non-dominated rows.

    Row i is dominated when some row j is >= i on every objective and > i on at
    least one. Ties are therefore both kept: identical points do not dominate
    each other, and discarding one would hide a real alternative.

    Rows with a NaN in any objective are never on the frontier and never
    dominate anything -- a missing measurement is not a good score. Callers
    should drop them explicitly (see `points()`), which forces the omission to
    be stated rather than absorbed.
    """
    objectives = list(objectives)
    if not objectives:
        raise ValueError("a frontier needs at least one objective")
    x = _oriented(df, objectives)
    n = len(x)
    ok = ~np.isnan(x).any(axis=1)
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        if not ok[i]:
            continue
        others = x[ok]
        ge = (others >= x[i]).all(axis=1)
        gt = (others > x[i]).any(axis=1)
        keep[i] = not bool((ge & gt).any())
    return keep


def frontier(df: pd.DataFrame, objectives) -> pd.DataFrame:
    """The non-dominated rows, sorted by the first objective."""
    sub = df[frontier_mask(df, objectives)].copy()
    first = list(objectives)[0]
    return sub.sort_values(first, ascending=DIRECTION[first] == "min")


# ---------------------------------------------------------------------------
# Joining host quality to platform cost
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Points:
    df: pd.DataFrame
    quality_metric: str
    dropped: dict          # tier -> why it is absent from this figure


def points(quality: pd.DataFrame, cost: pd.DataFrame, *, quality_metric: str,
           cost_metrics=("latency_algo_ms", "energy_mj_per_s")) -> Points:
    """Join host-side quality onto per-platform cost. Rule 1.

    `quality` is keyed by (algorithm, variant) and carries `quality_metric`.
    `cost`    is keyed by (platform, algorithm, variant) and carries the cost
              columns -- and NOT a quality column, which is refused.
    """
    if quality_metric not in TIER_OF:
        raise ValueError(
            f"{quality_metric!r} is not a declared quality metric. Known: "
            f"{sorted(TIER_OF)}"
        )
    tier = TIER_OF[quality_metric]
    strays = [c for c in cost.columns if c in TIER_OF]
    if strays:
        raise ValueError(
            f"the cost table carries quality column(s) {strays}. Quality is "
            f"measured ONCE host-side per (algorithm, variant) and platforms "
            f"contribute cost only -- that is a locked decision. Per-platform "
            f"quality is reported only where numerical equivalence EXCEEDS "
            f"tolerance, which is a finding and belongs in its own table, not "
            f"on a frontier."
        )
    other_tier = [c for c in quality.columns
                  if c in TIER_OF and TIER_OF[c] != tier]
    if other_tier:
        raise ValueError(
            f"the quality table mixes tiers: {quality_metric!r} is {tier} but "
            f"{other_tier} belong to other tiers. A frontier over pooled tiers "
            f"is meaningless (16 kHz wideband, 8 kHz narrowband and 44.1 kHz "
            f"separation are different measurements). Split the figure."
        )

    keys = ["algorithm", "variant"]
    for name, t in (("quality", quality), ("cost", cost)):
        missing = [k for k in keys if k not in t.columns]
        if missing:
            raise ValueError(f"the {name} table is missing key column(s) {missing}")
    if "platform" not in cost.columns:
        raise ValueError("the cost table must carry a `platform` column")

    df = cost.merge(quality[keys + [quality_metric]], on=keys, how="left")
    unmatched = df[df[quality_metric].isna()]
    if len(unmatched):
        pairs = sorted(set(map(tuple, unmatched[keys].to_numpy())))
        raise ValueError(
            f"no host-side quality for {len(pairs)} (algorithm, variant) "
            f"pair(s): {pairs[:5]}{' ...' if len(pairs) > 5 else ''}. Measure "
            f"quality on the host first; a frontier cannot invent it."
        )

    dropped = {}
    for metric in cost_metrics:
        tiers, why = NOT_ON_TIER.get(metric, ((), ""))
        for plat in tiers:
            if (df.platform == plat).any():
                dropped[plat] = f"{metric} is n/a: {why}"
                df = df[df.platform != plat]
    return Points(df.reset_index(drop=True), quality_metric, dropped)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
# Distinguished by MARKER and LINESTYLE as well as colour: a journal figure has
# to survive greyscale printing, and colour-only encoding does not.
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
_LINES = ("-", "--", "-.", ":")

_AXIS_LABEL = {
    "pesq_mos_lqo": "PESQ (wideband, MOS-LQO)",
    "pesq_raw": "PESQ (narrowband, raw MOS)",
    "stoi": "STOI", "estoi": "ESTOI", "si_sdr": "SI-SDR (dB)",
    "sdr": "SDR (dB)", "sir": "SIR (dB)", "sar": "SAR (dB)",
    "latency_algo_ms": "algorithm latency (ms)",
    "latency_e2e_ms": "end-to-end latency (ms)",
    "energy_mj_per_s": "energy per audio-second (mJ/s)",
    "rtf": "real-time factor",
    "power_mw": "average power (mW)",
    "flash_bytes": "flash (B)", "ram_bytes": "RAM (B)",
    "jitter_ms": "frame-to-frame jitter (ms)",
    "cycles_per_frame": "cycles/frame",
}


def plot(pts: Points, x: str, y: str, *, group: str = "platform",
         out: Path | None = None, title: str | None = None, ax=None):
    """Scatter of every configuration with the frontier drawn through it.

    `y` is the quality axis, `x` a cost axis. One marker/line style per group,
    so the figure reads in greyscale.
    """
    import matplotlib
    if out is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pts.df
    for name in (x, y):
        if name not in df.columns:
            raise ValueError(f"column {name!r} is not in the joined table")

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.2))

    groups = list(dict.fromkeys(df[group]))
    for i, g in enumerate(groups):
        sub = df[df[group] == g]
        m, ls = _MARKERS[i % len(_MARKERS)], _LINES[i % len(_LINES)]
        ax.scatter(sub[x], sub[y], marker=m, s=38, label=str(g),
                   facecolors="none", edgecolors=f"C{i}", linewidths=1.3)
        fr = frontier(sub, (x, y))
        if len(fr) > 1:
            ax.plot(fr[x], fr[y], ls=ls, lw=1.2, color=f"C{i}", alpha=0.85)

    ax.set_xlabel(_AXIS_LABEL.get(x, x))
    ax.set_ylabel(_AXIS_LABEL.get(y, y))
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)

    # Every omission is on the figure, not in a caption someone may drop.
    if pts.dropped:
        note = "; ".join(f"{k} omitted — {v}" for k, v in pts.dropped.items())
        ax.text(0.0, -0.22, note, transform=ax.transAxes, fontsize=6.5,
                va="top", wrap=True)

    if out is not None:
        ax.figure.savefig(out, dpi=200, bbox_inches="tight")
    return ax


# ---------------------------------------------------------------------------
# Validation -- constructed inputs with a PREDICTED answer
# ---------------------------------------------------------------------------
def validate() -> bool:
    ok = True

    def check(label, got, want):
        nonlocal ok
        good = bool(np.all(got == want))
        ok &= good
        print(f"    {label:<58} {'PASS' if good else '*** FAIL ***'}")
        if not good:
            print(f"      got  {got}\n      want {want}")

    print("\n  (1) Frontier geometry, on points whose answer is known by hand.")
    # Quality up, cost right. B is beaten by A on both; D and E are tied.
    df = pd.DataFrame({
        "name":            ["A",  "B",  "C",  "D",  "E",  "F"],
        "si_sdr":          [10.0, 9.0, 12.0, 11.0, 11.0, 8.0],
        "latency_algo_ms": [5.0,  6.0, 20.0,  9.0,  9.0, 2.0],
    })
    #   A (10, 5)  non-dominated: nothing is both cheaper and better
    #   B ( 9, 6)  DOMINATED by A -- worse quality, higher latency
    #   C (12,20)  non-dominated: best quality outright
    #   D (11, 9)  non-dominated
    #   E (11, 9)  identical to D -- ties are BOTH kept
    #   F ( 8, 2)  non-dominated: cheapest outright
    got = df.name[frontier_mask(df, ("si_sdr", "latency_algo_ms"))].tolist()
    check("A C D E F on the frontier, B dominated, tie kept",
          got, ["A", "C", "D", "E", "F"])

    print("\n  (2) Direction is respected -- flipping an objective's sense must")
    print("      change the answer, or the direction table is decoration.")
    flipped = df.rename(columns={"latency_algo_ms": "rtf"})
    flipped["rtf"] = df.latency_algo_ms / 100.0
    got2 = flipped.name[frontier_mask(flipped, ("si_sdr", "rtf"))].tolist()
    check("same shape under a second 'min' objective", got2, ["A", "C", "D", "E", "F"])
    as_max = df.rename(columns={"si_sdr": "latency_e2e_ms"})
    got3 = as_max.name[frontier_mask(as_max, ("latency_e2e_ms", "latency_algo_ms"))].tolist()
    check("both objectives 'min' -> only the cheapest corner", got3, ["F"])

    print("\n  (3) A NaN is not a good score.")
    dfn = df.copy()
    dfn.loc[dfn.name == "C", "si_sdr"] = np.nan
    got4 = dfn.name[frontier_mask(dfn, ("si_sdr", "latency_algo_ms"))].tolist()
    check("row with a missing objective is off the frontier",
          got4, ["A", "D", "E", "F"])

    print("\n  (4) Energy: the corrected form against the PDF's, on a duty cycle")
    print("      where they visibly differ.")
    e_ok = energy_mj_per_audio_s(p_active_mw=120.0, p_idle_mw=8.0, rtf=0.10)
    e_pdf = energy_mj_per_audio_s_pdf(p_avg_mw=120.0, rtf=0.10)
    # 120*0.10 + 8*0.90 = 12.0 + 7.2 = 19.2 ;  PDF form gives 12.0
    check("E = P_a*RTF + P_i*(1-RTF) = 19.2 mJ/s", round(float(e_ok), 6), 19.2)
    check("PDF section 3.6 form = 12.0 mJ/s (idle ignored)", round(float(e_pdf), 6), 12.0)
    print(f"      the naive form understates this point by "
          f"{100*(1-e_pdf/e_ok):.0f}% -- which is the whole reason duty cycle is")
    print("      logged from the first run.")

    print("\n  (5) The locked rules are enforced, not documented.")
    q = pd.DataFrame({"algorithm": ["dtln"], "variant": ["int8"], "si_sdr": [12.0]})
    c = pd.DataFrame({"platform": ["h723"], "algorithm": ["dtln"], "variant": ["int8"],
                      "latency_algo_ms": [4.0], "energy_mj_per_s": [20.0]})
    for label, fn in (
        ("cost table carrying quality is refused",
         lambda: points(q, c.assign(pesq_mos_lqo=2.9), quality_metric="si_sdr")),
        ("quality table mixing tiers is refused",
         lambda: points(q.assign(sdr=6.0), c, quality_metric="si_sdr")),
        ("unmatched (algorithm, variant) is refused",
         lambda: points(q, c.assign(variant="int4"), quality_metric="si_sdr")),
        ("undeclared objective direction is refused",
         lambda: frontier_mask(df.assign(mystery=1.0), ("mystery",))),
        ("RTF outside [0,1] is refused",
         lambda: energy_mj_per_audio_s(100.0, 5.0, 1.4)),
    ):
        try:
            fn()
        except ValueError:
            print(f"    {label:<58} PASS")
        else:
            ok = False
            print(f"    {label:<58} *** FAIL (no refusal) ***")

    print("\n  (6) The RPi5 is DROPPED from an end-to-end-latency figure and the")
    print("      omission is carried on the object, not left to a caption.")
    c2 = pd.DataFrame({
        "platform": ["h723", "s3", "rpi5"],
        "algorithm": ["dtln"]*3, "variant": ["int8"]*3,
        "latency_e2e_ms": [12.0, 14.0, np.nan],
        "latency_algo_ms": [4.0, 6.0, 0.3],
        "energy_mj_per_s": [20.0, 30.0, 900.0],
    })
    p_e2e = points(q, c2, quality_metric="si_sdr",
                   cost_metrics=("latency_e2e_ms",))
    check("rpi5 absent from the e2e figure", sorted(p_e2e.df.platform), ["h723", "s3"])
    check("and the reason is recorded", "rpi5" in p_e2e.dropped, True)
    p_algo = points(q, c2, quality_metric="si_sdr",
                    cost_metrics=("latency_algo_ms",))
    check("rpi5 PRESENT on an algorithm-latency figure",
          sorted(p_algo.df.platform), ["h723", "rpi5", "s3"])

    return ok


def _demo(out: Path) -> None:
    """A figure drawn from INVENTED numbers, to exercise the plotting path.

    Labelled as invented on the figure itself. No measurement exists yet: no
    platform has run an algorithm and no power has been measured.
    """
    rng = np.random.default_rng(20260916)
    algos = [("spectral_sub", "f32", 6.0), ("wiener", "f32", 7.5),
             ("mmse_stsa", "f32", 8.4), ("rnnoise", "int8", 12.1),
             ("dtln", "int8", 15.3)]
    plats = [("f411", 1.0, 40.0), ("h723", 0.14, 180.0),
             ("s3", 0.3, 120.0), ("rpi5", 0.01, 3000.0)]
    q = pd.DataFrame([{"algorithm": a, "variant": v, "si_sdr": s} for a, v, s in algos])
    rows = []
    for p, scale, pw in plats:
        for a, v, _ in algos:
            rtf = min(0.95, scale * (0.05 + 0.25 * rng.random()))
            rows.append({"platform": p, "algorithm": a, "variant": v,
                         "latency_algo_ms": 16.0 * rtf,
                         "energy_mj_per_s": float(energy_mj_per_audio_s(pw, pw * 0.15, rtf))})
    pts = points(q, pd.DataFrame(rows), quality_metric="si_sdr")
    ax = plot(pts, "energy_mj_per_s", "si_sdr",
              title="INVENTED DATA — plotting path only, no measurement exists yet")
    ax.set_xscale("log")
    ax.figure.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n  wrote {out}  (invented numbers, labelled as such on the figure)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="run the constructed-input checks")
    ap.add_argument("--demo", type=Path, default=None,
                    help="write a demo figure from INVENTED numbers")
    a = ap.parse_args(argv)

    if not a.validate and a.demo is None:
        ap.error("nothing to do: pass --validate and/or --demo PATH")

    ok = True
    if a.validate:
        print("=== eval_harness Pareto frontier: constructed-input validation ===")
        ok = validate()
        print(f"\n=== {'PARETO PATH VALIDATED' if ok else 'FAILED'} ===")
    if a.demo is not None:
        _demo(a.demo)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
