"""
frontiers.py -- Pareto frontiers on real data through eval_harness/pareto.py (task 25).

    .venv/bin/python -m analysis.frontiers

The join is pareto.points(): host quality per (algorithm, variant) from
results/quality_classes.csv, platform cost per (platform, algorithm, variant)
from results/cost.csv -- the cost table carries no quality column, so the locked
rule is enforced by the module, not by this script.

VoiceBank only. It is the one corpus every algorithm was scored on (RNNoise
through the voicebank16k <-> voicebank48k pairing, filter-matched references)
and the one every cost run streamed. DNS and NOIZEUS have quality rows but no
cost run of their own geometry (NOIZEUS is 8 kHz; DNS ran only DTLN on two
tiers), so a frontier there would join one corpus's quality to another's cost.

Two frontiers per tier: over the configurations that FIT (RTF < 1, fewer than 1
late hop in 10^4) -- the deployable frontier, drawn -- and over all measured
configurations, listed in frontier_membership.csv. Points that miss real time are
drawn as grey crosses, never joined.

ENERGY IS AN AXIS since 2026-11-06 (task 22 measured on every tier, 2026-11-04/05).
E = P_active x RTF + P_idle x (1 - RTF) per audio-second (pareto.energy_mj_per_audio_s),
with each board's P_idle / P_active from its RNNoise power session (analysis/power.py,
results/power.csv): rows with a session of their own are MEASURED (filled markers), every
other FIT row is DERIVED at its RTF from the board's constants, and a NO FIT row has no
real-time energy and is drawn as a grey cross at its measured cost-to-process where a
session exists. Energy is comparable across tiers (unlike e2e latency), so the energy
figure also draws the ladder-wide frontier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_harness import pareto  # noqa: E402

from analysis import registry as R  # noqa: E402
from analysis.collect import OUT  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
ENERGY_NOTE = ("Energy per audio-second: E = P_a·RTF + P_i·(1−RTF), P_i / P_a per board from its RNNoise power session "
               "(INA226 at the 5 V input on the MCUs, incl. regulator/LED/bridge, shunt ±1 %; PMIC telemetry on the Pi, "
               "which omits converter loss and 5 V-direct loads). Filled = the row's own session; open = derived at the "
               "row's RTF; NO FIT rows have no real-time energy (grey ×: measured cost to process, wall > audio).")
RTF_NOTE = "Energy axis: frontier_energy_<metric> (same points, E = P_a·RTF + P_i·(1−RTF) from the power sessions)."
SHORT = {("specsub", "MCRA"): "SpecSub", ("wiener", "MCRA"): "Wiener", ("mmse", "MCRA"): "MMSE",
         ("rnnoise", R.FLOAT_GRU): "RNNoise", ("rnnoise", R.INT8_GRU): "RNNoise",
         ("dtln", R.DTLN_P): "DTLN-p34", ("dtln", R.DTLN_F): "DTLN-full"}
LABEL = {"as built": "", "as shipped": "", "TFLM reference kernels": " ref", "CMSIS-NN s16 (DTCM arenas)": " CMSIS-NN"}
METRICS = ("si_sdr", "pesq_mos_lqo", "stoi")
MARGIN = {"si_sdr": 0.50, "pesq_mos_lqo": 0.10, "stoi": 0.01}   # the locked TOST margins


def eps_frontier(df, metric, x="rtf"):
    """Margin-aware frontier: a point is dropped if a STRICTLY CHEAPER point (on cost axis `x`) is no worse
    than it by more than the locked margin (q_j >= q_i - margin). Strict Pareto (pareto.frontier_mask) keeps
    a costlier point that is better by 0.0005 PESQ; this keeps only the cheapest configuration within each
    margin band -- the frontier a reader can act on, given that smaller quality differences are declared
    equivalent by the project's own TOST rule."""
    m = MARGIN[metric]
    r, q = df[x].to_numpy(), df[metric].to_numpy()
    keep = np.ones(len(df), bool)
    for i in range(len(df)):
        if np.any((r < r[i]) & (q >= q[i] - m)):
            keep[i] = False
    return keep


def _label_clusters(ax, df, metric, x="rtf", by_platform=False, stagger=False):
    """One label per cluster of near-coincident points (same algorithm, cost axis `x` within 15 %).
    `stagger`: avoid label-on-label collisions by trying a few offsets (the column-width panels)."""
    used = np.zeros(len(df), bool)
    placed = []                                    # (x0, y0, x1, y1) in points, stagger only
    rows = list(df.itertuples())
    for i, r in enumerate(rows):
        if used[i]:
            continue
        tol = 0.07 if r.fit else 0.35          # NO FIT points: one label per algorithm cluster
        grp = [j for j, s in enumerate(rows) if not used[j] and s.algorithm == r.algorithm and s.fit == r.fit
               and (not by_platform or s.platform == r.platform)
               and abs(np.log10(getattr(s, x) / getattr(r, x))) < tol
               and abs(getattr(s, metric) - getattr(r, metric)) < 0.3 * MARGIN[metric] + 1e-9]
        for j in grp:
            used[j] = True
        names = [rows[j].config for j in grp]
        base = SHORT_ALGO.get(r.algorithm, "")
        if len(names) == 1:
            text = names[0]
        elif base and all(n.startswith(base) for n in names):
            text = base + " " + "/".join(n[len(base):].strip() or "ship" for n in names)
        else:
            text = "/".join(names)
        if by_platform:
            text = f"{R.PLATFORM_SHORT[r.platform]} {text}"
        offset = (3, 3)
        if stagger:
            fs = 6.5
            w, h = 0.52 * fs * len(text), fs
            px, py = ax.transData.transform((getattr(r, x), getattr(r, metric))) * 72.0 / ax.figure.dpi
            right = ((3, 3), (3, -9), (3, 10), (-w - 3, 3), (-w - 3, -9), (3, -17))
            if len(grp) > 1:                       # a cluster: lift the label clear of its other markers
                right = ((3, 8), (3, -12), (-w - 3, 8), (-w - 3, -12), (3, 18))
            left = ((-w - 3, -9), (-w - 3, 3), (-w - 3, 10), (3, 3), (3, -9), (-w - 3, -17))
            ax_x0, ax_x1 = [v * 72.0 / ax.figure.dpi for v in (ax.bbox.x0, ax.bbox.x1)]
            for dx, dy in (left if px > ax_x0 + 0.62 * (ax_x1 - ax_x0) else right):
                box = (px + dx, py + dy, px + dx + w, py + dy + h)
                if not any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in placed):
                    offset = (dx, dy)
                    break
            placed.append((px + offset[0], py + offset[1], px + offset[0] + w, py + offset[1] + h))
        ax.annotate(text, (getattr(r, x), getattr(r, metric)), fontsize=6.5, xytext=offset, textcoords="offset points",
                    color="0.2" if r.fit else "0.5")


SHORT_ALGO = {"rnnoise": "RNNoise", "dtln": "DTLN-p34"}


def _tables():
    qc = pd.read_csv(OUT / "quality_classes.csv")
    q = qc[qc.corpus.isin(["voicebank16k", "voicebank48k"]) & (qc.algorithm != "noisy")]
    noisy = qc[(qc.algorithm == "noisy") & (qc.corpus == "voicebank16k")].iloc[0]
    c = pd.read_csv(OUT / "cost.csv")
    c = c[c.algorithm != "noisy"].copy()
    c["config"] = [f"{SHORT[(a, v)]}{LABEL.get(l, ' ' + l)}" for a, v, l in zip(c.algorithm, c.variant, c.label)]
    c["latency_e2e_ms"] = c.e2e_ms
    cost = c[["platform", "algorithm", "variant", "label", "config", "rtf", "latency_e2e_ms", "latency_algo_ms", "fit",
              "paced", "duty_cycle", "energy_mj_per_s", "energy_measured_mj_s", "energy_measured_wall_over_audio",
              "energy_source", "energy_active_share", "energy_pa20_pct"]]
    return q, cost, noisy


def membership(q, cost):
    rows = []
    for metric in METRICS:
        pts = pareto.points(q, cost, quality_metric=metric, cost_metrics=("rtf",)).df
        for plat in R.PLATFORMS:
            sub = pts[pts.platform == plat]
            all_m = pareto.frontier_mask(sub, ("rtf", metric))
            fit = sub[sub.fit]
            fit_m = pd.Series(pareto.frontier_mask(fit, ("rtf", metric)), index=fit.index) if len(fit) else pd.Series(dtype=bool)
            eps_m = pd.Series(eps_frontier(fit, metric), index=fit.index) if len(fit) else pd.Series(dtype=bool)
            for (i, r), on_all in zip(sub.iterrows(), all_m):
                rows.append(dict(metric=metric, platform=plat, config=r.config, rtf=r.rtf, quality=r[metric],
                                 fit=r.fit, on_frontier_all=bool(on_all),
                                 on_frontier_fit=bool(fit_m.get(i, False)),
                                 on_frontier_fit_margin=bool(eps_m.get(i, False))))
    return pd.DataFrame(rows)


def fig_rtf(q, cost, noisy, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), sharey=True)
    for ax, plat in zip(axes.flat, R.PLATFORMS):
        pts = pareto.points(q, cost[cost.platform == plat], quality_metric=metric, cost_metrics=("rtf",))
        fitp = pareto.Points(pts.df[pts.df.fit].reset_index(drop=True), metric, pts.dropped)
        if len(fitp.df):
            pareto.plot(fitp, "rtf", metric, group="platform", ax=ax)
        miss = pts.df[~pts.df.fit]
        ax.scatter(miss.rtf, miss[metric], marker="x", color="0.45", s=36, label="misses real time")
        _label_clusters(ax, pts.df, metric)
        ax.axhline(noisy[metric], color="0.6", lw=0.8, ls=":", label="noisy input")
        ax.axvspan(1.0, 10.0, color="0.92", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(5e-4 if plat == "rpi5" else 5e-3, 8)
        ax.set_title(R.PLATFORM_NAME[plat], fontsize=9.5)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"VoiceBank-DEMAND (824 utt.): host quality vs measured RTF per tier — "
                 f"frontier over configurations that FIT; shaded RTF ≥ 1", fontsize=10)
    fig.text(0.01, 0.005, RTF_NOTE, fontsize=7)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"frontier_rtf_{metric}.{ext}", dpi=200)
    plt.close(fig)


def fig_rtf_panels(q, cost, noisy, metric):
    """One panel per tier at column width (3.5 x 2.7 in, fonts at their final size), for the
    paper: the 2x2 grid of fig_rtf() at single-column width was unreadable. Same points, same
    frontier rule, no suptitle; the tier name is the panel title. Written 2026-11-06."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rc = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7}
    with plt.rc_context(rc):
        for plat in R.PLATFORMS:
            fig, ax = plt.subplots(figsize=(3.5, 2.7))
            pts = pareto.points(q, cost[cost.platform == plat], quality_metric=metric, cost_metrics=("rtf",))
            fitp = pareto.Points(pts.df[pts.df.fit].reset_index(drop=True), metric, pts.dropped)
            if len(fitp.df):
                pareto.plot(fitp, "rtf", metric, group="platform", ax=ax)
            miss = pts.df[~pts.df.fit]
            ax.scatter(miss.rtf, miss[metric], marker="x", color="0.45", s=30, label="misses real time")
            ax.axhline(noisy[metric], color="0.6", lw=0.8, ls=":", label="noisy input")
            ax.axvspan(1.0, 10.0, color="0.92", zorder=0)
            ax.set_xscale("log")
            ax.set_xlim(5e-4 if plat == "rpi5" else 5e-3, 8)
            ax.margins(y=0.16)
            ax.set_title(R.PLATFORM_NAME[plat])
            handles, labels = ax.get_legend_handles_labels()
            labels = ["fits real time, frontier drawn" if l == plat else l for l in labels]
            ax.legend(handles, labels, fontsize=6.5, frameon=True, framealpha=0.9, edgecolor="none", loc="lower right")
            fig.tight_layout(pad=0.4)
            _label_clusters(ax, pts.df, metric, stagger=True)   # after layout: offsets are in points
            fig.savefig(FIG / f"frontier_rtf_{metric}_{plat}.pdf")
            plt.close(fig)


def fig_e2e(q, cost, noisy, metric="si_sdr"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = pareto.points(q, cost[cost.fit], quality_metric=metric, cost_metrics=("latency_e2e_ms",))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    pareto.plot(pts, "latency_e2e_ms", metric, group="platform", ax=ax)
    for (plat, algo), g in pts.df.groupby(["platform", "algorithm"], sort=False):
        if algo in ("rnnoise", "dtln") and len(g) > 1:     # one label per cluster of builds
            r = g.sort_values("latency_e2e_ms").iloc[0]
            text = f"{SHORT_ALGO[algo]} {'/'.join(x.split(' ', 1)[1] if ' ' in x else 'ship' for x in g.config)}"
            ax.annotate(text, (r.latency_e2e_ms, r[metric]), fontsize=6, xytext=(3, -9), textcoords="offset points")
            continue
        for r in g.itertuples():
            ax.annotate(f"{r.config}{'' if r.paced else '*'}", (r.latency_e2e_ms, getattr(r, metric)), fontsize=6,
                        xytext=(3, 2 if plat != "f411" else -8), textcoords="offset points")
    ax.axhline(noisy[metric], color="0.6", lw=0.8, ls=":")
    ax.set_title("VoiceBank: host quality vs measured end-to-end latency (MCU tiers, FIT configurations)",
                 fontsize=9)
    fig.text(0.01, 0.005, "* unpaced run (classical on every MCU, H723 DTLN): e2e includes host-side backlog, an upper bound. "
             + RTF_NOTE, fontsize=6.2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"frontier_e2e_{metric}.{ext}", dpi=200)
    plt.close(fig)
    return pts.dropped


def energy_membership(q, cost):
    """Per-tier and ladder-wide frontier membership on the energy axis (FIT rows with an energy), strict and
    margin-aware, plus the composite efficiency of the reference methodology (SI-SDR per mJ·ms, with the
    transport-independent algorithm latency so the Pi is included)."""
    rows = []
    for metric in METRICS:
        pts = pareto.points(q, cost[cost.fit & cost.energy_mj_per_s.notna()], quality_metric=metric,
                            cost_metrics=("energy_mj_per_s",)).df
        all_m = pd.Series(pareto.frontier_mask(pts, ("energy_mj_per_s", metric)), index=pts.index)
        all_eps = pd.Series(eps_frontier(pts, metric, "energy_mj_per_s"), index=pts.index)
        for plat in R.PLATFORMS:
            sub = pts[pts.platform == plat]
            if not len(sub):
                continue
            tier_m = pd.Series(pareto.frontier_mask(sub, ("energy_mj_per_s", metric)), index=sub.index)
            tier_eps = pd.Series(eps_frontier(sub, metric, "energy_mj_per_s"), index=sub.index)
            for i, r in sub.iterrows():
                comp = (float(pareto.composite_efficiency(r[metric], r.energy_mj_per_s, r.latency_algo_ms))
                        if metric == "si_sdr" else np.nan)
                rows.append(dict(metric=metric, platform=plat, config=r.config, rtf=r.rtf,
                                 energy_mj_per_s=r.energy_mj_per_s, quality=r[metric],
                                 measured=r.energy_measured_mj_s == r.energy_measured_mj_s,
                                 energy_source=r.energy_source, active_share=r.energy_active_share,
                                 pa20_pct=r.energy_pa20_pct,
                                 on_tier_frontier=bool(tier_m[i]), on_tier_frontier_margin=bool(tier_eps[i]),
                                 on_ladder_frontier=bool(all_m[i]), on_ladder_frontier_margin=bool(all_eps[i]),
                                 composite_si_sdr_per_mj_ms=comp))
    return pd.DataFrame(rows)


def fig_energy(q, cost, noisy, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fitc = cost[cost.fit & cost.energy_mj_per_s.notna()]
    pts = pareto.points(q, fitc, quality_metric=metric, cost_metrics=("energy_mj_per_s",))
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    pareto.plot(pts, "energy_mj_per_s", metric, group="platform", ax=ax)       # per-tier frontiers
    df = pts.df
    meas = df[df.energy_measured_mj_s.notna()]
    for i, plat in enumerate(R.PLATFORMS):
        m = meas[meas.platform == plat]
        if len(m):
            ax.scatter(m.energy_mj_per_s, m[metric], marker="o", s=22, color=f"C{i}", zorder=4)
    fr = pareto.frontier(df, ("energy_mj_per_s", metric)).sort_values("energy_mj_per_s")
    ax.plot(fr.energy_mj_per_s, fr[metric], ls="--", lw=1.0, color="0.15", alpha=0.9, label="ladder-wide frontier")
    # NO FIT rows with a measured session: the cost to process, not a real-time figure
    nf = pareto.points(q, cost[~cost.fit & cost.energy_measured_mj_s.notna()], quality_metric=metric,
                       cost_metrics=("energy_measured_mj_s",)).df
    ax.scatter(nf.energy_measured_mj_s, nf[metric], marker="x", color="0.45", s=36,
               label="NO FIT: measured cost to process (wall > audio)")
    for k, r in enumerate(nf.sort_values("energy_measured_mj_s").itertuples()):
        ax.annotate(f"{R.PLATFORM_SHORT[r.platform]} {r.config} ship, NO FIT ({r.energy_measured_wall_over_audio:.2f}× wall)",
                    (r.energy_measured_mj_s, getattr(r, metric)), fontsize=6, xytext=(3, -9 - 8 * k),
                    textcoords="offset points", color="0.45")
    _label_clusters(ax, df, metric, x="energy_mj_per_s", by_platform=True)
    ax.axhline(noisy[metric], color="0.6", lw=0.8, ls=":", label="noisy input")
    ax.set_xscale("log")
    ax.set_xlim(90, 3500)
    ax.set_title(f"VoiceBank-DEMAND (824 utt.): host quality vs energy per audio-second, all four tiers — "
                 f"configurations that FIT", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.text(0.01, 0.005, ENERGY_NOTE, fontsize=6, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"frontier_energy_{metric}.{ext}", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    q, cost, noisy = _tables()
    mem = membership(q, cost)
    mem.to_csv(OUT / "frontier_membership.csv", index=False)
    for m in METRICS:
        fig_rtf(q, cost, noisy, m)
        fig_rtf_panels(q, cost, noisy, m)
    dropped = fig_e2e(q, cost, noisy)
    print("e2e figure dropped:", dropped)
    em = energy_membership(q, cost)
    em.to_csv(OUT / "frontier_energy_membership.csv", index=False)
    for m in METRICS:
        fig_energy(q, cost, noisy, m)
    pd.set_option("display.width", 200)
    for m in METRICS:
        s = mem[(mem.metric == m)]
        print(f"\n=== {m}: deployable frontier (FIT points) per tier")
        for plat in R.PLATFORMS:
            on = s[(s.platform == plat) & s.on_frontier_fit].sort_values("rtf")
            off = s[(s.platform == plat) & s.on_frontier_all & ~s.fit].sort_values("rtf")
            eps = s[(s.platform == plat) & s.on_frontier_fit_margin].sort_values("rtf")
            print(f"  {plat:5s} margin-aware FIT frontier: " + ", ".join(r.config for r in eps.itertuples()))
            print(f"  {plat:5s} FIT frontier: " + ", ".join(f"{r.config} ({r.rtf:.4g}, {r.quality:.3f})" for r in on.itertuples())
                  + (f" | all-points frontier adds (NO FIT): " + ", ".join(r.config for r in off.itertuples()) if len(off) else ""))
    for m in METRICS:
        s = em[em.metric == m]
        print(f"\n=== {m}: ENERGY frontier (FIT rows; E = P_a·RTF + P_i·(1−RTF); ● measured session)")
        for plat in R.PLATFORMS:
            on = s[(s.platform == plat) & s.on_tier_frontier].sort_values("energy_mj_per_s")
            eps = s[(s.platform == plat) & s.on_tier_frontier_margin].sort_values("energy_mj_per_s")
            print(f"  {plat:5s} margin-aware: " + ", ".join(r.config for r in eps.itertuples()))
            print(f"  {plat:5s} strict: " + ", ".join(f"{r.config}{'●' if r.measured else ''} ({r.energy_mj_per_s:.0f} mJ/s, {r.quality:.3f})"
                                                     for r in on.itertuples()))
        lad = s[s.on_ladder_frontier].sort_values("energy_mj_per_s")
        lad_eps = s[s.on_ladder_frontier_margin].sort_values("energy_mj_per_s")
        print("  LADDER strict: " + ", ".join(f"{R.PLATFORM_SHORT[r.platform]} {r.config} ({r.energy_mj_per_s:.0f}, {r.quality:.3f})" for r in lad.itertuples()))
        print("  LADDER margin-aware: " + ", ".join(f"{R.PLATFORM_SHORT[r.platform]} {r.config}" for r in lad_eps.itertuples()))
        if m == "si_sdr":
            top = s.sort_values("composite_si_sdr_per_mj_ms", ascending=False).head(6)
            print("  composite SI-SDR / (E × algo latency), top 6: " + ", ".join(
                f"{R.PLATFORM_SHORT[r.platform]} {r.config} {r.composite_si_sdr_per_mj_ms:.4g}" for r in top.itertuples()))


if __name__ == "__main__":
    main()
