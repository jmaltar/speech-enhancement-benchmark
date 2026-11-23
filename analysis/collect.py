"""
collect.py -- consolidate every measured result into one table set (task 21).

    .venv/bin/python -m analysis.collect            # write analysis/results/*.csv
    .venv/bin/python -m analysis.collect --inventory  # also list undeclared runs

Reads only machine-readable sources: the host scorer's per-utterance CSVs under
$BENCHMARK_DATA_ROOT/enhanced/<tag>/<corpus>.csv, the boards' per-utterance batch
CSVs under benchmark_*/results/, and the separation tier's score / run CSVs. Which
file is which result is declared in registry.py, never inferred.

Writes:
  results/quality_summary.csv     every scored tag x corpus x metric: n, mean, sd,
                                  median, 95 % BCa (stats.bootstrap_ci)
  results/quality_utterances.csv.gz  per-utterance rows of the declared classes
  results/quality_classes.csv     the host quality of each (algorithm, variant),
                                  the table pareto.points() joins against
  results/cost.csv                one row per (platform, label) cost run, with the
                                  energy columns power.attach() adds (task 22)
  results/power.csv               one row per steady-state power session (power.py)
  results/separation.csv          the separation tier (never pooled with speech)

THE COST STATISTIC. Two conventions are in use across the RESULTS.md files: the
classical and H723-DTLN rows quote the median over utterances of each utterance's
per-hop MEDIAN; the RNNoise rows and the S3/Pi DTLN rows quote the median over
utterances of each utterance's per-hop MEAN. They differ by < 0.3 % on every run
but MMSE (2.3 %, data-dependent gain cost). This table carries both plus the
hop-weighted mean, and uses the MEDIAN OF MEANS as primary: energy integrates over
time, so the mean hop is the quantity E = P_active x RTF + P_idle x (1 - RTF)
needs. The audit (audit.py) matches every quoted figure against the statistic its
source used and lists the convention.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_harness import stats  # noqa: E402

from analysis import registry as R  # noqa: E402
from analysis import power  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
METRICS = ("pesq_mos_lqo", "pesq_raw_mos", "stoi", "estoi", "si_sdr")
FIT_OVERRUN_FRAC = 1e-4   # FIT = RTF < 1 and fewer than 1 hop in 10^4 late (the S3 +FW boot hop)


# --------------------------------------------------------------------------- quality
def load_scored(tag: str, corpus: str) -> pd.DataFrame:
    p = R.ENH / tag / f"{corpus}.csv"
    df = pd.read_csv(p)
    if (df.corpus != corpus).any():
        raise SystemExit(f"{p} carries rows of another corpus -- never pooled")
    bad = int((~df.ok).sum())
    if bad:
        raise SystemExit(f"{p}: {bad} failed rows; a summary over a partial run is a biased sample")
    return df


def quality_summary() -> pd.DataFrame:
    rows = []
    for d in sorted(p for p in R.ENH.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.csv")):
            corpus = f.stem
            df = pd.read_csv(f)
            if "ok" not in df or "corpus" not in df:
                continue
            ok = df[df.ok]
            for m in METRICS:
                if m not in ok:
                    continue
                x = ok[m].to_numpy(float)
                desc = stats.describe(x)
                lo, hi = stats.bootstrap_ci(x)
                rows.append(dict(tag=d.name, corpus=corpus, metric=m, n=desc["n"], n_failed=int((~df.ok).sum()),
                                 mean=desc["mean"], sd=desc["sd"], median=desc["median"],
                                 bca_lo=lo, bca_hi=hi, declared=d.name in R.QCLASS_BY_TAG
                                 or any(t == d.name for t, *_ in R.PER_PLATFORM_QUALITY)))
    return pd.DataFrame(rows)


def quality_utterances() -> pd.DataFrame:
    tags = {(q.tag, c) for q in R.QCLASSES for c in q.corpora}
    tags |= {(t, c) for t, _, _, _, c in R.PER_PLATFORM_QUALITY}
    tags |= {(t, c) for _, _, c, t, _ in R.TOST_CLAIMS} | {(t, c) for c, t, _, _, _ in R.TOST_CLAIMS}
    tags |= {("bypass", "voicebank48k")}
    out = []
    for tag, corpus in sorted(tags):
        df = load_scored(tag, corpus)
        keep = [c for c in ("id", "corpus", "noise", "snr_db", "snr_bin", *METRICS) if c in df]
        out.append(df[keep].assign(tag=tag))
    return pd.concat(out, ignore_index=True)


def quality_classes(qs: pd.DataFrame) -> pd.DataFrame:
    """Host quality per (algorithm, variant): the table the frontier joins against."""
    rows = []
    for q in R.QCLASSES:
        for corpus in q.corpora:
            sub = qs[(qs.tag == q.tag) & (qs.corpus == corpus)].set_index("metric")
            r = dict(algorithm=q.algorithm, variant=q.variant, tag=q.tag, corpus=corpus,
                     n=int(sub["n"].iloc[0]), model_bytes=q.model_bytes, model_src=q.model_src, note=q.note)
            for m in sub.index:
                r[m] = sub.loc[m, "mean"]
                r[f"{m}_bca_lo"], r[f"{m}_bca_hi"] = sub.loc[m, "bca_lo"], sub.loc[m, "bca_hi"]
            rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- cost
def _ref_tag(ref: str) -> str:
    parts = str(ref).split("/")
    return parts[-3] if len(parts) >= 3 else ""


def cost_row(run: R.CostRun) -> dict:
    p = R.ROOT / run.file
    d = pd.read_csv(p)
    n, n_ok = len(d), int((d.status == "ok").sum())
    if n_ok != n:
        raise SystemExit(f"{run.file}: {n - n_ok} failed utterances")
    clk = R.CLOCK_HZ[run.platform]
    k = 2.4 if R.PROC_UNIT[run.platform] == "ns" else 1.0      # ns -> cycles at 2.4 GHz
    hop = R.HOP_S[run.algorithm]
    budget = clk * hop
    med_mean = float(d.proc_mean.median()) * k
    med_med = float(d.proc_median_data.median()) * k
    wmean = float((d.proc_mean * d.hops).sum() / d.hops.sum()) * k
    # The first utterance of a run on a freshly flashed board carries the one-off init hop
    # (rnnoise_create / weight copy: 8-11 M cycles on the F411 and S3 variant runs). It is
    # start-up cost, not jitter, so the tail statistics exclude utterance 0 and report it apart.
    tail = d.iloc[1:]
    worst = float(tail.proc_max.max()) * k
    first_max = float(d.proc_max.iloc[0]) * k
    hops, nres = int(d.hops.sum()), int(d.nres.sum())
    ovr = int(d.overruns.sum())
    rtf = med_mean / budget
    row = dict(
        platform=run.platform, algorithm=run.algorithm, variant=run.variant, label=run.label,
        quality_tag=R.QCLASS_BY_KEY[(run.algorithm, run.variant)].tag,
        own_quality_tag=run.extra.get("own_quality", ""),
        file=run.file, date=run.date, paced=run.paced, n_utt=n,
        hops=hops, nres=nres, hop_ms=hop * 1e3, clock_hz=clk, budget_cyc=budget,
        cyc_hop=med_mean, cyc_hop_med_of_medians=med_med, cyc_hop_weighted_mean=wmean,
        rtf=rtf, rtf_med_of_medians=med_med / budget,
        latency_algo_ms=med_mean / clk * 1e3,
        overruns=ovr, overrun_frac=ovr / max(nres, 1),
        fit=bool(rtf < 1 and ovr / max(nres, 1) < FIT_OVERRUN_FRAC),
        e2e_ms=float(d.e2e_median_us.median()) / 1e3 if run.platform != "rpi5" else np.nan,
        e2e_note=("n/a: the Pi is fed files and streams no USB (CLAUDE.md, Audio I/O)"
                  if run.platform == "rpi5" else
                  ("paced" if run.paced else "UNPACED batch: includes host-side backlog, an upper bound")),
        jitter_mode="paced" if run.paced else "batch (no deadline)",
        jitter_worst_minus_median_us=(worst - med_med) / clk * 1e6,
        jitter_p99_utt_max_minus_median_us=(float(np.percentile(tail.proc_max * k, 99)) - med_med) / clk * 1e6,
        first_utt_max_cyc=first_max, first_utt_is_boot_hop=bool(first_max > 2 * float(tail.proc_max.max()) * k),
        duty_cycle=(float(d.duty_cycle.median()) if "duty_cycle" in d else min(1.0, rtf)),
        duty_source=("logged (duty_cycle column)" if "duty_cycle" in d else "derived: min(1, RTF)"),
        max_abs_lsb=int(d.max_abs_lsb.max()), n_diff_median=float(d.n_diff.median()),
        nmse_db_worst=float(d.nmse_db.max()), ref_tag=_ref_tag(d.ref.iloc[0]),
        model_bytes=R.QCLASS_BY_KEY[(run.algorithm, run.variant)].model_bytes,
        source=run.source,
    )
    twin = R.PI_BATCH_TWINS.get(run.file)
    if twin:
        b = pd.read_csv(R.ROOT / twin)
        row["batch_cyc_hop_med_of_medians"] = float(b.proc_median_data.median()) * k
        row["batch_cyc_hop"] = float(b.proc_mean.median()) * k
        row["batch_jitter_worst_minus_median_us"] = (float(b.proc_max.iloc[1:].max()) - float(b.proc_median_data.median())) * k / clk * 1e6
    return row


def cost_table() -> pd.DataFrame:
    return pd.DataFrame([cost_row(r) for r in R.COST_RUNS])


# --------------------------------------------------------------------------- separation
def separation_table() -> pd.DataFrame:
    rows = []
    for who, key in (("host", "quality_host"), ("rpi5 (scored on host)", "quality_rpi5")):
        d = pd.read_csv(R.ROOT / R.SEPARATION[key])
        d = d[d.ok]
        for m in ("vocals_sdr", "vocals_sir", "vocals_sar", "accompaniment_sdr", "accompaniment_sir", "accompaniment_sar"):
            x = d[m].to_numpy(float)
            lo, hi = stats.bootstrap_ci(x, statistic=np.median)
            rows.append(dict(kind="quality", source=who, metric=m, n_tracks=len(x),
                             median=float(np.median(x)), mean=float(x.mean()), bca_median_lo=lo, bca_median_hi=hi,
                             file=R.SEPARATION[key]))
    d4 = pd.read_csv(R.ROOT / R.SEPARATION["quality_4target"])
    d4 = d4[d4.ok]
    for m in ("vocals_sdr", "drums_sdr", "bass_sdr", "other_sdr"):
        x = d4[m].to_numpy(float)
        rows.append(dict(kind="quality (four-target anchor)", source="host", metric=m, n_tracks=len(x),
                         median=float(np.median(x)), mean=float(x.mean()), file=R.SEPARATION["quality_4target"]))
    c = pd.read_csv(R.ROOT / R.SEPARATION["cost_rpi5"])
    c = c[c.status == "ok"]
    rows.append(dict(kind="cost", source="rpi5", metric="rtf (median over tracks)", n_tracks=len(c),
                     median=float(c.rtf.median()), mean=float(c.rtf.mean()),
                     q1=float(c.rtf.quantile(0.25)), q3=float(c.rtf.quantile(0.75)),
                     total_rtf=float(c.wall_s.sum() / c.seconds.sum()), threads=int(c.threads.iloc[0]),
                     throttled_any=bool((c.throttled != "throttled=0x0").any()), file=R.SEPARATION["cost_rpi5"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- inventory
def inventory() -> list[str]:
    declared_q = {q.tag for q in R.QCLASSES} | {t for t, *_ in R.PER_PLATFORM_QUALITY}
    declared_q |= {t for _, t, _, _, _ in R.TOST_CLAIMS} | {t for _, _, _, t, _ in R.TOST_CLAIMS}
    lines = ["scored tags not declared (diagnostic / Arm B rows; in quality_summary.csv only):"]
    for d in sorted(p.name for p in R.ENH.iterdir() if p.is_dir()):
        if d not in declared_q and d not in ("umxhq", "umxhq_rpi5"):
            lines.append(f"  {d}")
    declared_c = {r.file for r in R.COST_RUNS} | set(R.PI_BATCH_TWINS.values())
    lines.append("cost CSVs not declared (smokes, partials, stamped, placement diagnostics):")
    for p in sorted(R.ROOT.glob("benchmark_*/results/**/*.csv")):
        rel = str(p.relative_to(R.ROOT))
        if rel not in declared_c:
            try:
                if "proc_mean" in pd.read_csv(p, nrows=1):
                    lines.append(f"  {rel}")
            except Exception:
                pass
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inventory", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    qs = quality_summary()
    qs.to_csv(OUT / "quality_summary.csv", index=False)
    print(f"quality_summary.csv: {len(qs)} rows over {qs.tag.nunique()} tags")
    qu = quality_utterances()
    qu.to_csv(OUT / "quality_utterances.csv.gz", index=False)
    print(f"quality_utterances.csv.gz: {len(qu)} rows over {qu.tag.nunique()} tags")
    qc = quality_classes(qs)
    qc.to_csv(OUT / "quality_classes.csv", index=False)
    pt = power.power_table()
    pt.to_csv(OUT / "power.csv", index=False)
    print(f"power.csv: {len(pt)} sessions")
    ct = power.attach(cost_table(), pt)
    ct.to_csv(OUT / "cost.csv", index=False)
    print(f"cost.csv: {len(ct)} rows (energy columns from the power sessions)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(ct[["platform", "algorithm", "label", "cyc_hop", "cyc_hop_med_of_medians", "rtf", "fit",
                  "overruns", "e2e_ms", "duty_cycle"]].to_string(index=False))
    st = separation_table()
    st.to_csv(OUT / "separation.csv", index=False)
    print(f"separation.csv: {len(st)} rows")
    if a.inventory:
        inv = inventory()
        (OUT / "inventory.txt").write_text("\n".join(inv) + "\n")
        print("\n".join(inv[:6]), "...", f"({len(inv)} lines, results/inventory.txt)")


if __name__ == "__main__":
    main()
