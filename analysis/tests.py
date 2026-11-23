"""
tests.py -- the statistics over the consolidated tables (tasks 20 and 25).

    .venv/bin/python -m analysis.tests

Four outputs, each with the protection its claim shape needs (Metric Stack
Decision 6, 2026-10-12, and stats.py's rules -- the test is fixed by the metric,
effect size and CI beside every p):

  results/headline.csv     HEADLINE family per corpus: every algorithm pair at
                           its reference configuration x every metric of the
                           corpus's stack, Holm over that family (caller-applied,
                           `stats.HOLM_FAMILY`). Corpora are separate families.
  results/percell.csv      the same pairs per (noise x SNR) cell / DNS SNR bin:
                           EXPLORATORY, unadjusted p, labelled as such.
  results/tost_claims.csv  the "no worse" claims the project makes (compression
                           variant vs its baseline): TOST against the locked
                           margins; no Holm (the margin is the protection).
  results/per_platform.csv task 15: each board-scored row against its host
                           reference, TOST, plus the sample-level deviation.
  results/sample_size.csv  task 20: per cell N, SD of paired differences, MDE at
                           80 % power, and the N a TOST claim needs at 80 % power.

Pairing is always by utterance id. The one cross-corpus pairing allowed is
voicebank16k <-> voicebank48k (same 824 utterances, filter-matched references;
eval_harness.compare.SAME_UTTERANCES); RNNoise enters the VoiceBank family that way.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_harness import stats  # noqa: E402
from eval_harness.compare import SAME_UTTERANCES  # noqa: E402

from analysis import registry as R  # noqa: E402
from analysis.collect import OUT, load_scored  # noqa: E402

Z975, Z95, Z80, Z90 = sps.norm.ppf(0.975), sps.norm.ppf(0.95), sps.norm.ppf(0.80), sps.norm.ppf(0.90)
WILCOXON_ARE = 3 / np.pi       # asymptotic relative efficiency of signed-rank vs t under normality


def _scores(tag, corpus):
    return load_scored(tag, corpus).set_index("id")


def _pair(ca, ta, cb, tb):
    if ca != cb and frozenset({ca, cb}) not in SAME_UTTERANCES:
        raise SystemExit(f"{ca} and {cb} are different utterances -- never paired")
    A, B = _scores(ta, ca), _scores(tb, cb)
    ids = A.index.intersection(B.index)
    if len(ids) != len(A) or len(ids) != len(B):
        raise SystemExit(f"{ta}/{ca} vs {tb}/{cb}: only {len(ids)} of {len(A)}/{len(B)} ids pair")
    return A.loc[ids], B.loc[ids]


def _cell_key(df, corpus):
    if corpus.startswith("dns"):
        return df["snr_bin"].astype(str)
    return df["noise"].astype(str) + "@" + df["snr_db"].map(lambda v: f"{v:g}")


def headline():
    rows, cells = [], []
    for fam, members in R.HEADLINE.items():
        metrics = R.HEADLINE_METRICS[fam]
        fam_rows = []
        for (aa, ca, ta), (ab, cb, tb) in itertools.combinations(members, 2):
            A, B = _pair(ca, ta, cb, tb)
            for m in metrics:
                c = stats.compare(A[m].to_numpy(), B[m].to_numpy(), m)
                fam_rows.append(dict(family=fam, a=aa, b=ab, tag_a=ta, tag_b=tb, corpus_a=ca, corpus_b=cb,
                                     **c.as_dict()))
            # per-cell, exploratory
            key = _cell_key(A, ca)
            for cell in sorted(key.unique()):
                sel = (key == cell).to_numpy()
                for m in metrics:
                    a, b = A[m].to_numpy()[sel], B[m].to_numpy()[sel]
                    if len(a) < 20:
                        continue
                    c = stats.compare(a, b, m)
                    d = b - a
                    cells.append(dict(family=fam, a=aa, b=ab, cell=cell, label="EXPLORATORY, unadjusted",
                                      sd_diff=float(np.std(d, ddof=1)), **c.as_dict()))
        adj = stats.holm([r["p_diff"] for r in fam_rows])
        for r, p in zip(fam_rows, adj):
            r["p_holm"] = p
            r["significant_holm"] = p < stats.ALPHA
            # `verdict` (from stats.compare) reads the RAW p; the headline verdict reads Holm's
            r["verdict_holm"] = stats.verdict(r["significant_holm"], r["equivalent"])
            r["holm_family"] = f"{stats.HOLM_FAMILY}:{fam} ({len(fam_rows)} tests)"
        rows += fam_rows
    return pd.DataFrame(rows), pd.DataFrame(cells)


def tost_claims():
    rows = []
    for ca, ta, cb, tb, claim in R.TOST_CLAIMS:
        A, B = _pair(ca, ta, cb, tb)
        for m in R.HEADLINE_METRICS["voicebank"]:
            c = stats.compare(A[m].to_numpy(), B[m].to_numpy(), m)
            rows.append(dict(claim=claim, corpus=cb, a=ta, b=tb, **c.as_dict()))
    return pd.DataFrame(rows)


def per_platform(cost: pd.DataFrame):
    rows = []
    for tag, plat, label, ref, corpus in R.PER_PLATFORM_QUALITY:
        A, B = _pair(corpus, ref, corpus, tag)
        dev = cost[(cost.platform == plat) & (cost.own_quality_tag == tag)]
        for m in R.HEADLINE_METRICS["voicebank"]:
            a, b = A[m].to_numpy(), B[m].to_numpy()
            c = stats.compare(a, b, m)
            rows.append(dict(tag=tag, platform=plat, label=label, reference=ref, corpus=corpus,
                             utt_metric_identical=int(np.sum(a == b)),
                             max_abs_lsb_vs_oracle=(int(dev.max_abs_lsb.iloc[0]) if len(dev) and corpus != "dns" else np.nan),
                             nmse_db_worst_vs_oracle=(float(dev.nmse_db_worst.iloc[0]) if len(dev) and corpus != "dns" else np.nan),
                             **c.as_dict()))
    return pd.DataFrame(rows)


def sample_size(cells: pd.DataFrame, tost: pd.DataFrame, plat: pd.DataFrame):
    """Task 20. Two questions, kept apart.

    (1) DIFFERENCE claims, per cell: with N and the SD of paired differences, what is the smallest
        effect a two-sided test detects at 80 % power (paired-t approximation; Wilcoxon metrics
        divided by sqrt(ARE = 3/pi))? Compared with the locked margins.
    (2) EQUIVALENCE claims: at the observed difference and SD, what N gives TOST 80 % power?
        n = (z_{1-a} + z_{1-b})^2 sd^2 / (margin - |diff|)^2, z_{1-b/2} when diff = 0.
        Compared with the pooled N each claim was made at and with the N >= 20 floor.
    """
    out = []
    for r in cells.itertuples():
        se = r.sd_diff / np.sqrt(r.n)
        mde = (Z975 + Z80) * se / (np.sqrt(WILCOXON_ARE) if r.test == "wilcoxon" else 1.0)
        out.append(dict(kind="difference, per cell", family=r.family, pair=f"{r.a} vs {r.b}", cell=r.cell,
                        metric=r.metric, n=r.n, sd_diff=r.sd_diff, mde80=mde, margin=r.margin,
                        mde_over_margin=mde / r.margin))

    def need_n(sd, diff, margin, test):
        # `diff` is the comparison's own location estimate (Hodges-Lehmann for the signed-rank
        # metrics, the mean for SI-SDR), so the power figure is about the quantity TOST tested;
        # signed-rank efficiency enters as 1/ARE on n.
        gap = margin - abs(diff)
        if gap <= 0:
            return np.inf
        zb = Z90 if abs(diff) < 1e-12 else Z80
        n = ((Z95 + zb) * sd / gap) ** 2
        return float(np.ceil(n / (WILCOXON_ARE if test == "wilcoxon" else 1.0)))

    for src, df, who in (("tost claim", tost, "claim"), ("per-platform", plat, "tag")):
        # n_needed_tost80 is the n at which TOST has 80 % power at the observed location and SD
        for r in df.itertuples():
            A, B = (_pair(r.corpus, r.a, r.corpus, r.b) if src == "tost claim"
                    else _pair(r.corpus, r.reference, r.corpus, r.tag))
            d = B[r.metric].to_numpy() - A[r.metric].to_numpy()
            sd = float(np.std(d, ddof=1))
            nn = need_n(sd, float(r.diff), r.margin, r.test)
            # what a per-cell (N = 40, the VoiceBank minimum is 40-42; 20 is the floor) TOST would need
            out.append(dict(kind=f"equivalence ({src})", family=r.corpus, pair=getattr(r, who), cell="pooled",
                            metric=r.metric, n=r.n, sd_diff=sd, mean_diff=float(np.mean(d)), location=float(r.diff),
                            test=r.test, margin=r.margin,
                            n_needed_tost80=nn, enough_at_pooled_n=bool(r.n >= nn), enough_at_n20=bool(nn <= 20),
                            enough_at_n40=bool(nn <= 40)))
    return pd.DataFrame(out)


def main():
    cost = pd.read_csv(OUT / "cost.csv")
    head, cells = headline()
    head.to_csv(OUT / "headline.csv", index=False)
    cells.to_csv(OUT / "percell.csv", index=False)
    tc = tost_claims()
    tc.to_csv(OUT / "tost_claims.csv", index=False)
    pp = per_platform(cost)
    pp.to_csv(OUT / "per_platform.csv", index=False)
    ss = sample_size(cells, tc, pp)
    ss.to_csv(OUT / "sample_size.csv", index=False)

    pd.set_option("display.width", 220)
    for fam in R.HEADLINE:
        h = head[head.family == fam]
        print(f"\n=== {fam}: {len(h)} tests, Holm; significant after Holm: {int(h.significant_holm.sum())}")
        print(h[["a", "b", "metric", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "effect", "p_diff", "p_holm",
                 "significant_holm", "equivalent"]].round(4).to_string(index=False))
    print(f"\nper-cell exploratory: {len(cells)} tests")
    print("\n=== TOST claims")
    print(tc[["claim", "corpus", "metric", "mean_a", "mean_b", "diff", "ci_lo", "ci_hi", "p_tost", "verdict"]]
          .round(4).to_string(index=False))
    print("\n=== per-platform (task 15)")
    print(pp[["tag", "corpus", "metric", "utt_metric_identical", "diff", "ci_lo", "ci_hi", "p_tost", "equivalent"]]
          .round(5).to_string(index=False))


if __name__ == "__main__":
    main()
