"""
stats.py -- the statistics layer (task 6).

Three rules from CLAUDE.md, implemented rather than documented:

  1. THE TEST IS FIXED BY THE METRIC, not chosen per comparison. Paired Wilcoxon
     signed-rank for PESQ and STOI (bounded, skewed, not plausibly normal);
     paired t for SI-SDR (unbounded, roughly symmetric). Choosing a test after
     seeing which one gives a nicer p-value is the classic way to manufacture a
     result, so `compare()` looks the test up from the metric name and there is
     no argument to override it.

  2. EFFECT SIZES, NOT JUST p-VALUES. Every comparison returns the estimated
     difference with a confidence interval and a standardised effect size. With
     N = 824 a difference of 0.002 PESQ is "significant" and means nothing.

  3. EQUIVALENCE NEEDS AN EQUIVALENCE TEST. "Platform B is no worse than A" is
     not supported by a non-significant difference test -- absence of evidence
     is not evidence of absence. It needs TOST against a margin fixed BEFORE any
     data was seen. Those margins were locked and dated 2026-09-12 and are
     hard-coded below; passing a different one requires saying so explicitly.

A note on what TOST actually answers. The difference test asks "can I rule out
zero difference?". TOST asks the opposite question -- "can I rule out a
difference big enough to matter?" -- and a comparison can, legitimately and
informatively, come out:

    significant + equivalent      a real but unimportant difference
    not significant + equivalent  genuinely the same, to the stated precision
    not significant + inconclusive  underpowered; this is NOT "no difference"
    significant + not equivalent  a difference that matters

The third row is the one that gets misreported in the literature, so
`verdict()` names it explicitly rather than letting a bare p > 0.05 be read as
a null result.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats as sps

# --- LOCKED equivalence margins (2026-09-12, before any eval data existed) ---
# The SI-SDR margin sits just inside TinyLSTMs' own accepted 0.55 dB
# degradation for 11.9x compression, so every equivalence claim here is tighter
# than the closest published precedent. Do not revise after seeing data.
MARGIN = {
    "pesq_mos_lqo": 0.10,
    "pesq_raw_mos": 0.10,
    "stoi":         0.01,
    "estoi":        0.01,
    "si_sdr":       0.50,
}

# Which test each metric takes. Fixed, not selectable.
TEST = {
    "pesq_mos_lqo": "wilcoxon",
    "pesq_raw_mos": "wilcoxon",
    "stoi":         "wilcoxon",
    "estoi":        "wilcoxon",
    "si_sdr":       "t",
}

ALPHA = 0.05


@dataclass
class Comparison:
    metric: str
    n: int
    test: str
    mean_a: float
    mean_b: float
    diff: float           # point estimate of b - a (mean, or Hodges-Lehmann)
    ci_lo: float          # (1 - 2*alpha) CI -- the interval TOST is read from
    ci_hi: float
    ci_level: float
    p_diff: float         # two-sided difference test
    effect: float         # Cohen's d_z, or matched-pairs rank-biserial r
    effect_name: str
    margin: float
    p_tost: float
    equivalent: bool
    significant: bool
    verdict: str

    def as_dict(self):
        return asdict(self)


# Bootstrap settings. Fixed seed, because a confidence interval that moves
# between runs of the same data is the resampler talking, not the data.
BOOTSTRAP_SEED = 20260913
BOOTSTRAP_RESAMPLES = 10000


def bootstrap_ci(x: np.ndarray, level: float = 0.95,
                 statistic=np.mean,
                 resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """BCa bootstrap interval for `statistic`, assuming nothing about the shape
    of the distribution.

    Why it is here: PESQ and STOI are BOUNDED and skewed -- which is exactly
    why their significance tests are rank-based -- and a t-interval on a
    bounded skewed quantity is symmetric by construction whether or not the
    sampling distribution is. At N = 824 the difference is small, and reporting
    the symmetric one anyway would be assuming the thing this project's own
    rules decline to assume elsewhere. Bias-corrected and accelerated, so the
    skew is corrected rather than merely tolerated.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    r = sps.bootstrap((x,), statistic, confidence_level=level, method="BCa",
                      n_resamples=resamples,
                      random_state=np.random.default_rng(seed))
    return float(r.confidence_interval.low), float(r.confidence_interval.high)


def describe(x: np.ndarray, level: float = 0.95,
             bootstrap: bool = False) -> dict:
    """Summary of one sample. `ci_lo`/`ci_hi` are the t-interval; with
    `bootstrap=True` the distribution-free BCa interval is added alongside
    rather than replacing it, so the two can be compared in the write-up."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    m, sd = float(x.mean()), float(x.std(ddof=1))
    half = float(sps.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n))
    out = {"n": n, "mean": m, "sd": sd, "median": float(np.median(x)),
           "ci_lo": m - half, "ci_hi": m + half, "ci_level": level}
    if bootstrap:
        lo, hi = bootstrap_ci(x, level)
        out["boot_lo"], out["boot_hi"] = lo, hi
        out["boot_resamples"] = BOOTSTRAP_RESAMPLES
    return out


def hodges_lehmann(d: np.ndarray) -> float:
    """Median of Walsh averages -- the location estimate that goes WITH the
    signed-rank test, in the way the mean goes with the t-test. Reporting a
    mean beside a Wilcoxon p-value quietly mixes two different estimands."""
    d = np.asarray(d, float)
    i, j = np.triu_indices(len(d), k=0)
    return float(np.median((d[i] + d[j]) / 2.0))


def _hl_ci(d: np.ndarray, level: float) -> tuple[float, float]:
    """Distribution-free CI for the Hodges-Lehmann estimate, from the order
    statistics of the Walsh averages (the standard Tukey construction)."""
    d = np.asarray(d, float)
    n = len(d)
    i, j = np.triu_indices(n, k=0)
    w = np.sort((d[i] + d[j]) / 2.0)
    m = len(w)
    # Normal approximation to the signed-rank critical value; exact enough well
    # below the N >= 20 rule this project enforces everywhere.
    z = sps.norm.ppf(0.5 + level / 2)
    se = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    k = int(np.floor(n * (n + 1) / 4.0 - z * se))
    k = max(0, min(k, m - 1))
    return float(w[k]), float(w[m - 1 - k])


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: (W+ - W-) / sum(ranks).
    Bounded -1..1, and unlike a p-value it does not grow with N."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = sps.rankdata(np.abs(d))
    wp, wn = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wn) / (wp + wn))


def cohen_dz(d: np.ndarray) -> float:
    """Standardised mean difference for paired samples.

    The zero-variance guard is a TOLERANCE and not `sd > 0`, because a constant
    difference vector -- a port with a fixed bias, which is exactly what task 15
    goes looking for -- has a floating-point sd of ~2e-16 rather than of exactly
    zero. The exact-zero guard let that through and printed d_z = 3.1e15.
    """
    d = np.asarray(d, float)
    sd = float(d.std(ddof=1))
    scale = float(np.max(np.abs(d))) if d.size else 0.0
    if sd <= 16 * np.finfo(float).eps * max(scale, 1.0):
        return 0.0
    return float(d.mean() / sd)


def tost(d: np.ndarray, margin: float, kind: str, alpha: float = ALPHA) -> float:
    """Two one-sided tests. Returns the TOST p-value = max(p_lower, p_upper).

    Equivalence is declared when BOTH one-sided tests reject, i.e. when the
    whole (1 - 2*alpha) CI lies inside +/- margin.
    """
    d = np.asarray(d, float)
    if kind == "t":
        p_lo = sps.ttest_1samp(d + margin, 0.0, alternative="greater").pvalue
        p_hi = sps.ttest_1samp(d - margin, 0.0, alternative="less").pvalue
    elif kind == "wilcoxon":
        # Both shifted samples must contain a non-zero difference for the
        # signed-rank test to be defined.
        def _w(x, alt):
            x = x[x != 0]
            if len(x) == 0:
                return 1.0
            return sps.wilcoxon(x, alternative=alt, zero_method="wilcox").pvalue
        p_lo = _w(d + margin, "greater")
        p_hi = _w(d - margin, "less")
    else:
        raise ValueError(f"unknown test kind {kind!r}")
    return float(max(p_lo, p_hi))


def verdict(significant: bool, equivalent: bool) -> str:
    if significant and equivalent:
        return "different but equivalent (real, below the margin that matters)"
    if significant and not equivalent:
        return "DIFFERENT (exceeds the pre-specified margin)"
    if not significant and equivalent:
        return "EQUIVALENT (same to the stated precision)"
    return ("INCONCLUSIVE -- underpowered. This is NOT a no-difference result: "
            "the data cannot rule out a difference larger than the margin.")


def compare(a: np.ndarray, b: np.ndarray, metric: str,
            margin: float | None = None, alpha: float = ALPHA) -> Comparison:
    """Paired comparison of b against a on one metric, on matched samples.

    `a` and `b` must already be aligned element-for-element: pair them by
    utterance id before calling, never by position in a file.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"unpaired samples: {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 20:
        raise ValueError(
            f"n={n} after dropping non-finite values; this project's minimum is "
            f"20 per cell and a comparison below it should not be reported.")

    kind = TEST.get(metric)
    if kind is None:
        raise ValueError(
            f"no test is defined for metric {metric!r}. The test is a locked "
            f"per-metric decision, not something to default.")
    m = MARGIN[metric] if margin is None else float(margin)

    d = b - a
    lvl = 1 - 2 * alpha          # the interval TOST is read from: 90% at a=0.05

    if kind == "t":
        # A bit-exact port gives differences that are all exactly zero, and
        # ttest_rel returns NaN there (0/0). NaN compares False against every
        # threshold, so it reads as "not significant" and looks right -- until
        # it reaches holm(), where it sorts unpredictably and corrupts the
        # adjustment of every other comparison in the family. The honest
        # p-value for "no difference whatsoever" is 1.0.
        p_diff = float(sps.ttest_rel(b, a).pvalue) if np.any(d != 0) else 1.0
        point = float(d.mean())
        half = float(sps.t.ppf(0.5 + lvl / 2, n - 1) * d.std(ddof=1) / np.sqrt(n))
        lo, hi = point - half, point + half
        eff, eff_name = cohen_dz(d), "Cohen's d_z"
    else:
        nz = d[d != 0]
        p_diff = float(sps.wilcoxon(nz, zero_method="wilcox").pvalue) if len(nz) else 1.0
        point = hodges_lehmann(d)
        lo, hi = _hl_ci(d, lvl)
        eff, eff_name = rank_biserial(d), "rank-biserial r"

    p_tost = tost(d, m, kind, alpha)
    equivalent = bool(p_tost < alpha)
    significant = bool(p_diff < alpha)

    return Comparison(
        metric=metric, n=n, test=kind,
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        diff=point, ci_lo=lo, ci_hi=hi, ci_level=lvl,
        p_diff=p_diff, effect=eff, effect_name=eff_name,
        margin=m, p_tost=p_tost, equivalent=equivalent,
        significant=significant, verdict=verdict(significant, equivalent))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values.

    NOT applied automatically. The family it is applied to is a protocol
    decision, DECIDED 2026-10-12 and recorded in HOLM_FAMILY below and in
    CLAUDE.md (Metric Stack Decisions 6) before any difference test exists:

      * HEADLINE claims are the per-corpus POOLED algorithm-vs-algorithm
        difference tests, one per (algorithm pair, metric). Holm is applied
        over that family, per corpus. Corpora are never pooled, so they are
        never one family either.
      * PER-CELL (noise x SNR) difference tests are EXPLORATORY: reported
        with unadjusted p, the CI and the effect size, and labelled as such.
        They are read as trends, never as findings in their own right.
      * TOST equivalence claims are untouched either way -- their protection
        is the pre-specified margin, not a p-value.

    Why: the matrix has 20 cells per corpus and several algorithms, so
    per-cell testing without adjustment produces false positives at a rate
    nobody intends, while adjusting a family that includes the exploratory
    cells would cost power on exactly the claims the paper is about.
    """
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (len(p) - rank) * p[idx]
        running = max(running, val)
        out[idx] = min(1.0, running)
    return [float(v) for v in out]


# The multiple-comparison protocol, decided 2026-10-12 (see holm.__doc__).
# A caller assembling the headline family passes its p-values to holm();
# anything per-cell is reported unadjusted and labelled exploratory.
HOLM_FAMILY = "headline-pooled-per-corpus"
EXPLORATORY_UNADJUSTED = ("per-cell",)


# scipy warns rather than refuses above this N, so the truncation below is a
# silent policy and has to be a visible one: `shapiro_n` goes into the result.
SHAPIRO_MAX = 5000
SUBSAMPLE_SEED = 20260913


def check_normality(x: np.ndarray, name: str = "") -> dict:
    """Verify the t-test's assumption instead of asserting it.

    Shapiro-Wilk is hypersensitive at large N -- it will reject almost any real
    data at N = 824 -- so the skew/kurtosis of the DIFFERENCES are reported
    alongside, since those are what actually decide whether the paired t is
    behaving. Judge on the shape, not on the p-value.

    Above SHAPIRO_MAX the sample is subsampled AT RANDOM, never as `x[:N]`.
    Results arrive in manifest order, so the head is one speaker at the easy end
    of the SNR range -- the same biased-sample trap as `--limit 40` (CLAUDE.md
    trap 14), and it would be testing normality on the easiest tenth of the run.
    The seed is fixed so the figure is reproducible, and skew/kurtosis are
    always computed on the FULL sample regardless.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) > SHAPIRO_MAX:
        xs = np.random.default_rng(SUBSAMPLE_SEED).choice(
            x, SHAPIRO_MAX, replace=False)
    else:
        xs = x
    sw = sps.shapiro(xs)
    return {"name": name, "n": int(len(x)), "shapiro_n": int(len(xs)),
            "shapiro_p": float(sw.pvalue),
            "skew": float(sps.skew(x)), "excess_kurtosis": float(sps.kurtosis(x))}


def table(cmps: list[Comparison]) -> pd.DataFrame:
    return pd.DataFrame([c.as_dict() for c in cmps])


def render(cmps: list[Comparison]) -> str:
    lines = []
    w = max(len(c.metric) for c in cmps) + 1
    lines.append(f"{'metric':<{w}}{'A':>9}{'B':>9}{'B-A':>10}"
                 f"{'90% CI':>20}{'margin':>8}{'p(diff)':>10}{'p(TOST)':>10}  verdict")
    for c in cmps:
        ci = f"[{c.ci_lo:+.4f}, {c.ci_hi:+.4f}]"
        lines.append(f"{c.metric:<{w}}{c.mean_a:>9.4f}{c.mean_b:>9.4f}{c.diff:>+10.4f}"
                     f"{ci:>20}{c.margin:>8.2f}{c.p_diff:>10.3g}{c.p_tost:>10.3g}  "
                     f"{c.verdict}")
        lines.append(f"{'':<{w}}{c.effect_name} = {c.effect:+.4f}   "
                     f"(paired {c.test}, n = {c.n})")
    return "\n".join(lines)
