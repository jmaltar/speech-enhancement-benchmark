"""Mutation-test eval_harness/test_stats.py (run from the project root:
`.venv/bin/python -m eval_harness.mutate_stats`).

 each mutation below is a plausible
real defect; the suite must fail on every one. A mutation that survives means
the corresponding test is decoration."""
import pathlib, shutil, subprocess, sys

SRC = pathlib.Path("eval_harness/stats.py")
BAK = SRC.with_suffix(".py.mutbak")

MUTATIONS = [
  ("TOST lower test sign flipped (the classic bug)",
   'p_lo = sps.ttest_1samp(d + margin, 0.0, alternative="greater").pvalue',
   'p_lo = sps.ttest_1samp(d - margin, 0.0, alternative="greater").pvalue'),
  ("TOST wilcoxon lower test alternative flipped",
   'p_lo = _w(d + margin, "greater")', 'p_lo = _w(d + margin, "less")'),
  ("TOST returns min instead of max of the two one-sided p",
   'return float(max(p_lo, p_hi))', 'return float(min(p_lo, p_hi))'),
  ("equivalence read off the DIFFERENCE test (absence of evidence)",
   'equivalent = bool(p_tost < alpha)', 'equivalent = bool(p_diff > alpha)'),
  ("equivalence comparison inverted",
   'equivalent = bool(p_tost < alpha)', 'equivalent = bool(p_tost > alpha)'),
  ("Hodges-Lehmann drops the self-pairs (k=0 -> k=1)",
   'i, j = np.triu_indices(len(d), k=0)', 'i, j = np.triu_indices(len(d), k=1)'),
  ("HL CI uses the wrong critical value",
   'z = sps.norm.ppf(0.5 + level / 2)', 'z = sps.norm.ppf(level)'),
  ("HL CI off by one in the order statistic",
   'return float(w[k]), float(w[m - 1 - k])', 'return float(w[k]), float(w[m - 2 - k])'),
  ("SI-SDR margin loosened to TinyLSTMs' own 0.55 dB",
   '"si_sdr":       0.50,', '"si_sdr":       0.55,'),
  ("PESQ margin loosened after the fact",
   '"pesq_mos_lqo": 0.10,', '"pesq_mos_lqo": 0.20,'),
  ("SI-SDR switched to the wrong test",
   '"si_sdr":       "t",', '"si_sdr":       "wilcoxon",'),
  ("N >= 20 rule relaxed", 'if n < 20:', 'if n < 2:'),
  ("n counted before dropping non-finite values",
   '    ok = np.isfinite(a) & np.isfinite(b)\n    a, b = a[ok], b[ok]\n    n = len(a)',
   '    n = len(a)\n    ok = np.isfinite(a) & np.isfinite(b)\n    a, b = a[ok], b[ok]'),
  ("unpaired shapes no longer refused",
   'raise ValueError(f"unpaired samples: {a.shape} vs {b.shape}")', 'pass'),
  ("unknown metric silently defaults to the t-test",
   '    kind = TEST.get(metric)\n    if kind is None:', '    kind = TEST.get(metric, "t")\n    if kind is None:'),
  ("rank-biserial normalised by n instead of the rank sum",
   'return float((wp - wn) / (wp + wn))', 'return float((wp - wn) / len(d))'),
  ("Holm degenerates to Bonferroni",
   'val = (len(p) - rank) * p[idx]', 'val = len(p) * p[idx]'),
  ("Holm loses its running maximum",
   'running = max(running, val)', 'running = val'),
  ("CI level widened to 1 - alpha (95%, not the TOST interval)",
   'lvl = 1 - 2 * alpha', 'lvl = 1 - alpha'),
  ("point estimate for the wilcoxon branch reverts to the mean",
   '        point = hodges_lehmann(d)', '        point = float(d.mean())'),
  ("INCONCLUSIVE stops disclaiming a null result",
   'This is NOT a no-difference result: ', ''),
  ("cohen_dz zero-variance guard reverted to exact zero (the fix today)",
   '    if sd <= 16 * np.finfo(float).eps * max(scale, 1.0):', '    if sd <= 0.0:'),
  ("p_diff NaN guard reverted (the other fix today)",
   'p_diff = float(sps.ttest_rel(b, a).pvalue) if np.any(d != 0) else 1.0',
   'p_diff = float(sps.ttest_rel(b, a).pvalue)'),
  ("shapiro subsample reverted to the biased head (trap 14)",
   'xs = np.random.default_rng(SUBSAMPLE_SEED).choice(\n            x, SHAPIRO_MAX, replace=False)', 'xs = x[:SHAPIRO_MAX]'),
  ("shapiro_n dropped from the result (truncation goes silent again)",
   '"n": int(len(x)), "shapiro_n": int(len(xs)),', '"n": int(len(x)),'),
  ("skew computed on the subsample instead of the full sample",
   '"skew": float(sps.skew(x)), "excess_kurtosis": float(sps.kurtosis(x))}',
   '"skew": float(sps.skew(xs)), "excess_kurtosis": float(sps.kurtosis(xs))}'),
  ("bootstrap ignores the requested confidence level",
   'r = sps.bootstrap((x,), statistic, confidence_level=level, method="BCa",',
   'r = sps.bootstrap((x,), statistic, confidence_level=0.95, method="BCa",'),
  ("bootstrap seed varies per call (a CI that moves between runs)",
   'random_state=np.random.default_rng(seed))', 'random_state=np.random.default_rng())'),
  ("bootstrap silently returns the symmetric t-interval",
   '    return float(r.confidence_interval.low), float(r.confidence_interval.high)',
   '    h = sps.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))\n    return float(x.mean() - h), float(x.mean() + h)'),
  ("describe(bootstrap=True) overwrites the t-interval instead of adding to it",
   '        out["boot_lo"], out["boot_hi"] = lo, hi',
   '        out["ci_lo"], out["ci_hi"] = lo, hi\n        out["boot_lo"], out["boot_hi"] = lo, hi'),
  ("describe() uses the population sd", 'sd = float(x.mean()), float(x.std(ddof=1))',
   'sd = float(x.mean()), float(x.std(ddof=0))'),
]

def run():
    r = subprocess.run([sys.executable, "-m", "eval_harness.test_stats"],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr

shutil.copy(SRC, BAK)
orig = BAK.read_text()
survived, killed = [], 0
try:
    for name, old, new in MUTATIONS:
        if orig.count(old) != 1:
            survived.append((name, f"PATTERN MATCHED {orig.count(old)}x -- mutation not applied"))
            continue
        SRC.write_text(orig.replace(old, new))
        rc, out = run()
        if rc == 0:
            survived.append((name, "SURVIVED -- suite still passed"))
        else:
            killed += 1
            last = [l for l in out.splitlines() if l.startswith("FAIL:") or l.startswith("ERROR:")]
            print(f"killed  {name}\n        by {len(last)} test(s), first: "
                  f"{last[0][:90] if last else 'crash'}")
finally:
    SRC.write_text(orig)
    BAK.unlink()

print(f"\n{killed}/{len(MUTATIONS)} mutations killed")
for n, why in survived:
    print(f"  SURVIVOR: {n} -- {why}")
rc, out = run()
print("restored stats.py, suite:", "OK" if rc == 0 else "BROKEN")
