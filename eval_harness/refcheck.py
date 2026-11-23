"""
refcheck.py -- which clean reference does the RNNoise row get scored against?

A LOCKED decision of 2026-09-12 (CLAUDE.md, "Metric Stack Decisions" 5) chose a
primary and then required this check before any RNNoise number is quoted:

    Primary: score downsampled RNNoise output against the NATIVE 16 kHz clean
    mirror, the same reference every other algorithm is scored against, so the
    column is internally comparable. But that mirror was resampled from 48 kHz
    by a third party (JacobLinCool/VoiceBank-DEMAND-16k @ 4497db3) with an
    UNRECORDED filter, and the existing verification checked durations only,
    never sample values. So also score against the 48 kHz clean downsampled by
    our own filter and report the delta. If it exceeds the 0.10 PESQ margin that
    is a finding about the mirror's provenance and the primary choice has to be
    revisited.

Why this runs BEFORE RNNoise exists
-----------------------------------
As specified, the check scores RNNoise output against both candidate
references -- which would park it behind task 9. It does not have to be. Hold
the degraded signal fixed and vary only the reference, and the quantity in
question is isolated exactly; replace "RNNoise output" with the noisy input and
the check runs today. That is the same "needs no model at all" move that
validated the metric stack in the first place (see validate.py), applied to a
different question.

But the degraded signal's OWN provenance flips the sign of the bias
--------------------------------------------------------------------
This is the part a two-way comparison gets wrong. In the real RNNoise row the
degraded signal reaches 16 kHz through OUR filter, so it is filter-matched to
our candidate reference and MISmatched to the mirror. If the stand-in degraded
signal is the mirror's own noisy file, the mismatch sits on the other side and
the delta comes back with the opposite sign. So the two reference candidates are
crossed with two degraded-signal candidates, and all four are scored:

                            reference
                     mirror clean 16k   our downsample of clean 48k
    deg = mirror noisy 16k     S1                    S2
    deg = our downsample       S3                    S4
          of noisy 48k

  * S1 is fully mirror-filtered. It is EXACTLY the published "Noisy" row that
    validate.py reproduced to +0.0007 PESQ on 2026-09-12, so it is a built-in
    regression anchor: if S1 does not come back at 1.9707 / 0.9210, the bug is
    in this file and not in the mirror. Asserted below, not eyeballed.
  * S4 is fully our-filtered: the internally consistent alternative.
  * S3 is the primary choice under the RNNoise geometry, and S3-vs-S4 (C2) is
    therefore THE number this check exists to produce.
  * S1-vs-S3 (C3) is free and answers a different question worth having: at a
    fixed reference, how much does the choice of downsampler move the score of
    the signal itself? That is the cost our filter is being trusted with.

Two corpora are read, and that is not pooling
---------------------------------------------
manifest.py deliberately offers no API for fetching two corpora at once,
"because the next thing a caller would do is pool them". This reads
voicebank16k and voicebank48k and joins them BY ID -- the same 824 utterances at
two rates, never two samples concatenated. Every metric here is computed at
16 kHz in wideband mode, because the 48 kHz audio is consumed only by passing
through the locked 48->16 kHz resampler first. No 48 kHz number is produced, so
there is nothing to pool.

Usage:
    python -m eval_harness.refcheck                    # all 824, ~2 min
    python -m eval_harness.refcheck --limit 60         # random 60, not the head
    python -m eval_harness.refcheck --out refcheck.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import audio, manifest, metrics, stats

# Everything is scored at 16 kHz wideband. Both VoiceBank entries map to "wb" in
# manifest.PESQ_MODE, and metrics.pesq_score refuses wb at any rate but 16 kHz,
# so this label cannot silently select the wrong mode.
SCORED_AS = "voicebank16k"
SCORED_RATE = 16_000

METRICS = ("pesq_mos_lqo", "stoi", "estoi", "si_sdr")

# The four scorings, as (tag, degraded, reference).
SCORINGS = ("s1", "s2", "s3", "s4")

# S1 is the configuration validate.py already measured over all 824 utterances
# on 2026-09-12. Recorded here to 4 dp as CLAUDE.md carries them; a deviation
# beyond rounding means this file's loading path differs from validate.py's,
# which would invalidate every delta below.
VALIDATED_S1 = {"pesq_mos_lqo": 1.9707, "stoi": 0.9210,
                "estoi": 0.7867, "si_sdr": 8.4455}
VALIDATED_TOL = 1e-3

# A linear-phase FIR introduces no net delay and resample_poly compensates for
# its own, so a nonzero best lag between the two references is a property of
# whatever filter the mirror was built with -- a provenance finding, not noise.
# +/-16 samples at 16 kHz is +/-1 ms, far more than any sane group-delay
# difference, and the scan range is reported so the bound is visible.
LAG_SCAN = 16

# Cutting a sample to fit is how this project has produced a biased number three
# times (CLAUDE.md trap 14: --limit taking the head of an ordered manifest, and
# check_normality truncating to the head above N=5000). So --limit draws at
# random from a fixed seed and says how many points survived. The full run is
# ~2 min; there is no good reason to quote a truncated one.
LIMIT_SEED = 20260915


# Which reference the RNNoise row is scored against. None = open, and the run
# exits non-zero as a standing reminder.
#
# SIGNED OFF 2026-09-16: "ours". This check found the locked primary choice
# ("mirror") to be the filter-MISMATCHED cell -- 0.71 dB SI-SDR, 1.43x the
# locked +/-0.50 dB margin, exceeded on 382 of 824 utterances -- while
# PESQ/STOI/ESTOI are measured indifferent at ~1/50th of their margins. So the
# amendment costs nothing on the metrics that could have carried it and removes
# a real bias on the one that does. This is metric decision 5's own stated
# consequence firing, not a post-hoc revision. Recorded identically in
# CLAUDE.md; a decision that lives in only one of the two places is how they
# drift.
RNNOISE_REFERENCE: str | None = "ours"

_REFERENCE_MEANING = {
    "ours": ("the 48 kHz clean downsampled by our own locked filter (S4)",
             "filter-matched to RNNoise output, so no phase-mismatch tax"),
    "mirror": ("the native 16 kHz clean mirror (S3)",
               "filter-MISMATCHED to RNNoise output: accepts the SI-SDR tax "
               "measured below, and must be declared as a known bias"),
}


def _nmse_db(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mean-square error of b against a, in dB. The same quantity
    task 15 reports for cross-platform equivalence, used here on two candidate
    references instead of two platforms."""
    den = float(a @ a)
    if den <= 0.0:
        return float("nan")
    num = float((a - b) @ (a - b))
    return float("-inf") if num <= 0.0 else 10.0 * np.log10(num / den)


def _best_lag(a: np.ndarray, b: np.ndarray, scan: int = LAG_SCAN):
    """Integer sample offset minimising NMSE over +/-scan, and that NMSE."""
    n = min(len(a), len(b))
    best_lag, best = 0, float("inf")
    for lag in range(-scan, scan + 1):
        if lag >= 0:
            x, y = a[lag:n], b[:n - lag]
        else:
            x, y = a[:n + lag], b[-lag:n]
        e = _nmse_db(x, y)
        if e < best:
            best, best_lag = e, lag
    return best_lag, best


def _frac_lag(a: np.ndarray, b: np.ndarray, scan: int = LAG_SCAN) -> float:
    """Sub-sample delay of b relative to a: parabolic fit on the cross-
    correlation peak. An integer scan cannot see this, and it is the whole
    reason _best_lag returning 0 is not evidence of alignment."""
    n = 1 << int(np.ceil(np.log2(2 * max(len(a), len(b)))))
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    r = np.fft.irfft(B * np.conj(A), n)
    r = np.concatenate([r[-scan:], r[:scan + 1]])       # lags -scan..+scan
    k = int(np.argmax(r))
    if k == 0 or k == len(r) - 1:
        return float("nan")
    ym1, y0, yp1 = float(r[k - 1]), float(r[k]), float(r[k + 1])
    den = ym1 - 2 * y0 + yp1
    if den == 0.0:
        return float(k - scan)
    return float((k - scan) + 0.5 * (ym1 - yp1) / den)


def _shift(x: np.ndarray, tau: float) -> np.ndarray:
    """Delay x by tau samples (may be fractional) by phase rotation."""
    n = len(x)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * tau), n)


# Welch parameters for the magnitude/coherence estimate. 1024 points at 16 kHz
# is 64 ms -- long enough for 15.6 Hz resolution, short enough that a few
# hundred frames average over the whole utterance.
_SPEC_N, _SPEC_H = 1024, 512
_SPEC_W = np.hanning(_SPEC_N)
SPEC_BAND = (100.0, 6000.0)      # the band both filters pass flat, by design


def _spectral_agreement(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(mean magnitude ratio in dB, mean magnitude coherence) over SPEC_BAND.

    This is the measurement that separates the two ways two resamplers can
    disagree, and they have opposite consequences here:

      * a MAGNITUDE difference (droop, weak alias rejection) changes what the
        metrics see -- PESQ weights 7-8 kHz and would move.
      * a PHASE difference (group delay) leaves every magnitude-domain metric
        untouched but wrecks any waveform-domain one, because SI-SDR subtracts
        a projection in the time domain.

    NMSE alone cannot tell them apart: it sums both into one number and a pure
    phase difference reads as a large error.
    """
    f = np.fft.rfftfreq(_SPEC_N, 1.0 / SCORED_RATE)
    band = (f > SPEC_BAND[0]) & (f < SPEC_BAND[1])
    num = np.zeros(len(f), complex)
    pa = np.zeros(len(f))
    pb = np.zeros(len(f))
    for o in range(0, min(len(a), len(b)) - _SPEC_N, _SPEC_H):
        A = np.fft.rfft(a[o:o + _SPEC_N] * _SPEC_W)
        B = np.fft.rfft(b[o:o + _SPEC_N] * _SPEC_W)
        num += A * np.conj(B)
        pa += np.abs(A) ** 2
        pb += np.abs(B) ** 2
    if pb[band].sum() <= 0.0 or pa[band].sum() <= 0.0:
        return float("nan"), float("nan")
    mag = 10.0 * np.log10(pa[band].sum() / pb[band].sum())
    coh = float(np.mean(np.abs(num[band]) / np.sqrt(pa[band] * pb[band])))
    return float(mag), coh


def _score(job):
    uid, clean16, noisy16, clean48, noisy48 = job
    try:
        c16, _ = audio.load(clean16, expect_rate=16_000)
        n16, _ = audio.load(noisy16, expect_rate=16_000)
        c48, _ = audio.load(clean48, expect_rate=48_000)
        n48, _ = audio.load(noisy48, expect_rate=48_000)

        # The only sample-rate conversion this project performs, and the only
        # one audio.py will do: 577-tap Kaiser(12) at 7800 Hz, ratio exactly 1/3.
        c_ours = audio.downsample_48_to_16(c48)
        n_ours = audio.downsample_48_to_16(n48)

        row = {"id": uid, "ok": True}

        # (A) The two candidate references, compared as signals rather than
        # through a metric. This is the provenance measurement proper: a metric
        # can be insensitive to a difference that is nonetheless real.
        a, b = audio.align(c16, c_ours)
        row["ref_nmse_db"] = _nmse_db(a, b)
        row["ref_max_dev_lsb"] = float(np.max(np.abs(a - b))) * 32768.0
        row["ref_rms_mirror"] = float(np.sqrt(a @ a / len(a)))
        lag, lag_nmse = _best_lag(c16, c_ours)
        row["ref_best_lag"] = lag
        row["ref_nmse_db_at_best_lag"] = lag_nmse

        # A constant fractional delay, and what NMSE survives removing it.
        # Edges are trimmed because the phase rotation is circular.
        row["ref_gain_ratio"] = float((a @ b) / (b @ b)) if (b @ b) > 0 else np.nan
        tau = _frac_lag(a, b)
        row["ref_frac_lag"] = tau
        if np.isfinite(tau):
            g = _SPEC_H
            bs = _shift(b, -tau)
            row["ref_nmse_db_delag"] = _nmse_db(a[g:-g], bs[g:-g])
        # Magnitude vs phase: the decomposition NMSE cannot give.
        mag, coh = _spectral_agreement(a, b)
        row["ref_mag_ratio_db"] = mag
        row["ref_coherence"] = coh

        # (B) The four scorings. Degraded signal crossed with reference.
        for tag, deg, ref in (("s1", n16, c16), ("s2", n16, c_ours),
                              ("s3", n_ours, c16), ("s4", n_ours, c_ours)):
            r, d = audio.align(ref, deg)
            for k, v in metrics.all_metrics(r, d, SCORED_RATE, SCORED_AS).items():
                if k != "pesq_mode":
                    row[f"{tag}_{k}"] = v
        return row
    except Exception as e:                       # recorded, never silently dropped
        return {"id": uid, "ok": False, "error": f"{type(e).__name__}: {e}"}


def build_jobs(root: Path, limit: int | None):
    """Pair the 16 and 48 kHz test splits by id. Neither is trusted to be in the
    same order as the other, and a missing id is an error rather than a skip."""
    m = manifest.load(root)
    u16 = manifest.utterances(m, "voicebank16k")
    u48 = manifest.utterances(m, "voicebank48k")

    by48 = {u.id: u for u in u48}
    if len(by48) != len(u48):
        raise AssertionError("duplicate ids in the 48 kHz test split")
    missing = sorted(u.id for u in u16 if u.id not in by48)
    if missing:
        raise AssertionError(
            f"{len(missing)} of {len(u16)} 16 kHz test ids have no 48 kHz "
            f"counterpart (first: {missing[:3]}). The whole point of the 48 kHz "
            f"set is that it is the SAME utterances; without the join there is "
            f"nothing to compare."
        )
    if len(u16) != len(u48):
        raise AssertionError(
            f"split sizes differ: {len(u16)} at 16 kHz, {len(u48)} at 48 kHz"
        )

    paired = [(u.id, u.clean_path, u.noisy_path,
               by48[u.id].clean_path, by48[u.id].noisy_path) for u in u16]
    drawn = None
    if limit and limit < len(paired):
        rng = np.random.default_rng(LIMIT_SEED)
        idx = np.sort(rng.choice(len(paired), size=limit, replace=False))
        paired = [paired[i] for i in idx]
        drawn = (limit, len(u16))
    return paired, drawn


def run(root: Path, limit: int | None, workers: int) -> pd.DataFrame:
    jobs, drawn = build_jobs(root, limit)
    if drawn:
        print(f"  --limit: random {drawn[0]} of {drawn[1]} utterances "
              f"(seed {LIMIT_SEED}), NOT the head of the manifest")
    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = pool.map(_score, jobs, chunksize=4)
    dt = time.time() - t0

    df = pd.DataFrame(rows)
    bad = df[~df.ok]
    print(f"  {len(df)} utterances x 4 scorings in {dt:.1f}s ({workers} workers, "
          f"{dt/max(1,len(df))*1000:.0f} ms each)")
    if len(bad):
        print(f"  *** {len(bad)} FAILED ***")
        for e, n in bad.error.value_counts().items():
            print(f"      {n:4d}  {e}")
    return df


def report_signals(df: pd.DataFrame) -> None:
    """(A) How different are the two candidate references, as signals?"""
    ok = df[df.ok]
    print("\n(A) The two candidate 16 kHz clean references, compared as signals")
    print("    mirror (JacobLinCool @ 4497db3, unrecorded filter)  vs")
    print("    ours   (48 kHz DataShare clean -> locked 577-tap Kaiser(12))\n")
    for col, unit, fmt in (("ref_nmse_db", "dB", "{:9.3f}"),
                           ("ref_nmse_db_delag", "dB", "{:9.3f}"),
                           ("ref_max_dev_lsb", "LSB@int16", "{:9.3f}"),
                           ("ref_gain_ratio", "x", "{:9.4f}"),
                           ("ref_frac_lag", "samples", "{:9.4f}"),
                           ("ref_mag_ratio_db", "dB, 100-6k", "{:9.4f}"),
                           ("ref_coherence", "0-1, 100-6k", "{:9.5f}")):
        v = ok[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if not len(v):
            continue
        d = stats.describe(v, 0.95)
        print(f"    {col:<26} mean " + fmt.format(d["mean"]) +
              f"  median " + fmt.format(d["median"]) +
              f"  min " + fmt.format(v.min()) +
              f"  max " + fmt.format(v.max()) + f"   [{unit}]")
    lags = ok.ref_best_lag.value_counts().sort_index()
    print(f"    ref_best_lag               " +
          ", ".join(f"{int(k):+d}: {int(n)}" for k, n in lags.items()) +
          f"   (scan +/-{LAG_SCAN} samples)")
    if set(lags.index) != {0}:
        print("    ^ NONZERO INTEGER LAG: the mirror's resampler is offset from ours")
        print("      by whole samples. The NMSE at zero lag above then overstates the")
        print("      disagreement, and every waveform-domain metric is affected.")

    # The interpretation, driven by the numbers rather than asserted.
    mag = float(ok.ref_mag_ratio_db.abs().mean())
    coh = float(ok.ref_coherence.mean())
    nm = float(ok.ref_nmse_db.mean())
    print()
    if mag < 0.05 and coh > 0.999:
        print(f"    READ THIS TOGETHER: NMSE is {nm:.1f} dB, which looks alarming, but")
        print(f"    the magnitude ratio is {mag:.4f} dB and coherence {coh:.5f} over")
        print(f"    {SPEC_BAND[0]:.0f}-{SPEC_BAND[1]:.0f} Hz. The two references are")
        print("    SPECTRALLY IDENTICAL in the speech band and differ only in PHASE --")
        print("    the mirror's resampler is not linear-phase, and its group delay")
        print("    varies with frequency, so no single lag correction removes it")
        print("    (see ref_nmse_db_delag above).")
        print("    Consequence, and it is not symmetric across the metric stack:")
        print("      * PESQ / STOI / ESTOI are magnitude-domain and will barely move.")
        print("      * SI-SDR subtracts a time-domain projection, so it pays the full")
        print("        cost of a phase mismatch -- but ONLY when the reference and the")
        print("        degraded signal came through DIFFERENT resamplers.")
        print("    That is exactly the asymmetry (B) and (D) below quantify.")
    else:
        print(f"    Magnitude ratio {mag:.4f} dB, coherence {coh:.5f}: the two")
        print("    references differ in MAGNITUDE, not only phase. That would move")
        print("    PESQ as well, and is a stronger finding than a phase offset.")


def report_means(df: pd.DataFrame) -> None:
    ok = df[df.ok]
    print("\n(B) The four scorings")
    print("    S1 mirror deg / mirror ref   (= the validated 'Noisy' row)")
    print("    S2 mirror deg / our ref")
    print("    S3 our deg    / mirror ref   (RNNoise geometry, PRIMARY choice)")
    print("    S4 our deg    / our ref      (RNNoise geometry, our filter both sides)\n")
    print(f"    {'metric':<16}" + "".join(f"{t.upper():>11}" for t in SCORINGS))
    for metric in METRICS:
        cells = "".join(f"{ok[f'{t}_{metric}'].mean():>11.4f}" for t in SCORINGS)
        print(f"    {metric:<16}{cells}")


def check_anchor(df: pd.DataFrame) -> bool:
    """S1 must reproduce the validated 'Noisy' row. If it does not, the fault is
    here, not in the mirror -- so this is checked before any delta is believed."""
    ok = df[df.ok]
    full = len(ok) == 824
    print("\n(C) Regression anchor: S1 against the 2026-09-12 validation run")
    if not full:
        print(f"    n = {len(ok)}, not the full 824. The recorded row is a mean over")
        print("    the COMPLETE split, so it cannot be matched by a subsample and is")
        print("    shown for orientation only -- it gates nothing here.")
    allok = True
    for k, target in VALIDATED_S1.items():
        got = float(ok[f"s1_{k}"].mean())
        d = got - target
        good = abs(d) <= VALIDATED_TOL
        allok &= good
        mark = ("PASS" if good else "*** FAIL ***") if full else "(n/a)"
        print(f"    {k:<16} ours {got:8.4f}  recorded {target:8.4f}  "
              f"delta {d:+8.5f}  {mark}")
    if full and not allok:
        print("    S1 is the same computation validate.py performed over the same")
        print("    824 files. A deviation means refcheck's loading or alignment path")
        print("    differs from validate.py's, and every delta below is suspect.")
    return allok if full else True


COMPARISONS = (
    ("C1", "s1", "s2", "reference swap, MIRROR-filtered degraded signal",
     "deg is the mirror's own noisy file, so A is filter-matched and B is not."),
    ("C2", "s3", "s4", "reference swap, OUR-filtered degraded signal"
                       "   <-- the RNNoise geometry",
     "deg reaches 16 kHz through our locked filter, exactly as RNNoise output "
     "will. A is the primary choice (mirror ref), B the consistent alternative."),
    ("C3", "s1", "s3", "degraded-signal swap at a fixed (mirror) reference",
     "Not about the reference at all: what does the choice of DOWNSAMPLER cost "
     "the signal? This is the price our filter is being trusted with."),
)


def compare_all(df: pd.DataFrame) -> dict:
    ok = df[df.ok]
    out = {}
    print("\n(D) Paired comparisons, B against A, TOST against the locked margins")
    for name, ta, tb, title, why in COMPARISONS:
        print(f"\n  {name}  {title}")
        print(f"      {why}")
        cmps = [stats.compare(ok[f"{ta}_{m}"].to_numpy(),
                              ok[f"{tb}_{m}"].to_numpy(), m) for m in METRICS]
        out[name] = cmps
        print("\n".join("      " + l for l in stats.render(cmps).splitlines()))
    return out


def decide(cmps: dict) -> bool:
    """The gate the locked decision names: C2's PESQ delta against +/-0.10."""
    c2 = {c.metric: c for c in cmps["C2"]}
    pesq = c2["pesq_mos_lqo"]
    print("\n(E) Verdict on the primary reference choice")
    print(f"    The locked rule: if the delta exceeds the {pesq.margin:.2f} PESQ margin,")
    print("    that is a finding about the mirror's provenance and the primary")
    print("    choice (score against the 16 kHz mirror) has to be revisited.\n")
    within = abs(pesq.diff) <= pesq.margin
    print(f"    C2 PESQ delta {pesq.diff:+.4f}  vs margin +/-{pesq.margin:.2f}"
          f"   -> {'WITHIN' if within else 'EXCEEDS'}")
    print(f"    C2 PESQ TOST  p = {pesq.p_tost:.3g}  -> {pesq.verdict}")
    exceeded = [c.metric for c in cmps["C2"] if abs(c.diff) > c.margin]
    inconclusive = [c.metric for c in cmps["C2"] if not c.equivalent]
    if exceeded:
        print(f"\n    EXCEEDS THE MARGIN on: {', '.join(exceeded)}")
        print("    Do not quote an RNNoise number against the mirror until this is")
        print("    resolved. Record it as a provenance finding about the 16 kHz")
        print("    mirror and revisit the primary reference choice.")
    else:
        print(f"\n    No metric's C2 delta exceeds its locked margin.")
        print("    The primary choice stands: score the RNNoise row against the")
        print("    native 16 kHz clean mirror, the same reference every other")
        print("    algorithm uses, and cite this check as the evidence.")
    if inconclusive:
        print(f"\n    NOTE: TOST does not return EQUIVALENT for: "
              f"{', '.join(inconclusive)}.")
        print("    A delta inside the margin is not the same claim as a")
        print("    demonstrated equivalence; report whichever is actually supported.")

    print("\n(F) Recorded decision: RNNOISE_REFERENCE =", repr(RNNOISE_REFERENCE))
    if RNNOISE_REFERENCE is None:
        print("    Not yet signed off. The locked primary choice stands on paper but")
        print("    this check has shown it to be the filter-mismatched cell, so the")
        print("    run exits non-zero until the question is closed either way.")
        return False
    what, why = _REFERENCE_MEANING[RNNOISE_REFERENCE]
    print(f"    Score the RNNoise row against {what}")
    print(f"    -- {why}.")
    if RNNOISE_REFERENCE == "mirror" and exceeded:
        print("    The tax above is therefore ACCEPTED and declared, not resolved.")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=manifest.DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=None,
                    help="random subsample (fixed seed), not the head of the manifest")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--out", type=Path, default=None,
                    help="write the per-utterance table here")
    a = ap.parse_args(argv)

    print("=== RNNoise reference sensitivity check "
          "(CLAUDE.md, metric decision 5) ===\n")
    prov = metrics.assert_provenance()
    print("Provenance (asserted, not assumed):")
    for k, v in prov.items():
        print(f"  {k:<18} {v}")
    print()

    df = run(a.root, a.limit, a.workers)
    if not df.ok.any():
        print("\nevery utterance failed; nothing to report")
        return 1

    report_signals(df)
    report_means(df)
    anchor = check_anchor(df)
    cmps = compare_all(df)
    safe = decide(cmps)

    if a.out:
        df.to_csv(a.out, index=False)
        print(f"\n  per-utterance table -> {a.out}")

    ok = anchor and safe
    print("\n=== " + ("REFERENCE CHOICE CONFIRMED" if ok else
                      "ACTION REQUIRED -- see above") + " ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
