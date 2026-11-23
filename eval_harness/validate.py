"""
validate.py -- harness validation BEFORE any algorithm exists (task 6).

The cheapest possible check on the whole metric stack, and the first thing that
should ever be run through it: score the NOISY input against the clean
reference. That "Noisy" row is published in essentially every
VoiceBank-DEMAND paper, requires no model at all, and exercises every part of
the path that a real result would -- file loading, channel handling, length
alignment, mode selection, the metric libraries themselves, and the averaging.

If it reproduces the literature, the stack is sound. If it does not, the bug has
been found at the cheapest possible moment rather than after an algorithm is
blamed for it.

The DNS arm rides the same path and is BINNED, never celled: 150 clips carry
150 distinct noise instances, so a (noise x SNR) cell holds exactly one clip.
It enters as an aggregate generalisation check stratified by 5 dB SNR bin. The
key is looked up per corpus in manifest.STRATIFY_BY rather than inferred.

Usage:
    python -m eval_harness.validate                # VoiceBank + NOIZEUS
    python -m eval_harness.validate --corpus dns   # the generalisation arm
    python -m eval_harness.validate --corpus noizeus --limit 60
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import audio, manifest, metrics, stats

# Published "Noisy" reference rows. These are the values this harness is being
# checked against; they are targets, not outputs.
PUBLISHED = {
    "voicebank16k": {
        "pesq_mos_lqo": (1.97, 0.06, "wideband PESQ of the noisy test set, the "
                                     "standard 'Noisy' row in the VoiceBank-DEMAND "
                                     "literature"),
        "stoi": (0.921, 0.010, "STOI of the noisy test set, same row"),
    },
    # DNS, added 2026-09-16. The DNS arm was built with no anchor because none
    # was known; one exists after all, in Westhausen & Meyer (DTLN,
    # arXiv:2005.07551v2) Table 2, "DNS test set no reverb", row "Noisy":
    # PESQ 2.45, SI-SDR 9.07, STOI 91.52.
    #
    # TWO of its three columns are usable and the third is NOT, and which is
    # which is the whole point of listing it:
    #
    #   STOI   0.9152 measured vs 0.9152 published -- agreement to four decimal
    #          places, which also settles that this is the same 150-clip set
    #          despite the paper describing "300 samples" per category.
    #   SI-SDR 9.2295 dB measured vs 9.07 published -- 0.16 dB, within a
    #          plausible alignment/implementation difference.
    #   PESQ   1.5822 measured vs 2.45 published -- a gap of 0.87, EIGHT AND A
    #          HALF TIMES the locked 0.10 margin, and it is NOT a harness error.
    #          DTLN's reference [31] for PESQ is "ITU-T P.862 ... NARROW-BAND
    #          telephone networks", not P.862.2. Their column is narrowband and
    #          ours is wideband, and the two are not comparable by construction.
    #          This is the same trap as the NOIZEUS raw-vs-MOS-LQO question, in
    #          a second corpus, and it is why the mode is fixed by the corpus
    #          here rather than passed in by a caller.
    #
    # So DNS is anchored on STOI and SI-SDR and deliberately NOT on PESQ.
    "dns": {
        "stoi": (0.9152, 0.010, "STOI of the noisy DNS-Challenge INTERSPEECH-2020 "
                                "synthetic no-reverb test set, Westhausen & Meyer "
                                "(DTLN) Table 2 'Noisy'"),
        "si_sdr": (9.07, 0.75, "SI-SDR of the same row. Wider tolerance than the "
                               "metric's TOST margin on purpose: this is a "
                               "cross-implementation check, not an equivalence claim"),
    },
}


# NOIZEUS: the published per-algorithm PESQ numbers, settled 2026-10-12.
#
# Hu & Loizou 2007 (Speech Comm. 49, 588-601) -- the paper that defines the
# corpus -- publishes SUBJECTIVE P.835 ratings only (13 algorithms x babble/car/
# street/train x 5/10 dB, 16 sentences, Dynastat listeners, as bar charts), and
# Hu & Loizou 2006 (Interspeech) / 2008 (IEEE TASLP 16(1)) publish correlation
# coefficients, not scores. None of the three carries a PESQ table, and none
# carries a noisy ("unprocessed") PESQ row, so this corpus has no model-free
# anchor of the VoiceBank/DNS kind.
#
# The first-party per-algorithm PESQ table on NOIZEUS is Lu & Loizou 2008
# (Speech Comm. 50, 453-466), Table 2: 30 NOIZEUS sentences, babble/car/street
# at 0/5/10 dB (white noise too, which is not in NOIZEUS), scored with PESQ
# "(ITU, 2000)" -- i.e. raw P.862 -- through Loizou's MATLAB port. The SS and
# MMSE rows are identical across the paper's two beta columns (beta only acts
# on the smoothed variants), so one value per cell. No Wiener row exists.
#
# Scale: RAW P.862 (compare pesq_raw_mos, never pesq_mos_lqo). Algorithms:
#   SS   = conventional power spectral subtraction, their eq. (5) (Boll-type)
#   MMSE = Ephraim & Malah 1984 MMSE-STSA, decision-directed, alpha = 0.98
# Role: a task-8 anchor for the classical arm, at RANKING precision (MMSE above
# SS in every cell, by 0.03-0.35), not at the 0.10 margin -- see
# LOIZOU_PORT_SAMPLE for why.
NOIZEUS_PUBLISHED_ALGORITHMS = {
    "source": "Lu & Loizou 2008, Speech Communication 50, Table 2",
    "scale": "raw P.862, Loizou MATLAB port (composite.zip)",
    "spectral_subtraction": {
        ("babble", 0): 1.73, ("babble", 5): 2.04, ("babble", 10): 2.37,
        ("car", 0): 1.69, ("car", 5): 1.98, ("car", 10): 2.31,
        ("street", 0): 1.70, ("street", 5): 2.00, ("street", 10): 2.36,
    },
    "mmse_stsa": {
        ("babble", 0): 1.76, ("babble", 5): 2.12, ("babble", 10): 2.51,
        ("car", 0): 1.93, ("car", 5): 2.28, ("car", 10): 2.66,
        ("street", 0): 1.80, ("street", 5): 2.20, ("street", 10): 2.58,
    },
}

# VoiceBank-DEMAND: a published PESQ for a CLASSICAL algorithm on the exact
# test set, in the exact convention this harness uses. Pascual, Bonafonte &
# Serra, "SEGAN" (Interspeech 2017, arXiv:1703.09452), Table 1: their baseline
# is "the Wiener method based on a priori SNR estimation" (Scalart & Filho) and
# PESQ is stated in the text as "the wide-band version recommended in ITU-T
# P.862.2" -- so unlike NOIZEUS and DNS this column IS comparable as named.
# Noisy 1.97 on the same row is the anchor validate.py already reproduces.
# Our Wiener (task 8, 2026-10-12): 2.282 with the VAD tracker, 2.173 with
# MCRA -- both within the 0.10 margin of 2.22, from either side.
# The RNNoise arm's model-free anchor (added 2026-10-19 with task 9). The arm
# is scored at 16 kHz after the locked 577-tap downsample of BOTH the 48 kHz
# clean reference and the 48 kHz signal under test (Metric Stack Decision 5,
# amended and signed off 2026-09-16). Its bypass row is therefore refcheck.py's
# S4 cell -- our-filtered noisy against our-filtered clean -- measured over all
# 824 utterances on 2026-09-15 and recorded in eval_harness/RESULTS.md. The
# bypass run of enhance.py on voicebank48k must reproduce it; a deviation
# means the downsample or the 960/480 framing has moved.
RNNOISE_ARM_NOISY_ANCHOR = {
    "source": "refcheck.py S4 (2026-09-15), all 824 VoiceBank test utterances",
    "pesq_mos_lqo": 1.9645, "stoi": 0.9211, "estoi": 0.7867, "si_sdr": 8.4448,
    "tol": 1e-3,
}

# DNS: the published DTLN rows, for task 10's float oracle (added 2026-10-19).
# Same 150-clip no-reverb set as PUBLISHED['dns'] (STOI of the noisy row
# matches ours to four places, so it IS the same set). The paper's Table 2
# 'DTLN' row is the Keras model.h5 evaluated whole-file under TF 2.1; the
# repository README adds rows for the authors' own TF-lite float export (the
# pair this project's oracle runs) and their dynamic-range quantized export.
# STOI and SI-SDR are comparable; the PESQ column is NARROWBAND P.862 and is
# carried only so nobody compares it with our wideband figure by accident.
DNS_PUBLISHED_ALGORITHMS = {
    "source": "Westhausen & Meyer 2020 (DTLN), Table 2 row 'DTLN'; github.com/breizhn/DTLN "
              "README 'Results' table for the TF-lite rows; DNS INTERSPEECH-2020 synthetic no-reverb",
    "scale": "STOI and SI-SDR comparable; pesq_nb is narrowband P.862 (their ref. [31]) -- NOT our wideband",
    "noisy":      {"stoi": 0.9152, "si_sdr": 9.07,  "pesq_nb": 2.45},
    "dtln-keras": {"stoi": 0.9476, "si_sdr": 16.34, "pesq_nb": 3.04},   # Table 2, model.h5, TF 2.1
    "dtln-f32":   {"stoi": 0.9475, "si_sdr": 16.20, "pesq_nb": 2.98},   # README, "TF-light"
    "dtln-drq":   {"stoi": 0.9447, "si_sdr": 15.71, "pesq_nb": 2.95},   # README, "TF-light quantized"
}

VOICEBANK_PUBLISHED_ALGORITHMS = {
    "source": "Pascual et al. 2017 (SEGAN), Table 1, VoiceBank-DEMAND test set",
    "scale": "PESQ MOS-LQO, wideband P.862.2 (stated in the paper)",
    "wiener": {"pesq_mos_lqo": 2.22, "csig": 3.23, "cbak": 2.68, "covl": 2.67, "ssnr": 5.07},
    "noisy":  {"pesq_mos_lqo": 1.97, "csig": 3.35, "cbak": 2.44, "covl": 2.63, "ssnr": 1.68},
    # RNNoise (task 9, 2026-10-19): no first-party VoiceBank row exists (Valin
    # evaluated on his own unpublished mix). Third-party: arXiv:2506.01023
    # Table 3 lists the shipped RNNoise at 2.34 wideband PESQ on this test set.
    # Read from a search snippet, PDF not on disk -- check before citing.
    # (arXiv:2405.16834's 2.48 is a RETRAINED RNNoise and is not an anchor.)
    "rnnoise": {"pesq_mos_lqo": 2.34, "source": "arXiv:2506.01023 Table 3 (third party, unverified)"},
}

# The one calibration point between Loizou's MATLAB PESQ (the tool behind
# essentially every NOIZEUS PESQ number in the literature) and the ITU v2.0
# reference this harness vendors. Loizou's composite.zip ships sp09.wav,
# sp09_babble_sn10.wav (both md5-identical to the corpus files on disk) and
# enhanced_logmmse.wav, and its readme states pesq(8000, sp09, enhanced) =
# 2.2557. Scored here (2026-10-12) the same pair gives:
LOIZOU_PORT_SAMPLE = {
    "pair": ("sp09.wav", "enhanced_logmmse.wav"),
    "pesq_loizou_port": 2.2557,       # readme.txt, raw scale
    "pesq_itu_v2_raw": 2.0707,        # ours, via the P.862.1 inverse
    "pesq_itu_v2_mos_lqo": 1.8627,    # ours, as returned
    "delta_raw": +0.185,              # port minus ITU, 1.9x the 0.10 margin
    # Insensitive to length handling: full / truncated / zero-padded give
    # 2.0707 / 2.0730 / 2.0707. One sample; no MATLAB here to run the port over
    # the 960 corpus files, which is what would turn this into a calibration.
}


# Why a corpus has no published row to check against. Per corpus, because the
# reasons differ and are not interchangeable: one is a missing paper, the other
# is a corpus that was never designed to have an anchor.
NO_ANCHOR_NOTE = {
    "noizeus": (
        "(No published 'Noisy' row exists for NOIZEUS -- settled 2026-10-12 against",
        " the papers themselves, now in papers/. Hu & Loizou 2007 is",
        " SUBJECTIVE (P.835) only; Hu & Loizou 2006/2008 report correlations, not",
        " scores. Their PESQ is RAW P.862 -- defined in their own eq. (2)/(3) as",
        " a0 - a1*Dsym - a2*Dasym -- and P.862.1/MOS-LQO appears nowhere, so the",
        " comparable column here is pesq_raw_mos. The one first-party per-algorithm",
        " PESQ table is Lu & Loizou 2008 Table 2, carried in",
        " NOIZEUS_PUBLISHED_ALGORITHMS for task 8. It was computed with Loizou's",
        " MATLAB port, which on its own worked example reads +0.185 above the ITU",
        " v2.0 reference (LOIZOU_PORT_SAMPLE), so a NOIZEUS PESQ comparison with",
        " the literature carries ~0.2 of implementation uncertainty, 2x the margin.)",
    ),
    "dns": (
        "(PESQ specifically has no usable published counterpart here. DTLN's",
        " Table 2 does carry a 'Noisy' row for this exact test set, and its STOI",
        " and SI-SDR are checked above -- but its PESQ column is NARROWBAND",
        " (their reference [31] is P.862 narrow-band, not P.862.2), so it cannot",
        " be compared with the wideband figure this corpus is scored in.",
        " DNS remains a GENERALISATION check read as a trend across SNR bins;",
        " what it is here for is noise VARIETY, 150 unique instances.)",
    ),
    "voicebank48k": (
        "(The 48 kHz set exists solely for RNNoise and is scored after the locked",
        " 48->16 kHz downsample; the published 'Noisy' row is the 16 kHz one.)",
    ),
}


def _score(job):
    corpus, rate, clean_p, noisy_p, uid, noise, snr_db, snr_bin = job
    try:
        ref, sr_r = audio.load(clean_p)
        deg, sr_d = audio.load(noisy_p)
        if sr_r != sr_d:
            raise ValueError(f"rate mismatch {sr_r} vs {sr_d}")
        if sr_r != rate:
            raise ValueError(f"manifest says {rate} Hz, file is {sr_r} Hz")
        ref, deg = audio.align(ref, deg)
        r = metrics.all_metrics(ref, deg, rate, corpus)
        # Carry the experimental factors through, so the per-utterance CSV can be
        # stratified by (noise x SNR) without a second join. Noise type is a
        # factor in the matrix, not just SNR -- and on VoiceBank the usable
        # contrast is non-stationary cafe/psquare vs quasi-stationary bus/office,
        # because the test split has no babble and no street.
        r.update(id=uid, corpus=corpus, noise=noise, snr_db=snr_db,
                 snr_bin=snr_bin, ok=True)
        return r
    except Exception as e:                       # recorded, never silently dropped
        return {"id": uid, "corpus": corpus, "noise": noise, "snr_db": snr_db,
                "snr_bin": snr_bin, "ok": False,
                "error": f"{type(e).__name__}: {e}"}


def run(corpus: str, root: Path, limit: int | None, workers: int) -> pd.DataFrame:
    m = manifest.load(root)
    us = manifest.utterances(m, corpus, limit=limit)
    jobs = [(corpus, u.rate_hz, u.clean_path, u.noisy_path, u.id,
             u.noise, u.snr_db, u.snr_bin) for u in us]

    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = pool.map(_score, jobs, chunksize=4)
    dt = time.time() - t0

    df = pd.DataFrame(rows)
    bad = df[~df.ok]
    print(f"  {len(df)} utterances in {dt:.1f}s ({workers} workers, "
          f"{dt/max(1,len(df))*1000:.0f} ms each)")
    if len(bad):
        print(f"  *** {len(bad)} FAILED ***")
        for e, n in bad.error.value_counts().items():
            print(f"      {n:4d}  {e}")
    return df


def summarise(df: pd.DataFrame, corpus: str) -> dict:
    """Per-metric summary, through stats.describe() rather than a hand-rolled
    interval -- one implementation, and the same one the comparison layer uses.

    BOTH intervals are printed on purpose. The t-interval is what earlier
    revisions reported; the BCa bootstrap assumes nothing about the shape, and
    PESQ and STOI are bounded and skewed, which is precisely why their
    significance tests are rank-based. Printing the gap between the two answers
    the question rather than asserting an answer: at N = 824 is the normal
    approximation good enough? The largest edge discrepancy is reported below
    the table, against the tightest locked margin (STOI, +/-0.01).
    """
    ok = df[df.ok]
    out = {"corpus": corpus, "n": int(len(ok))}
    cols = [c for c in ("pesq_mos_lqo", "pesq_raw_mos", "stoi", "estoi", "si_sdr")
            if c in ok.columns]
    print(f"\n  {'metric':<16}{'mean':>9}{'t-CI 95%':>20}{'BCa 95%':>20}"
          f"{'median':>9}{'min':>8}{'max':>8}")
    worst, worst_c = 0.0, ""
    for c in cols:
        v = ok[c].dropna().to_numpy()
        d = stats.describe(v, 0.95, bootstrap=True)
        d.update(min=float(v.min()), max=float(v.max()))
        out[c] = d
        gap = max(abs(d["boot_lo"] - d["ci_lo"]), abs(d["boot_hi"] - d["ci_hi"]))
        if gap > worst:
            worst, worst_c = gap, c
        tci = f"[{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]"
        bci = f"[{d['boot_lo']:.4f}, {d['boot_hi']:.4f}]"
        print(f"  {c:<16}{d['mean']:>9.4f}{tci:>20}{bci:>20}"
              f"{d['median']:>9.4f}{v.min():>8.3f}{v.max():>8.3f}")
    print(f"  largest t-vs-BCa edge discrepancy: {worst:.5f} ({worst_c}), "
          f"{stats.BOOTSTRAP_RESAMPLES} resamples, seed {stats.BOOTSTRAP_SEED}")
    return out


def stratify(df: pd.DataFrame, corpus: str, metric: str = "pesq_mos_lqo",
             bootstrap: bool = True) -> pd.DataFrame | None:
    """Per-cell results, on the key that is LOCKED for this corpus.

    The key comes from manifest.stratify_key_for() and is not inferred from
    which columns happen to be populated. That inference was a live defect
    until 2026-09-15: DNS carries a `noise` label for all 150 clips -- 150
    DISTINCT ones -- so "cell by noise if a noise label exists" put DNS into
    150 cells of N = 1 while the docstring above it stated the opposite rule.

    Intervals are BCa bootstrap per cell, not just mean and sd. CLAUDE.md
    records why: at N = 824 the normal approximation was shown adequate (the
    largest t-vs-BCa edge gap was a few thousandths of a metric's margin), but
    "at a per-cell N of 40 it may not, and the per-cell intervals are where
    this will matter". So they are computed where the cells actually are.
    """
    ok = df[df.ok]
    key = list(manifest.stratify_key_for(corpus))
    manifest.assert_stratifiable(corpus, key)

    if not key:
        print(f"\n  {corpus} has no factor structure: reporting the aggregate "
              f"only (n = {len(ok)}).")
        return None

    missing = [k for k in key if k not in ok.columns or ok[k].isna().all()]
    if missing:
        raise AssertionError(
            f"{corpus} is stratified by {key} but {missing} is absent or empty. "
            f"For VoiceBank this means the DataShare logfile join failed, and "
            f"without it the noise-type factor is not recoverable at all."
        )

    def _cell_order(idx):
        """Sort cells sensibly. An SNR bin label sorts by its LOWER EDGE, not
        as a string -- lexically "10-14 dB" precedes "5-9 dB", which puts the
        bins out of SNR order in every table and invites a misreading."""
        vals = idx if isinstance(idx, tuple) else (idx,)
        out = []
        for v in vals:
            t = str(v)
            lead = t.split("-")[0].split()[0] if t else ""
            try:
                out.append((0, float(lead), t))
            except ValueError:
                out.append((1, 0.0, t))
        return out

    rows = []
    for idx, g in sorted(ok.groupby(key, dropna=False), key=lambda kv: _cell_order(kv[0])):
        v = g[metric].dropna().to_numpy()
        lbl = " ".join(str(i) for i in (idx if isinstance(idx, tuple) else (idx,)))
        r = {"cell": lbl, "n": len(v), "mean": float(v.mean()),
             "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan")}
        # A bootstrap on a handful of points is noise dressed as an interval.
        if bootstrap and len(v) >= 8:
            try:
                r["ci_lo"], r["ci_hi"] = stats.bootstrap_ci(v)
            except Exception as e:                  # recorded, not swallowed
                r["ci_lo"] = r["ci_hi"] = float("nan")
                r["ci_note"] = f"{type(e).__name__}"
        else:
            r["ci_lo"] = r["ci_hi"] = float("nan")
        rows.append(r)
    t = pd.DataFrame(rows)

    what = "bins" if key == ["snr_bin"] else "cells"
    print(f"\n  {metric} by {' x '.join(key)}  ({what}, BCa 95% per {what[:-1]}):")
    for r in rows:
        ci = (f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
              if np.isfinite(r["ci_lo"]) else "[n too small]")
        flag = "  <-- N < 20" if r["n"] < 20 else ""
        print(f"    {r['cell']:<22} n={r['n']:<4} mean {r['mean']:.4f}"
              f"  sd {r['sd']:.4f}  {ci}{flag}")
    under = t[t.n < 20]
    print(f"    {len(t)} {what}, min N = {int(t.n.min())}, "
          f"{'ALL >= 20' if under.empty else f'{len(under)} UNDER 20'}"
          f"  (project minimum is 20 per cell)")
    return t


def check(summary: dict, corpus: str) -> bool:
    ref = PUBLISHED.get(corpus)
    if not ref:
        print(f"\n  No published 'Noisy' row recorded for {corpus}; reporting only.")
        for line in NO_ANCHOR_NOTE.get(corpus, ()):
            print(f"  {line}")
        return True
    print(f"\n  Against the published 'Noisy' row:")
    allok = True
    for k, (target, tol, what) in ref.items():
        got = summary[k]["mean"]
        d = got - target
        ok = abs(d) <= tol
        allok &= ok
        print(f"    {k:<16} ours {got:7.4f}  published {target:6.3f}  "
              f"delta {d:+7.4f}  tol +/-{tol}  {'PASS' if ok else '*** FAIL ***'}")
        print(f"        ({what})")
    return allok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append",
                    help="corpus to validate (repeatable); default: voicebank16k, noizeus")
    ap.add_argument("--root", type=Path, default=manifest.DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--out", type=Path, default=None, help="write per-utterance CSV here")
    a = ap.parse_args(argv)

    print("=== eval_harness validation: the published 'Noisy' baseline ===\n")
    prov = metrics.assert_provenance()
    print("Provenance (asserted, not assumed):")
    for k, v in prov.items():
        print(f"  {k:<18} {v}")

    corpora = a.corpus or ["voicebank16k", "noizeus"]
    summaries, allok = [], True
    for c in corpora:
        print(f"\n--- {c} (PESQ {manifest.pesq_mode_for(c)}) ---")
        df = run(c, a.root, a.limit, a.workers)
        s = summarise(df, c)
        stratify(df, c)
        allok &= check(s, c)
        summaries.append(s)
        if a.out:
            p = a.out.with_name(f"{a.out.stem}_{c}{a.out.suffix or '.csv'}")
            df.to_csv(p, index=False)
            print(f"\n  per-utterance results -> {p}")

    print("\n=== " + ("VALIDATION PASSED" if allok else "VALIDATION FAILED") + " ===")
    print(json.dumps(summaries, indent=1)[:0] or "", end="")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
