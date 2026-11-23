"""
separation.py -- the vocal-isolation tier: SDR / SIR / SAR via museval (task 6,
serving task 11).

A SEPARATE TIER, not another corpus. MUSDB18-HQ is 44.1 kHz stereo, its metrics
are BSS Eval rather than PESQ/STOI, and CLAUDE.md is explicit that its numbers
are never pooled with the 16 kHz speech results. Three things follow and all
three are enforced below rather than trusted:

  * NO RESAMPLING. 44.1 kHz throughout. The locked 48->16 kHz resampler belongs
    to the RNNoise metric path and must not touch this tier.
  * NO MONO COLLAPSE. `audio.load_multichannel` is used, not `audio.load`.
    BSS Eval is defined on the multichannel image of each source; averaging the
    channels would change what SDR means, not merely lose the stereo field.
  * THE ACCOMPANIMENT REFERENCE IS FORMED HERE, from the stem sum. The manifest
    carries only `noisy_path` = mixture and `clean_path` = vocals, because the
    track layout is fixed; drums + bass + other are siblings. Deliberately NOT
    `mixture - vocals`: the mixture is not a bit-exact sum of the stems (3 LSB
    peak, ~-80 dBFS, independent float->int16 rounding), so that subtraction
    would fold the rounding into the reference.

Two measured facts shaped this module. Both cost a wrong answer to learn.

1. "SCORE THE MIXTURE AS THE ESTIMATE" IS NOT A VALID FLOOR HERE, and it looks
   like one. It is the natural analogue of the speech arm's "Noisy" row -- the
   no-model baseline that needs nothing but the corpus -- and measured on
   `AM Contra - Heart Peripheral` it returns

       SDR   vocals -64.4 dB,  accompaniment +65.8 dB,  SAR 69.0 dB

   which is not a floor but nonsense. The reason is the criterion: BSS Eval
   *images* (mode v4) allows each estimate to be matched against a 512-tap
   FILTERED version of the reference subspace. Given the whole subspace, those
   filters can subtract the vocals out of the mixture, so the accompaniment
   estimate scores +66 dB and the vocals estimate scores -64 dB. Identical
   estimates for two sources make the decomposition degenerate. **There is no
   model-free floor for this tier, and inventing one would put a fabricated
   number in the results table.** So this module validates itself differently:

2. VALIDATION IS AN ORACLE PLUS A PREDICTED DEGRADATION, and both discriminate.
   `oracle_check` feeds the true stems as the estimates: SDR must come back
   enormous, which no accidentally-wrong reference construction would produce
   (sum the wrong stems and it collapses). `injected_check` adds white noise to
   the true vocals at a known per-frame SNR and checks the returned SDR tracks
   it -- white noise is not in the span of the filtered references, so the
   projection cannot remove it. Between them they exercise reference
   construction, channel handling, framing, the museval call and the frame
   aggregation, with a *predicted* answer rather than a plausible-looking one.

Aggregation follows the museval/SiSEC convention: **median over frames within a
track, then median over tracks**, never the mean. BSS Eval frames where a source
is silent return NaN (MUSDB vocals are silent for much of many tracks), and a
mean over frames would be dominated by whichever frames happened to be scored.

Runtime, measured: **~0.11 s of compute per second of audio**, flat across
content (silent-intro, dense-vocal and whole-track excerpts all land within
0.094-0.114 s/s) and across estimate type (oracle, injected-noise and the
degenerate mixture case are all 3.4 s for a 30 s excerpt). The 50-track test
split is ~3.3 h of audio, so a full pass is **~22 min single-threaded** and a
few minutes across workers.

That figure was wrong by 20x on first measurement -- 47 s for a 30 s excerpt --
and the cause is worth recording, because it is one of this project's own rules
pointed back at itself: the probe ran while a 14-worker validation job had the
machine, so it timed contention rather than museval. The identical call returns
3.4 s on an idle host and the SDR values are bit-identical between the two runs,
which is how it was caught. **A timing measurement taken while something else is
running is not a measurement.** Workers default to 6 for a different reason --
memory, not CPU: a 4-minute stereo track is 74 MB per source at float32 and
museval holds several at once, against 14 GB of host RAM.

Usage:
    python -m eval_harness.separation --validate            # oracle + injected
    python -m eval_harness.separation --validate --tracks 3
    python -m eval_harness.separation --estimates-dir OUT   # score a separator
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import audio, manifest, stats

CORPUS = "musdb18hq"
RATE = 44_100
CHANNELS = 2

# The five stems are a fixed layout, which is what lets the manifest carry only
# two paths. `vocals` is the target; the other three sum to the accompaniment.
TARGET_STEM = "vocals"
ACCOMP_STEMS = ("drums", "bass", "other")
SOURCES = ("vocals", "accompaniment")

# museval defaults, which ARE the published MUSDB convention: 1 s windows, no
# overlap, BSS Eval v4 (images). Not knobs -- changing any of them makes the
# numbers incomparable with every published MUSDB result, which is the only
# reason to use this metric rather than SI-SDR.
WIN = RATE
HOP = RATE
MODE = "v4"

# Oracle: true stems as estimates. Anything less than this is a broken path, not
# a modest result. Observed in practice well above 100 dB; the threshold is set
# far below what a correct path gives and far above what a wrong one gives.
ORACLE_MIN_SDR_DB = 60.0

# Injected-noise check: how close the returned SDR must sit to the SNR actually
# present per frame. MEASURED at a max |delta| of 0.16 dB over four points on
# two tracks, so the tolerance is 0.5 dB and not the 3.0 first guessed -- which
# would have passed a path wrong by 2.8 dB. The value of a predictive check is
# that it is tight: a badly broken path misses by tens of dB (the mixture case
# above returns -64 and +66), but a subtly wrong one would not.
INJECT_TOL_DB = 0.5
INJECT_SEED = 20260915


def _museval():
    """Imported lazily and version-checked, like metrics.assert_provenance."""
    import importlib.metadata as md
    import museval
    v = md.version("museval")
    if v != "0.4.1":
        raise AssertionError(
            f"museval is {v}, not the pinned 0.4.1. BSS Eval's framing and "
            f"filter length are part of what makes these numbers comparable "
            f"with published MUSDB results; an unpinned version is not."
        )
    return museval


def tracks(root: Path | str = manifest.DEFAULT_ROOT, *, split: str = "test",
           limit: int | None = None) -> list[manifest.Utterance]:
    """The MUSDB tracks of one split, as manifest records.

    Note for any future fine-tuning work, which this module does not do: the
    MUSDB split is NOT artist-disjoint. Hollow Ground, Skelpolu and The Long
    Wait appear on both sides (5 train / 3 test tracks). Harmless while
    separators stay pretrained (Arm A); a trap the moment one is trained.
    """
    m = manifest.load(root)
    role = "eval" if split == "test" else "train"
    us = manifest.utterances(m, CORPUS, role=role)
    if limit and limit < len(us):
        # Fixed-seed random draw, never the head -- trap 14.
        rng = np.random.default_rng(INJECT_SEED)
        idx = np.sort(rng.choice(len(us), size=limit, replace=False))
        us = [us[i] for i in idx]
    return us


def references_for(u: manifest.Utterance) -> tuple[np.ndarray, np.ndarray, int]:
    """(vocals, accompaniment, rate). The accompaniment is the stem sum."""
    d = Path(u.noisy_path).parent
    voc, sr = audio.load_multichannel(d / f"{TARGET_STEM}.wav", RATE)
    acc = np.zeros_like(voc)
    for stem in ACCOMP_STEMS:
        x, sr2 = audio.load_multichannel(d / f"{stem}.wav", RATE)
        if x.shape != voc.shape or sr2 != sr:
            raise AssertionError(
                f"{d.name}: stem {stem} is {x.shape} @ {sr2} Hz against vocals "
                f"{voc.shape} @ {sr} Hz. The stems of a MUSDB track are the same "
                f"length by construction, so this is an extraction fault."
            )
        acc += x
    if voc.shape[1] != CHANNELS:
        raise AssertionError(f"{d.name}: {voc.shape[1]} channels, expected {CHANNELS}")
    return voc, acc, sr


def _frame_snr_db(sig: np.ndarray, noise: np.ndarray, win: int = WIN) -> float:
    """Median per-frame SNR, framed exactly as museval frames its metrics.

    This is the *prediction* the injected-noise check is tested against, which
    is why it is computed here rather than assumed to equal the global SNR:
    MUSDB vocals are silent for long stretches, so the global figure and the
    median frame figure are different numbers.
    """
    n = min(len(sig), len(noise)) // win
    out = []
    for i in range(n):
        a = sig[i * win:(i + 1) * win]
        b = noise[i * win:(i + 1) * win]
        ps, pn = float((a * a).sum()), float((b * b).sum())
        if ps > 0.0 and pn > 0.0:
            out.append(10.0 * np.log10(ps / pn))
    return float(np.median(out)) if out else float("nan")


def evaluate_track(refs: np.ndarray, ests: np.ndarray) -> dict:
    """museval on one track, aggregated to the per-track median over frames.

    refs/ests are (nsrc, nsampl, nchan). Frames where a source is silent come
    back NaN and are skipped by nanmedian -- the museval convention -- and the
    surviving frame count is reported so a track scored on three frames cannot
    pass as a track scored on two hundred.
    """
    mus = _museval()
    sdr, isr, sir, sar = mus.evaluate(refs, ests, win=WIN, hop=HOP, mode=MODE)
    out = {}
    for i, name in enumerate(SOURCES):
        with np.errstate(invalid="ignore"):
            out[f"{name}_sdr"] = float(np.nanmedian(sdr[i]))
            out[f"{name}_isr"] = float(np.nanmedian(isr[i]))
            out[f"{name}_sir"] = float(np.nanmedian(sir[i]))
            out[f"{name}_sar"] = float(np.nanmedian(sar[i]))
        # NaN means the frame was not scored (a silent source). inf means a
        # PERFECT match, which is what the oracle returns and is not a failure
        # -- counting inf as non-finite reported "0/162 frames scored" for a
        # run that in fact scored every frame exactly.
        out[f"{name}_frames"] = int((~np.isnan(sdr[i])).sum())
        out[f"{name}_exact"] = int(np.isposinf(sdr[i]).sum())
    out["frames_total"] = int(sdr.shape[1])
    return out


# --- the two validation jobs ------------------------------------------------
def _oracle_job(u):
    try:
        voc, acc, _ = references_for(u)
        refs = np.stack([voc, acc])
        r = evaluate_track(refs, refs.copy())      # estimates ARE the references
        r.update(track=Path(u.noisy_path).parent.name, kind="oracle", ok=True)
        return r
    except Exception as e:
        return {"track": Path(u.noisy_path).parent.name, "kind": "oracle",
                "ok": False, "error": f"{type(e).__name__}: {e}"}


def _inject_job(args):
    u, snr_db = args
    try:
        voc, acc, _ = references_for(u)
        rng = np.random.default_rng(INJECT_SEED)
        n = rng.standard_normal(voc.shape).astype(voc.dtype)
        # Scale to the requested SNR on total power, then MEASURE what the
        # per-frame median actually is -- that measured value is the prediction.
        pv, pn = float((voc * voc).sum()), float((n * n).sum())
        if pv <= 0.0:
            raise ValueError("silent vocals stem")
        n *= np.sqrt(pv / pn) * (10.0 ** (-snr_db / 20.0))
        est_voc = voc + n
        refs = np.stack([voc, acc])
        ests = np.stack([est_voc, acc])            # accompaniment left exact
        r = evaluate_track(refs, ests)
        r.update(track=Path(u.noisy_path).parent.name, kind=f"inject{snr_db:+g}",
                 snr_db_requested=snr_db,
                 snr_db_per_frame=_frame_snr_db(voc, n), ok=True)
        return r
    except Exception as e:
        return {"track": Path(u.noisy_path).parent.name,
                "kind": f"inject{snr_db:+g}", "ok": False,
                "error": f"{type(e).__name__}: {e}"}


def validate(root: Path, n_tracks: int, snrs: tuple[float, ...],
             workers: int) -> bool:
    """Oracle + injected-noise validation. No separator required."""
    us = tracks(root, limit=n_tracks)
    print(f"  {len(us)} track(s), {workers} worker(s); "
          f"museval {MODE}, win={WIN} hop={HOP}")
    for u in us:
        print(f"    {Path(u.noisy_path).parent.name}")

    jobs_o = list(us)
    jobs_i = [(u, s) for u in us for s in snrs]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = pool.map(_oracle_job, jobs_o) + pool.map(_inject_job, jobs_i)
    print(f"  {len(rows)} evaluations in {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    bad = df[~df.ok]
    if len(bad):
        print(f"  *** {len(bad)} FAILED ***")
        for _, r in bad.iterrows():
            print(f"      {r.track} [{r.kind}] {r.error}")

    ok = df[df.ok]
    allok = not len(bad)

    print("\n  (1) Oracle -- true stems as estimates. A correct path returns a")
    print("      huge SDR; a wrong accompaniment sum collapses it.")
    o = ok[ok.kind == "oracle"]
    for _, r in o.iterrows():
        passed = min(r.vocals_sdr, r.accompaniment_sdr) >= ORACLE_MIN_SDR_DB
        allok &= bool(passed)
        print(f"      {r.track[:34]:<34} vocals {r.vocals_sdr:8.1f} dB   "
              f"accomp {r.accompaniment_sdr:8.1f} dB   "
              f"{r.vocals_frames}/{r.frames_total} frames  "
              f"{'PASS' if passed else '*** FAIL ***'}")

    print(f"\n  (2) Injected white noise into the vocals estimate. SDR must track")
    print(f"      the SNR actually present per frame, within {INJECT_TOL_DB} dB.")
    print("      The REQUESTED figure scales the noise on total power; the")
    print("      per-frame figure is what museval's 1 s framing actually sees, and")
    print("      on a track with sparse vocals the two differ enormously (one test")
    print("      track asks for 0 dB and presents -36.7). Predicting against the")
    print("      measured per-frame value is the point -- assuming the requested")
    print("      one would fail on exactly the tracks that matter.")
    inj = ok[ok.kind != "oracle"].sort_values(["track", "snr_db_requested"])
    for _, r in inj.iterrows():
        d = r.vocals_sdr - r.snr_db_per_frame
        passed = abs(d) <= INJECT_TOL_DB
        allok &= bool(passed)
        print(f"      {r.track[:26]:<26} req {r.snr_db_requested:+5.1f}  "
              f"per-frame {r.snr_db_per_frame:6.2f}  SDR {r.vocals_sdr:6.2f}  "
              f"delta {d:+5.2f}  {'PASS' if passed else '*** FAIL ***'}")
    if len(inj) > 2:
        d = (inj.vocals_sdr - inj.snr_db_per_frame).to_numpy()
        print(f"      delta over {len(d)} points: mean {d.mean():+.2f} dB, "
              f"max |delta| {np.abs(d).max():.2f} dB")

    print("\n  (3) The accompaniment estimate was left EXACT in (2), so its SDR")
    print("      stays at the oracle value -- a per-source check that the two")
    print("      sources are not being crossed.")
    for _, r in inj.iterrows():
        print(f"      {r.track[:26]:<26} accomp SDR {r.accompaniment_sdr:8.1f} dB")
    return allok


# --- scoring a real separator ----------------------------------------------
def _score_job(args):
    u, est_dir = args
    name = Path(u.noisy_path).parent.name
    try:
        voc, acc, _ = references_for(u)
        d = Path(est_dir) / name
        ev, _ = audio.load_multichannel(d / "vocals.wav", RATE)
        ap = d / "accompaniment.wav"
        if ap.exists():
            ea, _ = audio.load_multichannel(ap, RATE)
        else:
            # A vocals-only separator: the accompaniment estimate is what is
            # left of the mixture. Recorded in the output so the row says which.
            mix, _ = audio.load_multichannel(u.noisy_path, RATE)
            n = min(len(mix), len(ev))
            ea = mix[:n] - ev[:n]
        n = min(len(voc), len(ev), len(ea))
        if abs(len(voc) - len(ev)) > 16:
            raise ValueError(
                f"estimate is {len(ev)} samples against reference {len(voc)}: "
                f"that is an alignment fault, not rounding")
        r = evaluate_track(np.stack([voc[:n], acc[:n]]),
                           np.stack([ev[:n], ea[:n]]))
        r.update(track=name, accomp_source="file" if ap.exists() else "mixture-minus",
                 ok=True)
        return r
    except Exception as e:
        return {"track": name, "ok": False, "error": f"{type(e).__name__}: {e}"}


def run(root: Path, est_dir: Path, limit: int | None, workers: int) -> pd.DataFrame:
    us = tracks(root, limit=limit)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = pool.map(_score_job, [(u, est_dir) for u in us])
    df = pd.DataFrame(rows)
    print(f"  {len(df)} tracks in {time.time()-t0:.0f}s")
    bad = df[~df.ok]
    if len(bad):
        print(f"  *** {len(bad)} FAILED ***")
        for _, r in bad.iterrows():
            print(f"      {r.track}: {r.error}")
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Median over tracks -- the museval convention -- with a BCa interval.

    The median is the headline because that is what MUSDB results are reported
    as and what makes them comparable. The bootstrap interval is on the median
    for the same reason.
    """
    ok = df[df.ok]
    out = {"corpus": CORPUS, "n_tracks": int(len(ok))}
    print(f"\n  {'metric':<22}{'median':>9}{'BCa 95% (median)':>24}"
          f"{'mean':>9}{'min':>9}{'max':>9}")
    for src in SOURCES:
        for met in ("sdr", "isr", "sir", "sar"):
            c = f"{src}_{met}"
            if c not in ok.columns:
                continue
            v = ok[c].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if not len(v):
                continue
            med = float(np.median(v))
            try:
                lo, hi = stats.bootstrap_ci(v, statistic=np.median)
            except Exception:
                lo = hi = float("nan")
            out[c] = {"median": med, "ci_lo": lo, "ci_hi": hi,
                      "mean": float(v.mean()), "n": len(v)}
            ci = f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "[n/a]"
            print(f"  {c:<22}{med:>9.3f}{ci:>24}{v.mean():>9.3f}"
                  f"{v.min():>9.3f}{v.max():>9.3f}")
    if len(ok) < 20:
        print(f"\n  NOTE: {len(ok)} tracks is below this project's N >= 20 rule. "
              f"The full test split is 50 and clears it on its own.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=manifest.DEFAULT_ROOT)
    ap.add_argument("--validate", action="store_true",
                    help="oracle + injected-noise validation; needs no separator")
    ap.add_argument("--estimates-dir", type=Path, default=None,
                    help="<dir>/<track>/vocals.wav[, accompaniment.wav]")
    ap.add_argument("--tracks", type=int, default=2,
                    help="how many tracks (random, fixed seed); default 2 for --validate")
    ap.add_argument("--snrs", type=float, nargs="+", default=(0.0, 10.0),
                    help="injected SNRs in dB for the validation")
    ap.add_argument("--workers", type=int, default=6,
                    help="6 by default: museval is memory-bound, not CPU-bound")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    print("=== eval_harness separation tier: SDR/SIR/SAR via museval ===")
    print("    MUSDB18-HQ, 44.1 kHz stereo. NEVER pooled with the 16 kHz "
          "speech results.\n")
    import importlib.metadata as md
    print(f"    museval {md.version('museval')}, musdb {md.version('musdb')}, "
          f"BSS Eval {MODE}, win={WIN}, hop={HOP}")
    print(f"    accompaniment reference = sum of {ACCOMP_STEMS}, formed at "
          f"eval time\n")

    if a.validate:
        ok = validate(a.root, a.tracks, tuple(a.snrs), a.workers)
        print("\n=== " + ("SEPARATION PATH VALIDATED" if ok
                          else "SEPARATION VALIDATION FAILED") + " ===")
        return 0 if ok else 1

    if a.estimates_dir is None:
        ap.error("give --validate or --estimates-dir")
    # Echo the separator's provenance so that a flag which can be false is seen before the output is
    # scored (trap 23): which forward path produced these estimates, and whether its equivalence to the
    # shipped path was checked -- and with what outcome -- on the machine that ran it.
    prov_path = a.estimates_dir / "provenance.json"
    if prov_path.exists():
        import json
        prov = json.loads(prov_path.read_text())
        cond = prov.get("conditions", {})
        print(f"    estimates: model {prov.get('model')}, forward {prov.get('forward')!r}, "
              f"verified_lowmem {prov.get('verified_lowmem')!r}, lowmem_check {prov.get('lowmem_check', 'n/a')!r}")
        print(f"    produced on {cond.get('node')} ({cond.get('machine')}, torch {cond.get('torch')}, "
              f"{cond.get('threads')} threads, governor {cond.get('governor')})\n")
    else:
        print(f"    WARNING: no provenance.json in {a.estimates_dir}\n")
    df = run(a.root, a.estimates_dir, a.tracks if a.tracks != 2 else None, a.workers)
    summarise(df)
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"\n  per-track results -> {a.out}")
    return 0 if df.ok.all() else 1


if __name__ == "__main__":
    sys.exit(main())
