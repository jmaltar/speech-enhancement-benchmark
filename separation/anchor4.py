"""anchor4.py -- the four-target BSS Eval anchor for the separation tier (task 11).

The tier's quality row is TWO-SOURCE: vocals against an accompaniment reference formed from the
stem sum (`eval_harness/separation.py`). The authors of Open-Unmix publish umxhq vocals SDR 6.25 dB
on the same split under the STANDARD FOUR-TARGET evaluation -- vocals / drums / bass / other, each
estimate against its own stem reference -- which is a different reference subspace: BSS Eval v4
matches every estimate against 512-tap filtered versions of ALL the references, so with four
references the projection has a different span and the vocals number is not the same quantity.
This script computes that four-target figure from the four stems `umx_run.py` wrote (no
re-separation), so the two-source row can be set beside the literature like for like.

Same framing, same museval, same aggregation as the two-source scorer: 1 s window / 1 s hop,
BSS Eval v4 images, median over frames within a track, median over tracks, BCa on the median.
Everything is imported from `separation.py` rather than restated.

Memory: museval computes its projection filters over the WHOLE track (`framewise_filters=False`,
the v4 default), so its peak scales with track length and with the number of sources. The
two-source scorer peaked at 8.97 GB on the 430 s track with one worker; four sources need more.
Each track is therefore scored in its own subprocess, shortest track first, with the child's
peak RSS recorded in the row, and the CSV is flushed and fsync'd per row so a kill loses one
track and `--resume` picks up where it stopped. Run it under a capped `systemd-run --user` unit
and never beside another job (host-ram rule).

Usage:
    python separation/anchor4.py --estimates-dir $BENCHMARK_DATA_ROOT/enhanced/umxhq/musdb18hq \
        --out separation/results/host_umxhq_score4_YYYY-MM-DD.csv [--resume] [--shortest N]
        [--two-source separation/results/host_umxhq_score_2026-10-29.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_harness import audio, manifest, stats  # noqa: E402
from eval_harness import separation as sep  # noqa: E402

STEMS = ("vocals", "drums", "bass", "other")
METRICS = ("sdr", "isr", "sir", "sar")
FIELDS = (["track", "ok", "error", "frames_total", "n_samples", "maxrss_gb", "wall_s"]
          + [f"{s}_{m}" for s in STEMS for m in METRICS]
          + [f"{s}_frames" for s in STEMS])


def _track_dir(u: manifest.Utterance) -> Path:
    return Path(u.noisy_path).parent


def _n_frames(u: manifest.Utterance) -> int:
    import soundfile as sf
    return sf.info(str(u.noisy_path)).frames


def score_one(u: manifest.Utterance, est_dir: Path) -> dict:
    """Four references, four estimates, one museval call -- the standard MUSDB evaluation."""
    d = _track_dir(u)
    refs, ests = [], []
    for s in STEMS:
        r, sr = audio.load_multichannel(d / f"{s}.wav", sep.RATE)
        e, sr2 = audio.load_multichannel(Path(est_dir) / d.name / f"{s}.wav", sep.RATE)
        if sr != sep.RATE or sr2 != sep.RATE or r.shape[1] != sep.CHANNELS or e.shape[1] != sep.CHANNELS:
            raise AssertionError(f"{d.name}/{s}: {r.shape}@{sr} vs {e.shape}@{sr2}")
        if abs(len(r) - len(e)) > 16:
            raise ValueError(f"{d.name}/{s}: estimate {len(e)} samples against reference {len(r)}: "
                             f"alignment fault, not rounding")
        refs.append(r)
        ests.append(e)
    n = min(min(len(r) for r in refs), min(len(e) for e in ests))
    refs = np.stack([r[:n] for r in refs])
    ests = np.stack([e[:n] for e in ests])
    del r, e
    mus = sep._museval()
    t0 = time.time()
    sdr, isr, sir, sar = mus.evaluate(refs, ests, win=sep.WIN, hop=sep.HOP, mode=sep.MODE)
    out = {"track": d.name, "ok": True, "error": "", "frames_total": int(sdr.shape[1]),
           "n_samples": int(n), "wall_s": round(time.time() - t0, 1)}
    for i, s in enumerate(STEMS):
        with np.errstate(invalid="ignore"):
            for name, arr in zip(METRICS, (sdr, isr, sir, sar)):
                out[f"{s}_{name}"] = float(np.nanmedian(arr[i]))
        out[f"{s}_frames"] = int((~np.isnan(sdr[i])).sum())
    out["maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)
    return out


def _child(a):
    us = {_track_dir(u).name: u for u in sep.tracks(a.root)}
    u = us[a.one]
    try:
        row = score_one(u, a.estimates_dir)
    except Exception as e:  # reported, never swallowed
        row = {"track": a.one, "ok": False, "error": f"{type(e).__name__}: {e}"}
    print("ANCHOR4 " + json.dumps(row))
    return 0 if row["ok"] else 1


def _read_done(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return {r["track"]: r for r in csv.DictReader(f) if r.get("ok") == "True"}


def summarise(rows: list[dict], two_source: Path | None) -> None:
    ok = [r for r in rows if str(r.get("ok")) == "True"]
    print(f"\n  four-target BSS Eval v4, {len(ok)} track(s); median over tracks of the per-track "
          f"median over 1 s frames, BCa 95 % on the median")
    print(f"  {'metric':<22}{'median':>9}{'BCa 95% (median)':>24}{'mean':>9}{'min':>9}{'max':>9}")
    for s in STEMS:
        for m in METRICS:
            v = np.array([float(r[f'{s}_{m}']) for r in ok], dtype=float)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            med = float(np.median(v))
            try:
                lo, hi = stats.bootstrap_ci(v, statistic=np.median)
                ci = f"[{lo:.3f}, {hi:.3f}]"
            except Exception:
                ci = "[n/a]"
            print(f"  {s + '_' + m:<22}{med:>9.3f}{ci:>24}{v.mean():>9.3f}{v.min():>9.3f}{v.max():>9.3f}")
    peaks = [float(r["maxrss_gb"]) for r in ok if r.get("maxrss_gb") not in (None, "")]
    if peaks:
        print(f"\n  child peak RSS: max {max(peaks):.2f} GB over {len(peaks)} track(s)")
    if two_source is not None and two_source.exists():
        with two_source.open() as f:
            two = {r["track"]: r for r in csv.DictReader(f) if r.get("ok") == "True"}
        pairs = [(float(r["vocals_sdr"]), float(two[r["track"]]["vocals_sdr"]))
                 for r in ok if r["track"] in two]
        if pairs:
            d = np.array([p[0] - p[1] for p in pairs])
            try:
                lo, hi = stats.bootstrap_ci(d, statistic=np.median)
                ci = f"[{lo:.3f}, {hi:.3f}]"
            except Exception:
                ci = "[n/a]"
            print(f"\n  vocals SDR, four-target minus two-source, {len(d)} paired tracks: "
                  f"median {np.median(d):+.3f} dB {ci}, mean {d.mean():+.3f}, "
                  f"range [{d.min():+.3f}, {d.max():+.3f}]")
            print(f"  two-source vocals SDR median on the same tracks: "
                  f"{np.median([p[1] for p in pairs]):.3f}; four-target: {np.median([p[0] for p in pairs]):.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=manifest.DEFAULT_ROOT)
    ap.add_argument("--estimates-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shortest", type=int, default=0, help="score only the N shortest tracks (a memory probe)")
    ap.add_argument("--two-source", type=Path, default=None, help="the two-source score CSV to pair against")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)  # child mode
    a = ap.parse_args(argv)
    if a.one:
        return _child(a)
    if a.out is None:
        ap.error("--out is required")

    print("=== separation tier: FOUR-TARGET BSS Eval anchor (vocals/drums/bass/other) ===")
    import importlib.metadata as md
    print(f"    museval {md.version('museval')}, BSS Eval {sep.MODE}, win={sep.WIN}, hop={sep.HOP}; "
          f"one child process per track, shortest first\n")
    us = sorted(sep.tracks(a.root), key=_n_frames)
    if a.shortest:
        us = us[:a.shortest]
    done = _read_done(a.out) if a.resume else {}
    new = a.out if not a.out.exists() or not a.resume else None
    f = a.out.open("a" if a.resume else "w", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    if new is not None or f.tell() == 0:
        w.writeheader(); f.flush(); os.fsync(f.fileno())
    rows = list(done.values())
    for u in us:
        name = _track_dir(u).name
        if name in done:
            continue
        secs = _n_frames(u) / sep.RATE
        print(f"  {name} ({secs:.0f} s) ...", end=" ", flush=True)
        p = subprocess.run([sys.executable, __file__, "--one", name, "--root", str(a.root),
                            "--estimates-dir", str(a.estimates_dir)],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("ANCHOR4 ")), None)
        if line is None:
            row = {"track": name, "ok": False,
                   "error": f"child exited {p.returncode} with no result: {p.stderr.strip()[-300:]}"}
        else:
            row = json.loads(line[len("ANCHOR4 "):])
        w.writerow(row); f.flush(); os.fsync(f.fileno())
        rows.append(row)
        if row["ok"]:
            print(f"vocals {row['vocals_sdr']:.3f} dB, peak {row['maxrss_gb']} GB, {row['wall_s']} s")
        else:
            print(f"FAILED: {row['error']}")
    f.close()
    summarise(rows, a.two_source)
    print(f"\n  per-track results -> {a.out}")
    return 0 if all(str(r.get("ok")) == "True" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
