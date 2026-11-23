"""Score a directory of enhanced WAVs produced OFF the host -- a board batch kept
with `abf_host.py batch --keep` -- through the same metric stack as enhance.py,
and file the result under <root>/enhanced/<tag>/ so compare.py can pair it
against the host oracle's run of the same model.

    python -m eval_harness.score_dir --corpus voicebank16k --dir benchmark_h723/results/dtln_keep \
        --variant dtln-i16 --tag dtln-h723-p020-i16pt

Board files are named <id>.<variant>.wav by the batch driver; they are copied
to <root>/enhanced/<tag>/<corpus>/<id>.wav. Every utterance of the corpus
must be present -- a partial directory is refused rather than scored (trap 14).
This is task 15's metric-level equivalence check: the port's output is scored
exactly as the host's, and the paired TOST against the host's scores decides.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_harness import audio, manifest, metrics  # noqa: E402


def _job(args):
    corpus, rate, clean_p, out_p, uid, noise, snr_db, snr_bin, tag = args
    try:
        ref, sr_r = audio.load(clean_p)
        deg, sr_d = audio.load(out_p)
        if sr_r != sr_d or sr_r != rate:
            raise ValueError(f"rate mismatch {sr_r}/{sr_d}/{rate}")
        if len(ref) != len(deg):
            raise ValueError(f"length {len(deg)} != reference {len(ref)}")
        if rate == 48_000:
            # The RNNoise arm (voicebank48k): both sides through the locked
            # resampler, scored wideband -- the same branch enhance.py takes,
            # so a board's 48 kHz output is scored exactly as the oracle's
            # (decision 5, amended 2026-09-16; added here 2026-10-28).
            ref, deg, rate = audio.downsample_48_to_16(ref), audio.downsample_48_to_16(deg), 16_000
        m = metrics.all_metrics(ref, deg, rate, corpus)
        m.update(id=uid, corpus=corpus, algo=tag, noise=noise, snr_db=snr_db, snr_bin=snr_bin, ok=True)
        return m
    except Exception as e:
        return {"id": uid, "corpus": corpus, "algo": tag, "noise": noise, "snr_db": snr_db,
                "snr_bin": snr_bin, "ok": False, "error": f"{type(e).__name__}: {e}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dir", type=Path, required=True, help="batch --keep output directory")
    ap.add_argument("--variant", required=True, help="the variant in the kept file names, <id>.<variant>.wav")
    ap.add_argument("--tag", required=True, help="enhanced/<tag>/ to file the scored run under")
    ap.add_argument("--root", type=Path, default=manifest.DEFAULT_ROOT)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    a = ap.parse_args(argv)
    metrics.assert_provenance()
    us = manifest.utterances(manifest.load(a.root), a.corpus)
    out_dir = a.root / "enhanced" / a.tag / a.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [u.id for u in us if not (a.dir / f"{u.id}.{a.variant}.wav").exists()]
    if missing:
        sys.exit(f"{len(missing)} of {len(us)} utterances missing from {a.dir} (e.g. {missing[:3]}): refused")
    jobs = []
    for u in us:
        dst = out_dir / f"{u.id}.wav"
        shutil.copyfile(a.dir / f"{u.id}.{a.variant}.wav", dst)
        jobs.append((a.corpus, u.rate_hz, u.clean_path, dst, u.id, u.noise, u.snr_db, u.snr_bin, a.tag))
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    bad = df[~df.ok]
    print(f"=== {a.corpus} / {a.tag}: {len(df)} utterances in {time.time()-t0:.1f}s; failed {len(bad)}")
    if len(bad):
        for e, n in bad.error.value_counts().items():
            print(f"   {n:4d}  {e}")
    ok = df[df.ok]
    for m in ("pesq_mos_lqo", "stoi", "estoi", "si_sdr"):
        print(f"   {m:13s} mean {ok[m].mean():.4f}")
    df.to_csv(a.root / "enhanced" / a.tag / f"{a.corpus}.csv", index=False)
    print(f"wrote {a.root / 'enhanced' / a.tag / (a.corpus + '.csv')}")


if __name__ == "__main__":
    main()
