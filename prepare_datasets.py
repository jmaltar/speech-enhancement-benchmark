#!/usr/bin/env python3
"""Extract and index the noise-removal datasets into a single manifest.

Covers five manifest corpora (four datasets -- VoiceBank appears at two rates):

  voicebank16k   HuggingFace parquet mirror (train 11,572 + test 824), 16 kHz.
                 Audio inside the parquet is already RIFF WAV / PCM_16 / 16 kHz,
                 so extraction is a byte dump -- no decode, no resample.
  voicebank48k   Official Edinburgh DataShare test set, 48 kHz. Feeds the RNNoise
                 arm, which is hardwired to 48 kHz and must not be handed audio
                 upsampled from the 16 kHz mirror.
  noizeus        Hu & Loizou, 8 kHz. Indexed in place; nothing is copied.
  musdb18hq      MUSDB18-HQ, 44.1 kHz stereo, 100 train + 50 test tracks x 5
                 stems. The vocal-isolation tier: a separation corpus, not a
                 speech-enhancement one, so it carries no noise type and no SNR
                 and is never pooled with the four corpora above. Indexed in
                 place from wherever it was unzipped (see MUSDB_DIR).
  dns            DNS-Challenge (INTERSPEECH 2020) synthetic test set, 16 kHz,
                 150 clips x 10 s per condition, with ground-truth references.
                 Sparse-checked-out from the `interspeech2020/master` branch;
                 indexed in place. `no_reverb` is the generalisation check
                 (role=eval); `with_reverb` is downloaded but marked role=aux,
                 since dereverberation is out of scope and must not drift into
                 the primary matrix.

Noise type and SNR are not present in the parquet mirror -- they come from the
DataShare logfiles (`log_testset.txt`, `log_trainset_28spk.txt`), joined here by
utterance id.

The PTQ calibration subset is drawn from *train* only and flagged in the
manifest, so a full-integer INT8 calibration run cannot silently consume the
evaluation split.

Usage:
    python prepare_datasets.py                     # extract everything
    python prepare_datasets.py --manifest-only     # re-index without extracting
    python prepare_datasets.py --verify            # checks only, no writes
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import sys
import wave
import zipfile
from pathlib import Path

# --- one data root ----------------------------------------------------------
#
# Every corpus lives under a single root, whether it is indexed in place or
# extracted: `Upper-Case` names are the as-downloaded corpora, `lower_case`
# names are what this script derives from them. The root sits on the data
# volume rather than under ~/Desktop because MUSDB18-HQ alone is 31 GB
# extracted -- and because one root beats four scattered ones.
#
# Override with $BENCHMARK_DATA_ROOT or --data-root.
DEFAULT_DATA_ROOT = Path("/mnt/data/benchmark-data")

DATA_ROOT = Path(os.environ.get("BENCHMARK_DATA_ROOT") or DEFAULT_DATA_ROOT)

# --- corpus locations, all relative to the root ------------------------------
VB16_DIR = VB48_DIR = NOIZEUS_DIR = DNS_DIR = MUSDB_DIR = LOGFILES = Path()


def apply_data_root(root: Path) -> None:
    """Point every corpus path at `root`. Called once, before anything reads."""
    global DATA_ROOT, VB16_DIR, VB48_DIR, NOIZEUS_DIR, DNS_DIR, MUSDB_DIR, LOGFILES
    DATA_ROOT = root
    VB16_DIR = root / "VoiceBank-DEMAND-16k"
    VB48_DIR = root / "VoiceBank-DEMAND-48k"
    NOIZEUS_DIR = root / "noizeus_corpora" / "NOIZEUS"
    DNS_DIR = root / "DNS-Challenge-testset" / "datasets" / "test_set" / "synthetic"
    MUSDB_DIR = root / "musdb18hq"
    LOGFILES = VB16_DIR / "logfiles"


apply_data_root(DATA_ROOT)

# Whether a corpus carries a usable *categorical* noise factor.
#
# VoiceBank and NOIZEUS label every utterance with one of a small closed set of
# noise types, so (noise x SNR) is a real factorial cell structure. DNS does
# not: its 150 synthetic clips carry 150 distinct noise instances at SNRs spread
# continuously over 0-19 dB in 1 dB steps, so every (noise x SNR) cell holds
# exactly one clip. DNS therefore enters the study as an aggregate
# generalisation check stratified by SNR *bin*, never as a matrix cell -- and
# its noise variety is the point of it, not a defect.
NOISE_FACTOR = {
    "voicebank16k": True,
    "voicebank48k": True,
    "noizeus": True,
    "dns": False,
    "musdb18hq": False,
}

# Corpora that are source separation rather than speech enhancement. They have
# no noise type and no SNR, use SDR/SIR/SAR instead of PESQ/STOI, and run at
# their own sample rate -- so they take a separate branch everywhere and their
# numbers are never pooled with the 16 kHz speech results.
SEPARATION_CORPORA = {"musdb18hq"}

# MUSDB18-HQ ships exactly these per track. `mixture` is the input, `vocals` the
# target source; the other three sum to the accompaniment (the interference).
MUSDB_STEMS = ("mixture", "vocals", "drums", "bass", "other")
MUSDB_TRACKS = {"train": 100, "test": 50}

# SNR bin width in dB, for corpora whose SNRs are continuous rather than fixed.
SNR_BIN_WIDTH = {"dns": 5.0}

MANIFEST_FIELDS = [
    "corpus",
    "split",
    "role",
    "id",
    "speaker",
    "noise",
    "snr_db",
    "snr_bin",
    "rate_hz",
    "channels",
    "frames",
    "duration_s",
    "clean_path",
    "noisy_path",
    "calib",
]


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# logfile join: id -> (noise, snr_db)
# ---------------------------------------------------------------------------


def read_log(path: Path) -> dict[str, tuple[str, float]]:
    """Parse a DataShare logfile: `wavfile noise SNR(dB)` per line."""
    out: dict[str, tuple[str, float]] = {}
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                out[parts[0]] = (parts[1], float(parts[2]))
    return out


def load_logs() -> dict[str, dict[str, tuple[str, float]]]:
    test = LOGFILES / "log_testset.txt"
    train = LOGFILES / "log_trainset_28spk.txt"
    missing = [p for p in (test, train) if not p.exists()]
    if missing:
        sys.exit(
            "missing logfile(s): "
            + ", ".join(str(p) for p in missing)
            + "\nFetch logfiles.zip from Edinburgh DataShare (DOI 10.7488/ds/2117)."
        )
    return {"test": read_log(test), "train": read_log(train)}


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def snr_bin(corpus: str, snr_db: float) -> str:
    """Bin label for stratification. Discrete-SNR corpora bin to themselves."""
    width = SNR_BIN_WIDTH.get(corpus)
    if width is None or snr_db != snr_db:  # no binning, or NaN
        return f"{snr_db:g}"
    lo = int(snr_db // width * width)
    return f"{lo}-{lo + int(width) - 1} dB"


def wav_meta(raw: bytes) -> tuple[int, int, int]:
    """(rate, frames, channels) from WAV bytes, via stdlib -- no sample decode."""
    with wave.open(io.BytesIO(raw)) as w:
        return w.getframerate(), w.getnframes(), w.getnchannels()


def wav_meta_path(path: Path) -> tuple[int, int, int]:
    """(rate, frames, channels) from a WAV file, reading the header only.

    The speech corpora are small enough to slurp whole, but a MUSDB18-HQ stem
    runs to ~230 MB and there are 750 of them -- 31 GB of pointless reads if we
    went through `wav_meta`.
    """
    with wave.open(os.fspath(path)) as w:
        return w.getframerate(), w.getnframes(), w.getnchannels()


def extract_voicebank16k(out_root: Path, skip_existing: bool) -> dict[str, dict]:
    """Byte-dump the parquet audio to WAV. Returns id -> row fragments per split."""
    import pyarrow.parquet as pq

    found: dict[str, dict] = {"train": {}, "test": {}}
    data_dir = VB16_DIR / "data"
    shards = sorted(data_dir.glob("*.parquet"))
    if not shards:
        sys.exit(f"no parquet shards under {data_dir}")

    for shard in shards:
        split = "test" if shard.name.startswith("test") else "train"
        pf = pq.ParquetFile(shard)
        n = pf.metadata.num_rows
        log(f"  {shard.name}: {n} rows -> {split}")
        written = 0
        for batch in pf.iter_batches(batch_size=32):
            cols = batch.to_pydict()
            for i, uid in enumerate(cols["id"]):
                paths = {}
                rate = frames = channels = None
                for kind in ("clean", "noisy"):
                    raw = cols[kind][i]["bytes"]
                    dest = out_root / "voicebank16k" / split / kind / f"{uid}.wav"
                    if not (skip_existing and dest.exists()):
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(raw)
                        written += 1
                    if rate is None:
                        rate, frames, channels = wav_meta(raw)
                    paths[kind] = dest
                found[split][uid] = {
                    "rate_hz": rate,
                    "channels": channels,
                    "frames": frames,
                    "clean_path": paths["clean"],
                    "noisy_path": paths["noisy"],
                }
        log(f"    wrote {written} wav files")
    return found


def extract_voicebank48k(out_root: Path, skip_existing: bool) -> dict[str, dict]:
    """Unzip the DataShare 48 kHz test WAVs."""
    found: dict[str, dict] = {}
    zips = {
        "clean": VB48_DIR / "clean_testset_wav.zip",
        "noisy": VB48_DIR / "noisy_testset_wav.zip",
    }
    missing = [str(p) for p in zips.values() if not p.exists()]
    if missing:
        log("  skipping 48 kHz set, not present: " + ", ".join(missing))
        return found

    per_kind: dict[str, dict[str, dict]] = {}
    for kind, zpath in zips.items():
        per_kind[kind] = {}
        with zipfile.ZipFile(zpath) as z:
            names = [i.filename for i in z.infolist() if i.filename.lower().endswith(".wav")]
            log(f"  {zpath.name}: {len(names)} wav entries")
            written = 0
            for name in names:
                uid = Path(name).stem
                dest = out_root / "voicebank48k" / "test" / kind / f"{uid}.wav"
                if skip_existing and dest.exists():
                    raw = dest.read_bytes()
                else:
                    raw = z.read(name)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(raw)
                    written += 1
                rate, frames, channels = wav_meta(raw)
                per_kind[kind][uid] = {
                    "rate_hz": rate,
                    "channels": channels,
                    "frames": frames,
                    "path": dest,
                }
            log(f"    wrote {written} wav files")

    for uid, c in per_kind["clean"].items():
        n = per_kind["noisy"].get(uid)
        if n is None:
            continue
        found[uid] = {
            "rate_hz": c["rate_hz"],
            "channels": c["channels"],
            "frames": c["frames"],
            "clean_path": c["path"],
            "noisy_path": n["path"],
        }
    return found


def index_noizeus() -> list[dict]:
    """Index NOIZEUS in place. Conditions are encoded in the directory names."""
    rows: list[dict] = []
    if not NOIZEUS_DIR.exists():
        log(f"  skipping NOIZEUS, not present: {NOIZEUS_DIR}")
        return rows

    clean_dir = NOIZEUS_DIR / "clean_noizeus" / "wav"
    for cond in sorted(NOIZEUS_DIR.iterdir()):
        if not cond.is_dir() or cond.name == "clean_noizeus":
            continue
        noise, _, snr = cond.name.rpartition("_")
        if not snr.endswith("dB"):
            continue
        snr_db = float(snr[:-2])
        for wav in sorted((cond / "wav").glob("*.wav")):
            # sp01_babble_sn5.wav -> clean reference is sp01.wav
            sent = wav.stem.split("_")[0]
            clean = clean_dir / f"{sent}.wav"
            if not clean.exists():
                continue
            rate, frames, channels = wav_meta(wav.read_bytes())
            rows.append(
                {
                    "corpus": "noizeus",
                    "split": "test",
                    "role": "eval",
                    "id": wav.stem,
                    "speaker": sent,
                    "noise": noise,
                    "snr_db": snr_db,
                    "snr_bin": snr_bin("noizeus", snr_db),
                    "rate_hz": rate,
                    "channels": channels,
                    "frames": frames,
                    "duration_s": round(frames / rate, 4),
                    "clean_path": clean,
                    "noisy_path": wav,
                    "calib": 0,
                }
            )
    log(f"  indexed {len(rows)} NOIZEUS files")
    return rows


def index_dns() -> list[dict]:
    """Index the DNS synthetic test set in place.

    Condition metadata lives in the noisy filenames:
        clnsp102_traffic_248091_3_snr0_tl-21_fileid_268.wav
    SNR and the `fileid` join key parse reliably. The noise token does not --
    it is a readable label for some clips (`traffic`, `car`) and an opaque
    AudioSet/freesound id for others (`G3hs6PFniM0`), so DNS supports
    SNR-stratified generalisation but not a categorical noise-type factor.
    """
    rows: list[dict] = []
    if not DNS_DIR.exists():
        log(f"  skipping DNS, not present: {DNS_DIR}")
        return rows

    pointer_skips = 0
    for cond, role in (("no_reverb", "eval"), ("with_reverb", "aux")):
        noisy_dir, clean_dir = DNS_DIR / cond / "noisy", DNS_DIR / cond / "clean"
        if not noisy_dir.is_dir():
            continue
        n_cond = 0
        for wav in sorted(noisy_dir.glob("*.wav")):
            stem = wav.stem
            m_id = re.search(r"fileid_(\d+)$", stem)
            m_snr = re.search(r"_snr(-?\d+)_", stem)
            if not m_id:
                continue
            clean = clean_dir / f"clean_fileid_{m_id.group(1)}.wav"
            if not clean.exists():
                continue

            # a 131-byte git-lfs pointer is not audio -- report rather than index
            if wav.stat().st_size < 1024 or clean.stat().st_size < 1024:
                pointer_skips += 1
                continue

            # noise token = between the leading clnsp id and the _snr field
            m_noise = re.match(r"clnsp\d+_(.+?)_snr-?\d+_", stem)
            rate, frames, channels = wav_meta(wav.read_bytes())
            rows.append(
                {
                    "corpus": "dns",
                    "split": cond,
                    "role": role,
                    "id": f"fileid_{m_id.group(1)}",
                    "speaker": stem.split("_")[0],
                    "noise": m_noise.group(1) if m_noise else "UNKNOWN",
                    "snr_db": (v := float(m_snr.group(1)) if m_snr else float("nan")),
                    "snr_bin": snr_bin("dns", v),
                    "rate_hz": rate,
                    "channels": channels,
                    "frames": frames,
                    "duration_s": round(frames / rate, 4),
                    "clean_path": clean,
                    "noisy_path": wav,
                    "calib": 0,
                }
            )
            n_cond += 1
        log(f"  indexed {n_cond} DNS {cond} files (role={role})")

    if pointer_skips:
        log(f"  WARNING: skipped {pointer_skips} unfetched git-lfs pointers -- run `git lfs pull`")
    return rows


def index_musdb(musdb_dir: Path) -> list[dict]:
    """Index MUSDB18-HQ in place. Nothing is copied -- it is 31 GB extracted.

    Layout is `<split>/<Artist> - <Title>/<stem>.wav`, five stems per track.
    Mapped onto the shared manifest schema the way the vocal-isolation task
    actually uses it: `mixture` is the input (noisy_path) and `vocals` is the
    target source (clean_path), so the enhancement and separation arms can share
    one loader. The remaining three stems are not given columns of their own --
    the layout is fixed, so they are `Path(noisy_path).parent / f"{stem}.wav"`,
    and the accompaniment reference is their sum, formed at evaluation time.

    `noise` and `snr_db` are left empty rather than faked: there is no additive
    noise and no mixing SNR here, and writing one in would let these rows be
    pooled into speech cells by accident.
    """
    rows: list[dict] = []
    if not musdb_dir.exists():
        log(f"  skipping MUSDB18-HQ, not present: {musdb_dir}")
        return rows

    for split, role in (("train", "train"), ("test", "eval")):
        split_dir = musdb_dir / split
        if not split_dir.is_dir():
            log(f"  skipping MUSDB18-HQ/{split}, not present")
            continue
        n_split = 0
        for track in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            stems = {k: track / f"{k}.wav" for k in MUSDB_STEMS}
            missing = [k for k, v in stems.items() if not v.exists()]
            if missing:
                log(f"  WARNING: {split}/{track.name} missing stem(s): {', '.join(missing)}")
                continue
            rate, frames, channels = wav_meta_path(stems["mixture"])
            rows.append(
                {
                    "corpus": "musdb18hq",
                    "split": split,
                    "role": role,
                    "id": track.name,
                    # artist, so train/test overlap is checkable the same way
                    # speaker disjointness is on VoiceBank
                    "speaker": track.name.split(" - ")[0],
                    "noise": "",
                    "snr_db": "",
                    "snr_bin": "",
                    "rate_hz": rate,
                    "channels": channels,
                    "frames": frames,
                    "duration_s": round(frames / rate, 4),
                    "clean_path": stems["vocals"],
                    "noisy_path": stems["mixture"],
                    "calib": 0,
                }
            )
            n_split += 1
        log(f"  indexed {n_split} MUSDB18-HQ {split} tracks (role={role})")
    return rows


def musdb_sum_check(rows: list[dict], n_tracks: int, seed: int) -> list[str]:
    """Verify MUSDB18-HQ's own invariant: mixture == bass + drums + other + vocals.

    This is the one check that proves the *extraction* is sound rather than just
    complete -- a truncated or interleaved stem shows up here and nowhere else.
    Streamed in blocks; a full track is ~1.1 GB across five stems.
    """
    import numpy as np

    musdb = [r for r in rows if r["corpus"] == "musdb18hq"]
    if not musdb or n_tracks <= 0:
        return []
    picks = random.Random(seed).sample(musdb, min(n_tracks, len(musdb)))
    parts = [k for k in MUSDB_STEMS if k != "mixture"]
    out = []
    for r in sorted(picks, key=lambda r: r["id"]):
        track = Path(r["noisy_path"]).parent
        handles = [wave.open(os.fspath(track / f"{k}.wav")) for k in MUSDB_STEMS]
        try:
            widths = {h.getsampwidth() for h in handles}
            lengths = {h.getnframes() for h in handles}
            if widths != {2}:
                out.append(f"{r['split']}/{r['id']}: SKIPPED, sample width {widths} (expected 16-bit)")
                continue
            if len(lengths) != 1:
                out.append(f"{r['split']}/{r['id']}: STEM LENGTH MISMATCH {sorted(lengths)}")
                continue
            worst, total, n = 0, 0, 0
            while True:
                blocks = [h.readframes(1 << 20) for h in handles]
                if not blocks[0]:
                    break
                mix, *stem_blocks = [np.frombuffer(b, dtype="<i2").astype(np.int32) for b in blocks]
                delta = np.abs(mix - sum(stem_blocks))
                worst = max(worst, int(delta.max()))
                total += int(delta.sum())
                n += delta.size
            mean = total / n if n else 0.0
            out.append(f"{r['split']}/{r['id']}: max {worst} LSB, mean {mean:.4f} LSB over {n:,} samples")
        finally:
            for h in handles:
                h.close()
    return out


# ---------------------------------------------------------------------------
# calibration subset -- train only, stratified, deterministic
# ---------------------------------------------------------------------------


def pick_calibration(rows: list[dict], n_total: int, seed: int) -> set[str]:
    """Stratified draw over (noise x snr) from the voicebank16k *train* split."""
    pool: dict[tuple[str, float], list[str]] = {}
    for r in rows:
        if r["corpus"] == "voicebank16k" and r["split"] == "train":
            pool.setdefault((r["noise"], r["snr_db"]), []).append(r["id"])
    if not pool:
        return set()

    per_cell = max(1, n_total // len(pool))
    rng = random.Random(seed)
    chosen: set[str] = set()
    for cell in sorted(pool):
        ids = sorted(pool[cell])  # sort first so the draw is reproducible
        chosen.update(rng.sample(ids, min(per_cell, len(ids))))
    log(f"  calibration: {len(chosen)} utterances, {per_cell} per cell over {len(pool)} cells")
    return chosen


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def verify(
    rows: list[dict],
    logs: dict,
    vb16: dict,
    vb48: dict,
    musdb_sum_tracks: int = 0,
    seed: int = 0,
) -> int:
    problems = 0

    def check(ok: bool, msg: str) -> None:
        nonlocal problems
        log(("  ok   " if ok else "  FAIL ") + msg)
        if not ok:
            problems += 1

    def note(msg: str) -> None:
        """A property worth recording that is not a pass/fail condition."""
        log("  note " + msg)

    log("verification")

    # id sets against the logfiles
    for split, expect in (("test", logs["test"]), ("train", logs["train"])):
        got = set(vb16.get(split, {}))
        if got:
            check(
                got == set(expect),
                f"voicebank16k/{split}: {len(got)} ids vs {len(expect)} logfile entries",
            )

    if vb48:
        check(
            set(vb48) == set(logs["test"]),
            f"voicebank48k/test: {len(vb48)} ids vs {len(logs['test'])} logfile entries",
        )

        # the 16 kHz mirror should be a clean 3:1 decimation of the 48 kHz source
        shared = set(vb48) & set(vb16.get("test", {}))
        worst = 0.0
        for uid in shared:
            d48 = vb48[uid]["frames"] / vb48[uid]["rate_hz"]
            d16 = vb16["test"][uid]["frames"] / vb16["test"][uid]["rate_hz"]
            worst = max(worst, abs(d48 - d16))
        check(
            worst < 1e-3,
            f"16k/48k duration agreement over {len(shared)} ids: max delta {worst * 1e3:.3f} ms",
        )

    # rate homogeneity per corpus
    for corpus, expect_rate in (
        ("voicebank16k", 16000),
        ("voicebank48k", 48000),
        ("noizeus", 8000),
        ("dns", 16000),
        ("musdb18hq", 44100),
    ):
        rates = {r["rate_hz"] for r in rows if r["corpus"] == corpus}
        if rates:
            check(rates == {expect_rate}, f"{corpus}: all files at {expect_rate} Hz (saw {rates})")

    # channel count: the speech corpora are mono, MUSDB18-HQ is stereo
    for corpus, expect_ch in (
        ("voicebank16k", 1),
        ("voicebank48k", 1),
        ("noizeus", 1),
        ("dns", 1),
        ("musdb18hq", 2),
    ):
        chans = {r["channels"] for r in rows if r["corpus"] == corpus}
        if chans:
            check(chans == {expect_ch}, f"{corpus}: all files {expect_ch}-channel (saw {chans})")

    # MUSDB18-HQ: track counts, stem completeness, artist overlap
    musdb = [r for r in rows if r["corpus"] == "musdb18hq"]
    if musdb:
        for split, expect_n in MUSDB_TRACKS.items():
            got = [r for r in musdb if r["split"] == split]
            if got:
                check(len(got) == expect_n, f"musdb18hq/{split}: {len(got)} tracks (expect {expect_n})")

        incomplete = [
            r["id"]
            for r in musdb
            if not all((Path(r["noisy_path"]).parent / f"{k}.wav").exists() for k in MUSDB_STEMS)
        ]
        check(
            not incomplete,
            f"musdb18hq: all {len(musdb)} tracks carry {len(MUSDB_STEMS)} stems"
            + (f" -- missing in {incomplete[:3]}" if incomplete else ""),
        )

        # MUSDB18 is widely assumed to be artist-disjoint across the split. It is
        # not: three artists appear on both sides. It costs us nothing while the
        # separators stay pretrained (Arm A), but any MUSDB fine-tuning would be
        # training and testing on the same artists, so it is recorded here rather
        # than discovered later.
        tr_art = {r["speaker"] for r in musdb if r["split"] == "train"}
        te_art = {r["speaker"] for r in musdb if r["split"] == "test"}
        if tr_art and te_art:
            shared_art = sorted(tr_art & te_art)
            if shared_art:
                note(
                    f"musdb18hq: {len(shared_art)} artist(s) in BOTH splits -- "
                    + ", ".join(shared_art)
                    + " (not artist-disjoint; matters only if MUSDB is used for training)"
                )
            else:
                check(True, "musdb18hq: artists disjoint across train/test")

        if musdb_sum_tracks:
            log(f"  ...  musdb18hq stem-sum check on {musdb_sum_tracks} track(s), streaming")
            for line in musdb_sum_check(rows, musdb_sum_tracks, seed):
                note(f"musdb18hq mixture vs sum-of-stems -- {line}")

    # calibration must never touch the eval split
    calib = {r["id"] for r in rows if r["calib"]}
    eval_ids = {r["id"] for r in rows if r["role"] == "eval"}
    check(not (calib & eval_ids), f"calibration/eval disjoint ({len(calib)} calib utterances)")

    # speaker disjointness
    tr = {r["speaker"] for r in rows if r["corpus"] == "voicebank16k" and r["split"] == "train"}
    te = {r["speaker"] for r in rows if r["corpus"] == "voicebank16k" and r["split"] == "test"}
    if tr and te:
        check(not (tr & te), f"speakers disjoint ({len(tr)} train / {len(te)} test)")

    return problems


def summarise(rows: list[dict]) -> None:
    log("\nmanifest summary")
    by_corpus: dict[str, list[dict]] = {}
    for r in rows:
        by_corpus.setdefault(r["corpus"], []).append(r)

    for corpus in sorted(by_corpus):
        sub = by_corpus[corpus]
        hours = sum(r["duration_s"] for r in sub) / 3600
        unit = "tracks" if corpus in SEPARATION_CORPORA else "utterances"
        log(f"\n  {corpus}: {len(sub)} {unit}, {hours:.2f} h")
        for split in sorted({r["split"] for r in sub}):
            ss = [r for r in sub if r["split"] == split]

            if corpus in SEPARATION_CORPORA:
                # no noise type, no SNR, no cells -- tracks and stems instead
                artists = {r["speaker"] for r in ss}
                ch = sorted({r["channels"] for r in ss})
                rates = sorted({r["rate_hz"] for r in ss})
                hrs = sum(r["duration_s"] for r in ss) / 3600
                log(
                    f"    {split}: {len(ss)} tracks | {len(artists)} artists | "
                    f"{hrs:.2f} h | {rates} Hz | {ch} ch"
                )
                log(f"      {len(MUSDB_STEMS)} stems/track: {', '.join(MUSDB_STEMS)}")
                flag = "" if len(ss) >= 20 else "   <-- below N>=20"
                log(f"      separation tier: SDR/SIR/SAR, not pooled with the speech corpora{flag}")
                continue

            noises = sorted({r["noise"] for r in ss})
            snrs = sorted({r["snr_db"] for r in ss})
            log(f"    {split}: {len(ss)} utts | {len(noises)} noises | SNR {snrs}")

            if NOISE_FACTOR.get(corpus, True):
                if len(noises) <= 10:
                    log(f"      noises: {', '.join(noises)}")
                cells: dict[tuple, int] = {}
                for r in ss:
                    key = (r["noise"], r["snr_db"])
                    cells[key] = cells.get(key, 0) + 1
                lo, hi = min(cells.values()), max(cells.values())
                flag = "" if lo >= 20 else "   <-- below N>=20"
                log(f"      {len(cells)} (noise x SNR) cells, {lo}-{hi} utts per cell{flag}")
            else:
                # continuous SNR and one unique noise per clip: bin, do not cell
                log(f"      {len(noises)} unique noise instances -- no categorical noise factor")
                bins: dict[str, int] = {}
                for r in ss:
                    bins[r["snr_bin"]] = bins.get(r["snr_bin"], 0) + 1
                order = sorted(bins, key=lambda b: float(b.split("-")[0]))
                lo = min(bins.values())
                flag = "" if lo >= 20 else "   <-- below N>=20"
                log(f"      {len(bins)} SNR bins (stratify on these){flag}")
                for b in order:
                    log(f"        {b:10s} {bins[b]:3d} utts")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=None, help=f"root for every corpus (default {DATA_ROOT})")
    ap.add_argument("--out", type=Path, default=None, help="output root for extracted WAVs (default: --data-root)")
    ap.add_argument("--manifest", type=Path, default=None, help="manifest path (default: <out>/manifest.csv)")
    ap.add_argument("--calib-size", type=int, default=200, help="PTQ calibration utterances, drawn from train")
    ap.add_argument("--seed", type=int, default=20260908, help="seed for the calibration draw")
    ap.add_argument("--musdb-dir", type=Path, default=None, help="MUSDB18-HQ root (default: <data-root>/musdb18hq)")
    ap.add_argument(
        "--musdb-sum-check",
        type=int,
        default=3,
        metavar="N",
        help="verify mixture == sum of stems on N random tracks (0 to skip; ~1.1 GB read each)",
    )
    ap.add_argument("--manifest-only", action="store_true", help="re-index without extracting")
    ap.add_argument("--verify", action="store_true", help="verify only, write nothing")
    ap.add_argument("--no-skip-existing", action="store_true", help="re-write files that already exist")
    args = ap.parse_args()

    if args.data_root is not None:
        apply_data_root(args.data_root)
    log(f"data root: {DATA_ROOT}")
    if not DATA_ROOT.is_dir():
        sys.exit(f"data root does not exist: {DATA_ROOT}")

    out_root = args.out or DATA_ROOT
    manifest_path = args.manifest or (out_root / "manifest.csv")
    skip_existing = args.manifest_only or args.verify or not args.no_skip_existing

    logs = load_logs()
    log(f"logfiles: {len(logs['train'])} train / {len(logs['test'])} test entries\n")

    log("voicebank16k (parquet -> wav)")
    vb16 = extract_voicebank16k(out_root, skip_existing)

    log("\nvoicebank48k (zip -> wav)")
    vb48 = extract_voicebank48k(out_root, skip_existing)

    log("\nnoizeus (index in place)")
    rows = index_noizeus()

    log("\ndns (index in place)")
    rows += index_dns()

    log("\nmusdb18hq (index in place)")
    rows += index_musdb(args.musdb_dir or MUSDB_DIR)

    # voicebank rows, joined against the logfiles
    for split, entries in vb16.items():
        table = logs[split]
        for uid, meta in sorted(entries.items()):
            noise, snr = table.get(uid, ("UNKNOWN", float("nan")))
            rows.append(
                {
                    "corpus": "voicebank16k",
                    "split": split,
                    "role": "eval" if split == "test" else "train",
                    "id": uid,
                    "speaker": uid.split("_")[0],
                    "noise": noise,
                    "snr_db": snr,
                    "snr_bin": snr_bin("voicebank16k", snr),
                    "rate_hz": meta["rate_hz"],
                    "channels": meta["channels"],
                    "frames": meta["frames"],
                    "duration_s": round(meta["frames"] / meta["rate_hz"], 4),
                    "clean_path": meta["clean_path"],
                    "noisy_path": meta["noisy_path"],
                    "calib": 0,
                }
            )

    for uid, meta in sorted(vb48.items()):
        noise, snr = logs["test"].get(uid, ("UNKNOWN", float("nan")))
        rows.append(
            {
                "corpus": "voicebank48k",
                "split": "test",
                "role": "eval",
                "id": uid,
                "speaker": uid.split("_")[0],
                "noise": noise,
                "snr_db": snr,
                "snr_bin": snr_bin("voicebank48k", snr),
                "rate_hz": meta["rate_hz"],
                "channels": meta["channels"],
                "frames": meta["frames"],
                "duration_s": round(meta["frames"] / meta["rate_hz"], 4),
                "clean_path": meta["clean_path"],
                "noisy_path": meta["noisy_path"],
                "calib": 0,
            }
        )

    log("\ncalibration subset")
    calib_ids = pick_calibration(rows, args.calib_size, args.seed)
    for r in rows:
        if r["corpus"] == "voicebank16k" and r["split"] == "train" and r["id"] in calib_ids:
            r["calib"] = 1

    problems = verify(rows, logs, vb16, vb48, args.musdb_sum_check, args.seed)
    summarise(rows)

    if args.verify:
        log("\n--verify: nothing written")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for r in sorted(rows, key=lambda r: (r["corpus"], r["split"], r["id"])):
                w.writerow({k: (os.fspath(v) if isinstance(v, Path) else v) for k, v in r.items()})
        log(f"\nwrote {manifest_path} ({len(rows)} rows)")

    if problems:
        log(f"\n{problems} verification failure(s)")
        return 1
    log("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
