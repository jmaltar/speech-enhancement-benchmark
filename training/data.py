"""Training data for task 13: VoiceBank-DEMAND train, through the project's manifest.

Decisions, each stated so it can be attacked:

* CORPUS: VoiceBank-DEMAND TRAIN (11,572 utterances, 28 speakers, 9.39 h at
  16 kHz), the project's primary corpus -- not CHiME-2 (paid LDC licence; only
  TinyLSTMs' two hardware constants would transfer, and we re-measure those).
  Recorded as the working assumption on 2026-10-19 pending the user's call.
* HOLD-OUT: validation is SPEAKER-disjoint, two of the 28 train speakers drawn
  with a fixed seed. The test split (p232, p257) is never touched here, and
  the 200 calib=1 rows stay in training (they are train rows; PTQ calibration
  is not evaluation).
* SEGMENTS: fixed-length random crops (default 4 s; VoiceBank utterances
  average 2.9 s, so most are zero-padded at the end and the LSTM state resets
  per segment, as in the authors' generator). The crop offset is random per
  draw, never the head (trap 14).
* IN MEMORY: the split is ~1.1 GB as int16; it is loaded once (tqdm) and
  segments are cut from RAM, so the GPU never waits on the NTFS volume.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval_harness import manifest
from dtln.oracle import read_wav_int16

RATE = 16_000
VAL_SPEAKERS = 2
SPLIT_SEED = 20261019


@dataclass
class Split:
    ids: list[str]
    noisy: list[np.ndarray]     # int16
    clean: list[np.ndarray]     # int16
    speakers: list[str]

    def __len__(self):
        return len(self.ids)

    @property
    def seconds(self) -> float:
        return sum(len(x) for x in self.clean) / RATE


def load_train_val(root: Path | None = None, limit: int | None = None) -> tuple[Split, Split]:
    m = manifest.load(root or manifest.DEFAULT_ROOT)
    t = m[(m.corpus == "voicebank16k") & (m.split == "train")].reset_index(drop=True)
    assert len(t) == 11_572, len(t)
    speakers = sorted(t.speaker.unique())
    rng = np.random.default_rng(SPLIT_SEED)
    val_spk = set(rng.choice(speakers, size=VAL_SPEAKERS, replace=False).tolist())
    if limit:
        rng2 = np.random.default_rng(SPLIT_SEED + 1)        # random subset, never the head
        t = t.iloc[sorted(rng2.choice(len(t), size=min(limit, len(t)), replace=False))]
    parts = {"train": Split([], [], [], []), "val": Split([], [], [], [])}
    for r in tqdm(t.itertuples(), total=len(t), desc="loading VoiceBank train", unit="utt", leave=False):
        s = parts["val" if r.speaker in val_spk else "train"]
        rn, xn = read_wav_int16(r.noisy_path)
        rc, xc = read_wav_int16(r.clean_path)
        assert rn == rc == RATE and len(xn) == len(xc), r.id
        s.ids.append(r.id); s.noisy.append(xn); s.clean.append(xc); s.speakers.append(r.speaker)
    tr, va = parts["train"], parts["val"]
    assert not (set(tr.speakers) & set(va.speakers)), "validation must be speaker-disjoint"
    return tr, va


def segments(split: Split, n: int, seg_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n random (noisy, clean) segments of seg_len samples as float32 in [-1, 1):
    a random utterance, a random offset, zero-padded past the end."""
    X = np.zeros((n, seg_len), np.float32); Y = np.zeros((n, seg_len), np.float32)
    idx = rng.integers(0, len(split), size=n)
    for k, i in enumerate(idx):
        L = len(split.clean[i])
        off = int(rng.integers(0, max(1, L - seg_len + 1))) if L > seg_len else 0
        take = min(seg_len, L - off)
        X[k, :take] = split.noisy[i][off:off + take] / 32768.0
        Y[k, :take] = split.clean[i][off:off + take] / 32768.0
    return X, Y


def val_batches(split: Split, seg_len: int, batch: int, rng_seed: int = SPLIT_SEED + 2):
    """A FIXED set of validation segments (same crops every evaluation), one
    segment per utterance, so validation loss is comparable across epochs."""
    rng = np.random.default_rng(rng_seed)
    X = np.zeros((len(split), seg_len), np.float32); Y = np.zeros((len(split), seg_len), np.float32)
    for i in range(len(split)):
        L = len(split.clean[i])
        off = int(rng.integers(0, max(1, L - seg_len + 1))) if L > seg_len else 0
        take = min(seg_len, L - off)
        X[i, :take] = split.noisy[i][off:off + take] / 32768.0
        Y[i, :take] = split.clean[i][off:off + take] / 32768.0
    for s in range(0, len(split), batch):
        yield X[s:s + batch], Y[s:s + batch]
