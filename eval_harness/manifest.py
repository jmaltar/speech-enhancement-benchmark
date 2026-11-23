"""
manifest.py -- the one way this project reads its corpora.

Everything comes from <data-root>/manifest.csv, produced by prepare_datasets.py:
14,630 rows over five corpus entries (VoiceBank appears at both 16 and 48 kHz).

Three rules are enforced here rather than trusted to the caller, because each
one has already been recorded in CLAUDE.md as a way this project could silently
produce wrong numbers:

  1. READ IT WITH A CSV PARSER, never `cut -d,`. One MUSDB track id contains a
     comma.
  2. CALIBRATION ROWS ARE NOT EVAL ROWS. The PTQ calibration subset is drawn
     from train and flagged calib=1; using it for evaluation leaks the eval
     split into the quantization scales and silently flatters every INT8 number.
     Asserted, not assumed.
  3. CORPORA AT DIFFERENT SAMPLE RATES ARE NEVER POOLED. NOIZEUS is 8 kHz with
     narrowband PESQ, the 16 kHz corpora are wideband, MUSDB is a separation
     tier at 44.1 kHz with its own metrics. Pooling them would average numbers
     that are not on the same scale.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path(os.environ.get("BENCHMARK_DATA_ROOT", "/mnt/data/benchmark-data"))

# Locked audio configuration (CLAUDE.md, 2026-09-08). Frame/hop are given in
# TIME so the 8 kHz NOIZEUS arm keeps the same 32/16 ms window with a 256-point
# FFT -- what is validated there must be the same algorithm, not a different one.
FRAME_MS = 32.0
HOP_MS = 16.0

# Which PESQ mode each corpus is scored in. Not a default with exceptions: an
# explicit table, so an unlisted corpus raises instead of silently picking one.
PESQ_MODE = {
    "voicebank16k": "wb",
    "voicebank48k": "wb",   # scored after the locked 48->16 kHz downsample
    "dns": "wb",
    "noizeus": "nb",        # Hu & Loizou's published figures are narrowband
}

# Corpora that take the speech metric stack at all. MUSDB is a separation tier
# with SDR/SIR/SAR via museval and its numbers are never pooled with these.
SPEECH_CORPORA = tuple(PESQ_MODE)

# How each corpus is STRATIFIED. Explicit per corpus for the same reason the
# PESQ mode is: inferring it from the data gets DNS wrong, and gets it wrong
# silently.
#
# DNS is the one that bites. Its `noise` column IS populated -- with 150
# distinct noise instances over 150 clips -- so any rule of the form "cell by
# noise if a noise label exists" yields 150 cells of N = 1. There is no
# factorial structure to recover: SNR is continuous over 0-19 dB in 1 dB steps
# and every (noise x SNR) cell holds exactly one clip. Its noise variety is the
# POINT of that corpus, not a factor in it, so DNS enters as an aggregate
# generalisation check stratified by 5 dB SNR bin (36/44/40/30 clips, all >= 20).
#
# This was a live defect until 2026-09-15: stratify()'s docstring stated the
# rule and its code keyed on `noise.notna().any()`, which is True for DNS. The
# rule was documented and not enforced -- and a documented rule that the code
# contradicts is the failure class this project has already recorded twice
# (a borrowed pointer whose lifetime the header describes; a build switch whose
# help text advertises a speedup it does not deliver).
STRATIFY_BY = {
    "voicebank16k": ("noise", "snr_db"),   # 20 cells, N = 40-42
    "voicebank48k": ("noise", "snr_db"),   # same utterances, RNNoise arm only
    "noizeus":      ("noise", "snr_db"),   # 32 cells, N = 30
    "dns":          ("snr_bin",),          # 4 bins -- NEVER noise. See above.
    "musdb18hq":    (),                    # no factor structure; 50 tracks
}

# Columns no corpus may ever be stratified by, whatever a caller asks for.
FORBIDDEN_STRATIFY = {"dns": ("noise",)}


@dataclass(frozen=True)
class Utterance:
    corpus: str
    split: str
    role: str
    id: str
    clean_path: Path
    noisy_path: Path
    rate_hz: int
    noise: str | None
    snr_db: float | None
    snr_bin: str | None


def load(root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """Read the manifest with a real CSV parser and verify its invariants."""
    root = Path(root)
    path = root / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_datasets.py, or set "
            f"BENCHMARK_DATA_ROOT / pass root=."
        )
    m = pd.read_csv(path)

    required = {"corpus", "split", "role", "id", "rate_hz", "clean_path",
                "noisy_path", "calib"}
    missing = required - set(m.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    # Rule 2, asserted every load rather than trusted.
    calib = m[m.calib == 1]
    if len(calib) and (calib.role == "eval").any():
        raise AssertionError(
            "calibration rows are flagged role=eval: the PTQ calibration subset "
            "has leaked into the evaluation split, which would silently flatter "
            "every INT8 number reported."
        )
    return m


def utterances(m: pd.DataFrame, corpus: str, *, role: str = "eval",
               limit: int | None = None) -> list[Utterance]:
    """Eval-role rows of one corpus, as typed records.

    One corpus at a time, by design: there is no API here for fetching two
    corpora at once, because the next thing a caller would do is pool them.
    """
    if corpus not in SPEECH_CORPORA and corpus != "musdb18hq":
        raise ValueError(
            f"unknown corpus {corpus!r}; known: {sorted(SPEECH_CORPORA) + ['musdb18hq']}"
        )
    sel = m[(m.corpus == corpus) & (m.role == role)]
    if sel.empty:
        raise ValueError(f"no rows for corpus={corpus!r} role={role!r}")

    rates = sel.rate_hz.unique()
    if len(rates) != 1:
        raise AssertionError(
            f"{corpus} contains mixed sample rates {sorted(rates)}; the metric "
            f"mode is chosen per corpus and cannot be well defined here."
        )
    if limit:
        sel = sel.head(limit)

    def _opt(v):
        return None if pd.isna(v) else v

    return [
        Utterance(
            corpus=r.corpus, split=r.split, role=r.role, id=str(r.id),
            clean_path=Path(r.clean_path), noisy_path=Path(r.noisy_path),
            rate_hz=int(r.rate_hz),
            noise=_opt(getattr(r, "noise", None)),
            snr_db=_opt(getattr(r, "snr_db", None)),
            snr_bin=_opt(getattr(r, "snr_bin", None)),
        )
        for r in sel.itertuples(index=False)
    ]


def stratify_key_for(corpus: str) -> tuple[str, ...]:
    """The stratification columns for one corpus, as a locked decision.

    Returns () for a corpus with no factor structure, which callers must treat
    as "report the aggregate", not as "no rule recorded".
    """
    try:
        return STRATIFY_BY[corpus]
    except KeyError:
        raise ValueError(
            f"no stratification is defined for corpus {corpus!r}. This is "
            f"deliberate: how a corpus is celled or binned is a locked decision "
            f"per corpus, not something to infer from which columns happen to be "
            f"populated -- inferring it puts DNS into 150 cells of N = 1."
        ) from None


def assert_stratifiable(corpus: str, key) -> None:
    """Refuse a stratification the corpus cannot support.

    Exists so that asking for the wrong thing fails loudly instead of returning
    150 single-clip cells that look like a result.
    """
    bad = set(FORBIDDEN_STRATIFY.get(corpus, ())) & set(key)
    if bad:
        raise ValueError(
            f"{corpus} must never be stratified by {sorted(bad)}: it has 150 "
            f"unique noise instances over 150 clips, so every cell would hold "
            f"exactly one. Use {list(stratify_key_for(corpus))} -- DNS is an "
            f"aggregate generalisation check binned by SNR, not a factorial arm."
        )


def pesq_mode_for(corpus: str) -> str:
    try:
        return PESQ_MODE[corpus]
    except KeyError:
        raise ValueError(
            f"no PESQ mode is defined for corpus {corpus!r}. This is deliberate: "
            f"the mode is a locked per-corpus decision (wideband at 16 kHz, "
            f"narrowband at 8 kHz), not something to default."
        ) from None
