"""
audio.py -- loading, resampling and framing. One implementation, shared with
the training pipeline (task 13), so that what is trained and what is measured
cannot drift apart.

The resampler is the part worth reading. It is a LOCKED decision dated
2026-09-12 and it is not scipy's default, for a reason specific to this project:

  RNNoise is the only algorithm in the study whose output traverses a
  sample-rate conversion. It is a full-band 48 kHz enhancer; everything else is
  natively 16 kHz and never resamples. So a lossy downsample degrades exactly
  one row of the results table -- and it does so inside 7-8 kHz, precisely the
  band wideband PESQ weights.

Measured on 2026-09-12, 48 -> 16 kHz:

    filter                    droop @7.5k   reject @8.2k   alias rejection
    scipy default (61 tap)      -1.85 dB       -8.7 dB         32.0 dB
    this one (577 tap)          -0.00 dB      -119   dB        45.3 dB

CLAUDE.md already insists on the native 48 kHz test set so RNNoise is not
"handicapped in exactly the comparison the paper most needs to be fair". That
protected the input side. This protects the output side.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import firwin, resample_poly, get_window

# --- the locked anti-alias filter ------------------------------------------
RESAMPLE_NUMTAPS = 577
RESAMPLE_CUTOFF_HZ = 7800.0
RESAMPLE_BETA = 12.0
_SRC_HZ = 48_000

# 48000/16000 is exactly 1/3: integer decimation, no rational approximation.
_AA_FILTER = firwin(RESAMPLE_NUMTAPS, cutoff=RESAMPLE_CUTOFF_HZ / (_SRC_HZ / 2),
                    window=("kaiser", RESAMPLE_BETA))


def load(path: Path | str, expect_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Read a WAV as mono float64. Rate is returned, never guessed."""
    x, sr = sf.read(str(path), always_2d=False, dtype="float64")
    if x.ndim > 1:                      # MUSDB is the only stereo corpus
        x = x.mean(axis=1)
    if expect_rate is not None and sr != expect_rate:
        raise ValueError(f"{path}: expected {expect_rate} Hz, file is {sr} Hz")
    return x, sr


def load_multichannel(path: Path | str, expect_rate: int | None = None,
                      dtype: str = "float32") -> tuple[np.ndarray, int]:
    """Read a WAV as (nsampl, nchan) WITHOUT collapsing to mono.

    `load` above averages channels, which is right for every speech corpus here
    and wrong for exactly one thing: MUSDB is a stereo separation corpus and
    BSS Eval is defined on the multichannel image of each source. Averaging to
    mono would not merely lose the stereo field, it would change what SDR means.

    float32 rather than float64 by default: a 4-minute stereo track is 74 MB at
    float64 and museval holds several of them at once, so the evaluation is
    memory-bound before it is anything else. The stems are PCM_16 on disk, so
    float32 is lossless for them.

    Nothing here resamples. MUSDB stays at 44.1 kHz; the locked 48->16 kHz path
    is for RNNoise and must not touch this tier.
    """
    x, sr = sf.read(str(path), always_2d=True, dtype=dtype)
    if expect_rate is not None and sr != expect_rate:
        raise ValueError(f"{path}: expected {expect_rate} Hz, file is {sr} Hz")
    return x, sr


def downsample_48_to_16(x: np.ndarray) -> np.ndarray:
    """The locked RNNoise metric-path resampler. Do not substitute the default."""
    return resample_poly(x, 1, 3, window=_AA_FILTER)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    if (sr_in, sr_out) == (48_000, 16_000):
        return downsample_48_to_16(x)
    raise NotImplementedError(
        f"no locked resampler for {sr_in} -> {sr_out} Hz. The only conversion "
        f"this project performs is RNNoise's 48->16 kHz metric path; anything "
        f"else needs a decision recorded in CLAUDE.md first."
    )


def align(ref: np.ndarray, deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim to common length. Any trim beyond a few samples is a real problem."""
    n = min(len(ref), len(deg))
    if abs(len(ref) - len(deg)) > 16:
        raise ValueError(
            f"length mismatch of {abs(len(ref)-len(deg))} samples between "
            f"reference and degraded signal -- that is an alignment bug, not "
            f"rounding, and would corrupt every metric computed from it."
        )
    return ref[:n], deg[:n]


# --- framing, for task 13 and the algorithm ports ---------------------------
def frame_params(rate_hz: int) -> tuple[int, int, int]:
    """(frame, hop, nfft) for the locked 32 ms / 16 ms configuration.

    Given in TIME, so 16 kHz gives 512/256 and 8 kHz gives 256/128 with a
    256-point FFT -- the same window on a different rate, which is what makes
    the NOIZEUS arm validate the same algorithm rather than a different one.
    """
    frame = int(round(rate_hz * 0.032))
    hop = int(round(rate_hz * 0.016))
    if frame & (frame - 1):
        raise ValueError(f"{rate_hz} Hz gives a {frame}-sample frame, not a power of two")
    return frame, hop, frame


def hann(frame: int) -> np.ndarray:
    """Periodic Hann, which is what satisfies COLA at 50% overlap."""
    return get_window("hann", frame, fftbins=True)


def stft(x: np.ndarray, rate_hz: int) -> np.ndarray:
    frame, hop, nfft = frame_params(rate_hz)
    w = hann(frame)
    n = 1 + max(0, (len(x) - frame) // hop)
    out = np.empty((n, nfft // 2 + 1), dtype=np.complex128)
    for i in range(n):
        out[i] = np.fft.rfft(x[i * hop: i * hop + frame] * w, nfft)
    return out
