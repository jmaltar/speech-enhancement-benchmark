"""
metrics.py -- the quality metrics, with the locked decisions enforced in code.

Five decisions, dated 2026-09-12, fixed before any eval data existed (CLAUDE.md,
"Metric Stack Decisions"). What matters here is that three of them are enforced
rather than documented, because each has a silent failure mode:

  * MODE IS PER CORPUS, NOT A DEFAULT. Wideband at 16 kHz, narrowband at 8 kHz,
    never pooled. `pesq_score` refuses any other pairing instead of picking one.

  * python-pesq RETURNS MOS-LQO ONLY, in both modes -- cypesq.pyx ends
    `return err_info.mapped_mos` and the raw P.862 score is not exposed. For
    wideband that is not a choice (P.862.2 *defines* its output as MOS-LQO). For
    narrowband it is, and it matters: the P.862.1 mapping differs from raw by
    -0.21 to +0.28 over the raw 1.5-3.0 range NOIZEUS results occupy, which is
    2-3x the +/-0.10 PESQ equivalence margin. Comparing a mapped score against a
    published raw table would swamp the margin and nothing would warn you. So
    narrowband additionally reports raw MOS via the closed-form inverse.

  * THE ITU-vs-WRAPPER CHOICE DOES NOT EXIST. `pesq` 0.0.4 vendors the ITU
    reference C (P.862/.1/.2, Version 2.0 - October 2005) behind Cython.
    `assert_provenance()` checks those sources are actually present, so the
    paper can cite the recommendation rather than the wrapper.
"""
from __future__ import annotations

import numpy as np

import pesq as _pesq_mod
import pystoi

from . import manifest

# --- P.862.1 mapping, and its exact inverse --------------------------------
_A, _B, _C = 0.999, 4.999, (1.4945, 4.6607)


def mos_lqo_from_raw(raw: float) -> float:
    return _A + _B / (1.0 + np.exp(-_C[0] * raw + _C[1]))


def raw_from_mos_lqo(m: float) -> float:
    """Recover raw P.862 MOS from the MOS-LQO python-pesq returns.

    Exact inverse of the P.862.1 mapping; verified to round-trip to 6 decimal
    places over raw 1.0-4.0. Defined on the open interval (0.999, 5.998).
    """
    if not (_A < m < _A + _B):
        return float("nan")
    return (_C[1] - np.log(_B / (m - _A) - 1.0)) / _C[0]


def assert_provenance() -> dict:
    """Fail loudly if the metric stack is not the one the decisions assume."""
    from pathlib import Path

    pkg = Path(_pesq_mod.__file__).parent
    itu = ["dsp.c", "pesqdsp.c", "pesqmod.c", "pesqpar.h", "pesqmain.h"]
    absent = [f for f in itu if not (pkg / f).exists()]
    if absent:
        raise AssertionError(
            f"the installed `pesq` package does not vendor the ITU reference "
            f"sources (missing {absent}). The paper's provenance claim -- ITU-T "
            f"P.862 reference implementation v2.0 accessed through python-pesq -- "
            f"would be false."
        )
    if pystoi.__version__ != "0.4.1":
        raise AssertionError(
            f"pystoi is {pystoi.__version__}, not the pinned 0.4.1; its internal "
            f"10 kHz resampling and band edges are part of the recorded decision."
        )
    from .audio import RESAMPLE_NUMTAPS, RESAMPLE_CUTOFF_HZ, _AA_FILTER
    if len(_AA_FILTER) != RESAMPLE_NUMTAPS:
        raise AssertionError("the locked anti-alias filter has been altered")
    return {"pesq_itu_sources": "P.862/.1/.2 v2.0 (Oct 2005), vendored",
            "pystoi": pystoi.__version__,
            "resampler": f"resample_poly(1,3), {RESAMPLE_NUMTAPS}-tap "
                         f"Kaiser(12) @ {RESAMPLE_CUTOFF_HZ:g} Hz"}


# --- the metrics -----------------------------------------------------------
def pesq_score(ref: np.ndarray, deg: np.ndarray, rate_hz: int, corpus: str) -> dict:
    """PESQ with the mode fixed by the corpus, not by the caller.

    Returns MOS-LQO always, plus raw MOS for narrowband, where the distinction
    is real and larger than the equivalence margin.
    """
    mode = manifest.pesq_mode_for(corpus)
    if mode == "wb" and rate_hz != 16_000:
        raise ValueError(f"{corpus}: wideband PESQ requires 16 kHz, got {rate_hz}")
    if mode == "nb" and rate_hz != 8_000:
        raise ValueError(f"{corpus}: narrowband PESQ requires 8 kHz, got {rate_hz}")

    m = float(_pesq_mod.pesq(rate_hz, ref.astype(np.float32),
                             deg.astype(np.float32), mode))
    out = {"pesq_mos_lqo": m, "pesq_mode": mode}
    if mode == "nb":
        out["pesq_raw_mos"] = float(raw_from_mos_lqo(m))
    return out


def stoi_scores(ref: np.ndarray, deg: np.ndarray, rate_hz: int) -> dict:
    """STOI primary, ESTOI alongside -- the extra call is one line of cost."""
    return {"stoi":  float(pystoi.stoi(ref, deg, rate_hz, extended=False)),
            "estoi": float(pystoi.stoi(ref, deg, rate_hz, extended=True))}


def si_sdr(ref: np.ndarray, deg: np.ndarray) -> float:
    """Scale-invariant SDR in dB. The numerator of the composite efficiency
    metric (PDF section 3.7) and what the paired t-test applies to."""
    ref = ref - ref.mean()
    deg = deg - deg.mean()
    denom = float(ref @ ref)
    if denom <= 0.0:
        return float("nan")
    target = (float(deg @ ref) / denom) * ref
    noise = deg - target
    p_t, p_n = float(target @ target), float(noise @ noise)
    if p_n <= 0.0 or p_t <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(p_t / p_n))


def all_metrics(ref: np.ndarray, deg: np.ndarray, rate_hz: int, corpus: str) -> dict:
    out = {}
    out.update(pesq_score(ref, deg, rate_hz, corpus))
    out.update(stoi_scores(ref, deg, rate_hz))
    out["si_sdr"] = si_sdr(ref, deg)
    return out
