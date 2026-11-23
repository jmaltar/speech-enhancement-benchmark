"""Validate the RNNoise host oracle before any of its numbers are quoted.

    python -m rnnoise.check_rnnoise [--n 12] [--seed 20261019] [--all-bypass]

Four checks, each discriminating a different way the oracle could be wrong:

1. FRAMING ANCHOR -- `bypass` through rnnoise_wav at RNNoise's 960/480 geometry
   must return the 48 kHz input bit-identically (random draw, never the head:
   trap 14; --all-bypass runs all 824).

2. ALIGNMENT -- RNNoise applied to CLEAN speech should pass it nearly unchanged,
   so the cross-correlation of output against input over +/-2000 samples must
   peak at lag 0 after the host's drop-and-trim. A peak at +/-480 means the
   declared lag (frame - hop) does not match the library's real delay.

3. DETERMINISM -- two runs of the same file are bit-identical (no uninitialised
   state, no threading).

4. RELEASE DELTA -- v0.1.1 (as shipped) against v0.1 (paper-era tree), same
   weights, on the same noisy files: max |delta| in LSB and NMSE. The difference
   is the 2017-10-10 gain-decay limiter plus maintenance fixes; it is measured
   here at the sample level and in enhance.py at the metric level.

All in the 48 kHz domain -- the metric path's downsample is not involved.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_harness import manifest  # noqa: E402
from dtln.oracle import read_wav_int16  # noqa: E402  (plain int16 WAV I/O)

HERE = Path(__file__).resolve().parent
BIN = HERE / "build" / "rnnoise_wav"
BIN_V01 = HERE / "build" / "rnnoise_wav_v01"


def run(binp: Path, inp, out, proc="rnnoise"):
    r = subprocess.run([str(binp), "--in", str(inp), "--out", str(out), "--proc", proc],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    return read_wav_int16(out)[1]


def best_lag(x: np.ndarray, y: np.ndarray, scan: int = 2000) -> tuple[int, float]:
    """argmax over integer lags of corr(x, y shifted); positive = y lags x."""
    x = x.astype(np.float64); y = y.astype(np.float64)
    x -= x.mean(); y -= y.mean()
    best, bl = -np.inf, 0
    for L in range(-scan, scan + 1):
        if L >= 0:
            c = float(x[:len(x) - L] @ y[L:])
        else:
            c = float(x[-L:] @ y[:len(y) + L])
        if c > best:
            best, bl = c, L
    return bl, best / np.sqrt(float(x @ x) * float(y @ y))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--all-bypass", action="store_true")
    ap.add_argument("--root", type=Path, default=manifest.DEFAULT_ROOT)
    a = ap.parse_args(argv)
    for b in (BIN, BIN_V01):
        if not b.exists():
            raise SystemExit(f"{b} not built: cmake -S rnnoise -B rnnoise/build && cmake --build rnnoise/build")

    us_all = manifest.utterances(manifest.load(a.root), "voicebank48k")
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(us_all), size=min(a.n, len(us_all)), replace=False))
    us = [us_all[i] for i in pick]
    print(f"voicebank48k: {len(us)} utterances drawn (seed {a.seed}) of {len(us_all)}; "
          f"{sum(1 for u in us_all if read_wav_int16(u.noisy_path)[1].shape[0] % 480 == 0)} of all 824 are a whole number of 480-hops")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1. framing anchor
        pool = us_all if a.all_bypass else us
        bad = []
        for u in pool:
            x = read_wav_int16(u.noisy_path)[1]
            y = run(BIN, u.noisy_path, td / "b.wav", "bypass")
            if not np.array_equal(x, y):
                bad.append(u.id)
        print(f"\n[1] framing anchor at 960/480 (bypass bit-identical, 48 kHz): {len(pool) - len(bad)}/{len(pool)} "
              f"{'OK' if not bad else 'FAILED ' + str(bad[:5])}")
        if bad:
            raise SystemExit("fix the framing before anything else")

        # 2. alignment on clean speech
        print(f"\n[2] alignment: RNNoise(clean) vs clean, best integer lag over +/-2000 samples")
        print(f"    {'id':<12}{'lag':>6}{'corr':>8}{'rms out/in':>12}")
        lags = []
        for u in us:
            c = read_wav_int16(u.clean_path)[1]
            y = run(BIN, u.clean_path, td / "c.wav")
            L, corr = best_lag(c, y)
            lags.append(L)
            print(f"    {u.id:<12}{L:>6}{corr:>8.4f}{np.sqrt((y.astype(float)**2).mean() / (c.astype(float)**2).mean()):>12.4f}")
        # RNNoise is not an identity (high-pass biquad, pitch filter, band gains), so the
        # correlation peak may sit a sample off; a framing error would put it at +/-480.
        print(f"    max |lag| {max(abs(L) for L in lags)} samples (a lag error would read +/-480): "
              f"{'OK' if all(abs(L) <= 2 for L in lags) else 'FAILED'}")

        # 3. determinism
        u = us[0]
        y1 = run(BIN, u.noisy_path, td / "d1.wav"); y2 = run(BIN, u.noisy_path, td / "d2.wav")
        print(f"\n[3] determinism on {u.id}: {'bit-identical' if np.array_equal(y1, y2) else 'DIFFERS'}")

        # 4. release delta
        print(f"\n[4] v0.1.1 (as shipped) vs v0.1 (paper-era), same weights, noisy input:")
        print(f"    {'id':<12}{'maxLSB':>8}{'differ %':>10}{'NMSE dB':>9}")
        for u in us[:6]:
            y = run(BIN, u.noisy_path, td / "e1.wav"); z = run(BIN_V01, u.noisy_path, td / "e2.wav")
            d = y.astype(np.float64) - z.astype(np.float64)
            nz = np.count_nonzero(d)
            nmse = 10 * np.log10((d @ d) / float(z.astype(np.float64) @ z.astype(np.float64))) if nz else float("-inf")
            print(f"    {u.id:<12}{np.abs(d).max():>8.0f}{100 * nz / len(d):>10.1f}{nmse:>9.2f}")


if __name__ == "__main__":
    main()
