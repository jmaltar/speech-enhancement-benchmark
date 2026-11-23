"""Validate the DTLN float oracle before any of its numbers are quoted.

    python -m dtln.check_oracle [--n 12] [--corpus voicebank16k] [--seed 20261019]

Four checks, each discriminating a different way the oracle could be wrong:

1. FRAMING ANCHOR -- `bypass` through the oracle's own hop loop must return the
   input bit-identically on every utterance drawn (random draw, never the head:
   trap 14). A one-sample error in the lag, the flush or the trim fails it.

2. RUNTIME EQUIVALENCE, per frame -- the SAME frame sequence through three
   implementations of the same weights: the LiteRT float32 pair on the XNNPACK
   delegate (what the oracle runs), the same pair on the builtin reference
   kernels, and the authors' stateful SavedModel (TF's own float32 rfft/irfft
   inside the graph, LSTM state carried internally). Reported in int16 LSB at
   the stage-2 output. Separates "the oracle's arithmetic" from "the weights".

3. PUBLISHED-PROCESSING EQUIVALENCE, whole file -- the authors' run_evaluation.py
   path (Keras model.h5, whole-utterance predict, 384-zero pad both ends, slice
   [384:384+len]) against the oracle's hop-by-hop ABF1 stream, on the same
   utterances. Max LSB, differing-sample count, NMSE, and the metric deltas.
   The two framings are the same policy in different words; this measures how
   close the realisations are, which is what task 15's tolerance later rests on.

4. DRQ vs F32 -- the authors' dynamic-range int8 export against the float pair,
   same frames, so the cost of the as-shipped quantization is seen at the
   sample level before it is seen in the metrics.

Needs tensorflow (checks 2-3) -- installed 2026-10-19 for this task. Check 3
imports the authors' DTLN_model.py from a clone of breizhn/DTLN whose path is
given with --dtln-src; wavinfo (a training-data helper it imports) is stubbed
rather than installed.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_harness import audio, manifest, metrics  # noqa: E402
from dtln import oracle  # noqa: E402

FRAME, HOP, LAG = oracle.FRAME, oracle.HOP, oracle.LAG


def frames_of(x_int16: np.ndarray):
    """The exact frame sequence enhance() feeds the model, including flush."""
    n = len(x_int16)
    dh = oracle.data_hops(n, HOP)
    xf = np.zeros(dh * HOP, dtype=np.float32)
    xf[:n] = x_int16.astype(np.float32) / oracle.SCALE
    buf = np.zeros(FRAME, dtype=np.float32)
    for j in range(dh + oracle.FLUSH_HOPS):
        hop = xf[j * HOP:(j + 1) * HOP] if j < dh else np.zeros(HOP, np.float32)
        buf[:-HOP] = buf[HOP:]
        buf[-HOP:] = hop
        yield buf.copy()


def lsb(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))) * oracle.SCALE)


def check_runtimes(x: np.ndarray, pretrained: Path) -> dict:
    import tensorflow as tf
    xnn = oracle.DtlnTflite("f32", xnnpack=True)
    ref = oracle.DtlnTflite("f32", xnnpack=False)
    drq = oracle.DtlnTflite("drq", xnnpack=True)
    sm = tf.saved_model.load(str(pretrained / "dtln_saved_model")).signatures["serving_default"]
    # The SavedModel is stateful across calls; it was loaded fresh so its state is zero.
    d_xnn_ref = d_xnn_sm = d_drq = 0.0
    nfr = 0
    for fr in frames_of(x):
        a = xnn.process_frame(fr)
        b = ref.process_frame(fr)
        c = drq.process_frame(fr)
        s = sm(input_1=tf.constant(fr.reshape(1, FRAME)))["conv1d_1"].numpy().reshape(-1)
        d_xnn_ref = max(d_xnn_ref, lsb(a, b))
        d_xnn_sm = max(d_xnn_sm, lsb(a, s))
        d_drq = max(d_drq, lsb(a, c))
        nfr += 1
    return {"frames": nfr, "xnnpack_vs_builtin_lsb": d_xnn_ref,
            "tflite_vs_savedmodel_lsb": d_xnn_sm, "drq_vs_f32_lsb": d_drq}


def load_keras(dtln_src: Path, h5: Path):
    sys.modules.setdefault("wavinfo", types.SimpleNamespace(WavInfoReader=None))
    sys.path.insert(0, str(dtln_src))
    from DTLN_model import DTLN_model  # the authors' definition, unmodified
    mc = DTLN_model()
    with contextlib.redirect_stdout(io.StringIO()):
        mc.build_DTLN_model(norm_stft=False)
    mc.model.load_weights(str(h5))
    assert mc.model.count_params() == 986_753, mc.model.count_params()
    return mc.model


def keras_published_path(model, x: np.ndarray) -> np.ndarray:
    """run_evaluation.py's process_file, on int16 input, to int16 output."""
    xf = x.astype(np.float32) / oracle.SCALE
    zero_pad = np.zeros(384, dtype=np.float32)
    inp = np.concatenate((zero_pad, xf, zero_pad))
    pred = np.squeeze(model.predict_on_batch(inp[None].astype(np.float32)))
    y = pred[384:384 + len(x)]
    return np.rint(np.clip(y.astype(np.float64) * oracle.SCALE, -32768, 32767)).astype(np.int16)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--corpus", default="voicebank16k")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--root", type=Path, default=manifest.DEFAULT_ROOT)
    ap.add_argument("--dtln-src", type=Path, default=None,
                    help="clone of github.com/breizhn/DTLN for check 3 (skipped if absent)")
    a = ap.parse_args(argv)

    m = manifest.load(a.root)
    us = manifest.utterances(m, a.corpus)
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(us), size=min(a.n, len(us)), replace=False))
    us = [us[i] for i in pick]
    lens = [oracle.read_wav_int16(u.noisy_path)[1].shape[0] for u in us]
    print(f"{a.corpus}: {len(us)} utterances, random draw seed {a.seed}; lengths {min(lens)}..{max(lens)}, "
          f"{sum(1 for L in lens if L % HOP == 0)} of them a whole number of hops")

    # 1. framing anchor
    byp = oracle.make_model("bypass")
    bad = [u.id for u in us if not np.array_equal(oracle.read_wav_int16(u.noisy_path)[1],
                                                  oracle.enhance(oracle.read_wav_int16(u.noisy_path)[1], byp))]
    print(f"\n[1] framing anchor (bypass bit-identical): {len(us) - len(bad)}/{len(us)} "
          f"{'OK' if not bad else 'FAILED ' + str(bad)}")
    if bad:
        raise SystemExit("fix the framing before anything else")

    # 2. runtimes, per frame, on the first drawn utterance
    x0 = oracle.read_wav_int16(us[0].noisy_path)[1]
    r = check_runtimes(x0, oracle.PRETRAINED)
    print(f"\n[2] runtime equivalence on {us[0].id} ({r['frames']} frames incl. flush), max |delta| at the "
          f"stage-2 output in int16 LSB:")
    print(f"    LiteRT XNNPACK vs builtin kernels : {r['xnnpack_vs_builtin_lsb']:.4f} LSB")
    print(f"    LiteRT (XNNPACK) vs SavedModel    : {r['tflite_vs_savedmodel_lsb']:.4f} LSB")
    print(f"[4] DRQ int8 export vs f32, same frames : {r['drq_vs_f32_lsb']:.2f} LSB")

    # 3. the authors' whole-file path vs the ABF1 stream
    if a.dtln_src is None or not (a.dtln_src / "DTLN_model.py").exists():
        print("\n[3] SKIPPED: --dtln-src not given / DTLN_model.py not found")
        return
    model = load_keras(a.dtln_src, oracle.PRETRAINED / "model.h5")
    f32 = oracle.make_model("f32")
    print(f"\n[3] authors' run_evaluation.py path (Keras model.h5, whole file) vs the ABF1 streaming oracle:")
    print(f"    {'id':<12}{'n':>7}{'maxLSB':>8}{'differ':>8}{'NMSE dB':>9}{'dPESQ':>8}{'dSTOI':>8}{'dSISDR':>8}")
    worst = 0.0
    dm = {"pesq_mos_lqo": [], "stoi": [], "si_sdr": []}
    for u in us:
        x = oracle.read_wav_int16(u.noisy_path)[1]
        ys = oracle.enhance(x, f32)
        yk = keras_published_path(model, x)
        d = ys.astype(np.float64) - yk.astype(np.float64)
        mx = float(np.abs(d).max()); nz = int(np.count_nonzero(d))
        nmse = 10 * np.log10((d @ d) / float(yk.astype(np.float64) @ yk.astype(np.float64))) if nz else float("-inf")
        ref, _ = audio.load(u.clean_path)
        ms = metrics.all_metrics(ref, ys / oracle.SCALE, u.rate_hz, a.corpus)
        mk = metrics.all_metrics(ref, yk / oracle.SCALE, u.rate_hz, a.corpus)
        for k in dm:
            dm[k].append(ms[k] - mk[k])
        worst = max(worst, mx)
        print(f"    {u.id:<12}{len(x):>7}{mx:>8.0f}{nz:>8}{nmse:>9.2f}"
              f"{ms['pesq_mos_lqo'] - mk['pesq_mos_lqo']:>+8.4f}{ms['stoi'] - mk['stoi']:>+8.4f}"
              f"{ms['si_sdr'] - mk['si_sdr']:>+8.3f}")
    print(f"    worst sample deviation {worst:.0f} LSB; mean metric delta (stream - keras): "
          + ", ".join(f"{k} {np.mean(v):+.4f}" for k, v in dm.items()))


if __name__ == "__main__":
    main()
