"""Prove the Keras-3 training model IS the authors' DTLN before any training.

    source gpu_env.sh && python -m training.check_model [--n 6]

Three checks on random VoiceBank test utterances (the oracle's domain):

1. Our model with the authors' weights vs the authors' own Keras model, same
   input, whole-utterance predict: max |delta| in int16 LSB. Different LSTM
   code, same arithmetic -- expected float32 round-off (<= 1 LSB).
2. The same vs the float oracle's streaming output (dtln/oracle.py f32, ABF1
   framing), through run_evaluation.py's pad/slice. Check 3 in
   dtln/check_oracle.py already showed the two framings agree to the fourth
   metric decimal; this confirms the training model sits on that chain.
3. Round trip: to_authors_weights -> authors' model -> save_weights -> reload
   into a fresh training model -> identical output. This is the export path a
   trained model takes into dtln/quantize.py, so it must be exact.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_harness import manifest  # noqa: E402
from dtln import oracle  # noqa: E402
from training import model as M  # noqa: E402

PAD = M.FRAME - M.HOP


def whole_file(model, x_int16: np.ndarray) -> np.ndarray:
    """run_evaluation.py's process_file: pad 384 both ends, predict, slice."""
    xf = x_int16.astype(np.float32) / 32768.0
    inp = np.concatenate((np.zeros(PAD, np.float32), xf, np.zeros(PAD, np.float32)))
    y = np.squeeze(model.predict_on_batch(inp[None]))[PAD:PAD + len(x_int16)]
    return np.rint(np.clip(y.astype(np.float64) * 32768.0, -32768, 32767)).astype(np.int16)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261019)
    a = ap.parse_args(argv)
    import tensorflow as tf
    # Ampere+ GPUs run float32 matmuls in TF32 (10-bit mantissa) by default --
    # fine for training, not for an equivalence check. Off here, so a
    # disagreement is a model difference and not a precision mode.
    tf.config.experimental.enable_tensor_float_32_execution(False)
    print("GPU:", tf.config.list_physical_devices("GPU") or "NONE (source gpu_env.sh)", "| TF32 off")

    h5 = oracle.PRETRAINED / "model.h5"
    ours = M.load_pretrained(h5, dropout=0.0)
    theirs = M.authors_model(h5)
    print(f"params: ours {ours.count_params():,} (incl. masks) / theirs {theirs.count_params():,}")
    us = manifest.utterances(manifest.load(), "voicebank16k")
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(us), size=a.n, replace=False))
    f32 = oracle.make_model("f32")
    print(f"\n{'id':<12}{'ours-vs-authors LSB':>22}{'ours-vs-oracle LSB':>20}{'differ':>8}")
    worst_a = worst_o = 0
    for i in pick:
        u = us[i]; x = oracle.read_wav_int16(u.noisy_path)[1]
        yo = whole_file(ours, x); ya = whole_file(theirs, x); ys = oracle.enhance(x, f32)
        da = np.abs(yo.astype(int) - ya.astype(int)).max(); do = np.abs(yo.astype(int) - ys.astype(int))
        worst_a = max(worst_a, da); worst_o = max(worst_o, int(do.max()))
        print(f"{u.id:<12}{da:>22}{do.max():>20}{np.count_nonzero(do):>8}")
    print(f"worst: vs authors {worst_a} LSB, vs oracle {worst_o} LSB")

    # 3. export round trip
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rt.weights.h5"
        M.save_authors_h5(ours, p)
        again = M.load_pretrained(p, dropout=0.0)
        x = oracle.read_wav_int16(us[pick[0]].noisy_path)[1]
        d = np.abs(whole_file(again, x).astype(int) - whole_file(ours, x).astype(int)).max()
        print(f"export round trip (ours -> authors' h5 -> ours): max {d} LSB  {'OK' if d == 0 else 'FAILED'}")
        if worst_a > 1 or d != 0:
            raise SystemExit("the training model is not the authors' network; fix before training")


if __name__ == "__main__":
    main()
