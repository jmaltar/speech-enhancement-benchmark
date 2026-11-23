"""Full-integer re-quantization of DTLN -- the Arm A path TFLite Micro will accept.

    python -m dtln.quantize --dtln-src <clone of breizhn/DTLN> [--every 16]
                            [--weights model.h5] [--states split|packed] [--tag int8]

Why this exists: the authors' model_quant_*.tflite are dynamic-range (hybrid) --
int8 weights, float32 activations -- and TFLite Micro refuses that pairing
(tflm_check.py runs the refusal: "Input type: FLOAT32 with filter type: INT8
not supported"), so DTLN has NO as-shipped path onto any MCU tier. A
full-integer model needs a representative dataset; this project's is the
200-utterance calibration subset drawn from the TRAIN split and flagged
calib=1 (never the eval split: calibrating on it would leak it into the
quantization scales and flatter every INT8 number).

What it does, each step checked rather than assumed:

1. Rebuilds the authors' two-part stateful model (model_1: magnitude -> mask,
   model_2: time frame -> frame, each with external LSTM state) from their
   weights file, with the split written here in Keras-3-compatible form.
   `--states packed` reproduces the authors' layout, ONE (1, 2, 128, 2) tensor
   per stage holding h and c of both layers; `--states split` (default) exposes
   four (1, 128) tensors. The weights are identical either way.
2. Exports float32 and CHECKS IT against the shipped model_1/model_2.tflite
   frame by frame on a real utterance (for the paper weights; the norm weights
   have no shipped TFLite pair and are checked for finiteness only).
3. Collects the representative dataset by running the float export over the
   calibration utterances through the oracle's own frame loop, recording the
   inputs of both stages at every `--every`-th hop -- real LSTM states, not
   zeros.
4. Converts with full-integer quantization (TFLITE_BUILTINS_INT8, int8 I/O).
5. Prints sizes, op sets, per-tensor quantization parameters, and the byte
   breakdown of populated buffers by dtype -- so a converter silently leaving
   an operator in float (which the packed layout provoked) is visible.

The packed layout is kept as a recorded failure (`int8p`): PACK forces one
scale on h and c together, c spans +/-150 and h +/-1, so h gets a step four
times its range and the recurrence is destroyed. See RESULTS.md.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import os
import sys
import types
from pathlib import Path

import numpy as np
import pathlib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_harness import manifest  # noqa: E402
from dtln import oracle  # noqa: E402

FRAME, HOP = oracle.FRAME, oracle.HOP
NUM_LAYER, NUM_UNITS, ENC = 2, 128, 256
BINS = FRAME // 2 + 1


# ----------------------------------------------------------------------------
# 1. the two-part model, Keras-3 compatible, weights from the authors' h5
# ----------------------------------------------------------------------------
import keras as _keras  # noqa: E402  (module-level so the converter's export path can trace the layer)


class ILN_E2(_keras.layers.Layer):
    """Instant layer normalisation with the variance formed as E[x^2] - E[x]^2.

    Algebraically the authors' InstantLayerNormalization (mean/variance over
    the feature axis, eps 1e-7, gamma, beta in that weight order), but written
    so the converter cannot fuse (x - mean)^2 into SQUARED_DIFFERENCE, for
    which LiteRT has no int16 kernel ("SquaredDifference only supports FLOAT32
    and INT32"). The 16x8 exports need it; the float re-export check says what
    the re-association costs (expected: float32 round-off, <= 1 LSB).
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.epsilon = 1e-7

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=input_shape[-1:], initializer="ones", trainable=True, name="gamma")
        self.beta = self.add_weight(shape=input_shape[-1:], initializer="zeros", trainable=True, name="beta")

    def call(self, x):
        ops = _keras.ops
        mean = ops.mean(x, axis=-1, keepdims=True)
        ex2 = ops.mean(x * x, axis=-1, keepdims=True)
        var = ex2 - mean * mean
        return (x - mean) / ops.sqrt(var + self.epsilon) * self.gamma + self.beta


def build_split_models(dtln_src: Path, h5: Path, norm_stft: bool, packed: bool, iln: str = "authors"):
    import tensorflow as tf  # noqa: F401
    import keras
    from keras import layers, ops
    sys.modules.setdefault("wavinfo", types.SimpleNamespace(WavInfoReader=None))
    sys.path.insert(0, str(dtln_src))
    import DTLN_model as dm
    from DTLN_model import DTLN_model, InstantLayerNormalization

    # The authors' norm_stft branch calls tf.math.log on a symbolic tensor while
    # building the graph, which Keras 3 refuses. Redirect that ONE function to
    # keras.ops.log through a proxy of the tf namespace; everything else the
    # module touches (signal.rfft, math.reduce_mean, ...) runs on real tensors
    # inside Lambda/Layer calls and is left as is.
    def _proxy(mod, **over):
        ns = types.SimpleNamespace()
        for k in dir(mod):
            if k.startswith("_"):
                continue
            try:
                setattr(ns, k, getattr(mod, k))
            except Exception:
                pass
        for k, v in over.items():
            setattr(ns, k, v)
        return ns
    dm.tf = _proxy(tf, math=_proxy(tf.math, log=ops.log))

    full = DTLN_model()
    with contextlib.redirect_stdout(io.StringIO()):
        full.build_DTLN_model_stateful(norm_stft=norm_stft)
    full.model.load_weights(str(h5))
    ILN = InstantLayerNormalization if iln == "authors" else ILN_E2

    def state_inputs(prefix):
        if packed:
            return [layers.Input(batch_shape=(1, NUM_LAYER, NUM_UNITS, 2), name=f"{prefix}s")]
        return [layers.Input(batch_shape=(1, NUM_UNITS), name=f"{prefix}s{i}") for i in range(2 * NUM_LAYER)]

    def kernel_with_states(x, sin, mask_size):
        hs, cs = [], []
        for idx in range(NUM_LAYER):
            if packed:
                h0 = layers.Lambda(lambda s, i=idx: s[:, i, :, 0])(sin[0])
                c0 = layers.Lambda(lambda s, i=idx: s[:, i, :, 1])(sin[0])
            else:
                h0, c0 = sin[2 * idx], sin[2 * idx + 1]
            x, h, c = layers.LSTM(NUM_UNITS, return_sequences=True, unroll=True,
                                  return_state=True)(x, initial_state=[h0, c0])
            hs.append(h); cs.append(c)
        mask = layers.Activation("sigmoid")(layers.Dense(mask_size)(x))
        if packed:
            def pack(ts):
                hh = ops.stack(ts[:NUM_LAYER], axis=1)
                cc = ops.stack(ts[NUM_LAYER:], axis=1)
                return ops.stack([hh, cc], axis=-1)           # (1, L, U, 2), the authors' layout
            sout = [layers.Lambda(pack)(hs + cs)]
        else:
            sout = [t for pair in zip(hs, cs) for t in pair]   # h0, c0, h1, c1
        return mask, sout

    mag = layers.Input(batch_shape=(1, 1, BINS), name="a_mag")
    s1 = state_inputs("b_")
    mag_in = ILN()(layers.Lambda(lambda m: ops.log(m + 1e-7))(mag)) if norm_stft else mag
    mask1, so1 = kernel_with_states(mag_in, s1, BINS)
    model_1 = keras.Model([mag] + s1, [mask1] + so1, name="dtln_1")

    fr = layers.Input(batch_shape=(1, 1, FRAME), name="a_frame")
    s2 = state_inputs("b_")
    enc = layers.Conv1D(ENC, 1, strides=1, use_bias=False)(fr)
    encn = ILN()(enc)
    mask2, so2 = kernel_with_states(encn, s2, ENC)
    est = layers.Multiply()([enc, mask2])
    dec = layers.Conv1D(FRAME, 1, padding="causal", use_bias=False)(est)
    model_2 = keras.Model([fr] + s2, [dec] + so2, name="dtln_2")

    w = full.model.get_weights()
    n1 = (2 if norm_stft else 0) + NUM_LAYER * 3 + 2        # the authors' num_elements_first_core
    model_1.set_weights(w[:n1])
    model_2.set_weights(w[n1:])
    total = model_1.count_params() + model_2.count_params()
    assert total == (986_753 + (514 if norm_stft else 0)), total   # norm adds gamma/beta over 257 bins
    return model_1, model_2


# ----------------------------------------------------------------------------
# 3. representative dataset from the calibration subset, through the oracle
# ----------------------------------------------------------------------------
class Recorder(oracle.DtlnTflite):
    """The float export, recording the inputs of both stages every k-th hop."""

    def __init__(self, variant_dir: Path, f1: str, f2: str, every: int):
        oracle.VARIANTS["_rec"] = (f1, f2)
        super().__init__("_rec", model_dir=variant_dir)
        self.every, self.k = every, 0
        self.rep1: list[tuple[np.ndarray, list[np.ndarray]]] = []
        self.rep2: list[tuple[np.ndarray, list[np.ndarray]]] = []

    def stage1(self, mag):
        if self.k % self.every == 0:
            self.rep1.append((mag.copy(), [s.copy() for s in self.s1]))
        return super().stage1(mag)

    def stage2(self, est):
        if self.k % self.every == 0:
            self.rep2.append((est.copy(), [s.copy() for s in self.s2]))
        self.k += 1
        return super().stage2(est)


def collect_representative(root: Path, rec: Recorder):
    m = manifest.load(root)
    calib = m[m.calib == 1]
    assert len(calib) == 200 and (calib.split == "train").all(), "calib subset is 200 TRAIN rows"
    assert not (set(calib.id) & set(m[m.role == "eval"].id)), "calib must be disjoint from eval"
    for p in calib.noisy_path:
        oracle.enhance(oracle.read_wav_int16(p)[1], rec)
    print(f"representative dataset: {len(calib)} calib utterances, every {rec.every}th hop -> "
          f"{len(rec.rep1)} samples per stage")
    return rec.rep1, rec.rep2


# ----------------------------------------------------------------------------
# 4. conversion
# ----------------------------------------------------------------------------
def convert(model, rep, state_names, scheme="int8", per_tensor_dense=False):
    """scheme: 'int8' = full-integer int8 activations and weights, int8 I/O;
    '16x8' = int16 activations with int8 weights, int16 I/O -- the scheme
    TFLite reserves for exactly this failure (LSTM cell states need more than
    8 bits of resolution; the builtin LSTM op keeps its cell state int16 for
    the same reason, which an unrolled export cannot express).

    per_tensor_dense: one weight scale per FULLY_CONNECTED instead of the
    converter's default of one per output row. Found necessary 2026-10-21 on
    the H723: TFLite Micro's CMSIS-NN fully-connected kernel (commit 64b3de8,
    fully_connected.cc:115) refuses per-channel weights with int16 activations,
    while the reference kernel the host wheel runs accepts them -- so the
    per-channel 16x8 export scores on the host and cannot run on the MCU. The
    deployable artifact is the per-tensor one, scored separately."""
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if rep is not None:
        main = model.inputs[0].name.split(":")[0]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if per_tensor_dense:
            conv._experimental_disable_per_channel_quantization_for_dense_layers = True
        def gen():
            for x, states in rep:
                d = {main: x}
                d.update({n: s for n, s in zip(state_names, states)})
                yield d
        conv.representative_dataset = gen
        if scheme == "16x8":
            conv.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            conv.inference_input_type = tf.int16
            conv.inference_output_type = tf.int16
        else:
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
    return conv.convert()


def describe(path: Path):
    from ai_edge_litert.interpreter import Interpreter
    from tflite.Model import Model
    from tflite.BuiltinOperator import BuiltinOperator as BO
    from tflite.TensorType import TensorType as TT
    on = {v: k for k, v in vars(BO).items() if isinstance(v, int)}
    tn = {v: k for k, v in vars(TT).items() if isinstance(v, int)}
    buf = path.read_bytes()
    mdl = Model.GetRootAsModel(bytearray(buf), 0)
    codes = [on.get(max(mdl.OperatorCodes(i).BuiltinCode(), mdl.OperatorCodes(i).DeprecatedBuiltinCode()), "?")
             for i in range(mdl.OperatorCodesLength())]
    used = collections.Counter(); sg = mdl.Subgraphs(0)
    for o in range(sg.OperatorsLength()):
        used[codes[sg.Operators(o).OpcodeIndex()]] += 1
    by = collections.Counter()
    for t in range(sg.TensorsLength()):
        T = sg.Tensors(t); b = mdl.Buffers(T.Buffer())
        if b.DataLength():
            by[tn[T.Type()]] += b.DataLength()
    it = Interpreter(model_path=str(path)); it.allocate_tensors()
    print(f"  {path.name}: {len(buf):,} B; {sum(used.values())} ops, {len(used)} distinct: "
          + " ".join(f"{k}x{v}" for k, v in sorted(used.items())))
    print("    populated buffers: " + ", ".join(f"{k} {v:,} B" for k, v in by.most_common()))
    for kind, dets in (("in ", it.get_input_details()), ("out", it.get_output_details())):
        for d in dets:
            q = d["quantization_parameters"]
            qs = f"scale {q['scales'][0]:.6g} zp {q['zero_points'][0]}" if len(q["scales"]) else ""
            print(f"    {kind} {d['name']:<36} {str(tuple(int(v) for v in d['shape'])):<18} {d['dtype'].__name__:<8} {qs}")
    return len(buf), used, by


def check_reexport(out_dir: Path, f1: str, f2: str, utt_path, against: str | None) -> None:
    oracle.VARIANTS["_chk"] = (f1, f2)
    ours = oracle.DtlnTflite("_chk", model_dir=out_dir)
    _, x = oracle.read_wav_int16(utt_path)
    yb = oracle.enhance(x, ours)
    if against is None:
        print(f"  re-export on {Path(utt_path).name}: finite output, rms {np.sqrt((yb.astype(float) ** 2).mean()):.1f} LSB "
              f"(no shipped TFLite pair to compare with)")
        return
    ya = oracle.enhance(x, oracle.DtlnTflite(against))
    d = np.abs(ya.astype(int) - yb.astype(int))
    print(f"  re-export vs shipped '{against}' pair on {Path(utt_path).name}: max {d.max()} LSB, "
          f"{np.count_nonzero(d)} of {len(d)} samples differ")
    if d.max() > 1:
        raise SystemExit("the re-export is not the shipped network; stop here")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtln-src", type=Path, default=oracle.PRETRAINED.parent / "upstream",
                    help="directory holding the authors' DTLN_model.py (vendored at dtln/upstream)")
    ap.add_argument("--weights", default="model.h5",
                    help="model.h5 (paper), DTLN_norm_500h.h5, or an absolute path to a trained authors'-layout h5")
    ap.add_argument("--states", choices=("split", "packed"), default="split")
    ap.add_argument("--scheme", choices=("int8", "16x8"), default="int8")
    ap.add_argument("--iln", choices=("authors", "e2"), default="authors",
                    help="layer-norm variance form: authors' (x-mean)^2 -> SQUARED_DIFFERENCE, or E[x^2]-E[x]^2 (16x8 needs it)")
    ap.add_argument("--tag", default=None, help="output tag; default int8 / int8p / int8n, i16 / i16n for 16x8")
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--out", type=Path, default=oracle.PRETRAINED)
    ap.add_argument("--root", type=Path, default=manifest.DEFAULT_ROOT)
    a = ap.parse_args(argv)
    norm = "_norm_" in a.weights
    packed = a.states == "packed"
    tag = a.tag or (("i16n" if norm else "i16") if a.scheme == "16x8" else ("int8n" if norm else "int8p" if packed else "int8"))

    if a.scheme == "16x8" and a.iln == "authors":
        print("note: 16x8 with the authors' ILN emits SQUARED_DIFFERENCE, which LiteRT cannot run in int16; use --iln e2")
    wpath = pathlib.Path(a.weights)
    wpath = wpath if wpath.is_absolute() else oracle.PRETRAINED / a.weights   # trained weights live elsewhere
    paper = wpath.name == "model.h5"
    ftag = "f32n" if norm else ("reexport_f32" if (paper and not packed) else "reexport_f32p" if paper else "f32")
    m1, m2 = build_split_models(a.dtln_src, wpath, norm, packed, a.iln)
    print(f"[{a.weights}, states {a.states}, iln {a.iln}, scheme {a.scheme}] split models rebuilt: {m1.count_params():,} + {m2.count_params():,} params")
    sn1 = [t.name.split(":")[0] for t in m1.inputs[1:]]; sn2 = [t.name.split(":")[0] for t in m2.inputs[1:]]

    f1, f2 = f"model_{ftag}_1.tflite", f"model_{ftag}_2.tflite"
    (a.out / f1).write_bytes(convert(m1, None, sn1)); (a.out / f2).write_bytes(convert(m2, None, sn2))
    print("float export:"); describe(a.out / f1); describe(a.out / f2)
    us = manifest.utterances(manifest.load(a.root), "voicebank16k")
    check_reexport(a.out, f1, f2, us[500].noisy_path, "f32" if paper else None)

    rec = Recorder(a.out, f1, f2, a.every)
    rep1, rep2 = collect_representative(a.root, rec)

    p1, p2 = a.out / f"model_{tag}_1.tflite", a.out / f"model_{tag}_2.tflite"
    p1.write_bytes(convert(m1, rep1, sn1, a.scheme)); p2.write_bytes(convert(m2, rep2, sn2, a.scheme))
    print(f"\n{tag} (scheme {a.scheme}, states {a.states}):")
    n1, u1, b1 = describe(p1); n2, u2, b2 = describe(p2)
    fl = b1.get("FLOAT32", 0) + b2.get("FLOAT32", 0)
    print(f"  pair total {n1 + n2:,} B; {len(set(u1) | set(u2))} distinct ops: {sorted(set(u1) | set(u2))}")
    print(f"  float32 bytes left in the pair: {fl:,} B {'(an operator stayed in float -- see RESULTS.md)' if fl > 20000 else '(biases/params only)'}")


if __name__ == "__main__":
    main()
