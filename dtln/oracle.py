"""DTLN float oracle -- task 10. Host-only; the uncompressed quality reference.

    python -m dtln.oracle --in noisy.wav --out enhanced.wav [--variant f32|drq|bypass]

What this is
------------
The as-shipped DTLN (Westhausen & Meyer 2020, github.com/breizhn/DTLN @ 1de1f15a)
run frame by frame through the TFLite float32 pair `model_1.tflite` /
`model_2.tflite` -- the authors' own export of `model.h5`, the weights behind
their Table 2 row -- under the ABF1 FRAMING POLICY the ports use:

  * analysis buffer zero pre-filled (a leading pad of frame - hop samples),
  * input delivered in whole hops, the last hop zero-padded,
  * FLUSH injects frame - hop samples of silence, yielding frame/hop - 1 more
    output hops,
  * the host drops the first frame - hop output samples and trims to the real
    length.

The geometry comes from harness/abf_host.py (lag / flush_hops / data_hops),
the same three helpers the device binding and the host driver use, so there is
one definition of the policy and the oracle cannot drift from the ports (the
requirement task 7 placed on tasks 9-10; FRAME_FORMAT.md section 8).

Happily the authors' own evaluation script (run_evaluation.py) pads 384 zeros
before and after each file, predicts, then takes predicted[384:384+len] -- the
same policy in other words. So the ABF1-framed streaming oracle reproduces the
processing behind the published numbers rather than approximating it; the
check in dtln/check_oracle.py measures how closely.

Two documented deviations from the locked audio configuration
-------------------------------------------------------------
DTLN's native hop is 8 ms (block_shift = 128), not the locked 16 ms, and its
frames are RECTANGULAR -- tf.signal.frame with no analysis window, the second
stage's learned decoder standing in for the synthesis window. Both are baked
into the trained weights: running the model at a 256-sample hop or under a
Hann window would be a different, untrained algorithm. So, exactly as RNNoise
keeps its 48 kHz / 10 ms geometry inside the processing path, DTLN keeps
512 / 128 rectangular inside this module, the output comes back at 16 kHz
sample-aligned with the input, and the metric path sees nothing of it. The
frame is 512 samples either way, so per-frame cost is the locked configuration's;
only the frame rate doubles. Report which hop every DTLN row is costed at.

Numerics, stated rather than assumed
------------------------------------
Input int16 is scaled by 1/32768 to the [-1, 1) float the model was trained on
(soundfile's convention, which the authors' pipeline used); output is scaled
back by 32768, clamped and rounded with round-half-even (`np.rint`, the
behaviour of C `lrintf` in the default rounding mode the ports use). The
STFT is numpy's real FFT in float64 on the float32 buffer, magnitude and phase
cast to float32 for the network -- the authors' real_time_processing_tf_lite.py
arithmetic, step for step. The interpreter runs single-threaded on the XNNPACK
delegate LiteRT enables by default; `xnnpack=False` selects the builtin
reference kernels, and check_oracle.py reports the difference.

Variants
--------
  f32     model_1/model_2.tflite            -- THE float oracle (task 10)
  drq     model_quant_1/model_quant_2.tflite -- the authors' dynamic-range int8
          export, i.e. the as-shipped Arm A quantization. Runs here because the
          host interpreter accepts hybrid kernels; TFLite Micro does NOT, which
          is why no MCU tier has an as-shipped DTLN path (tflm_footprint/).
  bypass  no network: each frame is returned scaled by hop/frame, so the
          rectangular overlap-add reconstructs the input exactly. The regression
          anchor for the framing code in this file -- its output must be
          bit-identical to the input, as enhance.py asserts for the classical
          bypass.
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

# One definition of the geometry: the host driver's helpers, not a copy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))
from abf_host import data_hops, flush_hops, lag  # noqa: E402

RATE_HZ = 16_000
FRAME = 512            # block_len  -- fixed by the trained weights
HOP = 128              # block_shift -- fixed by the trained weights (8 ms)
PRETRAINED = Path(__file__).resolve().parent / "pretrained"
SCALE = 32768.0

VARIANTS = {
    "f32": ("model_1.tflite", "model_2.tflite"),
    "drq": ("model_quant_1.tflite", "model_quant_2.tflite"),
    # full-integer PTQ exports (dtln/quantize.py), all int8 I/O:
    "int8p": ("model_int8p_1.tflite", "model_int8p_2.tflite"),   # authors' PACKED (h,c) state layout -- collapses
    "int8":  ("model_int8_1.tflite", "model_int8_2.tflite"),     # states as separate tensors (the deployable form)
    "f32n":  ("model_f32n_1.tflite", "model_f32n_2.tflite"),     # authors' DTLN_norm_500h weights, float
    "int8n": ("model_int8n_1.tflite", "model_int8n_2.tflite"),   # DTLN_norm_500h, full-integer, split states
    # 16x8 PTQ exports: int16 activations, int8 weights, int16 I/O, split states
    "i16":   ("model_i16_1.tflite", "model_i16_2.tflite"),       # paper weights
    "i16n":  ("model_i16n_1.tflite", "model_i16n_2.tflite"),     # DTLN_norm_500h weights
    "bypass": None,
}

assert FRAME % HOP == 0, "COLA precondition (abf_geometry_ok)"
LAG = lag(FRAME, HOP)                 # 384 samples, 24 ms
FLUSH_HOPS = flush_hops(FRAME, HOP)   # 3


class DtlnTflite:
    """The two-stage network with external LSTM state, one frame at a time.

    State layout is read from the model: the authors' export carries ONE packed
    (1, 2, 128, 2) tensor per stage; dtln/quantize.py's exports carry four
    (1, 128) tensors (h0, c0, h1, c1) so that each can take its own int8 scale.
    Inputs are matched by name (`serving_default_<name>`, sorted); outputs by
    their ":N" position in the model's output list, states after the main output.
    """

    def __init__(self, variant: str = "f32", model_dir: Path = PRETRAINED,
                 threads: int = 1, xnnpack: bool = True):
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        kw = {"num_threads": threads}
        if not xnnpack:
            kw["experimental_op_resolver_type"] = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        f1, f2 = VARIANTS[variant]
        self.i1 = Interpreter(model_path=str(model_dir / f1), **kw)
        self.i2 = Interpreter(model_path=str(model_dir / f2), **kw)
        self.i1.allocate_tensors()
        self.i2.allocate_tensors()
        self.in1_x, self.in1_s, self.out1_x, self.out1_s = self._wire(self.i1, FRAME // 2 + 1)
        self.in2_x, self.in2_s, self.out2_x, self.out2_s = self._wire(self.i2, FRAME)
        self.reset()

    @staticmethod
    def _wire(it, main_len: int):
        """main input/output = the rank-3 tensor of length main_len; the rest are
        states, inputs sorted by name and outputs in index order."""
        ins, outs = it.get_input_details(), it.get_output_details()
        x = [d for d in ins if len(d["shape"]) == 3]
        assert len(x) == 1 and tuple(x[0]["shape"]) == (1, 1, main_len), [tuple(d["shape"]) for d in ins]
        s_in = sorted([d for d in ins if len(d["shape"]) != 3], key=lambda d: d["name"])
        y = [d for d in outs if len(d["shape"]) == 3]
        assert len(y) == 1, [tuple(d["shape"]) for d in outs]
        # output order = the model's output list; TFLite names them "<call>:N"
        # with N that position, while the tensor index is graph order and
        # means nothing here.
        def pos(d):
            tail = d["name"].rsplit(":", 1)[-1]
            return int(tail) if tail.isdigit() else 0
        s_out = sorted([d for d in outs if len(d["shape"]) != 3], key=pos)
        assert [tuple(d["shape"]) for d in s_in] == [tuple(d["shape"]) for d in s_out], "state in/out mismatch"
        return x[0], s_in, y[0], s_out

    def reset(self) -> None:
        self.s1 = [np.zeros(d["shape"], dtype=np.float32) for d in self.in1_s]
        self.s2 = [np.zeros(d["shape"], dtype=np.float32) for d in self.in2_s]

    # --- int8 boundary handling. A full-integer model with int8 I/O carries a
    # (scale, zero_point) per input/output; the host quantizes the float
    # magnitude/frame in and dequantizes the mask/frame out, exactly as a
    # deployment would. The LSTM states cross the boundary too: each OUTPUT
    # state and its INPUT get independent scales from the converter, so the
    # state is carried in float and requantized each frame (dequantize with the
    # output's parameters, quantize with the input's) -- a per-frame rounding a
    # deployment pays as well unless the two scales are tied. Recorded here so
    # the equivalence protocol knows it is part of the model, not of the port.
    _LIM = {np.int8: (-128, 127), np.int16: (-32768, 32767)}

    @classmethod
    def _q(cls, x: np.ndarray, d: dict) -> np.ndarray:
        if d["dtype"] not in cls._LIM:
            return x.astype(np.float32)
        s, z = d["quantization_parameters"]["scales"][0], d["quantization_parameters"]["zero_points"][0]
        lo, hi = cls._LIM[d["dtype"]]
        return np.clip(np.rint(x / s) + z, lo, hi).astype(d["dtype"])

    @classmethod
    def _dq(cls, x: np.ndarray, d: dict) -> np.ndarray:
        if d["dtype"] not in cls._LIM:
            return x
        s, z = d["quantization_parameters"]["scales"][0], d["quantization_parameters"]["zero_points"][0]
        return ((x.astype(np.float32) - z) * s).astype(np.float32)

    def _run(self, it, in_x, in_s, out_x, out_s, x, states):
        it.set_tensor(in_x["index"], self._q(x, in_x))
        for d, st in zip(in_s, states):
            it.set_tensor(d["index"], self._q(st, d))
        it.invoke()
        y = self._dq(it.get_tensor(out_x["index"]), out_x)
        new = [self._dq(it.get_tensor(d["index"]), d) for d in out_s]
        return y, new

    def stage1(self, mag: np.ndarray) -> np.ndarray:
        mask, self.s1 = self._run(self.i1, self.in1_x, self.in1_s, self.out1_x, self.out1_s, mag, self.s1)
        return mask

    def stage2(self, est: np.ndarray) -> np.ndarray:
        out, self.s2 = self._run(self.i2, self.in2_x, self.in2_s, self.out2_x, self.out2_s, est, self.s2)
        return out

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """frame: float32[FRAME] in [-1, 1). Returns the stage-2 time-domain
        block, float32[FRAME], to be overlap-added at HOP."""
        spec = np.fft.rfft(frame)                      # float64 arithmetic, as the authors'
        mag = np.abs(spec).reshape(1, 1, -1).astype(np.float32)
        phase = np.angle(spec)
        mask = self.stage1(mag)
        est = np.fft.irfft(mag * mask * np.exp(1j * phase), n=FRAME)
        est = est.reshape(1, 1, -1).astype(np.float32)
        return self.stage2(est).reshape(-1)


class Bypass:
    """Framing regression anchor: rectangular OLA of frame * hop/frame is the
    identity. 1/4 is exact in binary, so the reconstruction is bit-exact after
    rounding back to int16."""

    def reset(self) -> None:
        pass

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame * np.float32(HOP / FRAME)


class DtlnTflm(DtlnTflite):
    """The same two-stage pipeline with TFLite MICRO as the runtime -- the
    MicroInterpreter the firmware tiers run, via the tflite-micro wheel -- so
    a quantized model's quality is measured on the arithmetic an MCU would
    actually execute, not on LiteRT's. LiteRT is still opened, read-only, for
    tensor metadata (dtype, quantization parameters, order); TFLM indexes
    inputs and outputs by subgraph position, which is the order LiteRT's
    get_*_details() lists them in."""

    def __init__(self, variant: str, model_dir: Path = PRETRAINED, arena_kb: int = 256, **kw):
        super().__init__(variant, model_dir=model_dir, **kw)
        from tflite_micro.python.tflite_micro import runtime
        f1, f2 = VARIANTS[variant]
        self.m1 = runtime.Interpreter.from_file(str(model_dir / f1), arena_size=arena_kb * 1024)
        self.m2 = runtime.Interpreter.from_file(str(model_dir / f2), arena_size=arena_kb * 1024)
        self.pos1 = self._positions(self.i1)
        self.pos2 = self._positions(self.i2)

    @staticmethod
    def _positions(it):
        """tensor index -> subgraph position, for inputs and outputs."""
        return ({d["index"]: p for p, d in enumerate(it.get_input_details())},
                {d["index"]: p for p, d in enumerate(it.get_output_details())})

    def _run(self, it, in_x, in_s, out_x, out_s, x, states):
        micro, (pin, pout) = (self.m1, self.pos1) if it is self.i1 else (self.m2, self.pos2)
        micro.set_input(self._q(x, in_x), pin[in_x["index"]])
        for d, st in zip(in_s, states):
            micro.set_input(self._q(st, d), pin[d["index"]])
        micro.invoke()
        y = self._dq(micro.get_output(pout[out_x["index"]]), out_x)
        new = [self._dq(micro.get_output(pout[d["index"]]), d) for d in out_s]
        return y, new


def make_model(variant: str, **kw):
    """variant, optionally suffixed '-tflm' to run it on TFLite Micro."""
    if variant == "bypass":
        return Bypass()
    if variant.endswith("-tflm"):
        return DtlnTflm(variant[:-5], **kw)
    return DtlnTflite(variant, **kw)


def enhance(x: np.ndarray, model) -> np.ndarray:
    """int16[n] -> int16[n], sample-aligned, under the ABF1 policy."""
    if x.dtype != np.int16 or x.ndim != 1:
        raise TypeError("mono int16 only -- the stream format is fmt=1 int16")
    n = len(x)
    dh = data_hops(n, HOP)
    xf = np.zeros(dh * HOP, dtype=np.float32)
    xf[:n] = x.astype(np.float32) / SCALE                  # last hop zero-padded
    model.reset()
    in_buf = np.zeros(FRAME, dtype=np.float32)           # zero pre-fill = leading lag
    out_buf = np.zeros(FRAME, dtype=np.float32)
    emitted = np.empty((dh + FLUSH_HOPS) * HOP, dtype=np.float32)
    silence = np.zeros(HOP, dtype=np.float32)
    for j in range(dh + FLUSH_HOPS):
        hop = xf[j * HOP:(j + 1) * HOP] if j < dh else silence     # FLUSH: silence in
        in_buf[:-HOP] = in_buf[HOP:]
        in_buf[-HOP:] = hop
        block = model.process_frame(in_buf)
        out_buf[:-HOP] = out_buf[HOP:]
        out_buf[-HOP:] = 0.0
        out_buf += block
        emitted[j * HOP:(j + 1) * HOP] = out_buf[:HOP]           # one RESULT per hop
    y = emitted[LAG:][:n]                                          # drop-and-trim
    assert len(y) == n, (len(y), n)
    y = np.rint(np.clip(y.astype(np.float64) * SCALE, -32768.0, 32767.0))
    return y.astype(np.int16)


def read_wav_int16(path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: need 16-bit mono PCM")
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return rate, x


def write_wav_int16(path, rate: int, y: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(y.astype("<i2").tobytes())


def enhance_file(in_path, out_path, model) -> int:
    rate, x = read_wav_int16(in_path)
    if rate != RATE_HZ:
        raise ValueError(f"{in_path}: DTLN is a {RATE_HZ} Hz model; file is {rate} Hz -- "
                         "no resampling here (that would be an undocumented deviation)")
    write_wav_int16(out_path, rate, enhance(x, model))
    return len(x)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="f32", help="one of %s, optionally with '-tflm'" % sorted(VARIANTS))
    ap.add_argument("--no-xnnpack", action="store_true", help="builtin reference kernels")
    a = ap.parse_args(argv)
    kw = {} if a.variant == "bypass" else {"xnnpack": not a.no_xnnpack}
    n = enhance_file(a.inp, a.out, make_model(a.variant, **kw))
    print(f"{a.variant}: {n} samples -> {a.out}  (frame {FRAME}, hop {HOP}, lag {LAG}, "
          f"data hops {data_hops(n, HOP)} + flush {FLUSH_HOPS})")


if __name__ == "__main__":
    main()
