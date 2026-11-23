"""DTLN in Keras 3 with a hand-written LSTM cell -- the model task 13 trains.

Why not the authors' model file: its LSTM layers are keras.layers.LSTM, whose
kernel is opaque to the three TinyLSTMs compression steps -- learned-threshold
structured pruning (a mask on rows/units inside the gate matmul), QAT (fake
quantization of weights, inputs, states and gate activations, straight-through
rounding) and Skip-RNN (a learned update gate on the state). All three act
INSIDE the recurrence, so the cell has to be ours. Everything else (STFT,
masks, 1x1 convolutions, instant layer norm, overlap-add) is the authors'
graph, written with Keras ops.

Weight layout is the authors' exactly (kernel (in, 4U) / recurrent (U, 4U) /
bias (4U,) with gates ordered i, f, g, o; Dense kernel/bias; Conv1D kernels
(1, in, out); ILN gamma/beta), so `from_authors_weights()` takes the list the
authors' model.h5 produces through their own DTLN_model.py (vendored at
dtln/upstream/), and `to_authors_model()` hands trained weights back into that
model so model.save_weights() writes an h5 that dtln/quantize.py converts
unchanged. check_model.py proves the round trip reproduces the float oracle.

Compression hooks (all OFF by default; the model is then the authors' DTLN):
  cell.mask_in / cell.mask_rec   -- non-trainable 0/1 masks on the gate matmuls
  cell.fq                        -- fake-quantize weights and gate inputs (STE)
These are the attachment points for Arm B; the procedures themselves
(threshold learning, QAT schedules) come in train.py's follow-ups.
"""
from __future__ import annotations

import contextlib
import io
import sys
import types
from pathlib import Path

import numpy as np
import keras
from keras import layers, ops
import tensorflow as tf

FRAME, HOP, UNITS, ENC, LAYERS = 512, 128, 128, 256, 2
BINS = FRAME // 2 + 1
UPSTREAM = Path(__file__).resolve().parent.parent / "dtln" / "upstream"


def fake_quant(x, bits: int = 8, scale=None, channel_axis=None):
    """Symmetric per-tensor fake quantization with a straight-through estimator
    (TinyLSTMs section 3.2, eq. 4). Gradient passes as identity. With `scale`
    the step is FIXED (a calibrated range, as a deployed int8 graph has);
    without it the step follows the tensor's own max, per call. With
    `channel_axis` (weights only, 2026-10-22) the max is taken per slice along
    that axis -- the converter's per-output-channel weight scales, one per
    CONV_2D output channel (absmax / 127, zero point 0)."""
    qmax = 2.0 ** (bits - 1) - 1
    # The scale is a statistic of the tensor, not a path for gradients (the
    # first QAT run went to NaN at epoch 2 with the scale differentiable).
    if scale is not None:
        s = ops.stop_gradient(scale)
    elif channel_axis is None:
        s = ops.stop_gradient(ops.maximum(ops.max(ops.abs(x)), 1e-8) / qmax)
    else:
        red = [i for i in range(len(x.shape)) if i != channel_axis]
        s = ops.stop_gradient(ops.maximum(ops.max(ops.abs(x), axis=red, keepdims=True), 1e-8) / qmax)
    q = ops.clip(ops.round(x / s), -qmax - 1, qmax) * s
    return x + ops.stop_gradient(q - x)


class LSTMCellC(layers.Layer):
    """Keras-compatible LSTM cell with compression hooks. Same arithmetic as
    keras.layers.LSTM (tanh / sigmoid, gates i f g o).

    Structured pruning (TinyLSTMs section 3.1, eq. 3, at unit granularity): a
    unit j has a score s_j = ||all weights that feed its four gates||_2 (input
    and recurrent columns). It is alive when s_j > tau, tau a learned scalar
    per layer. Forward uses the hard indicator; backward the sigmoid
    surrogate sigma((s_j - tau) / temp) (straight-through). The unit mask
    multiplies h and c, so a dead unit contributes nothing downstream or to
    the recurrence, and `expected_units()` (sum of the soft mask) is what the
    size penalty in the loss is applied to. finalize_pruning() turns a trained
    soft decision into hard 0/1 masks on the weight matrices (rows, columns,
    biases), which is what a port removes to get a smaller dense matrix.
    """

    def __init__(self, units: int, **kw):
        super().__init__(**kw)
        self.units = units
        self.state_size = [units, units]
        self.output_size = units
        self.fq = False            # fake-quantize weights, inputs, state (QAT)
        self.fq_fixed = True       # fixed calibrated scales (calibrate_fq); False = per-call max (the 2026-10-19 runs)
        self.fq_abits = 8          # activation / state bit width under fixed scales (8 = int8 target, 16 = the 16x8 scheme)
        self.z_clip = 0.0          # clip the gate pre-activations at +/- z_clip (0 = off); lossless past |z| ~ 16
        self.c_clip = False        # clip the cell state at +/- alpha, alpha learned (PACT-style)
        self.prune = False         # learned-threshold unit pruning
        self.temp = 0.2            # sigmoid temperature on the score scale
        self._rec = None           # calibration: lists of per-call maxima while calibrate_fq runs

    def build(self, input_shape):
        d = int(input_shape[-1]); U = self.units
        self.kernel = self.add_weight(shape=(d, 4 * U), name="kernel", initializer="glorot_uniform")
        self.recurrent_kernel = self.add_weight(shape=(U, 4 * U), name="recurrent_kernel", initializer="orthogonal")
        self.bias = self.add_weight(shape=(4 * U,), name="bias", initializer="zeros")
        self.mask_in = self.add_weight(shape=(d, 4 * U), name="mask_in", initializer="ones", trainable=False)
        self.mask_rec = self.add_weight(shape=(U, 4 * U), name="mask_rec", initializer="ones", trainable=False)
        self.mask_bias = self.add_weight(shape=(4 * U,), name="mask_bias", initializer="ones", trainable=False)
        self.unit_mask = self.add_weight(shape=(U,), name="unit_mask", initializer="ones", trainable=False)
        self.tau_raw = self.add_weight(shape=(), name="tau_raw", initializer="zeros")   # trained only when prune=True
        self.s_ref = self.add_weight(shape=(), name="s_ref", initializer="ones", trainable=False)  # score scale at init
        # QAT with fixed scales: the calibrated ranges of this cell's input, gate
        # pre-activations and cell state (calibrate_fq), and the learned state clip
        self.x_absmax = self.add_weight(shape=(), name="x_absmax", initializer="ones", trainable=False)
        self.z_absmax = self.add_weight(shape=(), name="z_absmax", initializer="ones", trainable=False)
        self.c_absmax = self.add_weight(shape=(), name="c_absmax", initializer="ones", trainable=False)
        self.alpha = self.add_weight(shape=(), name="alpha", initializer="ones")   # trained only when c_clip

    @property
    def tau(self):
        """Threshold on the score scale: tau_raw x s_ref. Adam moves tau_raw by ~lr
        per step, so tau moves by ~lr x s_ref -- the same relative speed in every
        layer whatever its weight norms. s_ref is the layer's median score when
        pruning is switched on (set_pruning())."""
        return self.tau_raw * self.s_ref

    def set_pruning(self, on: bool, temp: float = 0.2, start_quantile: float = 0.0):
        self.prune = on
        if on:
            sc = self.scores().numpy()
            self.s_ref.assign(float(np.median(sc)))
            self.temp = temp * float(np.median(sc))                 # temperature on the score scale too
            self.tau_raw.assign(float(np.quantile(sc, start_quantile)) / float(np.median(sc)) if start_quantile > 0 else 0.0)

    def scores(self):
        """s_j: L2 norm over the four gate columns of unit j, input and recurrent."""
        W = self.kernel * self.mask_in; R = self.recurrent_kernel * self.mask_rec
        U = self.units
        w2 = ops.sum(ops.reshape(ops.square(W), (-1, 4, U)), axis=(0, 1))
        r2 = ops.sum(ops.reshape(ops.square(R), (-1, 4, U)), axis=(0, 1))
        return ops.sqrt(w2 + r2 + 1e-12)

    def soft_mask(self):
        return ops.sigmoid((self.scores() - self.tau) / self.temp)

    def expected_units(self):
        return ops.sum(self.soft_mask())

    def _unit_gate(self):
        if not self.prune:
            return self.unit_mask
        soft = self.soft_mask()
        hard = ops.cast(self.scores() > self.tau, soft.dtype)
        return (hard + (soft - ops.stop_gradient(soft))) * self.unit_mask

    def call(self, inputs, states):
        h, c = states
        W = self.kernel * self.mask_in
        R = self.recurrent_kernel * self.mask_rec
        b = self.bias * self.mask_bias
        x = inputs
        fixed = self.fq and self.fq_fixed
        ab = self.fq_abits; aq = 2.0 ** (ab - 1) - 1          # activation step = range / aq
        if self._rec is not None:
            self._rec["x"].append(ops.max(ops.abs(x)))
        if self.fq and not self.fq_fixed:
            # the 2026-10-19 scheme: per-call max scales on weights, input and both incoming states
            W, R, x, h, c = fake_quant(W), fake_quant(R), fake_quant(x), fake_quant(h), fake_quant(c)
        elif fixed:
            # weights per-tensor (as the converter's per-channel is at least as fine), the input at
            # its calibrated range; the states arrive quantized from the previous step (or zero)
            W, R, x = fake_quant(W), fake_quant(R), fake_quant(x, ab, scale=self.x_absmax / aq)
        z = ops.matmul(x, W) + ops.matmul(h, R) + b
        if self._rec is not None:
            self._rec["z"].append(ops.max(ops.abs(z)))
        if self.z_clip > 0:
            z = ops.clip(z, -self.z_clip, self.z_clip)
        if self.fq:   # the gate pre-activations (the +/-670 tensor PTQ could not survive)
            z = fake_quant(z, ab, scale=(self.z_clip if self.z_clip > 0 else self.z_absmax) / aq) if fixed else fake_quant(z)
        i, f, g, o = ops.split(z, 4, axis=-1)
        i, f, o = ops.sigmoid(i), ops.sigmoid(f), ops.sigmoid(o)
        c2 = f * c + i * ops.tanh(g)
        if self._rec is not None:
            self._rec["c"].append(ops.max(ops.abs(c2)))
        if self.c_clip:
            c2 = ops.minimum(ops.maximum(c2, -self.alpha), self.alpha)
        h2 = o * ops.tanh(c2)
        if fixed:   # the outgoing states, at the ranges a deployed graph holds them in
            c2 = fake_quant(c2, ab, scale=(self.alpha if self.c_clip else self.c_absmax) / aq)
            h2 = fake_quant(h2, ab, scale=1.0 / aq)
        m = self._unit_gate()
        h2 = h2 * m; c2 = c2 * m
        return h2, [h2, c2]


def lstm_cells(model: keras.Model) -> list["LSTMCellC"]:
    return [model.get_layer(f"lstm{i}").cell for i in range(2 * LAYERS)]


# --- the sites OUTSIDE the LSTM cells (2026-10-21) ----------------------------------
# Two findings on the H723 the same day: the 16x8 export costs 0.07-0.10 PESQ against
# the same model's float export whether or not the cells were trained quantized, and
# the encoder/decoder basis the learned thresholds never touched is 74 % of the hop.
# So (a) fake quantization at the boundaries the cells do not cover -- the dense
# mask layers, the encoder, the layer norm, the masked estimate, the decoder -- and
# (b) learned-threshold structured pruning of the 256-channel basis. All of it lives
# in layers of its own: to_authors_weights() reads the named layers' weight lists,
# and the h5 layout must stay the authors' 20 arrays.

class FQ(layers.Layer):
    """Activation fake-quant site at a FIXED calibrated range (absmax), off by
    default; `fixed_absmax` pins the range (1.0 for a sigmoid output, as the
    exported mask tensors are quantized at 1/32768)."""

    def __init__(self, fixed_absmax: float | None = None, **kw):
        super().__init__(**kw)
        self.on = False
        self.bits = 8
        self.fixed_absmax = fixed_absmax
        self._rec = None

    def build(self, input_shape):
        self.absmax = self.add_weight(shape=(), name="absmax", initializer="ones", trainable=False)
        if self.fixed_absmax is not None:
            self.absmax.assign(self.fixed_absmax)

    def call(self, x):
        if self._rec is not None:
            self._rec.append(ops.max(ops.abs(x)))
        if not self.on:
            return x
        aq = 2.0 ** (self.bits - 1) - 1
        return fake_quant(x, self.bits, scale=self.absmax / aq)


class DenseFQ(layers.Dense):
    """keras Dense whose kernel is fake-quantized (8-bit, per tensor) when .fq is set.
    Weight list unchanged (kernel, bias), so the authors' layout holds."""
    fq = False

    def call(self, inputs):
        k = fake_quant(self.kernel) if self.fq else self.kernel
        x = ops.matmul(inputs, k)
        if self.bias is not None:
            x = ops.add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv1x1FQ(layers.Conv1D):
    """The authors' 1x1 Conv1D (a matmul over the last axis) with an optional
    fake-quantized kernel. kernel shape (1, in, out) as Keras lays it out.
    `fq_per_channel` (2026-10-22): one scale per OUTPUT channel (axis 1 of
    kernel[0]), which is what the converter emits for these CONV_2D ops
    (encoder 256 scales, decoder 512); the default per-tensor form is what the
    2026-10-21 all-site run used."""
    fq = False
    fq_per_channel = False

    def call(self, inputs):
        k = self.kernel[0]
        if self.fq:
            k = fake_quant(k, channel_axis=1 if self.fq_per_channel else None)
        x = ops.matmul(inputs, k)
        if self.use_bias:
            x = ops.add(x, self.bias)
        return x


class BasisGate(layers.Layer):
    """Learned-threshold structured pruning of the encoder/decoder basis -- the
    TinyLSTMs unit mechanism (score, learned tau, sigmoid surrogate, hard forward)
    applied to the ENC analysis/synthesis channel pairs. Channel j's score is the
    L2 norm of encoder column j and decoder row j together. Returns (x * m, m); the
    mask also feeds the layer norm, which must normalise over LIVE channels only
    or the shrunk network would not be the masked one. `kernels` is a plain
    callable (not a tracked attribute) returning (encoder_kernel, decoder_kernel)."""

    def __init__(self, n: int, kernels=None, **kw):
        super().__init__(**kw)
        self.n = n
        self._kernels = kernels
        self.prune = False
        self.temp = 0.2

    def build(self, input_shape):
        self.basis_mask = self.add_weight(shape=(self.n,), name="basis_mask", initializer="ones", trainable=False)
        self.tau_raw = self.add_weight(shape=(), name="tau_raw", initializer="zeros")
        self.s_ref = self.add_weight(shape=(), name="s_ref", initializer="ones", trainable=False)

    @property
    def tau(self):
        return self.tau_raw * self.s_ref

    def scores(self):
        E, D = self._kernels()                      # (1, FRAME, ENC), (1, ENC, FRAME)
        E = E[0] * self.basis_mask[None, :]; D = D[0] * self.basis_mask[:, None]
        return ops.sqrt(ops.sum(ops.square(E), axis=0) + ops.sum(ops.square(D), axis=1) + 1e-12)

    def set_pruning(self, on: bool, temp: float = 0.2, start_quantile: float = 0.0):
        self.prune = on
        if on:
            sc = self.scores().numpy()
            self.s_ref.assign(float(np.median(sc)))
            self.temp = temp * float(np.median(sc))
            self.tau_raw.assign(float(np.quantile(sc, start_quantile)) / float(np.median(sc)) if start_quantile > 0 else 0.0)

    def soft_mask(self):
        return ops.sigmoid((self.scores() - self.tau) / self.temp)

    def expected_channels(self):
        return ops.sum(self.soft_mask())

    def gate(self):
        if not self.prune:
            return self.basis_mask
        soft = self.soft_mask()
        hard = ops.cast(self.scores() > self.tau, soft.dtype)
        return (hard + (soft - ops.stop_gradient(soft))) * self.basis_mask

    def call(self, x):
        m = self.gate()
        return x * m, m


class ILNMasked(layers.Layer):
    """Instant layer normalisation over the LIVE channels of a masked basis:
    inputs [x, m]; mean and variance over sum(m) channels, output re-masked.
    With m = ones it is the authors' ILN exactly; after a basis shrink the
    plain ILN over the live channels computes the same numbers."""

    def build(self, input_shape):
        n = input_shape[0][-1]
        self.gamma = self.add_weight(shape=(n,), initializer="ones", name="gamma")
        self.beta = self.add_weight(shape=(n,), initializer="zeros", name="beta")

    def call(self, inputs):
        x, m = inputs
        cnt = ops.maximum(ops.sum(m), 1.0)
        mean = ops.sum(x * m, axis=-1, keepdims=True) / cnt
        var = ops.sum(ops.square((x - mean) * m), axis=-1, keepdims=True) / cnt
        return ((x - mean) / ops.sqrt(var + 1e-7) * self.gamma + self.beta) * m


def basis_gate(model: keras.Model) -> "BasisGate":
    return model.get_layer("basis_gate")


def fq_sites(model: keras.Model) -> list["FQ"]:
    return [l for l in model.layers if isinstance(l, FQ)]


def fq_weight_layers(model: keras.Model):
    return [model.get_layer(n) for n in ("dense1", "encoder", "dense2", "decoder")]


def live_basis(model: keras.Model) -> int:
    return int(np.sum(basis_gate(model).basis_mask.numpy() > 0.5))


def finalize_basis(model: keras.Model) -> int:
    """Hard basis decision: zero encoder columns, decoder rows, layer-norm gamma/beta,
    dense2 output columns and bias, and lstm2's input rows of dead channels, and fix
    basis_mask. Returns live channels."""
    g = basis_gate(model)
    a = (g.scores().numpy() > float(np.asarray(g.tau))) & (g.basis_mask.numpy() > 0.5)
    af = a.astype(np.float32)
    enc = model.get_layer("encoder"); dec = model.get_layer("decoder")
    enc.kernel.assign(enc.kernel.numpy() * af[None, None, :])
    dec.kernel.assign(dec.kernel.numpy() * af[None, :, None])
    iln = model.get_layer("iln")
    iln.gamma.assign(iln.gamma.numpy() * af); iln.beta.assign(iln.beta.numpy() * af)
    d2 = model.get_layer("dense2")
    d2.kernel.assign(d2.kernel.numpy() * af[None, :]); d2.bias.assign(d2.bias.numpy() * af)
    c2 = model.get_layer("lstm2").cell
    c2.mask_in.assign(c2.mask_in.numpy() * af[:, None])
    g.basis_mask.assign(af); g.prune = False; g.tau_raw.assign(0.0)
    return int(a.sum())


def calibrate_fq(model: keras.Model, x) -> dict:
    """Set every cell's fixed quantization ranges from one float forward pass
    (eager) over a batch -- the same per-tensor calibration the converter does
    from its representative dataset -- and initialise a learned state clip at
    the observed state range. Returns the ranges per cell."""
    cells = lstm_cells(model)
    for c in cells:
        c._rec = {"x": [], "z": [], "c": []}
    sites = fq_sites(model)
    for s_ in sites:
        s_._rec = []
    model(x, training=False)
    out = {}
    for s_ in sites:
        if s_.fixed_absmax is None:
            s_.absmax.assign(max(float(max(float(v) for v in s_._rec)), 1e-6))
        out[s_.name] = {"absmax": float(s_.absmax.numpy())}
        s_._rec = None
    for k, c in enumerate(cells):
        # a dead (pruned-out) layer has zero range everywhere: floor it, or its scale is 0 and 0/0 = NaN
        r = {key: max(float(max(float(v) for v in vals)), 1e-6) for key, vals in c._rec.items()}
        c._rec = None
        c.x_absmax.assign(r["x"]); c.z_absmax.assign(r["z"]); c.c_absmax.assign(r["c"])
        if c.c_clip:
            c.alpha.assign(r["c"])
        out[f"lstm{k}"] = r
    return out


def fq_clips(model: keras.Model) -> dict:
    """What the export graph must carry: the gate clip and the per-layer state clip."""
    cells = lstm_cells(model)
    return {"z_clip": float(cells[0].z_clip),
            "c_clip": [float(c.alpha.numpy()) for c in cells] if cells[0].c_clip else None}


class ExportLSTMCell(layers.Layer):
    """The deployable LSTM cell: Keras LSTM arithmetic and weight layout
    (kernel, recurrent_kernel, bias) plus the clips a QAT run was trained with,
    so the converter sees the bounded gate and state tensors and emits the
    MINIMUM/MAXIMUM pair that bounds them on the target."""

    def __init__(self, units: int, z_clip: float = 0.0, c_clip: float | None = None, **kw):
        super().__init__(**kw)
        self.units = units; self.state_size = [units, units]; self.output_size = units
        self.z_clip = float(z_clip); self.c_clip = None if c_clip is None else float(c_clip)

    def build(self, input_shape):
        d = int(input_shape[-1]); U = self.units
        self.kernel = self.add_weight(shape=(d, 4 * U), name="kernel")
        self.recurrent_kernel = self.add_weight(shape=(U, 4 * U), name="recurrent_kernel")
        self.bias = self.add_weight(shape=(4 * U,), name="bias")

    def call(self, inputs, states):
        h, c = states
        z = ops.matmul(inputs, self.kernel) + ops.matmul(h, self.recurrent_kernel) + self.bias
        if self.z_clip > 0:
            z = ops.clip(z, -self.z_clip, self.z_clip)
        i, f, g, o = ops.split(z, 4, axis=-1)
        c2 = ops.sigmoid(f) * c + ops.sigmoid(i) * ops.tanh(g)
        if self.c_clip is not None:
            c2 = ops.clip(c2, -self.c_clip, self.c_clip)
        h2 = ops.sigmoid(o) * ops.tanh(c2)
        return h2, [h2, c2]


def live_units(model: keras.Model) -> list[int]:
    return [int(np.sum(c.unit_mask.numpy() > 0.5)) for c in lstm_cells(model)]


def finalize_pruning(model: keras.Model, keep: list[np.ndarray] | None = None) -> list[int]:
    """Turn the learned (or given) unit decisions into hard masks on the
    matrices: a dead unit's four gate columns, its recurrent input row, its bias
    entries, and the next layer's input rows that read it. Returns live units
    per cell. `keep` overrides the learned decision (fixed-threshold ablation)."""
    cells = lstm_cells(model)
    alive = []
    for k, c in enumerate(cells):
        a = (c.scores().numpy() > float(np.asarray(c.tau))) if keep is None else keep[k].astype(bool)
        a = a & (c.unit_mask.numpy() > 0.5)
        alive.append(a)
    for k, c in enumerate(cells):
        U = c.units; a = alive[k].astype(np.float32)
        col = np.tile(a, 4)                                  # gate columns of dead units
        c.mask_in.assign(c.mask_in.numpy() * col[None, :])
        c.mask_rec.assign(c.mask_rec.numpy() * col[None, :] * a[:, None])
        c.mask_bias.assign(c.mask_bias.numpy() * col)
        c.unit_mask.assign(c.unit_mask.numpy() * a)
        c.prune = False; c.tau_raw.assign(0.0)
        # the layer that reads this one (lstm1 reads lstm0, lstm3 reads lstm2)
        if k in (0, 2):
            nxt = cells[k + 1]
            nxt.mask_in.assign(nxt.mask_in.numpy() * a[:, None])
    return [int(a.sum()) for a in alive]


def unstructured_prune(model: keras.Model, n_keep: int) -> dict:
    """Core item 3's other arm: zero the smallest-magnitude entries of the four
    LSTM cells' input and recurrent matrices, globally ranked, keeping n_keep
    of them -- the same weight budget a structured run reached by removing
    whole units, but scattered. The masks are the same elementwise masks the
    structured path uses, so training and export need nothing new; what differs
    is that no matrix gets smaller, which is the point of measuring it."""
    cells = lstm_cells(model)
    mats = []
    for c in cells:
        mats += [(c.kernel, c.mask_in), (c.recurrent_kernel, c.mask_rec)]
    mags = np.concatenate([np.abs((w * m).numpy()).ravel() for w, m in mats])
    total = int(mags.size)
    assert 0 < n_keep <= total, (n_keep, total)
    thr = float(np.sort(mags)[::-1][n_keep - 1])
    kept = 0
    for w, m in mats:
        keep = (np.abs((w * m).numpy()) >= thr).astype(np.float32) * m.numpy()
        m.assign(keep); kept += int(keep.sum())
    return {"lstm_weights_total": total, "lstm_weights_kept": kept, "threshold": thr}


def lstm_weight_nonzeros(model: keras.Model) -> int:
    return int(sum(int(np.count_nonzero((c.kernel * c.mask_in).numpy())) + int(np.count_nonzero((c.recurrent_kernel * c.mask_rec).numpy()))
                   for c in lstm_cells(model)))


def lstm_dense_params_for_units(live_units: list[int]) -> int:
    """Input + recurrent matrix entries of the four cells at the given live units
    (the structured model's matrices), biases excluded -- what an unstructured
    run is matched to."""
    d_in = [BINS, live_units[0], ENC, live_units[2]]
    return int(sum(d * 4 * u + u * 4 * u for d, u in zip(d_in, live_units)))


def pruned_param_count(model: keras.Model) -> int:
    """Parameters of the smaller DENSE network a port would run: live rows and
    columns only, in every matrix the masks touch; the Dense layers after the
    LSTMs lose the rows of dead units."""
    cells = lstm_cells(model); n = 0
    nb = live_basis(model)
    d_in = [BINS, None, nb, None]
    for k, c in enumerate(cells):
        live = int(np.sum(c.unit_mask.numpy() > 0.5))
        din = d_in[k] if d_in[k] is not None else int(np.sum(cells[k - 1].unit_mask.numpy() > 0.5))
        n += din * 4 * live + live * 4 * live + 4 * live
    n += int(np.sum(cells[1].unit_mask.numpy() > 0.5)) * BINS + BINS           # dense1
    n += int(np.sum(cells[3].unit_mask.numpy() > 0.5)) * nb + nb               # dense2
    n += FRAME * nb + 2 * nb + nb * FRAME                                       # encoder, ILN, decoder
    return n


class ILN(layers.Layer):
    """Instant layer normalisation, the authors' form (variance as mean of
    squared deviation), eps 1e-7, weights gamma then beta."""

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=input_shape[-1:], initializer="ones", name="gamma")
        self.beta = self.add_weight(shape=input_shape[-1:], initializer="zeros", name="beta")

    def call(self, x):
        mean = ops.mean(x, axis=-1, keepdims=True)
        var = ops.mean(ops.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) / ops.sqrt(var + 1e-7) * self.gamma + self.beta


def _stft(x):
    frames = tf.signal.frame(x, FRAME, HOP)
    X = tf.signal.rfft(frames)
    return tf.abs(X), tf.math.angle(X)


def _istft_frames(mag_phase):
    mag, ph = mag_phase
    return tf.signal.irfft(tf.cast(mag, tf.complex64) * tf.exp(1j * tf.cast(ph, tf.complex64)))


def build(dropout: float = 0.25) -> keras.Model:
    """Whole-signal model: (B, T) noisy -> (B, T') enhanced, T' = (frames-1)*HOP+FRAME."""
    x = layers.Input(shape=(None,), name="noisy")
    mag, ph = layers.Lambda(_stft, name="stft")(x)
    h = mag
    for i in range(LAYERS):
        h = layers.RNN(LSTMCellC(UNITS, name=f"lstm{i}_cell"), return_sequences=True, name=f"lstm{i}")(h)
        if i < LAYERS - 1:
            h = layers.Dropout(dropout)(h)
    mask1 = layers.Activation("sigmoid")(DenseFQ(BINS, name="dense1")(h))
    mask1 = FQ(fixed_absmax=1.0, name="fq_mask1")(mask1)
    est_frames = layers.Lambda(_istft_frames, name="istft")([mag * mask1, ph])
    enc_layer = Conv1x1FQ(ENC, 1, use_bias=False, name="encoder")
    dec_layer = Conv1x1FQ(FRAME, 1, padding="causal", use_bias=False, name="decoder")
    dec_layer.build((None, None, ENC))          # built early: the basis gate scores its rows
    enc = enc_layer(est_frames)
    enc, bm = BasisGate(ENC, kernels=lambda: (enc_layer.kernel, dec_layer.kernel), name="basis_gate")(enc)
    enc = FQ(name="fq_enc")(enc)
    h = ILNMasked(name="iln")([enc, bm])
    h = FQ(name="fq_iln")(h)
    for i in range(LAYERS):
        h = layers.RNN(LSTMCellC(UNITS, name=f"lstm{2 + i}_cell"), return_sequences=True, name=f"lstm{2 + i}")(h)
        if i < LAYERS - 1:
            h = layers.Dropout(dropout)(h)
    mask2 = layers.Activation("sigmoid")(DenseFQ(ENC, name="dense2")(h))
    mask2 = FQ(fixed_absmax=1.0, name="fq_mask2")(mask2)
    est = FQ(name="fq_est")(enc * mask2)
    dec = FQ(name="fq_dec")(dec_layer(est))
    y = layers.Lambda(lambda f: tf.signal.overlap_and_add(f, HOP), name="ola")(dec)
    return keras.Model(x, y, name="dtln_k3")


# --- the authors' layout -------------------------------------------------------------
# get_weights() order of their build_DTLN_model(norm_stft=False), 20 arrays:
#   lstm1 (k, r, b), lstm2 (k, r, b), dense1 (k, b), encoder (k), iln (gamma, beta),
#   lstm3 (k, r, b), lstm4 (k, r, b), dense2 (k, b), decoder (k)
_ORDER = (["lstm0"] * 3 + ["lstm1"] * 3 + ["dense1"] * 2 + ["encoder"] + ["iln"] * 2
          + ["lstm2"] * 3 + ["lstm3"] * 3 + ["dense2"] * 2 + ["decoder"])


def authors_model(h5: Path | None = None):
    """The authors' Keras model (vendored DTLN_model.py, unmodified), optionally
    loaded with an h5 weights file. wavinfo (a training-data helper it imports)
    is stubbed, not installed."""
    sys.modules.setdefault("wavinfo", types.SimpleNamespace(WavInfoReader=None))
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from DTLN_model import DTLN_model
    # Their constructor sets memory growth on every GPU, which TensorFlow refuses
    # once the context exists or a memory cap (virtual device) is configured.
    # It is a convenience in their file, not part of the model: no-op it here.
    import tensorflow as tf
    _orig = tf.config.experimental.set_memory_growth
    tf.config.experimental.set_memory_growth = lambda *a_, **k_: None
    try:
        mc = DTLN_model()
    finally:
        tf.config.experimental.set_memory_growth = _orig
    with contextlib.redirect_stdout(io.StringIO()):
        mc.build_DTLN_model(norm_stft=False)
    if h5 is not None:
        mc.model.load_weights(str(h5))
    return mc.model


def from_authors_weights(model: keras.Model, weights: list[np.ndarray]) -> None:
    assert len(weights) == 20, len(weights)
    by_layer: dict[str, list[np.ndarray]] = {}
    for name, w in zip(_ORDER, weights):
        by_layer.setdefault(name, []).append(w)
    for name, ws in by_layer.items():
        layer = model.get_layer(name)
        if name.startswith("lstm"):
            cell = layer.cell
            cell.kernel.assign(ws[0]); cell.recurrent_kernel.assign(ws[1]); cell.bias.assign(ws[2])
        else:
            layer.set_weights(ws)
    # a basis-pruned run's export has zero encoder columns where channels died; the
    # layer norm must know (it normalises over live channels), so recover the mask
    enc = model.get_layer("encoder").kernel.numpy()[0]
    alive = np.any(enc != 0.0, axis=0).astype(np.float32)
    if alive.sum() < alive.size:
        basis_gate(model).basis_mask.assign(alive)


def to_authors_weights(model: keras.Model) -> list[np.ndarray]:
    out = []
    for name in dict.fromkeys(_ORDER):
        layer = model.get_layer(name)
        if name.startswith("lstm"):
            c = layer.cell
            out += [(c.kernel * c.mask_in).numpy(), (c.recurrent_kernel * c.mask_rec).numpy(),
                    (c.bias * c.mask_bias).numpy()]
        else:
            out += [w.numpy() for w in layer.weights]
    assert len(out) == 20
    return out


def load_pretrained(h5: Path, dropout: float = 0.25) -> keras.Model:
    """The authors' weights in our model: the starting point of every Arm B run."""
    # The authors' constructor calls tf.config.experimental.set_memory_growth,
    # which TensorFlow refuses once the GPU context exists -- so their model is
    # built FIRST, before ours creates any variable.
    w = authors_model(h5).get_weights()
    m = build(dropout)
    from_authors_weights(m, w)
    return m


def save_authors_h5(model: keras.Model, path: Path) -> None:
    """Trained weights in the authors' layout, as an h5 dtln/quantize.py converts."""
    am = authors_model()
    am.set_weights(to_authors_weights(model))
    am.save_weights(str(path))
