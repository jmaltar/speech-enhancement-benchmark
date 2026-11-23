"""Export a PRUNED run as the smaller dense network a port would actually run.

    python -m training.shrink --run training/runs/prune-learned [--every 16]

A pruning run's export (`best.authors.weights.h5`) is the full-size network with
zero rows and columns: correct, but it neither saves a byte of flash nor
converts -- a dead unit's state tensor has zero range, and the TFLite
quantizer aborts on it (the 16x8 conversion of prune-learned died with
SIGABRT). This script rebuilds the two-part stateful model at the LIVE sizes,
copies the surviving rows and columns, and exports float and 16x8 pairs into
<run>/shrunk/, with the TFLite Micro arena and the sizes printed. A layer with
zero live units is removed; the layer that read it receives a constant zero
input of width 1 (so the recurrence it runs on its own state is unchanged).

The result is what the budget's flash column should be charged for a pruned
model, and what `evaluate.py --shrunk` scores (added 2026-10-20).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dtln import oracle, quantize as Q  # noqa: E402
from training import model as M  # noqa: E402

FRAME, HOP, BINS, ENC = M.FRAME, M.HOP, M.BINS, M.ENC


def build_shrunk(net, live: list[np.ndarray], clips: dict | None = None, basis: np.ndarray | None = None):
    """Two Keras split models (stage 1, stage 2) at live sizes, weights copied.
    `clips` ({"z_clip": float, "c_clip": [per layer] | None}, a QAT run's) puts
    the clipped cell (M.ExportLSTMCell) in place of keras.layers.LSTM."""
    import keras
    from keras import layers
    cells = M.lstm_cells(net)
    bidx = np.arange(ENC) if basis is None else np.flatnonzero(basis)
    nb = len(bidx)
    z_clip = float(clips["z_clip"]) if clips else 0.0
    c_clip = clips.get("c_clip") if clips else None

    def lstm_weights(c, in_idx, out_idx):
        U = c.units
        W = (c.kernel * c.mask_in).numpy(); R = (c.recurrent_kernel * c.mask_rec).numpy(); b = (c.bias * c.mask_bias).numpy()
        cols = np.concatenate([g * U + out_idx for g in range(4)])
        if in_idx is None:                               # predecessor layer dead: constant zero input, width 1
            Wn = np.zeros((1, 4 * len(out_idx)), np.float32)
        elif isinstance(in_idx, str):                    # "all": the stage's full input
            Wn = W[:, cols]
        else:
            Wn = W[in_idx][:, cols]
        return Wn, R[out_idx][:, cols], b[cols]

    def stage(name, x_in, in_dim_desc, cell_pair, dense_layer, mask_size, prev_live_in, out_idx=None):
        states_in, states_out, h = [], [], x_in
        prev_idx = prev_live_in          # "all" = the stage's full input; None = dead predecessor; else live indices
        for k, c in cell_pair:
            idx = np.flatnonzero(live[k])
            if len(idx) == 0:
                # dead layer: its successor reads a constant zero of width 1
                h = layers.Lambda(lambda t: t[..., :1] * 0.0, name=f"zero_in_{k}")(h)
                prev_idx = None
                continue
            h0 = layers.Input(batch_shape=(1, len(idx)), name=f"b_s{2 * (k % 2)}")
            c0 = layers.Input(batch_shape=(1, len(idx)), name=f"b_s{2 * (k % 2) + 1}")
            states_in += [h0, c0]
            if z_clip > 0 or c_clip is not None:
                cell = M.ExportLSTMCell(len(idx), z_clip, None if c_clip is None else c_clip[k], name=f"lstm{k}_cell")
                rnn = layers.RNN(cell, return_sequences=True, unroll=True, return_state=True, name=f"lstm{k}")
            else:
                rnn = layers.LSTM(len(idx), return_sequences=True, unroll=True, return_state=True, name=f"lstm{k}")
            h, hs, cs = rnn(h, initial_state=[h0, c0])
            Wn, Rn, bn = lstm_weights(c, prev_idx, idx)
            rnn.set_weights([Wn, Rn, bn])
            states_out += [hs, cs]
            prev_idx = idx
        d = layers.Dense(mask_size, name=dense_layer.name)
        mask = layers.Activation("sigmoid")(d(h))
        Wd, bd = dense_layer.get_weights()
        if out_idx is not None:                          # a pruned basis: the mask dense loses dead output columns
            Wd, bd = Wd[:, out_idx], bd[out_idx]
        Wdn = np.zeros((1, mask_size), np.float32) if prev_idx is None else (Wd if isinstance(prev_idx, str) else Wd[prev_idx])
        d.set_weights([Wdn, bd])
        return mask, states_in, states_out

    # stage 1: magnitude -> mask
    mag = layers.Input(batch_shape=(1, 1, BINS), name="a_mag")
    mask1, s1_in, s1_out = stage("s1", mag, BINS, [(0, cells[0]), (1, cells[1])], net.get_layer("dense1"), BINS, "all")
    model_1 = keras.Model([mag] + s1_in, [mask1] + s1_out, name="dtln_1_shrunk")

    # stage 2: frame -> frame
    fr = layers.Input(batch_shape=(1, 1, FRAME), name="a_frame")
    enc = layers.Conv1D(nb, 1, use_bias=False, name="encoder")(fr)
    encn = Q.ILN_E2(name="iln")(enc)
    # lstm2 reads the live basis channels: "all" only when none died
    s2_prev = "all" if nb == ENC else bidx
    mask2, s2_in, s2_out = stage("s2", encn, nb, [(2, cells[2]), (3, cells[3])], net.get_layer("dense2"), nb, s2_prev,
                                 out_idx=None if nb == ENC else bidx)
    est = layers.Multiply()([enc, mask2])
    dec = layers.Conv1D(FRAME, 1, padding="causal", use_bias=False, name="decoder")(est)
    model_2 = keras.Model([fr] + s2_in, [dec] + s2_out, name="dtln_2_shrunk")
    Ek = net.get_layer("encoder").kernel.numpy(); Dk = net.get_layer("decoder").kernel.numpy()
    g_, b_ = net.get_layer("iln").get_weights()
    model_2.get_layer("encoder").set_weights([Ek[:, :, bidx]])
    model_2.get_layer("iln").set_weights([g_[bidx], b_[bidx]])
    model_2.get_layer("decoder").set_weights([Dk[:, bidx, :]])
    return model_1, model_2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None, help="export directory (default <run>/shrunk)")
    ap.add_argument("--int8", action="store_true", help="also export the full-integer int8 pair (model_int8_*.tflite)")
    ap.add_argument("--per-tensor-dense", action="store_true",
                    help="also export the 16x8 pair with per-tensor FULLY_CONNECTED weights (model_i16pt_*.tflite): "
                         "the form CMSIS-NN's s16 kernel accepts (H723 port, 2026-10-21)")
    ap.add_argument("--z-clip", type=float, default=None, help="override: gate clip (a QAT run's is read from summary.json)")
    ap.add_argument("--c-clip", default=None, help="override: state clip, one value or four comma-separated")
    a = ap.parse_args(argv)
    run = a.run.resolve(); out = (a.out.resolve() if a.out else run / "shrunk"); out.mkdir(parents=True, exist_ok=True)
    summ = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    clips = summ.get("fq_clips") or None
    if a.z_clip is not None or a.c_clip is not None:
        clips = dict(clips or {"z_clip": 0.0, "c_clip": None})
        if a.z_clip is not None:
            clips["z_clip"] = a.z_clip
        if a.c_clip is not None:
            v = [float(t) for t in a.c_clip.split(",")]
            clips["c_clip"] = v * 4 if len(v) == 1 else v
    if clips:
        print(f"clips: {clips}")
    net = M.load_pretrained(run / "best.authors.weights.h5", dropout=0.0)
    live = [c.unit_mask.numpy() > 0.5 for c in M.lstm_cells(net)]
    # a run's export has zeros where units died but unit_mask is reloaded as ones:
    # recover the decision from the zeroed gate columns
    live = []
    for c in M.lstm_cells(net):
        W = c.kernel.numpy(); R = c.recurrent_kernel.numpy(); U = c.units
        alive = np.array([np.any(W[:, [g * U + j for g in range(4)]]) or np.any(R[:, [g * U + j for g in range(4)]]) for j in range(U)])
        live.append(alive)
    counts = [int(l.sum()) for l in live]
    basis = M.basis_gate(net).basis_mask.numpy() > 0.5       # recovered from the zeroed encoder columns on load
    nb = int(basis.sum())
    print(f"live units {counts}, live basis channels {nb}")
    m1, m2 = build_shrunk(net, live, clips, None if nb == ENC else basis)
    params = m1.count_params() + m2.count_params()
    print(f"shrunk params {params:,} (full 986,753 -> {100 * params / 986_753:.1f} %)")

    sn1 = [t.name.split(":")[0] for t in m1.inputs[1:]]; sn2 = [t.name.split(":")[0] for t in m2.inputs[1:]]
    f1, f2 = out / "model_f32_1.tflite", out / "model_f32_2.tflite"
    f1.write_bytes(Q.convert(m1, None, sn1)); f2.write_bytes(Q.convert(m2, None, sn2))
    print("float:"); Q.describe(f1); Q.describe(f2)
    # the shrunk float pair must reproduce the full-size (zeroed) export. evaluate.py --shrunk
    # never makes one (a zeroed full-size export cannot be quantized), so build it here through
    # the same two-stage graph with every unit live -- the reference for the row/column selection
    full_dir = run / "tflite"
    if not (full_dir / "model_f32_1.tflite").exists():
        full_dir.mkdir(exist_ok=True)
        fm1, fm2 = build_shrunk(net, [np.ones(c.units, bool) for c in M.lstm_cells(net)], clips)
        (full_dir / "model_f32_1.tflite").write_bytes(Q.convert(fm1, None, [t.name.split(":")[0] for t in fm1.inputs[1:]]))
        (full_dir / "model_f32_2.tflite").write_bytes(Q.convert(fm2, None, [t.name.split(":")[0] for t in fm2.inputs[1:]]))
        print(f"full-size float export written to {full_dir} (reference for the shrink check)")
    oracle.VARIANTS["_full"] = ("model_f32_1.tflite", "model_f32_2.tflite")
    oracle.VARIANTS["_shr"] = ("model_f32_1.tflite", "model_f32_2.tflite")
    from eval_harness import manifest
    us = manifest.utterances(manifest.load(), "voicebank16k")
    x = oracle.read_wav_int16(us[500].noisy_path)[1]
    y_full = oracle.enhance(x, oracle.DtlnTflite("_full", model_dir=run / "tflite"))
    y_shr = oracle.enhance(x, oracle.DtlnTflite("_shr", model_dir=out))
    d = int(np.abs(y_full.astype(int) - y_shr.astype(int)).max())
    clipped = bool(clips) and (clips["z_clip"] > 0 or clips["c_clip"] is not None)
    print(f"shrunk vs full-size zeroed export on {us[500].id}: max {d} LSB" + ("  (clips active: a difference is expected)" if clipped else "")
          + ("  (basis pruned: the zeroed full-size export normalises over dead channels, so a difference is expected; "
             "the shrunk network is the training network, see the whole-file check)" if nb < ENC else ""))
    if d > 2 and not clipped and nb == ENC:
        raise SystemExit("the shrunk network is not the pruned network; stop")
    if nb < ENC:
        # the reference for a pruned basis is the training model itself (masked layer norm):
        # whole-file Keras path vs the streaming shrunk pair, which the oracle showed agree to
        # <= 32 LSB on <= 0.3 % of samples for the authors' own model (framing policy, not arithmetic)
        import tensorflow as tf
        xf = x.astype(np.float32) / 32768.0
        pad = np.concatenate([np.zeros(384, np.float32), xf, np.zeros(384, np.float32)])[None]
        yk = net(tf.constant(pad), training=False).numpy()[0][384:384 + len(x)]
        yk = np.clip(np.rint(yk * 32768.0), -32768, 32767).astype(int)
        dd = np.abs(yk - y_shr.astype(int)); frac = float(np.mean(dd > 2))
        print(f"shrunk vs the training model (whole-file) on {us[500].id}: max {int(dd.max())} LSB, {100 * frac:.2f} % of samples > 2 LSB")
        if dd.max() > 64 or frac > 0.01:
            raise SystemExit("the shrunk basis network is not the trained masked network; stop")
    rec = Q.Recorder(out, "model_f32_1.tflite", "model_f32_2.tflite", a.every)
    rep1, rep2 = Q.collect_representative(manifest.DEFAULT_ROOT, rec)
    p1, p2 = out / "model_i16_1.tflite", out / "model_i16_2.tflite"
    p1.write_bytes(Q.convert(m1, rep1, sn1, "16x8")); p2.write_bytes(Q.convert(m2, rep2, sn2, "16x8"))
    print("16x8:"); n1, _, _ = Q.describe(p1); n2, _, _ = Q.describe(p2)
    if a.per_tensor_dense:
        t1, t2 = out / "model_i16pt_1.tflite", out / "model_i16pt_2.tflite"
        t1.write_bytes(Q.convert(m1, rep1, sn1, "16x8", per_tensor_dense=True))
        t2.write_bytes(Q.convert(m2, rep2, sn2, "16x8", per_tensor_dense=True))
        print("16x8, per-tensor dense:"); Q.describe(t1); Q.describe(t2)
    i8 = None
    if a.int8:
        q1, q2 = out / "model_int8_1.tflite", out / "model_int8_2.tflite"
        q1.write_bytes(Q.convert(m1, rep1, sn1, "int8")); q2.write_bytes(Q.convert(m2, rep2, sn2, "int8"))
        print("int8:"); i1, _, _ = Q.describe(q1); i2, _, _ = Q.describe(q2); i8 = i1 + i2
    # TFLM arena
    from tflite_micro.python.tflite_micro import runtime
    import contextlib, io, re
    arenas = []
    import tempfile
    for p in (p1, p2):
        it = runtime.Interpreter.from_file(str(p), arena_size=512 * 1024)
        # TFLM prints from C onto the raw descriptors (stderr), beneath sys.stdout: capture both
        with tempfile.TemporaryFile(mode="w+") as tmp:
            saved = [os.dup(fd) for fd in (1, 2)]
            for fd in (1, 2):
                os.dup2(tmp.fileno(), fd)
            try:
                it.print_allocations(); sys.stdout.flush(); sys.stderr.flush()
            finally:
                for fd, sv in zip((1, 2), saved):
                    os.dup2(sv, fd); os.close(sv)
            tmp.seek(0); text = tmp.read()
        mm = re.search(r"Arena allocation total (\d+)", text)
        arenas.append(int(mm.group(1)) if mm else -1)
    summary = dict(live_units=counts, live_basis=nb, shrunk_params=params, f32_bytes=f1.stat().st_size + f2.stat().st_size,
                   i16_bytes=n1 + n2, int8_bytes=i8, tflm_arena_bytes=arenas, shrunk_vs_full_lsb=d, clips=clips)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"16x8 pair {n1 + n2:,} B, TFLM arena {arenas} B; summary {out / 'summary.json'}")


if __name__ == "__main__":
    main()
