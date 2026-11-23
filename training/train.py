"""Task 13 training loop -- float fine-tune, learned-threshold pruning, fixed-threshold
pruning and QAT, one script, one switch each.

    source gpu_env.sh
    python -m training.train --out training/runs/<name> [--epochs 10] [--steps 500] [--batch 32]
        [--init <authors-layout .h5>]        start from a previous run's export (default: pretrained)
        [--prune-lambda L [--prune-temp T]]  learned per-layer thresholds, size penalty L x expected units
        [--fixed-prune a,b,c,d]              keep the a/b/c/d highest-scoring units per LSTM (magnitude), masks fixed
        [--fq]                               fake-quantize the LSTM cells (QAT)
        [--lr 1e-3] [--seg-seconds 4] [--limit N] [--smoke]

Starts from the authors' pretrained weights (or --init), trains with their
objective -- negative SNR per segment -- plus, for a learned-pruning run,
lambda x sum of the soft unit masks (TinyLSTMs eq. 3's size term). Adam with
gradient-norm clipping 3; the learning rate halves when validation does not
improve for two epochs.

Progress: a tqdm bar per epoch (running loss, step rate, live units when
pruning), an outer bar over epochs, a bar over the fixed validation set.
Output: <out>/log.csv, a checkpoint per epoch, and at the end the BEST
validation checkpoint reloaded, pruning finalised into hard masks, and the
weights exported in the authors' layout as <out>/best.authors.weights.h5
plus <out>/summary.json (live units, equivalent dense parameter count).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dtln import oracle  # noqa: E402
from training import data, model as M  # noqa: E402


def snr_loss(y_true, y_pred):
    """The authors' snr_cost: -10 log10(|s|^2 / |s - s_hat|^2), mean over the batch."""
    import tensorflow as tf
    T = tf.shape(y_pred)[1]
    s = y_true[:, :T]
    num = tf.reduce_sum(tf.square(s), axis=-1)
    den = tf.reduce_sum(tf.square(s - y_pred), axis=-1) + 1e-7
    return -tf.reduce_mean(10.0 * tf.math.log(num / den + 1e-7) / tf.math.log(10.0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", type=Path, default=None, help="authors'-layout h5 to start from")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seg-seconds", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--prune-lambda", type=float, default=0.0)
    ap.add_argument("--prune-temp", type=float, default=0.2, help="sigmoid temperature, x the layer's median score")
    ap.add_argument("--prune-start-q", type=float, default=0.05,
                    help="initial threshold at this quantile of the layer's scores (0 = below every unit)")
    ap.add_argument("--fixed-prune", default=None, help="live units per LSTM, e.g. 96,96,80,80")
    ap.add_argument("--prune-basis-lambda", type=float, default=0.0,
                    help="learned-threshold pruning of the 256-channel encoder/decoder basis, size penalty L x expected channels")
    ap.add_argument("--fq-sites", choices=("cells", "all"), default="cells",
                    help="under --fq: 'cells' = the four LSTM cells only (as before); 'all' = also the mask dense layers, "
                         "encoder, layer norm, masked estimate and decoder (the 16x8 export's residual, 2026-10-21)")
    ap.add_argument("--unstructured-match", type=Path, default=None,
                    help="a structured run dir: keep as many LSTM weights as its live units give, by magnitude, scattered (core item 3)")
    ap.add_argument("--fq", action="store_true", help="QAT: fake quantization at FIXED calibrated scales (one batch)")
    ap.add_argument("--fq-dynamic", action="store_true", help="QAT as on 2026-10-19: per-call max scales (diverges; kept for the record)")
    ap.add_argument("--fq-bits", type=int, default=8, help="activation/state bits under --fq: 8 (int8 target) or 16 (the 16x8 scheme); weights stay 8")
    ap.add_argument("--z-clip", type=float, default=0.0, help="clip gate pre-activations at +/- this (16 is lossless)")
    ap.add_argument("--c-clip", action="store_true", help="learned cell-state clip per layer (PACT), init at the calibrated range")
    ap.add_argument("--clip-reg", type=float, default=1e-4, help="L2 penalty on the learned state clips")
    ap.add_argument("--wq-only", action="store_true",
                    help="2026-10-22: weight-only QAT of the three stage-2 tensors the 16x8 residual sits in "
                         "(encoder/decoder bases per-output-channel int8, mask dense2 per-tensor int8, STE); "
                         "EVERYTHING else frozen, no activation fake-quant")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--gpu-mem-mb", type=int, default=0,
                    help="cap this process's GPU memory so two runs can share the card (0 = grow as needed)")
    a = ap.parse_args(argv)
    if a.smoke:
        a.epochs, a.steps, a.limit = 2, 50, 600
    from tqdm import tqdm
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print("GPU:", gpus or "NONE -- source gpu_env.sh; training on the CPU is not the plan")
    for g in gpus:
        if a.gpu_mem_mb:
            tf.config.set_logical_device_configuration(g, [tf.config.LogicalDeviceConfiguration(memory_limit=a.gpu_mem_mb)])
        else:
            tf.config.experimental.set_memory_growth(g, True)
    tf.random.set_seed(a.seed); np.random.seed(a.seed)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, indent=1))
    tr, va = data.load_train_val(limit=a.limit)
    print(f"train {len(tr)} utterances ({tr.seconds / 3600:.2f} h, {len(set(tr.speakers))} speakers) | "
          f"val {len(va)} ({va.seconds / 60:.1f} min, speakers {sorted(set(va.speakers))})")
    seg = int(a.seg_seconds * data.RATE)
    seg = ((seg - M.FRAME) // M.HOP) * M.HOP + M.FRAME

    net = M.load_pretrained(a.init or (oracle.PRETRAINED / "model.h5"), dropout=a.dropout)
    cells = M.lstm_cells(net)
    mode = "float fine-tune"
    if a.fixed_prune:
        keep_n = [int(v) for v in a.fixed_prune.split(",")]
        assert len(keep_n) == 4
        keep = []
        for c, n in zip(cells, keep_n):
            s_ = c.scores().numpy(); idx = np.argsort(-s_)[:n]
            k = np.zeros(c.units, bool); k[idx] = True; keep.append(k)
        live = M.finalize_pruning(net, keep)
        mode = f"fixed magnitude pruning to {live} units, then fine-tune"
    unstructured = None
    if a.unstructured_match:
        ref = json.loads((a.unstructured_match / "summary.json").read_text())
        n_keep = M.lstm_dense_params_for_units(ref["live_units"])
        unstructured = M.unstructured_prune(net, n_keep)
        unstructured["matched_run"] = str(a.unstructured_match); unstructured["matched_live_units"] = ref["live_units"]
        mode = (f"unstructured magnitude pruning to {n_keep:,} of {unstructured['lstm_weights_total']:,} LSTM weights "
                f"(matched to {a.unstructured_match.name}'s {ref['live_units']}), then fine-tune")
    if a.prune_lambda > 0:
        for c in cells:
            c.set_pruning(True, temp=a.prune_temp, start_quantile=a.prune_start_q)
        mode = f"learned-threshold pruning, lambda {a.prune_lambda}, temp {a.prune_temp}, start quantile {a.prune_start_q}"
    gate = M.basis_gate(net)
    if a.prune_basis_lambda > 0:
        gate.set_pruning(True, temp=a.prune_temp, start_quantile=a.prune_start_q)
        mode += f" + learned-threshold BASIS pruning, lambda {a.prune_basis_lambda}"
    if a.fq_dynamic:
        a.fq = True
    for c in cells:
        c.z_clip = a.z_clip; c.c_clip = a.c_clip
    if a.z_clip > 0:
        mode += f" + gate clip {a.z_clip:g}"
    if a.c_clip:
        mode += " + learned state clip"
    calib = None
    if a.fq:
        for c in cells:
            c.fq = True; c.fq_fixed = not a.fq_dynamic; c.fq_abits = a.fq_bits
        mode += f" + fake quantization (QAT, {a.fq_bits}-bit activations) in all four LSTM cells, " + ("per-call max scales" if a.fq_dynamic else "fixed calibrated scales")
        if a.fq_sites == "all":
            for s_ in M.fq_sites(net):
                s_.on = True; s_.bits = a.fq_bits
            for l in M.fq_weight_layers(net):
                l.fq = True
            mode += " + fake quantization at the six non-cell sites (mask dense x2, encoder, layer norm, estimate, decoder; 8-bit weights)"
    if a.wq_only:
        assert not a.fq and a.prune_lambda == 0 and a.prune_basis_lambda == 0, "--wq-only is a stand-alone mode"
        WQ = ("encoder", "decoder", "dense2")
        for l in net.layers:
            l.trainable = l.name in WQ
        for n in ("encoder", "decoder"):
            net.get_layer(n).fq = True; net.get_layer(n).fq_per_channel = True
        net.get_layer("dense2").fq = True
        # 2026-10-22 (wqat3 post-mortem, training/scale_check.py): dense2's per-tensor fake-quant
        # scale must be the one the SHRUNK export gets. Its kernel rows for pruned-out units of
        # cells[3] are never read (their input is masked to zero) and never trained, but they keep
        # their fine-tune values and set the tensor max -- wqat3 trained on a grid 2.43x coarser
        # than it was deployed on. Zero them: no forward or backward value changes, the max is live.
        d2 = net.get_layer("dense2"); alive = cells[3].unit_mask.numpy() > 0.5
        k2 = d2.kernel.numpy(); k2[~alive, :] = 0.0; d2.kernel.assign(k2)
        print(f"dense2 dead rows zeroed: {int((~alive).sum())} of {alive.size}; live-row max |w| = {np.abs(k2).max():.4f}")
        tv = [v.path if hasattr(v, "path") else v.name for v in net.trainable_variables]
        print("trainable under --wq-only:", tv)
        assert 3 <= len(tv) <= 4, tv
        mode += " + weight-only QAT of encoder/decoder (int8 per output channel) and dense2 (int8 per tensor), everything else frozen"
    if a.fq or a.c_clip:
        xb, _ = data.segments(tr, a.batch, seg, np.random.default_rng(a.seed + 1))
        calib = M.calibrate_fq(net, tf.constant(xb))
        print("calibrated ranges (max |x|, |z|, |c| per cell):", {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in calib.items()})
    print("mode:", mode)
    # Clipping is done here, not by Adam's clipnorm, so the norm can be checked first.
    # Why (qat / qat2, 2026-10-19): both runs went NaN at epoch 2 although every gradient
    # was finite -- 12 entries of lstm0's kernel (input bins 7/15/18, six units). A gradient
    # near 1e19 is finite in float32 but its SQUARE is not, so Adam's second moment went to
    # inf on one step and inf - inf = NaN on the next; the finite-gradient guard could not
    # see it. The norms are therefore accumulated in float64, each gradient is clipped to
    # CLIP per variable exactly as Keras's clipnorm did for the float baseline (so the
    # compression runs keep the baseline's optimizer), and only a step whose norm is still
    # non-finite is skipped and counted. Norms of 1e6-1e9 are ROUTINE in this recurrence
    # (a 50-step smoke run: 19 steps above 1e4, largest 4.6e9), so a cap would discard a
    # third of the steps; the epoch's largest norm is logged instead.
    opt = tf.keras.optimizers.Adam(learning_rate=a.lr)   # Keras 3: opt.learning_rate is assignable
    opt.build(net.trainable_variables)
    lam = tf.constant(a.prune_lambda, tf.float32)
    CLIP = 3.0
    skipped = tf.Variable(0, dtype=tf.int64, trainable=False)
    gnorm_max = tf.Variable(0.0, dtype=tf.float32, trainable=False)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            loss = snr_loss(y, net(x, training=True))
            size = tf.add_n([c.expected_units() for c in cells]) if a.prune_lambda > 0 else tf.constant(0.0)
            total = loss + lam * size
            if a.prune_basis_lambda > 0:
                total = total + a.prune_basis_lambda * gate.expected_channels()
            if a.c_clip:
                total = total + a.clip_reg * tf.add_n([tf.square(c.alpha) for c in cells])
        grads = tape.gradient(total, net.trainable_variables)
        pairs = [(g, v) for g, v in zip(grads, net.trainable_variables) if g is not None]
        norms = [tf.sqrt(tf.reduce_sum(tf.square(tf.cast(g, tf.float64)))) for g, _ in pairs]
        gnorm = tf.cast(tf.sqrt(tf.add_n([n * n for n in norms])), tf.float32)
        ok = tf.math.is_finite(total) & tf.math.is_finite(gnorm)
        if ok:
            clipped = [(g * tf.cast(tf.minimum(tf.constant(1.0, tf.float64), CLIP / (n + 1e-30)), tf.float32), v) for (g, v), n in zip(pairs, norms)]
            opt.apply_gradients(clipped)
        else:
            skipped.assign_add(1)
        gnorm_max.assign(tf.maximum(gnorm_max, tf.where(tf.math.is_finite(gnorm), gnorm, np.float32(np.inf))))
        return loss, size

    @tf.function
    def eval_step(x, y):
        return snr_loss(y, net(x, training=False))

    def validate(desc):
        losses = []
        nb = (len(va) + a.batch - 1) // a.batch
        for x, y in tqdm(data.val_batches(va, seg, a.batch), total=nb, desc=desc, unit="batch", leave=False):
            losses.append(float(eval_step(tf.constant(x), tf.constant(y))))
        return float(np.mean(losses))

    def hard_live():
        lu = [int(np.sum((c.scores().numpy() > float(np.asarray(c.tau))) & (c.unit_mask.numpy() > 0.5))) for c in cells]
        if a.prune_basis_lambda > 0:
            lu.append(int(np.sum((gate.scores().numpy() > float(np.asarray(gate.tau))) & (gate.basis_mask.numpy() > 0.5))))
        return lu

    log = a.out / "log.csv"
    with open(log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "train_loss", "val_loss", "live_units", "lr", "seconds", "steps_per_s", "skipped_steps", "max_gnorm"])
        v0 = validate("val @ start")
        w.writerow([0, 0, "", f"{v0:.4f}", "/".join(map(str, hard_live())), a.lr, 0, "", 0, ""]); f.flush()
        tqdm.write(f"start: validation loss (neg. SNR) {v0:.3f} dB, live units {hard_live()}")
        best, best_ep, stale, rng = v0, 0, 0, np.random.default_rng(a.seed)
        t_all = time.time()
        for ep in tqdm(range(1, a.epochs + 1), desc="epochs", unit="epoch"):
            t0 = time.time(); run = []
            skipped.assign(0); gnorm_max.assign(0.0)
            bar = tqdm(range(a.steps), desc=f"epoch {ep}/{a.epochs}", unit="step", leave=False)
            for st in bar:
                x, y = data.segments(tr, a.batch, seg, rng)
                loss, size = train_step(tf.constant(x), tf.constant(y))
                if np.isfinite(float(loss)):
                    run.append(float(loss))
                if st % 10 == 0 or st == a.steps - 1:
                    post = dict(loss=f"{np.mean(run[-50:]):.3f}", rate=f"{(st + 1) / (time.time() - t0):.2f} step/s")
                    if a.prune_lambda > 0:
                        post["units"] = "/".join(map(str, hard_live()))
                    bar.set_postfix(**post)
            dt = time.time() - t0
            v = validate(f"val @ epoch {ep}")
            lu = hard_live()
            n_skip, g_max = int(skipped.numpy()), float(gnorm_max.numpy())
            w.writerow([ep, ep * a.steps, f"{np.mean(run):.4f}", f"{v:.4f}", "/".join(map(str, lu)),
                        float(opt.learning_rate), f"{dt:.1f}", f"{a.steps / dt:.3f}", n_skip, f"{g_max:.3g}"]); f.flush()
            net.save_weights(a.out / f"epoch-{ep:02d}.weights.h5")
            mark = ""
            # for a pruning run "best" must also respect the size the run is converging to:
            # validation alone would always prefer the un-pruned early epochs. Use the last
            # epoch's units as the target and keep the best val among epochs at <= that size+5%.
            if v < best:
                best, best_ep, stale, mark = v, ep, 0, "  <- best"
            else:
                stale += 1
                if stale >= 2:
                    opt.learning_rate.assign(float(opt.learning_rate) * 0.5); stale = 0
                    mark = f"  (lr -> {float(opt.learning_rate):.2e})"
            clips = f"  alpha {[round(float(c.alpha.numpy()), 2) for c in cells]}" if a.c_clip else ""
            tqdm.write(f"epoch {ep}: train {np.mean(run):.3f}  val {v:.3f} dB  units {lu}  "
                       f"{a.steps / dt:.2f} step/s ({dt:.0f} s)  max |g| {g_max:.3g}, skipped {n_skip}{clips}{mark}")
    tot = time.time() - t_all

    # finalise: for a learned-pruning run the LAST epoch is the operating point (the masks
    # are what was trained for); otherwise the best-validation epoch.
    final_ep = a.epochs if (a.prune_lambda > 0 or a.prune_basis_lambda > 0) else best_ep
    if final_ep > 0:
        net.load_weights(a.out / f"epoch-{final_ep:02d}.weights.h5")
    live = M.finalize_pruning(net) if a.prune_lambda > 0 else M.live_units(net)
    nb = M.finalize_basis(net) if a.prune_basis_lambda > 0 else M.live_basis(net)
    params = M.pruned_param_count(net)
    v_final = validate("val @ final (masks hard)")
    M.save_authors_h5(net, a.out / "best.authors.weights.h5")
    summary = dict(mode=mode, wq_only=a.wq_only, epochs=a.epochs, steps=a.steps, batch=a.batch, seg_seconds=seg / data.RATE,
                   start_val=v0, best_val=best, best_epoch=best_ep, final_epoch=final_ep, final_val=v_final,
                   live_units=live, live_basis=nb, dense_params=params, dense_params_full=986_753,
                   fq_sites=(a.fq_sites if a.fq else None), prune_basis_lambda=a.prune_basis_lambda,
                   fq_scheme=(None if not a.fq else "dynamic" if a.fq_dynamic else "fixed"), fq_bits=a.fq_bits, calibration=calib,
                   unstructured=unstructured, lstm_weight_nonzeros=M.lstm_weight_nonzeros(net),
                   fq_clips=M.fq_clips(net) if (a.z_clip > 0 or a.c_clip) else None,
                   minutes=tot / 60, steps_per_s=a.steps * a.epochs / tot,
                   train_utterances=len(tr), val_speakers=sorted(set(va.speakers)))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"done in {tot / 60:.1f} min: start {v0:.3f} -> best {best:.3f} (epoch {best_ep}) -> final (epoch {final_ep}, "
          f"hard masks) {v_final:.3f} dB; live units {live}, basis {nb}, dense params {params:,} of 986,753 "
          f"({100 * params / 986_753:.1f} %); exported {a.out / 'best.authors.weights.h5'}")


if __name__ == "__main__":
    main()
