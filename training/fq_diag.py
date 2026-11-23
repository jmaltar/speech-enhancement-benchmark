"""Which fake-quant site makes the gradients explode? (2026-10-20)

With --fq the per-step gradient norm reaches 1e9-1e12 against ~5 in float, although
the straight-through estimator is the identity. One fixed batch at the fine-tuned
weights; the six fake_quant call sites in LSTMCellC.call (W, R, x, h, c, z) are
enabled one at a time and all together, and the global gradient norm and the
largest per-variable norm are printed. Call order inside the cell is static, so
the sites are identified by a trace-time counter.

    source gpu_env.sh; python -m training.fq_diag [--seg-seconds 4] [--batch 32]
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training import data, model as M  # noqa: E402
from training.train import snr_loss  # noqa: E402

SITES = ["W", "R", "x", "h", "c", "z"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, default=Path("training/runs/ft-f32/best.authors.weights.h5"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seg-seconds", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=600)
    a = ap.parse_args(argv)
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    tr, _ = data.load_train_val(limit=a.limit)
    seg = int(a.seg_seconds * data.RATE); seg = ((seg - M.FRAME) // M.HOP) * M.HOP + M.FRAME
    x, y = data.segments(tr, a.batch, seg, np.random.default_rng(1))
    x, y = tf.constant(x), tf.constant(y)
    net = M.load_pretrained(a.init, dropout=0.0)
    cells = M.lstm_cells(net)
    for c in cells:
        c.fq = True
    real_fq = M.fake_quant
    state = {"on": set(), "n": 0}

    def selective(t, bits=8):
        site = SITES[state["n"] % 6]; state["n"] += 1
        return real_fq(t, bits) if site in state["on"] else t
    M.fake_quant = selective

    configs = [("none", set()), ("all", set(SITES))] + [(s, {s}) for s in SITES] + [("W+R", {"W", "R"}), ("h+c", {"h", "c"})]
    print(f"{'sites':>6} {'loss':>9} {'|g| global':>12} {'largest variable':>34} {'its |g|':>10}")
    for name, on in configs:
        state["on"] = on; state["n"] = 0

        @tf.function
        def step(x, y):
            with tf.GradientTape() as tape:
                loss = snr_loss(y, net(x, training=False))
            grads = tape.gradient(loss, net.trainable_variables)
            norms = [tf.sqrt(tf.reduce_sum(tf.square(tf.cast(g, tf.float64)))) for g in grads if g is not None]
            return loss, tf.sqrt(tf.add_n([n * n for n in norms])), tf.stack(norms)
        loss, gn, norms = step(x, y)
        vars_ = [v for v, g in zip(net.trainable_variables, [1] * len(net.trainable_variables))]
        k = int(np.argmax(norms.numpy()))
        print(f"{name:>6} {float(loss):9.3f} {float(gn):12.3g} {vars_[k].path:>34} {float(norms[k]):10.3g}")


if __name__ == "__main__":
    main()
