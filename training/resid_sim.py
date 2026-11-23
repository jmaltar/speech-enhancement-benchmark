"""Per-site ablation of the 16x8 export residual on the pruned DTLN (2026-10-22).

A numpy re-execution of the shrunk two-stage graph, op for op as the deployed
16x8 flatbuffer lays it out (model_i16pt_*.tflite), in float with OPTIONAL
quantization at each tensor using the flatbuffer's OWN scales: int16 symmetric
for activations, the int8 weights/scales the converter wrote for weights.
Every quantization point belongs to a named GROUP; a run switches groups on.
  all groups on  -> should reproduce the TFLM i16pt output (anchor A)
  all groups off -> should reproduce the LiteRT f32 output   (anchor B)
Interface matches dtln.oracle.DtlnTflite so oracle.enhance() drives it under
the ABF1 policy. Read-only on the project; writes WAVs to --out.
"""
from __future__ import annotations
import argparse, multiprocessing as mp, os, sys, time
from pathlib import Path
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]   # the project root
sys.path.insert(0, str(ROOT))
from dtln import oracle  # noqa: E402

SHRUNK = ROOT / "training/runs/prune-learned/shrunk"
GROUPS = ["s1_in", "s1_w", "s1_z", "s1_act", "s1_c", "s1_h", "s1_mask",
          "s2_in", "s2_encw", "s2_enc", "s2_lstm", "s2_mask", "s2_est", "s2_decw", "s2_out"]
QMAX16 = 32767.0

def _details(path):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path)); it.allocate_tensors()
    det = {d["index"]: d for d in it.get_tensor_details()}
    return it, det

def _scale(det, idx):
    s = det[idx]["quantization_parameters"]["scales"]
    return float(s[0]) if len(s) == 1 else np.asarray(s, np.float32)

class Weights:
    """float weights from the f32 pair, int8 weights + scales from the i16pt pair, by tensor index
    (the two exports share op order and tensor indices: dump_graph.py)."""
    def __init__(self):
        f1, d1 = _details(SHRUNK / "model_f32_1.tflite"); q1, e1 = _details(SHRUNK / "model_i16pt_1.tflite")
        f2, d2 = _details(SHRUNK / "model_f32_2.tflite"); q2, e2 = _details(SHRUNK / "model_i16pt_2.tflite")
        def pair(itf, itq, detq, idx):
            wf = itf.get_tensor(idx).astype(np.float32); wq = itq.get_tensor(idx)
            s = _scale(detq, idx)
            if wq.dtype == np.int8 and np.ndim(s) == 1:      # per-channel (conv): scale per output channel (axis 0)
                wdq = wq.astype(np.float32) * s.reshape((-1,) + (1,) * (wq.ndim - 1))
            else:
                wdq = wq.astype(np.float32) * s
            return wf, wdq
        # stage 1
        self.R0 = pair(f1, q1, e1, 12); self.W0 = pair(f1, q1, e1, 11); self.b0 = pair(f1, q1, e1, 14)
        self.R1 = pair(f1, q1, e1, 10); self.W1 = pair(f1, q1, e1, 9);  self.b1 = pair(f1, q1, e1, 13)
        self.Wd1 = pair(f1, q1, e1, 8); self.bd1 = pair(f1, q1, e1, 5)
        self.s1 = {k: _scale(e1, i) for k, i in dict(mag=3, h0=2, c0=1, h1=0, c1=4, zh0=19, zx0=21, zs0=22, zb0=23,
                   fc0=30, ig0=33, c0o=34, h0o=36, zh1=39, zx1=41, zs1=42, zb1=43, fc1=50, ig1=53, c1o=54, h1o=56,
                   d1=58, m1=59).items()}
        # stage 2
        self.We = pair(f2, q2, e2, 13); self.be = pair(f2, q2, e2, 14)
        self.R3 = pair(f2, q2, e2, 17); self.b3 = pair(f2, q2, e2, 18)
        self.Wd2 = pair(f2, q2, e2, 15); self.bd2 = pair(f2, q2, e2, 5)
        self.Wdec = pair(f2, q2, e2, 8); self.bdec = pair(f2, q2, e2, 9)
        self.s2 = {k: _scale(e2, i) for k, i in dict(fr=0, h3=1, c3=2, enc=28, zh3=44, zx3=46, zs3=47, zb3=48, fc3=55, ig3=58,
                   c3o=59, h3o=61, d2=63, m2=64, est=65, out=67).items()}
        # sanity: the dead lstm2 path feeds lstm3 a zero: its input kernel must be all zero
        assert not np.any(f2.get_tensor(16)), "lstm3 input kernel not zero"

def q16(x, s):
    return np.clip(np.rint(x / s), -32768, QMAX16) * s

class Sim:
    def __init__(self, W: Weights, on: set[str]):
        self.W = W; self.on = on
        self.reset()
    def reset(self):
        self.h0 = np.zeros(30, np.float32); self.c0 = np.zeros(30, np.float32)
        self.h1 = np.zeros(58, np.float32); self.c1 = np.zeros(58, np.float32)
        self.h3 = np.zeros(11, np.float32); self.c3 = np.zeros(11, np.float32)
    def Q(self, x, s, g):
        return q16(x, s) if g in self.on else x
    def w(self, pair, g):
        return pair[1] if g in self.on else pair[0]
    def lstm(self, x, h, c, Wp, Rp, bp, s, k, gw, gz, ga, gc, gh, zx_fixed=None):
        R, b = self.w(Rp, gw), self.w(bp, gw)
        zh = self.Q(R @ h, s[f"zh{k}"], gz)
        zx = self.Q(self.w(Wp, gw) @ x, s[f"zx{k}"], gz) if zx_fixed is None else zx_fixed
        z = self.Q(zx + zh, s[f"zs{k}"], gz)
        z = self.Q(z + b, s[f"zb{k}"], gz)
        U = len(h); i, f, g, o = z[:U], z[U:2*U], z[2*U:3*U], z[3*U:]
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))
        si, sf, so = (self.Q(sig(v), 1 / 32768, ga) for v in (i, f, o))
        tg = self.Q(np.tanh(g), 1 / 32768, ga)
        fc = self.Q(sf * c, s[f"fc{k}"], gc)
        ig = self.Q(si * tg, s[f"ig{k}"], gc)
        c2 = self.Q(fc + ig, s[f"c{k}o"], gc)
        tc = self.Q(np.tanh(c2), 1 / 32768, ga)
        h2 = self.Q(so * tc, s[f"h{k}o"], gh)
        # the states cross the stage boundary: requantized at the INPUT tensor's scale next frame (oracle._q)
        return h2.astype(np.float32), c2.astype(np.float32)
    def stage1(self, mag):
        W, s = self.W, self.W.s1; on = self.on
        x = self.Q(mag.reshape(-1).astype(np.float32), s["mag"], "s1_in")
        h0 = self.Q(self.h0, s["h0"], "s1_h"); c0 = self.Q(self.c0, s["c0"], "s1_c")
        self.h0, self.c0 = self.lstm(x, h0, c0, W.W0, W.R0, W.b0, s, 0, "s1_w", "s1_z", "s1_act", "s1_c", "s1_h")
        h1 = self.Q(self.h1, s["h1"], "s1_h"); c1 = self.Q(self.c1, s["c1"], "s1_c")
        self.h1, self.c1 = self.lstm(self.h0, h1, c1, W.W1, W.R1, W.b1, s, 1, "s1_w", "s1_z", "s1_act", "s1_c", "s1_h")
        d = self.Q(self.w(W.Wd1, "s1_w") @ self.h1 + self.w(W.bd1, "s1_w"), s["d1"], "s1_mask")
        m = self.Q(1.0 / (1.0 + np.exp(-d)), 1 / 32768, "s1_mask")
        return m.reshape(1, 1, -1).astype(np.float32)
    def stage2(self, est_frame):
        W, s = self.W, self.W.s2
        fr = self.Q(est_frame.reshape(-1).astype(np.float32), s["fr"], "s2_in")
        We = self.w(W.We, "s2_encw").reshape(256, 512); be = self.w(W.be, "s2_encw")
        enc = self.Q(We @ fr + be, s["enc"], "s2_enc")
        h3 = self.Q(self.h3, s["h3"], "s2_lstm"); c3 = self.Q(self.c3, s["c3"], "s2_lstm")
        self.h3, self.c3 = self.lstm(None, h3, c3, None, W.R3, W.b3, s, 3, "s2_lstm", "s2_lstm", "s2_lstm", "s2_lstm", "s2_lstm",
                                     zx_fixed=np.zeros(44, np.float32))
        d = self.Q(self.w(W.Wd2, "s2_mask") @ self.h3 + self.w(W.bd2, "s2_mask"), s["d2"], "s2_mask")
        m2 = self.Q(1.0 / (1.0 + np.exp(-d)), 1 / 32768, "s2_mask")
        est = self.Q(enc * m2, s["est"], "s2_est")
        Wd = self.w(W.Wdec, "s2_decw").reshape(512, 256); bd = self.w(W.bdec, "s2_decw")
        out = self.Q(Wd @ est + bd, s["out"], "s2_out")
        return out.reshape(1, 1, -1).astype(np.float32)
    def process_frame(self, frame):
        spec = np.fft.rfft(frame)
        mag = np.abs(spec).reshape(1, 1, -1).astype(np.float32)
        phase = np.angle(spec)
        mask = self.stage1(mag)
        est = np.fft.irfft(mag * mask * np.exp(1j * phase), n=oracle.FRAME).reshape(1, 1, -1).astype(np.float32)
        return self.stage2(est).reshape(-1)

_W = None
def _init(on):
    global _W, _ON
    _W = Weights(); _ON = on

def _job(a):
    uid, noisy, out = a
    rate, x = oracle.read_wav_int16(noisy)
    y = oracle.enhance(x, Sim(_W, _ON))
    oracle.write_wav_int16(out, rate, y)
    return uid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", default="", help="comma list of groups to quantize, or 'all'")
    ap.add_argument("--off", default="", help="with --on all: groups to leave float")
    ap.add_argument("--out", type=Path); ap.add_argument("--variant", default="sim")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    on = set(GROUPS) if a.on == "all" else set(filter(None, a.on.split(",")))
    on -= set(filter(None, a.off.split(",")))
    assert on <= set(GROUPS), on - set(GROUPS)
    from eval_harness import manifest
    us = manifest.utterances(manifest.load(), "voicebank16k")
    if a.check:
        W = Weights()
        oracle.VARIANTS["_pt"] = ("model_i16pt_1.tflite", "model_i16pt_2.tflite")
        oracle.VARIANTS["_f"] = ("model_f32_1.tflite", "model_f32_2.tflite")
        tflm = oracle.DtlnTflm("_pt", model_dir=SHRUNK); f32 = oracle.DtlnTflite("_f", model_dir=SHRUNK)
        for u in (us[0], us[300], us[700]):
            x = oracle.read_wav_int16(u.noisy_path)[1]
            yq = oracle.enhance(x, tflm).astype(int); yf = oracle.enhance(x, f32).astype(int)
            t0 = time.time(); ya = oracle.enhance(x, Sim(W, set(GROUPS))).astype(int); dt = time.time() - t0
            yn = oracle.enhance(x, Sim(W, set())).astype(int)
            def st(a, b):
                d = np.abs(a - b); return f"max {d.max()} LSB, {100 * np.mean(d > 0):.1f} % differ, nmse {10 * np.log10(np.sum(d ** 2) / max(np.sum(b ** 2), 1)):.1f} dB"
            print(f"{u.id} ({len(x)} samples, sim {dt:.2f} s):\n  all-on  vs TFLM i16pt : {st(ya, yq)}\n  all-off vs LiteRT f32 : {st(yn, yf)}\n"
                  f"  TFLM i16pt vs f32     : {st(yq, yf)}")
        return
    a.out.mkdir(parents=True, exist_ok=True)
    if a.limit:
        us = us[:a.limit]
    jobs = [(u.id, u.noisy_path, a.out / f"{u.id}.{a.variant}.wav") for u in us]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init, initargs=(on,)) as pool:
        n = sum(1 for _ in pool.imap_unordered(_job, jobs, chunksize=4))
    print(f"{a.variant}: on={sorted(on)} -> {n} files in {time.time() - t0:.0f} s -> {a.out}")

if __name__ == "__main__":
    main()
