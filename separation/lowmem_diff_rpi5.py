"""RPi5, 2026-10-29: how far the lowmem forward is from the shipped forward on this torch build, in float32
and after the PCM_16 quantisation the estimates are written with. Run from ~/benchmark/sep."""
import sys, time, resource
import numpy as np, soundfile as sf, torch, openunmix
sys.path.insert(0, ".")
from umx_run import forward_lowmem, RATE
torch.set_num_threads(4); torch.set_grad_enabled(False)
sep = openunmix.umxhq(targets=None, residual=False, niter=1, device="cpu", pretrained=True); sep.eval()
targets = list(sep.target_models.keys())
print("torch", torch.__version__, "targets", targets, flush=True)
def q16(x):
    return np.clip(np.rint(np.clip(x, -1.0, 32767.0 / 32768.0) * 32768.0), -32768, 32767).astype(np.int32)
def report(label, ref, got):
    d = (ref - got).abs()
    print(f"{label}: float max|d| {float(d.max()):.3e}, mean|d| {float(d.mean()):.3e}, "
          f"{float((d > 0).float().mean()) * 100:.2f} % of float samples differ, torch.equal {bool(torch.equal(ref, got))}")
    r, g = ref[0].numpy(), got[0].numpy()          # (src, ch, n)
    vi = targets.index("vocals"); acc = [i for i in range(len(targets)) if i != vi]
    for name, a, b in (("vocals", r[vi], g[vi]), ("accompaniment", r[acc].sum(0), g[acc].sum(0))):
        dd = np.abs(q16(a) - q16(b)); n = np.count_nonzero(dd)
        print(f"  {name}: PCM_16 max {dd.max()} LSB, {n} of {dd.size} samples differ ({n / dd.size * 100:.4f} %)", flush=True)
mix = __import__("os").path.expanduser("~/benchmark/musdb_test")
for track, secs in (("AM Contra - Heart Peripheral", 10), ("PR - Oh No", None), ("AM Contra - Heart Peripheral", None)):
    x, sr = sf.read(f"{mix}/{track}/mixture.wav", always_2d=True, dtype="float32"); assert sr == RATE
    if secs: x = x[: secs * RATE]
    audio = torch.from_numpy(x.T.copy())[None]
    t = time.perf_counter(); ref = sep(audio); t1 = time.perf_counter() - t
    t = time.perf_counter(); got = forward_lowmem(sep, audio); t2 = time.perf_counter() - t
    report(f"{track} {'%d s' % secs if secs else 'full %.1f s' % (x.shape[0] / RATE)} (shipped {t1:.1f} s, lowmem {t2:.1f} s)", ref, got)
    got2 = forward_lowmem(sep, audio)
    print(f"  lowmem run-to-run torch.equal: {bool(torch.equal(got, got2))}")
    ref2 = sep(audio)
    print(f"  shipped run-to-run torch.equal: {bool(torch.equal(ref, ref2))}")
    del ref, got, got2, ref2, audio
    print(f"  max RSS so far {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB", flush=True)
