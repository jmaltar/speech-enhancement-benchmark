#!/usr/bin/env python3
"""umx_run.py -- Open-Unmix over the MUSDB18-HQ test mixtures, host or Pi (task 11).

The vocal-isolation tier's separator, run as the other arms are run: the
model as its authors shipped it (Arm A), every track of the test split, the
separation call alone inside the timed region, the estimates written beside
the CSV in the layout eval_harness/separation.py scores
(<out>/<track>/vocals.wav + accompaniment.wav, 44.1 kHz stereo PCM_16).

    python separation/umx_run.py --mix-root /mnt/data/benchmark-data/musdb18hq/test \
        --out /mnt/data/benchmark-data/enhanced/umxhq/musdb18hq --model umxhq --threads 8

Same file on the Pi (its venv under ~/benchmark/sep). What is timed is
Separator.forward() on the whole track held in memory -- STFT, the four
target models, the one-iteration Wiener filter and the inverse transform --
never the file I/O. One untimed warm-up call precedes the first track (lazy
initialisation, as the harness's untimed prep). Thread count, core affinity,
governor and throttle flags are read and written to the CSV, not assumed;
the weight files' SHA-256 go to <out>/provenance.json so the Pi and the host
can be shown to have run byte-identical weights.

Quality is measured on the host (the one-oracle rule); the Pi contributes
wall-clock RTF, which for a torch workload is the cost figure. --limit draws
at random with the module's seed, never the head (trap 14), and labels the
CSV as a smoke.

Every CSV row is flushed and fsync'd as it is written, and --resume skips the
tracks the CSV already holds an "ok" row for (and re-separates any track whose
row is missing or failed), so an interrupted run loses at most the track in
flight -- the 2026-10-28 host run died with four tracks done and a 0-byte CSV
because the rows sat in a buffer. Launch long runs as a systemd-run --user
unit, not a shell job (trap 21).

Memory: the authors' Separator.forward() holds the four-source STFT twice
(a permute().contiguous() copy) and inverse-transforms all four sources in
one batched torch.istft, which peaks near 10 GB on a 400 s track and was
OOM-killed on the 14.3 GB host (2026-10-28, track 37/50). The default
--forward lowmem path runs the same calls on the same tensors but frees the
spectrograms before the inverse transform and inverse-transforms one source
at a time; --verify-lowmem runs both paths on the warm-up excerpt and on the
first track and refuses to continue unless the outputs are torch.equal
(bit-identical), so the estimates are still the as-shipped model's. The
provenance file records which path ran and the OUTCOME of the check. Measured:
bit-identical on the host (torch 2.14.1+cu130, x86-64, full 210 s track);
NOT bit-identical on the RPi5 (torch 2.15.0+cpu, AArch64): max |d| 2.2e-8 in
float32 on the warm-up excerpt, i.e. last-bit differences between the batched
and the per-source istft, so the Pi's batch ran without the flag and its
provenance says so (2026-10-29).
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import openunmix
from openunmix.filtering import wiener

RATE = 44_100
INJECT_SEED = 20260915          # eval_harness/separation.py's draw seed, reused for --limit
TARGET = "vocals"


def read_conditions() -> dict:
    d = {"threads": torch.get_num_threads(), "affinity": sorted(os.sched_getaffinity(0)),
         "torch": torch.__version__, "machine": platform.machine(), "node": platform.node()}
    try:
        d["governor"] = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        d["governor"] = "n/a"
    try:
        d["throttled"] = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True,
                                        timeout=5).stdout.strip()
    except Exception:
        d["throttled"] = "n/a"
    return d


def weight_hashes(model: str) -> dict:
    out = {}
    ck = Path(torch.hub.get_dir()) / "checkpoints"
    if ck.exists():
        for p in sorted(ck.glob("*.pth")):
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def forward_lowmem(sep, audio: torch.Tensor) -> torch.Tensor:
    """openunmix.model.Separator.forward, same calls in the same order, lower peak memory.

    Differences from the shipped method, none of which change a value: the
    magnitude and source spectrograms are dropped once the Wiener filter has
    consumed them, the (frames, bins, ch, 2, src) -> (src, ch, bins, frames, 2)
    permute is made contiguous one source at a time, and torch.istft runs per
    source instead of over the stacked batch (istft is row-independent)."""
    nb_sources = sep.nb_targets
    mix_stft = sep.stft(audio)                                  # (1, ch, bins, frames, 2)
    X = sep.complexnorm(mix_stft)
    spectrograms = torch.zeros(X.shape + (nb_sources,), dtype=audio.dtype, device=X.device)
    for j, target_module in enumerate(sep.target_models.values()):
        spectrograms[..., j] = target_module(X.detach().clone())
    del X
    spectrograms = spectrograms.permute(0, 3, 2, 1, 4)          # (1, frames, bins, ch, src)
    mix_stft = mix_stft.permute(0, 3, 2, 1, 4)                  # (1, frames, bins, ch, 2)
    if sep.residual:
        nb_sources += 1
    if nb_sources == 1 and sep.niter > 0:
        raise Exception("Cannot use EM if only one target is estimated.")
    nb_frames = spectrograms.shape[1]
    targets_stft = torch.zeros(mix_stft.shape + (nb_sources,), dtype=audio.dtype, device=mix_stft.device)
    win = sep.wiener_win_len if sep.wiener_win_len else nb_frames
    pos = 0
    while pos < nb_frames:
        cur_frame = torch.arange(pos, min(nb_frames, pos + win))
        pos = int(cur_frame[-1]) + 1
        targets_stft[0, cur_frame] = wiener(spectrograms[0, cur_frame], mix_stft[0, cur_frame], sep.niter,
                                            softmask=sep.softmask, residual=sep.residual)
    del spectrograms, mix_stft
    out = []
    for j in range(nb_sources):
        t = targets_stft[..., j].permute(0, 3, 2, 1, 4).contiguous()   # (1, ch, bins, frames, 2)
        out.append(sep.istft(t, length=audio.shape[2]))                 # (1, ch, n)
        del t
    del targets_stft
    return torch.stack(out, dim=1)                                      # (1, src, ch, n)


def write_pcm16(path: Path, x: np.ndarray):
    x = np.clip(x, -1.0, 32767.0 / 32768.0)
    sf.write(str(path), x.T if x.shape[0] == 2 else x, RATE, subtype="PCM_16")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix-root", type=Path, required=True, help="<track>/mixture.wav per test track")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="umxhq", choices=["umxhq", "umx", "umxl"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--niter", type=int, default=1, help="Wiener EM iterations (authors' default 1)")
    ap.add_argument("--limit", type=int, default=0, help="smoke only: random draw, labelled")
    ap.add_argument("--tag", default="")
    ap.add_argument("--forward", default="lowmem", choices=["lowmem", "shipped"],
                    help="lowmem (default): same arithmetic, half the peak RAM; shipped: the authors' method")
    ap.add_argument("--verify-lowmem", action="store_true",
                    help="run both forward paths on the warm-up excerpt and the first track; abort unless bit-identical")
    ap.add_argument("--resume", action="store_true",
                    help="keep the existing CSV and skip every track it already has an 'ok' row for")
    ap.add_argument("--all-stems", action="store_true",
                    help="also write drums/bass/other.wav (the host's quality pass: lets the standard "
                         "four-target museval anchor be computed beside the two-source row)")
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    ctor = getattr(openunmix, a.model)
    sep = ctor(targets=None, residual=False, niter=a.niter, device="cpu", pretrained=True)
    sep.eval()
    targets = list(sep.target_models.keys())
    if TARGET not in targets:
        sys.exit(f"{a.model} has no '{TARGET}' target: {targets}")

    tracks = sorted(p.parent for p in a.mix_root.glob("*/mixture.wav"))
    if not tracks:
        sys.exit(f"no <track>/mixture.wav under {a.mix_root}")
    if a.limit and a.limit < len(tracks):
        rng = np.random.default_rng(INJECT_SEED)
        idx = np.sort(rng.choice(len(tracks), size=a.limit, replace=False))
        tracks = [tracks[i] for i in idx]
    a.out.mkdir(parents=True, exist_ok=True)
    cond = read_conditions()
    if cond["governor"] not in ("performance", "n/a"):
        print(f"warning: governor is {cond['governor']!r}, not 'performance' -- the RTF column of this run "
              "is not a quotable cost figure", flush=True)
    prov = {"model": a.model, "niter": a.niter, "targets": targets, "weights_sha256": weight_hashes(a.model),
            "conditions": cond, "openunmix": getattr(openunmix, "__version__", "1.3.0"),
            "forward": a.forward,
            # the OUTCOME of the check, never the flag that requested it (trap 23: the Pi's provenance read
            # 'true' for a verification that had failed, because the flag was recorded before the check ran)
            "verified_lowmem": False, "lowmem_check": "not run"}
    print("conditions:", json.dumps(cond))

    # untimed warm-up on 10 s of the first track (lazy init, allocator, STFT plan)
    x0, sr = sf.read(str(tracks[0] / "mixture.wav"), always_2d=True, dtype="float32")
    assert sr == RATE
    run = (lambda audio: forward_lowmem(sep, audio)) if a.forward == "lowmem" else sep
    _ = run(torch.from_numpy(x0[: 10 * RATE].T.copy())[None])
    if a.verify_lowmem:
        for label, xa in (("warm-up 10 s", x0[: 10 * RATE]), (f"first track ({tracks[0].name})", x0)):
            audio = torch.from_numpy(xa.T.copy())[None]
            ref = sep(audio); got = forward_lowmem(sep, audio)
            same = ref.shape == got.shape and bool(torch.equal(ref, got))
            print(f"verify lowmem vs shipped on {label}: shape {tuple(got.shape)} "
                  f"{'bit-identical' if same else 'DIFFER, max |d| %.3e' % float((ref - got).abs().max())}",
                  flush=True)
            if not same:
                prov["lowmem_check"] = f"DIFFER on {label}: max |d| {float((ref - got).abs().max()):.3e}"
                (a.out / "provenance.json").write_text(json.dumps(prov, indent=1))
                sys.exit("lowmem forward is not bit-identical to the shipped forward -- refusing to run")
            del ref, got, audio
        prov["verified_lowmem"] = True
        prov["lowmem_check"] = f"bit-identical (torch.equal) on the warm-up 10 s and on the full first track ({tracks[0].name})"
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=1))

    tag = f"{a.model}{('-' + a.tag) if a.tag else ''}_t{a.threads}{'_smoke' if a.limit else ''}"
    csv_path = a.out / f"umx_run_{tag}.csv"
    fields = ["track", "seconds", "wall_s", "proc_s", "rtf", "threads", "affinity", "governor", "throttled",
              "vocals_peak", "status"]
    done: set[str] = set()
    if a.resume and csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
        done = {r["track"] for r in rows}
        # rewrite with the ok rows only, so a failed row is re-run rather than duplicated
        with open(csv_path, "w", newline="") as f:
            w0 = csv.DictWriter(f, fieldnames=fields); w0.writeheader(); w0.writerows(rows)
        print(f"resume: {len(done)} of {len(tracks)} tracks already in {csv_path.name}", flush=True)
    fcsv = open(csv_path, "a" if done else "w", newline="")
    w = csv.DictWriter(fcsv, fieldnames=fields)
    if not done:
        w.writeheader()
    fcsv.flush(); os.fsync(fcsv.fileno())
    t_all = time.perf_counter()
    for k, t in enumerate(tracks, 1):
        if t.name in done:
            print(f"[{k}/{len(tracks)}] {t.name[:40]:40s} skipped (resume)", flush=True)
            continue
        row = dict(track=t.name, threads=a.threads, affinity=str(cond["affinity"]), governor=cond["governor"],
                   throttled=cond["throttled"], status="ok")
        try:
            x, sr = sf.read(str(t / "mixture.wav"), always_2d=True, dtype="float32")
            assert sr == RATE and x.shape[1] == 2
            audio = torch.from_numpy(x.T.copy())[None]           # (1, 2, n)
            row["seconds"] = round(x.shape[0] / RATE, 3)
            # ---- timed region: the separation only ----
            tp0 = time.process_time_ns(); t0 = time.perf_counter_ns()
            est = run(audio)                                      # (1, targets, 2, n)
            t1 = time.perf_counter_ns(); tp1 = time.process_time_ns()
            # -------------------------------------------
            est = est[0].numpy()
            voc = est[targets.index(TARGET)]
            acc = est[[i for i, n in enumerate(targets) if n != TARGET]].sum(axis=0)
            d = a.out / t.name
            d.mkdir(exist_ok=True)
            write_pcm16(d / "vocals.wav", voc)
            write_pcm16(d / "accompaniment.wav", acc)
            if a.all_stems:
                for i, n in enumerate(targets):
                    if n != TARGET:
                        write_pcm16(d / f"{n}.wav", est[i])
            row.update(wall_s=round((t1 - t0) / 1e9, 3), proc_s=round((tp1 - tp0) / 1e9, 3),
                       rtf=round((t1 - t0) / 1e9 / (x.shape[0] / RATE), 4),
                       vocals_peak=round(float(np.abs(voc).max()), 4))
        except Exception as e:
            row["status"] = f"{type(e).__name__}: {e}"
        w.writerow(row)
        fcsv.flush(); os.fsync(fcsv.fileno())
        print(f"[{k}/{len(tracks)}] {t.name[:40]:40s} {row.get('seconds', '?')} s  wall {row.get('wall_s', '?')} s"
              f"  rtf {row.get('rtf', '?')}  {row['status']}", flush=True)
    fcsv.close()
    print(f"done: {len(tracks)} tracks in {time.perf_counter() - t_all:.0f} s -> {csv_path}  "
          f"(max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB -- the process, "
          "not the cgroup: Raspberry Pi OS boots with cgroup_disable=memory)", flush=True)


if __name__ == "__main__":
    main()
