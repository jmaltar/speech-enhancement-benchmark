#!/usr/bin/env python3
"""
abf_batch_file.py -- the corpus batch driver for the FILE transport (the RPi5).

The file-transport twin of `abf_host.py batch`: same manifest filter, same
stream builder, same reassembly, same comparison against the host oracle and
the same CSV columns, so a Pi row and an MCU row are read with one script. What
is swapped is the transport: instead of a serial port, each stream is written
to a file and handed to `cdsp_run`, which selects the processor from
CONFIG.algo_id through the shared registry and runs the ABF1 file binding.

Two columns are tier-specific and are filled accordingly, never faked:
  e2e_median_us  -- empty. End-to-end latency has no comparable value on a tier
                    fed by files (CLAUDE.md, Audio I/O Methodology).
  duty_cycle     -- proc_ticks_sum / (frames_out * hop_ns): the active fraction
                    of the audio time, the per-run log the power section wants
                    from the first run. Meaningful for paced and unpaced alike,
                    since it is compute time over audio time.
Three more are appended: `void` (the binding's own verdict on its conditions),
`sched` (what cdsp_run witnessed: policy, core, rtprio) and `paced`.

Paths in the manifest are the host's. `--remap OLD=NEW` rewrites a prefix so
the same manifest serves a working set staged on the Pi's tmpfs.
"""
import argparse
import csv
import os
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abf_host as H                                           # noqa: E402


def per_hop_proc(buf, data_hops):
    """proc ns per DATA hop (seq < data_hops) from the RESULT timing headers,
    the same quantity abf_host.run_stream collects as proc_data_hops."""
    proc = []
    for typ, seq, pl in H.unpack(buf):
        if typ == H.T["RESULT"] and seq < data_hops and len(pl) >= H.TIMING_BYTES:
            tm = struct.unpack("<IIIIII", pl[:H.TIMING_BYTES])
            proc.append((tm[2] - tm[1]) & 0xFFFFFFFF)
    return proc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest"); p.add_argument("outdir")
    p.add_argument("--runner", required=True, help="path to cdsp_run")
    p.add_argument("--corpus", default="voicebank16k"); p.add_argument("--role", default="eval")
    p.add_argument("--algo", choices=sorted(H.ALGO), default="bypass")
    p.add_argument("--noise", choices=sorted(H.NOISE), default="default")
    p.add_argument("--paced", action="store_true")
    p.add_argument("--frame", type=int, default=0); p.add_argument("--hop", type=int, default=0)
    p.add_argument("--ref-dir", help="host-oracle WAVs named <id>.wav")
    p.add_argument("--remap", action="append", default=[], help="OLD=NEW path prefix rewrite")
    p.add_argument("--limit", type=int, default=0, help="smoke test only: head of the manifest")
    p.add_argument("--keep", action="store_true"); p.add_argument("--every", type=int, default=100)
    p.add_argument("--core", type=int, default=2); p.add_argument("--rtprio", type=int, default=50)
    p.add_argument("--build", default="unknown"); p.add_argument("--platform", default="RPI5")
    p.add_argument("--no-measure", action="store_true", help="functional run only; stamped VOID")
    p.add_argument("--stages-log", help="append each run's stage-stamp lines (stdout lines starting 'stages' or 'rn_stages', printed only by stamped run tools) to this file, one per utterance")
    a = p.parse_args()

    def remap(path):
        for m in a.remap:
            old, new = m.split("=", 1)
            if path.startswith(old):
                return new + path[len(old):]
        return path

    rows = [r for r in csv.DictReader(open(a.manifest))
            if r["corpus"] == a.corpus and r["role"] == a.role]
    if not rows:
        sys.exit(f"no rows for corpus={a.corpus} role={a.role}")
    if a.limit:
        rows = rows[:a.limit]            # biased head cut: smoke test only (trap 14)
    algo = H.ALGO[a.algo] | (H.NOISE[a.noise] << 8)
    variant = H.variant_name(a.algo, a.noise)
    os.makedirs(a.outdir, exist_ok=True)
    work = os.path.join(a.outdir, "work"); os.makedirs(work, exist_ok=True)
    tag = f"{a.corpus}_{variant}{'_paced' if a.paced else ''}{'_smoke' if a.limit else ''}"
    csv_path = os.path.join(a.outdir, tag + ".csv")
    fields = ["id", "variant", "hops", "nres", "overruns", "crc_errors", "resyncs",
              "proc_min", "proc_max", "proc_mean", "proc_median_data", "e2e_median_us",
              "ref", "max_abs_lsb", "n_diff", "nmse_db", "status",
              "duty_cycle", "void", "sched", "paced"]
    w = csv.DictWriter(open(csv_path, "w", newline=""), fieldnames=fields)
    w.writeheader()
    t_start = time.perf_counter()
    worst = dict(max_abs_lsb=0, n_diff=0, nmse_db=float("-inf"), overruns=0, fails=0, void=0)
    hello_seen = False
    for k, r in enumerate(rows, 1):
        uid = r["id"]
        rate, ch, raw = H.read_wav(remap(r["noisy_path"]))
        frame = a.frame or rate * 32 // 1000
        hop = a.hop or frame // 2
        stream, n, dh = H.build_stream(rate, ch, raw, frame, hop, a.paced, algo, variant, uid[:23])
        inp = os.path.join(work, f"{uid}.{variant}.in.abf")
        outp = os.path.join(work, f"{uid}.{variant}.out.abf")
        open(inp, "wb").write(stream)
        row = dict(id=uid, variant=variant, hops=dh, ref="", status="ok", e2e_median_us="",
                   paced=int(a.paced))
        cmd = [a.runner, "--in", inp, "--out", outp, "--core", str(a.core),
               "--rtprio", str(a.rtprio), "--build", a.build, "--platform", a.platform]
        if not a.no_measure:
            cmd.append("--measure")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            row["status"] = "runner failed: " + (res.stderr.strip() or res.stdout.strip())[:120]
            worst["fails"] += 1; w.writerow(row)
            print(f"[{k}/{len(rows)}] {uid}: FAIL {row['status']}"); continue
        cond = next((l for l in res.stdout.splitlines() if l.startswith("conditions:")), "")
        status_line = next((l for l in res.stdout.splitlines() if l.startswith("STATUS")), "")
        if a.stages_log:
            with open(a.stages_log, "a") as sf:
                for l in res.stdout.splitlines():
                    if l.startswith(("stages", "rn_stages")):
                        sf.write(f"{uid}\t{l}\n")
        row["void"] = 0 if "measurement-grade" in status_line else 1
        worst["void"] += row["void"]
        toks = dict(t.split("=", 1) for t in cond.split()[1:] if "=" in t)
        row["sched"] = f"{toks.get('sched_policy','?')} core={toks.get('core','?')} rtprio={toks.get('rtprio','?')} mlock={toks.get('mlockall','?')}"
        if not hello_seen:
            hello_seen = True
            print(f"device   : {a.platform} build '{a.build}' tick_hz=1000000000 (ns)  {cond}")
            print(f"platform : " + next((l for l in res.stdout.splitlines() if l.startswith("platform")), ""))
        buf = open(outp, "rb").read()
        out = H.reassemble(buf)
        st = out["stats"]
        if st is None:
            row["status"] = "no STATS"; worst["fails"] += 1; w.writerow(row); continue
        frames_in, frames_out, overruns, crc, resyncs, pmin, pmax, _pad, psum, _so = st
        pd = per_hop_proc(buf, dh)
        hop_ns = hop * 1e9 / rate
        row.update(nres=out["nres"], overruns=overruns, crc_errors=crc, resyncs=resyncs,
                   proc_min=pmin, proc_max=pmax,
                   proc_mean=round(psum / frames_out, 1) if frames_out else "",
                   proc_median_data=H._q(pd, .5) if pd else "",
                   duty_cycle=round(psum / (frames_out * hop_ns), 5) if frames_out else "")
        worst["overruns"] += overruns
        body = out.get("body", b"")
        if len(body) != n * ch * 2:
            row["status"] = f"short output {len(body)//2//ch}/{n}"; worst["fails"] += 1
        else:
            if a.keep:
                H.write_wav(os.path.join(a.outdir, f"{uid}.{variant}.wav"), rate, ch, body)
            ref_path = None
            if a.ref_dir:
                ref_path = os.path.join(a.ref_dir, uid + ".wav")
            elif H.ALGO[a.algo] == 0:
                ref_path = remap(r["noisy_path"])      # bypass must return its input
            if ref_path and os.path.exists(ref_path):
                _, _, ref = H.read_wav(ref_path)
                c = H.compare_pcm(body, ref)
                row["ref"] = ref_path
                if c is None:
                    row["status"] = f"ref length {len(ref)//2//ch} != {n}"; worst["fails"] += 1
                else:
                    mx, nz, nmse = c
                    row.update(max_abs_lsb=mx, n_diff=nz, nmse_db=round(nmse, 2))
                    worst["max_abs_lsb"] = max(worst["max_abs_lsb"], mx)
                    worst["n_diff"] = max(worst["n_diff"], nz)
                    worst["nmse_db"] = max(worst["nmse_db"], nmse)
            elif ref_path:
                row["status"] = "no ref"
        w.writerow(row)
        if not a.keep:
            os.unlink(inp); os.unlink(outp)
        if k % a.every == 0 or k == len(rows):
            el = time.perf_counter() - t_start
            print(f"[{k}/{len(rows)}] {el:.0f}s  last {uid}: hops={dh} overruns={overruns} "
                  f"proc_mean={row['proc_mean']} ns max|d|={row.get('max_abs_lsb','-')} "
                  f"duty={row['duty_cycle']} void={row['void']} status={row['status']}", flush=True)
    print(f"done     : {len(rows)} runs in {time.perf_counter()-t_start:.0f}s -> {csv_path}")
    print(f"summary  : fails={worst['fails']} void_runs={worst['void']} total_overruns={worst['overruns']} "
          f"worst max|d|={worst['max_abs_lsb']} LSB  worst n_diff={worst['n_diff']}  "
          f"worst NMSE={worst['nmse_db']:.1f} dB")
    return 1 if worst["fails"] or worst["void"] else 0


if __name__ == "__main__":
    sys.exit(main())
