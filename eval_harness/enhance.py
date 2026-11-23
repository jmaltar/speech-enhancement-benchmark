"""Run the classical enhancers over a corpus and score them -- task 8's driver.

    python -m eval_harness.enhance --corpus voicebank16k [--algo bypass specsub wiener mmse dtln-f32 dtln-drq]
                                   [--bin classical/build/cdsp_wav] [--workers N]

Every utterance goes through `cdsp_wav`, i.e. through the ABF1 FILE BINDING
with the shared geometry helpers, so what is scored here is the PC float
oracle for this arm framed exactly as the ports will frame it. Enhanced WAVs
land under <data-root>/enhanced/<algo>/<corpus>/ (derived data lives under the
data root, never under the project folder).

Three things this script asserts rather than assumes:

  * `bypass` is run first as the REGRESSION ANCHOR: its output must be
    bit-identical to the noisy input, so its scores must equal the validated
    'Noisy' row to the last digit. A framing or geometry defect shows up here,
    on the cheapest possible run, instead of being blamed on an algorithm.
  * Every enhanced file is scored against the same clean reference as the
    noisy row, through metrics.all_metrics(), in the mode the corpus locks.
  * Comparisons are paired by utterance id through stats.compare(), with the
    test fixed per metric and TOST against the locked margins. These are the
    HEADLINE pooled per-corpus tests of Metric Stack Decision 6; Holm is
    applied across the family printed here (one per (algorithm, metric)).
    Per-cell tables are exploratory and are printed without adjustment and
    labelled so.

On NOIZEUS the per-cell RAW-MOS means for babble/car/street x 0/5/10 dB are
set beside Lu & Loizou 2008 Table 2 -- at RANKING precision, as validate.py
records: no published noisy row, and a MATLAB-port offset of unknown spread.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import audio, manifest, metrics, stats, validate

ALGOS = ("bypass", "specsub", "wiener", "mmse")
# Task 10 (2026-10-19): the DTLN float oracle and the authors' dynamic-range
# int8 export, run IN-PROCESS through dtln/oracle.py under the same ABF1
# framing policy (its geometry is imported from harness/abf_host.py). 16 kHz
# only -- refused on NOIZEUS rather than resampled.
DTLN_ALGOS = ("dtln-f32", "dtln-drq", "dtln-int8", "dtln-int8p", "dtln-f32n", "dtln-int8n", "dtln-i16", "dtln-i16n",
              "dtln-i16-tflm", "dtln-int8-tflm")   # -tflm: same model, TFLite Micro runtime
# Task 9 (2026-10-19): Xiph RNNoise through rnnoise_wav (the ABF1 file binding
# at its own 960/480 geometry at 48 kHz). `rnnoise` is the as-shipped v0.1.1
# release, `rnnoise-v01` the paper-era v0.1 tree with identical weights. The
# corpus is voicebank48k and the scoring is at 16 kHz after the locked
# downsample of reference and output alike (Metric Stack Decision 5).
RNNOISE_ALGOS = ("rnnoise", "rnnoise-v01", "rnnoise-nn8")
# rnnoise-nn8 (2026-10-31): the "+N" variant, the three GRUs on CMSIS-NN with
# int8 activations (rnnoise/nn8/rnn_nn8.h) -- the host twin of the F411 image
# of that name, scored here because quality is measured host-side.
PUBLISHED_KEY = {"specsub": "spectral_subtraction", "mmse": "mmse_stsa"}
_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = _ROOT / "classical" / "build" / "cdsp_wav"
RNNOISE_BIN = {"rnnoise": _ROOT / "rnnoise" / "build" / "rnnoise_wav",
               "rnnoise-v01": _ROOT / "rnnoise" / "build" / "rnnoise_wav_v01",
               "rnnoise-nn8": _ROOT / "rnnoise" / "build" / "rnnoise_wav_nn8"}

_DTLN_MODELS: dict = {}          # one interpreter pair per worker process
DTLN_RUN_DIR: Path | None = None  # --dtln-run: a trained run's tflite/ directory (task 13)


def _dtln_model(variant: str):
    """variant 'f32' etc. from dtln/pretrained, or 'run-<v>' from DTLN_RUN_DIR."""
    if variant not in _DTLN_MODELS:
        from dtln import oracle          # imports LiteRT; kept out of the parent
        if variant.startswith("run-"):
            v = variant[4:]
            base = v[:-5] if v.endswith("-tflm") else v
            oracle.VARIANTS[f"run-{base}"] = (f"model_{base}_1.tflite", f"model_{base}_2.tflite")
            m = oracle.make_model(f"run-{base}" + ("-tflm" if v.endswith("-tflm") else ""), model_dir=DTLN_RUN_DIR)
        else:
            m = oracle.make_model(variant)
        _DTLN_MODELS[variant] = (oracle, m)
    return _DTLN_MODELS[variant]


def _is_dtln(algo: str) -> bool:
    return algo in DTLN_ALGOS or algo.startswith("dtln-run-")


def _job(args):
    binp, algo, nest, corpus, rate, clean_p, noisy_p, out_p, uid, noise, snr_db, snr_bin = args
    try:
        Path(out_p).parent.mkdir(parents=True, exist_ok=True)
        if _is_dtln(algo):
            oracle, model = _dtln_model(algo.split("-", 1)[1])
            oracle.enhance_file(noisy_p, out_p, model)
        elif algo in RNNOISE_ALGOS or (algo == "bypass" and corpus == "voicebank48k"):
            b = RNNOISE_BIN.get(algo, RNNOISE_BIN["rnnoise"])
            r = subprocess.run([str(b), "--in", str(noisy_p), "--out", str(out_p),
                                "--proc", "bypass" if algo == "bypass" else "rnnoise"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"rnnoise_wav rc={r.returncode}: {r.stderr.strip()[:200]}")
        else:
            r = subprocess.run([str(binp), "--in", str(noisy_p), "--out", str(out_p), "--proc", algo,
                                "--noise", nest], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"cdsp_wav rc={r.returncode}: {r.stderr.strip()[:200]}")
        ref, sr_r = audio.load(clean_p)
        deg, sr_d = audio.load(out_p)
        if sr_r != sr_d or sr_r != rate:
            raise ValueError(f"rate mismatch {sr_r}/{sr_d}/{rate}")
        if len(ref) != len(deg):
            raise ValueError(f"length {len(deg)} != reference {len(ref)}: the drop-and-trim is wrong")
        if algo == "bypass":
            noisy, _ = audio.load(noisy_p)
            if not np.array_equal(noisy, deg):
                raise AssertionError("bypass output is not bit-identical to the input")
        if rate == 48_000:
            # The RNNoise arm: reference and signal under test through the SAME
            # locked resampler (decision 5, amended 2026-09-16), scored wideband.
            ref, deg, rate = audio.downsample_48_to_16(ref), audio.downsample_48_to_16(deg), 16_000
        m = metrics.all_metrics(ref, deg, rate, corpus)
        m.update(id=uid, corpus=corpus, algo=algo, noise=noise, snr_db=snr_db, snr_bin=snr_bin, ok=True)
        return m
    except Exception as e:
        return {"id": uid, "corpus": corpus, "algo": algo, "noise": noise, "snr_db": snr_db,
                "snr_bin": snr_bin, "ok": False, "error": f"{type(e).__name__}: {e}"}


def tag(algo: str, nest: str) -> str:
    """Output directory / CSV tag: the gain rule and the noise tracker are
    separate design choices and both are named, so two runs never collide."""
    if algo.startswith("dtln-run-"):
        d = DTLN_RUN_DIR
        base = d.parent.name + ("-shrunk" if d.name == "shrunk" else "") if d.name in ("tflite", "shrunk") else d.name
        return f"dtln-{base}-{algo[9:]}"
    return algo if algo == "bypass" or _is_dtln(algo) or algo in RNNOISE_ALGOS else f"{algo}-{nest}"


def run_algo(binp: Path, algo: str, nest: str, corpus: str, root: Path, out_root: Path,
             workers: int, limit: int | None) -> pd.DataFrame:
    m = manifest.load(root)
    us = manifest.utterances(m, corpus, limit=None)
    if limit:
        rng = np.random.default_rng(20261012)      # random, never the head (trap 14)
        idx = sorted(rng.choice(len(us), size=min(limit, len(us)), replace=False))
        us = [us[i] for i in idx]
    jobs = [(binp, algo, nest, corpus, u.rate_hz, u.clean_path, u.noisy_path,
             out_root / tag(algo, nest) / corpus / f"{u.id}.wav", u.id, u.noise, u.snr_db, u.snr_bin) for u in us]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = pool.map(_job, jobs, chunksize=4)
    dt = time.time() - t0
    df = pd.DataFrame(rows)
    bad = df[~df.ok]
    print(f"\n=== {corpus} / {tag(algo, nest)}: {len(df)} utterances in {dt:.1f}s ({workers} workers)")
    if len(bad):
        print(f"  *** {len(bad)} FAILED ***")
        for e, n in bad.error.value_counts().items():
            print(f"      {n:4d}  {e}")
    return df


def paired(noisy: pd.DataFrame, enh: pd.DataFrame, corpus: str) -> list[stats.Comparison]:
    a = noisy[noisy.ok].set_index("id")
    b = enh[enh.ok].set_index("id")
    ids = a.index.intersection(b.index)
    cols = [c for c in ("pesq_mos_lqo", "pesq_raw_mos", "stoi", "estoi", "si_sdr") if c in b.columns]
    return [stats.compare(a.loc[ids, c].to_numpy(), b.loc[ids, c].to_numpy(), c) for c in cols]


def noizeus_vs_published(dfs: dict[str, pd.DataFrame]) -> None:
    pub = validate.NOIZEUS_PUBLISHED_ALGORITHMS
    cells = sorted({k for a in PUBLISHED_KEY.values() for k in pub[a]}, key=lambda k: (k[0], k[1]))
    have = [a for a in PUBLISHED_KEY if a in dfs]
    if not have:
        return
    print(f"\n  NOIZEUS per-cell RAW MOS (ours, N=30 each) beside Lu & Loizou 2008 Table 2")
    print(f"  ({pub['scale']}; ranking precision -- see validate.NO_ANCHOR_NOTE)")
    hdr = f"  {'cell':<14}" + "".join(f"{a+' ours':>12}{a+' pub':>11}{'delta':>8}" for a in have)
    print(hdr)
    rank_ok = rank_n = 0
    for noise, snr in cells:
        line = f"  {noise:<8}{snr:>3.0f} dB "
        ours = {}
        for a in have:
            d = dfs[a]
            sel = d[d.ok & (d.noise == noise) & (d.snr_db == snr)]
            v = float(sel.pesq_raw_mos.mean()) if len(sel) else float("nan")
            ours[a] = v
            p = pub[PUBLISHED_KEY[a]][(noise, snr)]
            line += f"{v:>12.3f}{p:>11.2f}{v - p:>+8.3f}"
        if "specsub" in ours and "mmse" in ours:
            rank_n += 1
            pub_sign = np.sign(pub["mmse_stsa"][(noise, snr)] - pub["spectral_subtraction"][(noise, snr)])
            our_sign = np.sign(ours["mmse"] - ours["specsub"])
            rank_ok += int(pub_sign == our_sign)
            line += "   " + ("same order" if pub_sign == our_sign else "ORDER DIFFERS")
        print(line)
    if rank_n:
        print(f"  MMSE-vs-SS ordering agrees with the published table in {rank_ok}/{rank_n} cells")


def dns_vs_published(dfs: dict[str, pd.DataFrame]) -> None:
    """DTLN's own published rows on this exact test set (validate.DNS_PUBLISHED_ALGORITHMS):
    STOI and SI-SDR compared, the narrowband PESQ column shown but never compared."""
    pub = validate.DNS_PUBLISHED_ALGORITHMS
    rows = [("bypass", "noisy")] + [(a, a) for a in DTLN_ALGOS if a in dfs]
    if len(rows) == 1:
        return
    print(f"\n  DNS no-reverb: ours (N=150) beside the published DTLN rows")
    print(f"  ({pub['source']})\n  ({pub['scale']})")
    print(f"  {'row':<10}{'STOI ours':>10}{'pub':>8}{'delta':>8}{'SI-SDR ours':>13}{'pub':>8}{'delta':>8}"
          f"{'PESQ wb ours':>14}{'pub (NB!)':>10}")
    for algo, key in rows:
        d = dfs[algo][dfs[algo].ok]
        p = pub[key]
        print(f"  {key:<10}{d.stoi.mean():>10.4f}{p['stoi']:>8.4f}{d.stoi.mean() - p['stoi']:>+8.4f}"
              f"{d.si_sdr.mean():>13.2f}{p['si_sdr']:>8.2f}{d.si_sdr.mean() - p['si_sdr']:>+8.2f}"
              f"{d.pesq_mos_lqo.mean():>14.4f}{p['pesq_nb']:>10.2f}")
    k = pub["dtln-keras"]
    print(f"  {'(Table 2)':<10}{'':>10}{k['stoi']:>8.4f}{'':>8}{'':>13}{k['si_sdr']:>8.2f}{'':>8}{'':>14}{k['pesq_nb']:>10.2f}"
          f"   <- model.h5 whole-file, the paper's row")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--algo", nargs="+", default=list(ALGOS))
    ap.add_argument("--noise", default="mcra", choices=("mcra", "vad"),
                    help="noise PSD tracker: MCRA (default; needs no leading silence) or the "
                         "init-then-VAD tracker of Hu & Loizou's testbed")
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--root", type=Path, default=manifest.DEFAULT_ROOT)
    ap.add_argument("--out-root", type=Path, default=None, help="default <root>/enhanced")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--limit", type=int, default=None, help="random subset, fixed seed (never the head)")
    ap.add_argument("--dtln-run", type=Path, default=None,
                    help="a task-13 run's tflite/ dir; enables algos dtln-run-<variant>[-tflm]")
    a = ap.parse_args()
    global DTLN_RUN_DIR
    DTLN_RUN_DIR = a.dtln_run.resolve() if a.dtln_run else None

    prov = metrics.assert_provenance()
    print("Provenance (asserted, not assumed):")
    for k, v in prov.items():
        print(f"  {k:<18} {v}")
    if not a.bin.exists():
        raise SystemExit(f"{a.bin} not built: cmake -S classical -B classical/build && cmake --build classical/build")
    out_root = a.out_root or (a.root / "enhanced")
    algos = list(a.algo)
    if a.corpus == "noizeus" and any(_is_dtln(al) for al in algos):
        raise SystemExit("DTLN is a 16 kHz model and NOIZEUS is 8 kHz: not run rather than resampled "
                         "(a resampling pair would need its own provenance -- Metric Stack Decision 5)")
    if any(al in RNNOISE_ALGOS for al in algos) != (a.corpus == "voicebank48k"):
        raise SystemExit("RNNoise runs on voicebank48k and nothing else runs there: it is the one 48 kHz "
                         "enhancer, and the 48 kHz test set exists solely for it (CLAUDE.md, Datasets)")
    for al in algos:
        if al in RNNOISE_ALGOS and not RNNOISE_BIN[al].exists():
            raise SystemExit(f"{RNNOISE_BIN[al]} not built: cmake -S rnnoise -B rnnoise/build && cmake --build rnnoise/build")
    if "bypass" not in algos:
        algos = ["bypass"] + algos          # the regression anchor is not optional

    dfs: dict[str, pd.DataFrame] = {}
    for algo in algos:
        df = run_algo(a.bin, algo, a.noise, a.corpus, a.root, out_root, a.workers, a.limit)
        (out_root / tag(algo, a.noise)).mkdir(parents=True, exist_ok=True)
        df.to_csv(out_root / tag(algo, a.noise) / f"{a.corpus}.csv", index=False)
        dfs[algo] = df
        validate.summarise(df, a.corpus)
        if algo != "bypass":
            validate.stratify(df, a.corpus, "pesq_raw_mos" if a.corpus == "noizeus" else "pesq_mos_lqo")
            print("  (per-cell: EXPLORATORY, unadjusted -- Metric Stack Decision 6)")

    # bypass must reproduce the noisy row exactly; it is also the 'A' of every comparison
    noisy = dfs["bypass"]
    if (~noisy.ok).any():
        raise SystemExit("bypass regression anchor FAILED -- fix the framing before quoting any number")
    print("\n  bypass: every utterance bit-identical to its input; scores ARE the Noisy row.")

    # headline family: one comparison per (algorithm, metric), Holm across it
    fam: list[tuple[str, stats.Comparison]] = []
    for algo in algos:
        if algo == "bypass":
            continue
        for c in paired(noisy, dfs[algo], a.corpus):
            fam.append((algo, c))
    if fam:
        adj = stats.holm([c.p_diff for _, c in fam])
        print(f"\n=== {a.corpus}: enhanced (B, noise tracker {a.noise}) vs noisy (A), paired by "
              f"utterance -- HEADLINE family, Holm over {len(fam)} tests ({stats.HOLM_FAMILY})")
        for algo in algos:
            if algo == "bypass":
                continue
            cs = [c for al, c in fam if al == algo]
            print(f"\n  --- {algo}")
            print("  " + stats.render(cs).replace("\n", "\n  "))
        print("\n  Holm-adjusted p(diff):")
        for (algo, c), p in zip(fam, adj):
            flag = "significant" if p < stats.ALPHA else "n.s."
            print(f"    {algo:<12}{c.metric:<14} raw {c.p_diff:9.3g}  adj {p:9.3g}  {flag}")

    if a.corpus == "noizeus":
        noizeus_vs_published(dfs)
    if a.corpus == "dns":
        dns_vs_published(dfs)
    if a.corpus == "voicebank48k":
        anc = validate.RNNOISE_ARM_NOISY_ANCHOR
        d = noisy[noisy.ok]
        print(f"\n  RNNoise-arm regression anchor: bypass (our-filtered noisy vs our-filtered clean) against "
              f"{anc['source']}")
        allok = True
        for k in ("pesq_mos_lqo", "stoi", "estoi", "si_sdr"):
            v = float(d[k].mean()); ok = abs(v - anc[k]) <= anc["tol"]; allok &= ok
            print(f"    {k:<14} ours {v:8.4f}  recorded {anc[k]:8.4f}  delta {v - anc[k]:+8.5f}  {'PASS' if ok else '*** FAIL ***'}")
        if not allok:
            print("    the 48->16 kHz downsample or the 960/480 framing has moved -- fix before quoting RNNoise")
        if "rnnoise" in dfs:
            pub = validate.VOICEBANK_PUBLISHED_ALGORITHMS["rnnoise"]
            ours = float(dfs["rnnoise"][dfs["rnnoise"].ok].pesq_mos_lqo.mean())
            d = ours - pub["pesq_mos_lqo"]
            print(f"\n  VoiceBank RNNoise PESQ: ours {ours:.4f}  published {pub['pesq_mos_lqo']:.2f} ({pub['source']})  "
                  f"delta {d:+.4f}  {'within' if abs(d) <= stats.MARGIN['pesq_mos_lqo'] else 'OUTSIDE'} the 0.10 margin")
    if a.corpus == "voicebank16k" and "wiener" in dfs:
        pub = validate.VOICEBANK_PUBLISHED_ALGORITHMS
        ours = float(dfs["wiener"][dfs["wiener"].ok].pesq_mos_lqo.mean())
        d = ours - pub["wiener"]["pesq_mos_lqo"]
        print(f"\n  VoiceBank Wiener PESQ: ours {ours:.4f}  published {pub['wiener']['pesq_mos_lqo']:.2f} "
              f"({pub['source']}; {pub['scale']})  delta {d:+.4f}  "
              f"{'within' if abs(d) <= stats.MARGIN['pesq_mos_lqo'] else 'OUTSIDE'} the 0.10 margin")
    print("\n=== DONE")


if __name__ == "__main__":
    main()
