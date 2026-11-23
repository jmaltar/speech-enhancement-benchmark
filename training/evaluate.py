"""Score a trained run the way the baseline was scored: export, quantize, oracle, harness.

    python -m training.evaluate --run training/runs/<name> [--schemes f32,i16] [--corpora voicebank16k,dns]
                                [--against dtln-f32] [--limit N]

1. dtln/quantize.py on <run>/best.authors.weights.h5 -> <run>/tflite/model_f32_* (float export)
   and, per scheme, model_i16_* (16x8, split states, ILN_E2) / model_int8_* (full int8).
2. eval_harness.enhance on each corpus with algos dtln-run-<scheme>[-tflm] against --dtln-run,
   i.e. the same ABF1-framed oracle and metric path as the pretrained baseline; quantized
   schemes are scored on TFLite Micro (the MCU runtime), float on LiteRT.
3. eval_harness.compare of each scored variant against --against (default: the pretrained
   float oracle) -- the paired test with TOST against the locked margins.
Everything lands under <data-root>/enhanced/dtln-<run name>-<scheme>/ and <run>/eval.log.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def sh(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n"); f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(map(str, cmd))} -- see {log}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--schemes", default="f32,i16")
    ap.add_argument("--corpora", default="voicebank16k,dns")
    ap.add_argument("--against", default="dtln-f32")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shrunk", action="store_true",
                    help="a PRUNED run: export it at its live sizes (training.shrink) and score that; "
                         "tags become dtln-<run>-shrunk-<scheme>")
    a = ap.parse_args(argv)
    run = a.run.resolve(); tfl = run / ("shrunk" if a.shrunk else "tflite"); tfl.mkdir(exist_ok=True)
    log = run / "eval.log"; log.write_text("")
    w = run / "best.authors.weights.h5"
    assert w.exists(), w
    schemes = a.schemes.split(",")
    if a.shrunk:
        # shrink.py writes the float and 16x8 pairs itself (a zeroed full-size export
        # cannot be quantized: a dead unit's state has zero range and the converter aborts)
        assert set(schemes) <= {"f32", "i16", "i16pt", "int8"}, "a shrunk run is scored as f32 / i16 / i16pt / int8"
        if (not (tfl / "summary.json").exists() or ("int8" in schemes and not (tfl / "model_int8_1.tflite").exists())
                or ("i16pt" in schemes and not (tfl / "model_i16pt_1.tflite").exists())):
            sh([PY, "-m", "training.shrink", "--run", str(run)] + (["--int8"] if "int8" in schemes else [])
               + (["--per-tensor-dense"] if "i16pt" in schemes else []), log)
        schemes_to_convert = []
    else:
        schemes_to_convert = schemes
    for s in schemes_to_convert:
        if s == "f32":
            continue
        cmd = [PY, "-m", "dtln.quantize", "--weights", str(w), "--out", str(tfl), "--tag", s]
        cmd += ["--scheme", "16x8", "--iln", "e2"] if s == "i16" else ["--scheme", "int8"]
        sh(cmd, log)
    if "f32" in schemes and not a.shrunk and not (tfl / "model_f32_1.tflite").exists():
        sh([PY, "-m", "dtln.quantize", "--weights", str(w), "--out", str(tfl), "--tag", "i16", "--scheme", "16x8", "--iln", "e2"], log)
    algos = [f"dtln-run-{s}" + ("" if s == "f32" else "-tflm") for s in schemes]
    for corpus in a.corpora.split(","):
        cmd = [PY, "-m", "eval_harness.enhance", "--corpus", corpus, "--dtln-run", str(tfl), "--algo"] + algos
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        sh(cmd, log)
        for al in algos:
            tag = f"dtln-{run.name}{'-shrunk' if a.shrunk else ''}-{al.split('dtln-run-', 1)[1]}"
            sh([PY, "-m", "eval_harness.compare", "--corpus", corpus, "--a", a.against, "--b", tag], log)
    print(f"scored {run.name}: {schemes} on {a.corpora}; log {log}")
    txt = log.read_text()
    for line in txt.splitlines():
        if line.startswith("===") and ("vs" in line) or line.startswith(("pesq_mos_lqo", "stoi ", "si_sdr")):
            print("  " + line[:150])


if __name__ == "__main__":
    main()
