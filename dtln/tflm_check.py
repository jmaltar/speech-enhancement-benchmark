"""Does TFLite MICRO accept the model, and how big is its arena? Measured, not estimated.

    python -m dtln.tflm_check [--arena-kb 512] [--frames 100]

Loads each DTLN TFLite pair into the TFLite Micro host runtime (the `tflite-micro`
wheel: the same MicroInterpreter the firmware tiers run, with the
RecordingMicroAllocator behind print_allocations()) and reports, per model:

  * whether TFLM accepts it at all. The authors' hybrid model_quant_* pair is
    expected to be REFUSED -- tflm_footprint/RESULTS.md read the kTfLiteError
    in fully_connected.cc; this runs it and prints TFLM's own message;
  * the tensor arena TFLM actually allocates (its "Arena allocation total"),
    which closes the 45,056 B estimate the budget has carried since 2026-09-12;
  * kernel arithmetic: the SAME quantized inputs (main input and every state,
    taken frame by frame from a LiteRT run so no recurrence divergence enters)
    through TFLM and LiteRT, max |delta| in quantized steps at every output.
    An MCU runtime's int8 arithmetic should match the host oracle's exactly;
    where it does not, task 15 has to know before any port is compared.

TFLM indexes inputs and outputs by subgraph position and carries no names;
LiteRT's get_input_details()/get_output_details() lists are in that same
position order, which is what the mapping below relies on.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_harness import manifest  # noqa: E402
from dtln import oracle  # noqa: E402

PAIRS = [
    ("drq   authors' hybrid (int8 weights, float activations)", "drq"),
    ("int8p full-integer int8, authors' packed state layout", "int8p"),
    ("int8  full-integer int8, split states", "int8"),
    ("int8n full-integer int8, split states, norm weights", "int8n"),
    ("i16   16x8: int16 activations, int8 weights, split states", "i16"),
    ("i16n  16x8, norm weights", "i16n"),
    ("f32   shipped float pair (reference; does not fit any MCU)", "f32"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arena-kb", type=int, default=512)
    ap.add_argument("--frames", type=int, default=100)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)        # keep TFLM's C++ prints in order with ours
    from tflite_micro.python.tflite_micro import runtime
    from ai_edge_litert.interpreter import Interpreter as Lite

    us = manifest.utterances(manifest.load(), "voicebank16k")
    _, x = oracle.read_wav_int16(us[500].noisy_path)

    for label, variant in PAIRS:
        files = oracle.VARIANTS[variant]
        if not all((oracle.PRETRAINED / f).exists() for f in files):
            print(f"\n=== {label}: files missing, skipped"); continue
        print(f"\n=== {label}")
        # One LiteRT pass through the oracle, recording the quantized inputs and
        # outputs of both stages per frame.
        rec = {0: [], 1: []}
        class Tap(oracle.DtlnTflite):
            def _run(self, it, in_x, in_s, out_x, out_s, xx, states):
                stage = 0 if it is self.i1 else 1
                ins = {in_x["index"]: self._q(xx, in_x)}
                ins.update({d["index"]: self._q(st, d) for d, st in zip(in_s, states)})
                y, new = super()._run(it, in_x, in_s, out_x, out_s, xx, states)
                outs = {d["index"]: it.get_tensor(d["index"]).copy() for d in [out_x] + out_s}
                if len(rec[stage]) < a.frames:
                    rec[stage].append((ins, outs))
                return y, new
        tap = Tap(variant)
        oracle.enhance(x[: (a.frames + 8) * oracle.HOP], tap)

        for stage, fn in enumerate(files):
            p = oracle.PRETRAINED / fn
            print(f"  -- {fn}")
            try:
                it = runtime.Interpreter.from_file(str(p), arena_size=a.arena_kb * 1024)
            except Exception as e:
                print(f"     REFUSED by TFLite Micro: {str(e).strip()[:120]}")
                continue
            it.print_allocations()
            lite = Lite(model_path=str(p), num_threads=1); lite.allocate_tensors()
            li, lo = lite.get_input_details(), lite.get_output_details()
            worst = {d["index"]: 0 for d in lo}
            failed = None
            for ins, outs in rec[stage]:
                for pos, d in enumerate(li):
                    it.set_input(ins[d["index"]], pos)
                try:
                    it.invoke()
                except Exception as e:       # a kernel TFLM registers but cannot run for this dtype
                    failed = f"{type(e).__name__}: {str(e).strip()[:100]}"
                    break
                for pos, d in enumerate(lo):
                    yt = it.get_output(pos)
                    dd = np.abs(yt.astype(np.float64) - outs[d["index"]].astype(np.float64)).max()
                    worst[d["index"]] = max(worst[d["index"]], dd)
            it.reset()
            if failed:
                print(f"     accepted at allocation but INVOKE FAILED in TFLM: {failed}")
                continue
            desc = ", ".join(f"{str(tuple(int(v) for v in d['shape']))} {worst[d['index']]:.4g}" for d in lo)
            unit = "steps" if lo[0]["dtype"] in (np.int8, np.int16) else "float"
            print(f"     TFLM vs LiteRT, same inputs, {len(rec[stage])} frames, max |delta| per output ({unit}): {desc}")


if __name__ == "__main__":
    main()
