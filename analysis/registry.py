"""
registry.py -- the one place that says which scored run is which result (task 21).

Every number the consolidated tables carry starts here: a QUALITY CLASS maps an
(algorithm, variant) pair to the host-scored tag that measures it, and a COST RUN
maps a (platform, label) pair to the per-utterance CSV the board's batch wrote.
Nothing in collect.py guesses a file name. A run that is not declared here is not
in the tables, and `collect.py --inventory` lists every scored tag and every cost
CSV on disk that is NOT declared, so an omission is visible rather than silent.

Labels are the ones the per-tier RESULTS.md files use (`+F`, `+FS`, `+N`, `+M`,
`+W`, `+FW`, `+P`). Quality classes follow the locked rule (quality is measured
host-side per (model, compression) pair; platforms contribute cost):

  * a cost run whose port reproduces its host oracle within tolerance joins that
    oracle's quality (classical, RNNoise float GRU, DTLN on every tier);
  * the libm / placement variants (`+F`, `+FS`, `+W`, `+FW`) are cost variants of
    the float-GRU class -- byte-identical or within 1 LSB of the oracle and
    metric-EQUIVALENT to it on all four metrics (rnnoise/RESULTS.md, the S3 and
    F411 RESULTS) -- so they join `rnnoise`;
  * `+N` / `+M` (CMSIS-NN int8 GRU) form their own quality class whose host twin
    is `rnnoise-nn8`. The Cortex-M outputs are NOT sample-equivalent to it (FMA
    placement through the int8 quantiser), so they were scored on their own
    output under the per-platform clause; those rows live in the per-platform
    table and are EQUIVALENT to the twin, so the frontier uses the twin.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("BENCHMARK_DATA_ROOT", "/mnt/data/benchmark-data"))
ENH = DATA / "enhanced"

# Verified clocks (CLAUDE.md, Measured Hardware Constants: every clock confirmed against an
# independent time reference). The Pi's batch CSVs carry CLOCK_MONOTONIC nanoseconds, the
# MCUs' carry core cycles; cycles = ns x 2.4 on the Pi.
CLOCK_HZ = {"f411": 96e6, "h723": 550e6, "s3": 240e6, "rpi5": 2.4e9}
PROC_UNIT = {"f411": "cycles", "h723": "cycles", "s3": "cycles", "rpi5": "ns"}
PLATFORMS = ("f411", "h723", "s3", "rpi5")
PLATFORM_SHORT = {"f411": "F411", "h723": "H723", "s3": "S3", "rpi5": "RPi5"}
PLATFORM_NAME = {"f411": "STM32F411 (M4F, 96 MHz)", "h723": "STM32H723 (M7, 550 MHz)",
                 "s3": "ESP32-S3 (LX7, 240 MHz)", "rpi5": "Raspberry Pi 5 (A76, 2.4 GHz)"}

# Hop per algorithm family: the locked 16 ms classical hop, RNNoise's hardwired 10 ms at
# 48 kHz (documented deviation 1) and DTLN's native 8 ms hop (verified 2026-09-16).
HOP_S = {"noisy": 0.016, "specsub": 0.016, "wiener": 0.016, "mmse": 0.016,
         "rnnoise": 0.010, "dtln": 0.008}

ALGO_NAME = {"noisy": "Noisy (identity)", "specsub": "Spectral subtraction",
             "wiener": "Wiener", "mmse": "MMSE-STSA", "rnnoise": "RNNoise",
             "dtln": "DTLN", "umx": "Open-Unmix umxhq"}


@dataclass(frozen=True)
class QClass:
    """A host-scored (algorithm, variant) pair: the quality every platform inherits."""
    algorithm: str
    variant: str
    tag: str                      # enhanced/<tag>/
    corpora: tuple                # corpora it was scored on
    note: str = ""
    model_bytes: float = float("nan")    # deployable weight / artifact footprint
    model_src: str = ""


QCLASSES = [
    QClass("noisy", "identity", "bypass", ("voicebank16k", "noizeus", "dns"),
           "the Noisy row: bypass is bit-identical to the input on every utterance", 0, "no model"),
    QClass("specsub", "MCRA", "specsub-mcra", ("voicebank16k", "noizeus", "dns"),
           "MCRA is the default tracker and the one every port ran", 0, "no model"),
    QClass("wiener", "MCRA", "wiener-mcra", ("voicebank16k", "noizeus", "dns"), "", 0, "no model"),
    QClass("mmse", "MCRA", "mmse-mcra", ("voicebank16k", "noizeus", "dns"), "", 0, "no model"),
    QClass("rnnoise", "float GRU (as shipped)", "rnnoise", ("voicebank48k",),
           "Xiph v0.1.1; 48 kHz input, scored at 16 kHz wideband against OUR-filter reference "
           "(Metric Stack Decision 5, locked 2026-09-16)",
           87_503, "CLAUDE.md Task 4 memory table (RNNoise int8 weights)"),
    QClass("rnnoise", "int8 GRU (CMSIS-NN)", "rnnoise-nn8", ("voicebank48k",),
           "host twin of the +N build (rnnoise/nn8/)", 87_503, "same 87,503 int8 weights"),
    QClass("dtln", "float32 (oracle)", "dtln-f32", ("voicebank16k", "dns"),
           "the authors' float TFLite pair; runs on no tier as deployed",
           3_975_748, "CLAUDE.md Task 4 memory table (DTLN f32 tflite)"),
    QClass("dtln", "full 16x8 (TFLM)", "dtln-i16-tflm", ("voicebank16k", "dns"),
           "int16 activations, int8 weights, split states; the MCU oracle",
           1_149_416, "CLAUDE.md Algorithms (DTLN artifact 1,149,416 B, arena 68,512 B)"),
    QClass("dtln", "pruned 34 % 16x8 (TFLM)", "dtln-prune-learned-shrunk-i16pt-tflm",
           ("voicebank16k", "dns"),
           "Arm B learned-threshold structured pruning (lambda 0.02), per-tensor 16x8; "
           "TRAINED ON VOICEBANK, unlike the float oracle",
           396_800, "benchmark_h723/RESULTS.md:453 (396,800 B)"),
]
QCLASS_BY_KEY = {(q.algorithm, q.variant): q for q in QCLASSES}
QCLASS_BY_TAG = {q.tag: q for q in QCLASSES}


@dataclass(frozen=True)
class CostRun:
    platform: str
    algorithm: str
    variant: str                  # quality class (QCLASS key)
    label: str                    # the tier's own label for the build
    file: str                     # repo-relative per-utterance CSV
    date: str
    paced: bool
    source: str                   # RESULTS.md section that reports it
    extra: dict = field(default_factory=dict)


def _classical(plat, date, paced_files=False, src=""):
    out = []
    for a in ("specsub", "wiener", "mmse"):
        f = (f"benchmark_{plat}/results/voicebank16k_{a}-mcra"
             f"{'_paced' if paced_files else ''}.csv")
        out.append(CostRun(plat, a, "MCRA", "as built", f, date, paced_files, src))
    f = f"benchmark_{plat}/results/voicebank16k_bypass{'_paced' if paced_files else ''}.csv"
    out.append(CostRun(plat, "noisy", "identity", "bypass (harness floor)", f, date, paced_files, src))
    return out


FLOAT_GRU = "float GRU (as shipped)"
INT8_GRU = "int8 GRU (CMSIS-NN)"
DTLN_P = "pruned 34 % 16x8 (TFLM)"
DTLN_F = "full 16x8 (TFLM)"

COST_RUNS = [
    # ---- classical arm: MCU corpus runs 2026-10-14 (unpaced batch); Pi 2026-10-16 ----
    *_classical("f411", "2026-10-14", False, "benchmark_f411/RESULTS.md; classical/RESULTS.md"),
    *_classical("h723", "2026-10-14", False, "benchmark_h723/RESULTS.md"),
    *_classical("s3", "2026-10-14", False, "benchmark_s3/RESULTS.md"),
    *_classical("rpi5", "2026-10-16", True, "benchmark_rpi5/RESULTS.md (paced corpus run)"),
    # ---- RNNoise, voicebank48k, 10 ms hop, every run paced ----
    CostRun("f411", "rnnoise", FLOAT_GRU, "as shipped",
            "benchmark_f411/results/rnnoise/voicebank48k_rnnoise-v0.1.1_paced.csv", "2026-10-26", True,
            "benchmark_f411/RESULTS.md (unit 1)", {"own_quality": "rnnoise-f411"}),
    CostRun("f411", "rnnoise", FLOAT_GRU, "+F",
            "benchmark_f411/results/rnnoise_f411b/splibm_corpus/voicebank48k_rnnoise-v0.1.1_paced.csv",
            "2026-10-30", True, "benchmark_f411/RESULTS.md (unit 2)", {"own_quality": "rnnoise-f411-f"}),
    CostRun("f411", "rnnoise", FLOAT_GRU, "+FS",
            "benchmark_f411/results/rnnoise_f411b/spconst_corpus/voicebank48k_rnnoise-v0.1.1_paced.csv",
            "2026-10-30", True, "benchmark_f411/RESULTS.md (unit 2)", {"own_quality": "rnnoise-f411-fs"}),
    CostRun("f411", "rnnoise", INT8_GRU, "+N",
            "benchmark_f411/results/rnnoise_f411b/nn8_corpus/voicebank48k_rnnoise-v0.1.1_paced.csv",
            "2026-10-31", True, "benchmark_f411/RESULTS.md 17:15 (unit 2)", {"own_quality": "rnnoise-f411-n"}),
    CostRun("f411", "rnnoise", INT8_GRU, "+M",
            "benchmark_f411/results/rnnoise_f411b/nn8fs_corpus/voicebank48k_rnnoise-v0.1.1_paced.csv",
            "2026-10-31", True, "benchmark_f411/RESULTS.md (unit 2)", {"own_quality": "rnnoise-f411-m"}),
    CostRun("h723", "rnnoise", FLOAT_GRU, "as shipped",
            "benchmark_h723/results/rnnoise/voicebank48k_rnnoise-v0.1.1_paced.csv", "2026-10-27", True,
            "benchmark_h723/RESULTS.md"),
    CostRun("h723", "rnnoise", FLOAT_GRU, "+FS",
            "benchmark_h723/results/rnnoise/spconst_corpus/voicebank48k_rnnoise-v0.1.1_paced.csv",
            "2026-11-01", True, "benchmark_h723/RESULTS.md 2026-11-01", {"own_quality": "rnnoise-h723-fs"}),
    CostRun("h723", "rnnoise", INT8_GRU, "+N",
            "benchmark_h723/results/rnnoise/nn8_corpus/voicebank48k_rnnoise-v0.1.1_paced.csv",
            "2026-11-01", True, "benchmark_h723/RESULTS.md 2026-11-01", {"own_quality": "rnnoise-h723-n"}),
    CostRun("h723", "rnnoise", INT8_GRU, "+M",
            "benchmark_h723/results/rnnoise/nn8fs_corpus/voicebank48k_rnnoise-v0.1.1_paced.csv",
            "2026-11-01", True, "benchmark_h723/RESULTS.md 2026-11-01", {"own_quality": "rnnoise-h723-m"}),
    CostRun("s3", "rnnoise", FLOAT_GRU, "as shipped",
            "benchmark_s3/results/rnnoise/voicebank48k_rnnoise-v0.1.1_paced.csv", "2026-10-28", True,
            "benchmark_s3/RESULTS.md", {"own_quality": "rnnoise-s3"}),
    CostRun("s3", "rnnoise", FLOAT_GRU, "+W",
            "benchmark_s3/results/rnnoise/variant_ramw_corpus_2026-10-28/voicebank48k_rnnoise-v0.1.1_paced.csv",
            "2026-10-28", True, "benchmark_s3/RESULTS.md (output byte-identical to as shipped)"),
    CostRun("s3", "rnnoise", FLOAT_GRU, "+FW",
            "benchmark_s3/results/rnnoise/variant_both_corpus_2026-10-28/voicebank48k_rnnoise-v0.1.1_paced.csv",
            "2026-10-28", True, "benchmark_s3/RESULTS.md", {"own_quality": "rnnoise-s3-fw"}),
    CostRun("rpi5", "rnnoise", FLOAT_GRU, "as shipped",
            "benchmark_rpi5/results/rnnoise/voicebank48k_rnnoise-v0.1.1_paced.csv", "2026-10-26", True,
            "benchmark_rpi5/RESULTS.md"),
    # ---- DTLN, voicebank16k, 8 ms hop ----
    CostRun("h723", "dtln", DTLN_P, "CMSIS-NN s16 (DTCM arenas)",
            "benchmark_h723/results/dtln/voicebank16k_dtln-i16.csv", "2026-10-21", False,
            "benchmark_h723/RESULTS.md (first pass)", {"own_quality": "dtln-h723-p020-i16pt"}),
    CostRun("s3", "dtln", DTLN_P, "TFLM reference kernels",
            "benchmark_s3/results/dtln_full/voicebank16k_dtln-i16.csv", "2026-10-22", False,
            "benchmark_s3/RESULTS.md", {"own_quality": "dtln-s3-p020-i16pt"}),
    CostRun("s3", "dtln", DTLN_P, "+P",
            "benchmark_s3/results/dtln_s16_full/voicebank16k_dtln-i16.csv", "2026-10-26", False,
            "benchmark_s3/RESULTS.md (LX7 PIE int16 matvec; output byte-identical to the reference run)"),
    CostRun("rpi5", "dtln", DTLN_P, "TFLM reference kernels",
            "benchmark_rpi5/results/dtln/p020/voicebank16k_dtln-i16_paced.csv", "2026-10-27", True,
            "benchmark_rpi5/RESULTS.md", {"own_quality": "dtln-rpi5-p020-i16pt"}),
    CostRun("rpi5", "dtln", DTLN_F, "TFLM reference kernels",
            "benchmark_rpi5/results/dtln/full/voicebank16k_dtln-i16_paced.csv", "2026-10-27", True,
            "benchmark_rpi5/RESULTS.md", {"own_quality": "dtln-rpi5-full-i16"}),
]

# Unpaced twins of paced Pi runs: carried for the batch-vs-paced columns only.
PI_BATCH_TWINS = {
    "benchmark_rpi5/results/voicebank16k_specsub-mcra_paced.csv": "benchmark_rpi5/results/voicebank16k_specsub-mcra.csv",
    "benchmark_rpi5/results/voicebank16k_wiener-mcra_paced.csv": "benchmark_rpi5/results/voicebank16k_wiener-mcra.csv",
    "benchmark_rpi5/results/voicebank16k_mmse-mcra_paced.csv": "benchmark_rpi5/results/voicebank16k_mmse-mcra.csv",
    "benchmark_rpi5/results/voicebank16k_bypass_paced.csv": "benchmark_rpi5/results/voicebank16k_bypass.csv",
    "benchmark_rpi5/results/rnnoise/voicebank48k_rnnoise-v0.1.1_paced.csv": "benchmark_rpi5/results/rnnoise/voicebank48k_rnnoise-v0.1.1.csv",
    "benchmark_rpi5/results/dtln/p020/voicebank16k_dtln-i16_paced.csv": "benchmark_rpi5/results/dtln/p020/voicebank16k_dtln-i16.csv",
    "benchmark_rpi5/results/dtln/full/voicebank16k_dtln-i16_paced.csv": "benchmark_rpi5/results/dtln/full/voicebank16k_dtln-i16.csv",
}

# Separation tier: host quality, Pi cost. Never pooled with speech.
SEPARATION = {
    "quality_host": "separation/results/host_umxhq_score_2026-10-29.csv",
    "quality_rpi5": "separation/results/rpi5_umxhq_score_2026-10-29.csv",
    "quality_4target": "separation/results/host_umxhq_score4_2026-10-30.csv",
    "cost_rpi5": "separation/results/rpi5_umx_run_umxhq_t4_2026-10-29.csv",
}

# Per-platform quality rows (task 15): scored on the board's own output. Reference = the host
# tag the port is meant to reproduce. Only these appear in the per-platform table; the
# frontier never uses them (pareto.points refuses a quality-bearing cost table).
PER_PLATFORM_QUALITY = [
    # (tag, platform, label, host reference tag, corpus)
    ("rnnoise-f411", "f411", "as shipped", "rnnoise", "voicebank48k"),
    ("rnnoise-f411-f", "f411", "+F", "rnnoise", "voicebank48k"),
    ("rnnoise-f411-fs", "f411", "+FS", "rnnoise", "voicebank48k"),
    ("rnnoise-f411-n", "f411", "+N", "rnnoise-nn8", "voicebank48k"),
    ("rnnoise-f411-m", "f411", "+M", "rnnoise-nn8", "voicebank48k"),
    ("rnnoise-h723-fs", "h723", "+FS", "rnnoise", "voicebank48k"),
    ("rnnoise-h723-n", "h723", "+N", "rnnoise-nn8", "voicebank48k"),
    ("rnnoise-h723-m", "h723", "+M", "rnnoise-nn8", "voicebank48k"),
    ("rnnoise-s3", "s3", "as shipped", "rnnoise", "voicebank48k"),
    ("rnnoise-s3-fw", "s3", "+FW", "rnnoise", "voicebank48k"),
    ("dtln-h723-p020-i16pt", "h723", "CMSIS-NN s16", "dtln-prune-learned-shrunk-i16pt-tflm", "voicebank16k"),
    ("dtln-h723-p020-i16pt", "h723", "CMSIS-NN s16", "dtln-prune-learned-shrunk-i16pt-tflm", "dns"),
    ("dtln-s3-p020-i16pt", "s3", "TFLM reference kernels", "dtln-prune-learned-shrunk-i16pt-tflm", "voicebank16k"),
    ("dtln-rpi5-p020-i16pt", "rpi5", "TFLM reference kernels", "dtln-prune-learned-shrunk-i16pt-tflm", "voicebank16k"),
    ("dtln-rpi5-p020-i16pt", "rpi5", "TFLM reference kernels", "dtln-prune-learned-shrunk-i16pt-tflm", "dns"),
    ("dtln-rpi5-full-i16", "rpi5", "TFLM reference kernels", "dtln-i16-tflm", "voicebank16k"),
    ("dtln-rpi5-full-i16", "rpi5", "TFLM reference kernels", "dtln-i16-tflm", "dns"),
]

# Equivalence ("no worse") claims the project actually makes, beyond the per-platform ones:
# compression variant B against its baseline A, pooled per corpus. TOST against the locked
# margins; Holm is not applied (their protection is the margin, Metric Stack Decision 6).
TOST_CLAIMS = [
    # (corpus_a, tag_a, corpus_b, tag_b, claim)
    ("voicebank48k", "rnnoise", "voicebank48k", "rnnoise-v01", "release v0.1 vs v0.1.1 (same weights)"),
    ("voicebank48k", "rnnoise", "voicebank48k", "rnnoise-nn8", "+N int8 GRU (host twin) vs float GRU"),
    ("voicebank16k", "dtln-f32", "voicebank16k", "dtln-i16-tflm", "16x8 on TFLM vs float (Arm A)"),
    ("dns", "dtln-f32", "dns", "dtln-i16-tflm", "16x8 on TFLM vs float (Arm A)"),
    ("voicebank16k", "dtln-f32", "voicebank16k", "dtln-drq", "authors' dynamic-range int8 vs float (Arm A)"),
    ("dns", "dtln-f32", "dns", "dtln-drq", "authors' dynamic-range int8 vs float (Arm A)"),
    ("voicebank16k", "dtln-prune-learned-shrunk-f32", "voicebank16k", "dtln-prune-learned-shrunk-i16pt-tflm",
     "pruned model: 16x8 PTQ vs its float"),
    ("voicebank16k", "dtln-prune-learned-shrunk-i16pt-tflm", "voicebank16k", "dtln-wqat4-shrunk-i16pt-tflm",
     "Arm B (a): QAT vs PTQ at 16x8 on the pruned model"),
    ("voicebank16k", "dtln-f32", "voicebank16k", "dtln-prune-learned-shrunk-i16pt-tflm",
     "deployed pruned 16x8 vs the float oracle (VoiceBank: in-domain for the pruned model)"),
    ("dns", "dtln-f32", "dns", "dtln-prune-learned-shrunk-i16pt-tflm",
     "deployed pruned 16x8 vs the float oracle (DNS: out-of-domain for the pruned model)"),
]

# Headline family per corpus (Metric Stack Decision 6): algorithm pairs at reference
# configurations, one test per (pair, metric), Holm over the corpus's family. DTLN enters at its
# float oracle -- its deployment variants are compression claims (TOST_CLAIMS), not algorithms.
HEADLINE = {
    "voicebank": [("noisy", "voicebank16k", "bypass"), ("specsub", "voicebank16k", "specsub-mcra"),
                  ("wiener", "voicebank16k", "wiener-mcra"), ("mmse", "voicebank16k", "mmse-mcra"),
                  ("rnnoise", "voicebank48k", "rnnoise"), ("dtln", "voicebank16k", "dtln-f32")],
    "dns": [("noisy", "dns", "bypass"), ("specsub", "dns", "specsub-mcra"), ("wiener", "dns", "wiener-mcra"),
            ("mmse", "dns", "mmse-mcra"), ("dtln", "dns", "dtln-f32")],
    "noizeus": [("noisy", "noizeus", "bypass"), ("specsub", "noizeus", "specsub-mcra"),
                ("wiener", "noizeus", "wiener-mcra"), ("mmse", "noizeus", "mmse-mcra")],
}
HEADLINE_METRICS = {"voicebank": ("pesq_mos_lqo", "stoi", "estoi", "si_sdr"),
                    "dns": ("pesq_mos_lqo", "stoi", "estoi", "si_sdr"),
                    "noizeus": ("pesq_mos_lqo", "pesq_raw_mos", "stoi", "estoi", "si_sdr")}


# --------------------------------------------------------------------------- power (task 22)
# One steady-state power session per board on RNNoise (2026-11-04/05), through the spare-F411
# fixture (INA226, 0.1 ohm shunt at the 5 V input) on the MCUs and PMIC telemetry on the Pi.
# `idle` names the file whose mean is the model's P_idle: the state between hops of a paced
# stream, i.e. the POST-STREAM idle. The H723's session took its idle leg before any stream
# (the 828 mW fresh-boot state), so its model idle comes from the replug A/B's post-stream leg.
@dataclass(frozen=True)
class PowerSession:
    platform: str
    algorithm: str
    variant: str
    label: str
    dir: str                      # repo-relative results/power/<session>
    idle: str                     # file in `dir` whose mean is the model's P_idle
    method: str
    source: str
    note: str = ""


INA = "INA226 R100 at the board's 5 V input (includes regulator, LEDs, bridge; shunt +-1 %, uncalibrated)"
PMIC = "Pi PMIC telemetry (vcgencmd pmic_read_adc, V x I summed over rails at 1 Hz; excludes converter loss and 5 V-direct loads)"

POWER_SESSIONS = [
    PowerSession("f411", "rnnoise", FLOAT_GRU, "as shipped", "results/power/f411_rnnoise", "power_idle.csv", INA,
                 "power_logger_f411/README.md (F411 session 2026-11-04, paced leg 2026-11-05)",
                 "NO FIT row: the paced figure is the energy to process one audio-second in 3.45 s of wall time"),
    PowerSession("h723", "rnnoise", FLOAT_GRU, "as shipped", "results/power/h723_rnnoise", "replug_before.csv", INA,
                 "power_logger_f411/README.md (H723 session and replug A/B, 2026-11-04)",
                 "session idle leg ran before any stream (fresh-boot state, 825 mW); model idle = the post-stream state "
                 "(replug A/B leg A: the session's own board, warm, one paced corpus behind it, 748.5 mW; leg D after the replug 749.5)"),
    PowerSession("s3", "rnnoise", FLOAT_GRU, "as shipped", "results/power/s3_rnnoise", "power_idle.csv", INA,
                 "power_logger_f411/README.md (S3 session 2026-11-04)",
                 "NO FIT row: the paced figure is the energy to process one audio-second in 1.57 s of wall time"),
    PowerSession("s3", "rnnoise", FLOAT_GRU, "+FW", "results/power/s3_rnnoise_fw", "power_idle.csv", INA,
                 "power_logger_f411/README.md (S3 +FW session 2026-11-05)",
                 "FIT; paced = unpaced because the per-utterance transfer sets the batch pace (wall 1.166 x audio)"),
    PowerSession("rpi5", "rnnoise", FLOAT_GRU, "as shipped", "results/power/rpi5_rnnoise", "power_idle.csv", PMIC,
                 "benchmark_rpi5/RESULTS.md (RNNoise power on the Pi, 2026-11-04)",
                 "P_active includes the batch driver running on the measured machine: an upper bound"),
]

# Which session supplies a board's (P_idle, P_active) for rows that had no session of their own.
BOARD_POWER = {"f411": ("f411", "as shipped"), "h723": ("h723", "as shipped"),
               "s3": ("s3", "as shipped"), "rpi5": ("rpi5", "as shipped")}
