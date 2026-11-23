#!/usr/bin/env python3
"""
budget_model.py -- task 4: per-platform constraint budget tables and the
fit/no-fit predictions that must be written down BEFORE any model is touched.

Modelled on Table 1 of TinyLSTMs (Constraint / Specification / Rationale), whose
compute row is "<= 1.55 MOps/inf", i.e. their measured 155 MOps/s times a 10 ms
latency budget. The same construction is used here, but with OUR measured
constants and OUR locked framing, and with one difference that is the point of
this project: TinyLSTMs' cost model covers inference only and excludes the STFT
and mel front end, whereas every cost below is built from kernels we measured on
the actual part -- transforms and transcendentals included.

All hardware constants come from the on-hardware runs recorded in CLAUDE.md:
  F411  2026-09-10, 96 MHz, CMSIS-DSP 1.10.0, -O2 -g3 -DNDEBUG
  H723  2026-09-11, 550 MHz, DTCM + I$/D$ on, same flags and same harness
  S3    2026-09-11, 240 MHz, esp-dsp 1.5.2 + esp-nn 1.3.2, internal SRAM
  RPi5  2026-09-12, 2.4 GHz, gcc 14.2.0 -O2 -mcpu=cortex-a76, performance
        governor, core 2, no thermal throttling
Ops convention is BENCH_OPS_PER_MAC = 2 throughout, matching both the bench
harness and Valin's own "we count a multiply-add as two operations".

All four tiers of the ladder are present (RPi5 added 2026-09-12 when that tier
was reinstated and measured). The RPi5 row is deliberately NOT symmetric with
the other three and the asymmetries are annotated in place rather than smoothed
over: storage and RAM are not binding constraints there, its int8 constant comes
from compiler auto-vectorisation rather than a vendor kernel, and one of its
kernel constants is an estimate.

Run:  python3 budget_model.py
"""

# ----------------------------------------------------------------------------
# Locked audio configuration (CLAUDE.md, decided 2026-09-08)
# ----------------------------------------------------------------------------
SR          = 16_000
FRAME       = 512          # 32 ms
HOP         = 256          # 16 ms
BINS        = FRAME // 2 + 1   # 257
HOP_S       = HOP / SR     # 0.016 s

# RNNoise is a documented deviation: 48 kHz, 20 ms frame, 10 ms hop.
RNN_HOP_S   = 0.010

# ----------------------------------------------------------------------------
# Measured hardware constants
# ----------------------------------------------------------------------------
class Plat:
    def __init__(self, name, hz, flash, ram, ram_note, infra_flash, infra_ram, k,
                 short=None, libs="CMSIS-DSP + TinyUSB CDC",
                 intlib="`arm_mat_vec_mult_q7`", f32lib="`arm_mat_vec_mult_f32`",
                 per_bin=14, mem_binding=True, mem_note=None,
                 dtype_note="SIMD not datatype"):
        self.name, self.hz = name, hz
        # Per-bin gain-rule arithmetic that is not a named kernel (divides, max,
        # MAC updates). ESTIMATED on every tier -- f32 divide dominates and is
        # not pipelined on the MCUs. Was an if-chain keyed on platform identity,
        # which would silently have handed a fourth platform the S3's constant;
        # made explicit 2026-09-12 when the RPi5 row was added.
        self.per_bin = per_bin
        # False where flash/SRAM are simply not binding constraints (a Linux SBC
        # with an SD card and GBs of DRAM). Printing a "free bytes" figure there
        # would invite a fit/no-fit reading of a question that does not arise.
        self.mem_binding, self.mem_note = mem_binding, mem_note
        self.dtype_note = dtype_note
        self.short = short or name
        self.libs, self.intlib, self.f32lib = libs, intlib, f32lib
        self.flash, self.ram, self.ram_note = flash, ram, ram_note
        self.infra_flash, self.infra_ram = infra_flash, infra_ram
        self.k = k                      # measured kernel cycle counts
    def frame_budget(self, hop_s=HOP_S):
        return int(self.hz * hop_s)
    def free_flash(self):
        return self.flash - self.infra_flash
    def free_ram(self):
        return self.ram - self.infra_ram

# Kernel cycle counts, minimum of 64 reps, interrupts masked.
# "infra" = bare project + CMSIS-DSP (tables trimmed) + TinyUSB CDC, i.e. what a
# deployed build links before a single algorithm byte. The bench harness's 64 KB
# weight pool is NOT counted: a deployed build reclaims it.
F411 = Plat(
    "STM32F411CEU6", 96_000_000,
    flash=512*1024, ram=128*1024, ram_note="SRAM",
    infra_flash=48_200, infra_ram=91_600 - 65_536,   # minus the bench pool
    k=dict(rfft512=35_508, mag257=7_209, expf257=38_949, logf257=37_289,
           sqrtf257=5_940, tanhf257=51_879, mult512=3_895,
           cyc_per_mac_q7=2.69, cyc_per_mac_f32=5.80,
           cyc_per_op_fft_plainc=110_585/(5*512*9),
           cyc_per_op_dot_f32=3_251/(2*512)), short="F411", per_bin=20)

H723 = Plat(
    "STM32H723VGT6", 550_000_000,
    flash=1024*1024, ram=128*1024, ram_note="DTCM (AXI 320 KB more)",
    infra_flash=52_256, infra_ram=91_852 - 65_536,
    k=dict(rfft512=19_862, mag257=6_430, expf257=25_348, logf257=23_703,
           sqrtf257=6_184, tanhf257=47_353, mult512=1_826,
           cyc_per_mac_q7=1.28, cyc_per_mac_f32=4.03,
           q7_cycles_256=84_191,   # measured; the throughput line prints 856 from this,
                                   # the v1 predictor keeps the rounded 1.28 (2026-10-22)
           cyc_per_op_fft_plainc=68_867/(5*512*9),
           cyc_per_op_dot_f32=1_952/(2*512)), short="H723", per_bin=12)

# ESP32-S3R8 @ 240 MHz, measured 2026-09-11.
#
# Flash is the subtle one and is reported for the DEPLOYED partition, not the
# chip. The module carries 16 MB, but ESP-IDF's default partition table gives
# the factory app 0x100000 = 1 MB. That is a CONFIGURATION choice, not a
# platform limit -- a custom partition table can hand the app most of the 16 MB
# -- and it decides a verdict below, so both numbers are reported rather than
# one being quietly chosen.
#
# Kernel constants use the vendor libraries wherever a real deployment would
# link them, matching how the F411/H723 rows use CMSIS:
#   int8 matvec -> esp-nn  esp_nn_fully_connected_s8   0.795 cyc/MAC
#   f32  matvec -> per-row dsps_dotprod_f32            1.886 cyc/MAC
#                  (NOT dspm_mult_f32, which loses to plain C -- see CLAUDE.md)
#   rfft/dot/mult -> esp-dsp
# mag/expf/logf are plain C + libm: esp-dsp has no vectorised equivalent, the
# same structural situation as CMSIS-DSP's Helium-only transcendentals.
S3 = Plat(
    "ESP32-S3R8", 240_000_000,
    flash=1024*1024,                 # DEPLOYED app partition, not the 16 MB chip
    ram=341_760, ram_note="DIRAM (+8 MB PSRAM, but see jitter)",
    infra_flash=239_920, infra_ram=156_058 - 65_536,
    k=dict(rfft512=33_214, mag257=21_387, expf257=59_441, logf257=55_224,
           sqrtf257=18_818, tanhf257=86_991, mult512=4_174,
           cyc_per_mac_q7=0.795, cyc_per_mac_f32=1.886,
           cyc_per_op_fft_plainc=91_153/(5*512*9),
           cyc_per_op_dot_f32=867/1024),
    short="S3", libs="esp-dsp + esp-nn + TinyUSB CDC",
    intlib="`esp_nn_fully_connected_s8`",
    f32lib="per-row `dsps_dotprod_f32`")

S3_FLASH_CHIP = 16*1024*1024

# Raspberry Pi 5 Model B Rev 1.1 @ 2.4 GHz, measured 2026-09-12.
#
# THIS ROW IS NOT LIKE THE OTHER THREE, in three ways that must travel with any
# number taken from it:
#
# 1. STORAGE AND MEMORY ARE NOT BINDING. Flash is a 59.5 GB SD card (55.2 GB
#    free) and RAM is 8 GB. There is no fit/no-fit question to ask, so
#    mem_binding=False and the report prints "not binding" rather than a
#    free-bytes figure that invites one. The `flash`/`ram` values below are the
#    real capacities, present so the table can state them, not so they can be
#    subtracted from.
#
# 2. THE INT8 CONSTANT IS A COMPILER RESULT, NOT A LIBRARY ONE. The other three
#    rows use the vendor kernel a deployment would link (CMSIS q7, esp-nn). Here
#    the PLAIN-C loop is auto-vectorised by gcc 14.2 at -O2 into
#    `sdot v31.4s, v30.16b, v29.16b` -- 16 MACs per instruction -- confirmed by
#    disassembly. So 0.1017 cyc/MAC is real, but it also does NOT fold in the
#    zero-point and output requantisation that esp_nn_fully_connected_s8 and
#    arm_nn_vec_mat_mult_t_s8 do. It is therefore NOT a like-for-like deployment
#    figure and must be labelled wherever it is compared.
#
#    The f32 constant is the mirror image and the reason is the same decision:
#    float addition is not associative, so with -ffast-math declined (it would
#    forfeit the cross-platform checksum equivalence task 22 rests on) gcc keeps
#    the f32 reduction SCALAR -- `fmadd s31, s30, s29, s31`. 1.809 cyc/MAC.
#
# 3. rfft512 IS THE ONE ESTIMATED KERNEL ON THIS TIER. The probe measures a
#    plain-C COMPLEX 512-point FFT (20,400 cyc); it has no real-FFT row, because
#    the other tiers got theirs from CMSIS/esp-dsp and this tier has no vendor
#    pairing yet. A real transform of N real samples is conventionally half the
#    work, so 10,200 is used and FLAGGED. Replace it with a measured NEON/ACL
#    or FFTW figure when the pairing lands -- it appears twice in every
#    classical row (forward and inverse) and is the largest single term there.
RPI5 = Plat(
    "Raspberry Pi 5 (BCM2712)", 2_400_000_000,
    flash=59.5*1024*1024*1024, ram=8*1024*1024*1024,
    ram_note="LPDDR4X",
    infra_flash=0, infra_ram=0,
    k=dict(rfft512=10_200,              # ESTIMATE: fft_c_512 / 2. See note 3.
           mag257=622, expf257=3_956, logf257=4_090,
           sqrtf257=756, tanhf257=5_156, mult512=178,
           cyc_per_mac_q7=6_665/65_536,     # 0.1017, auto-SDOT plain C
           cyc_per_mac_f32=29_643/16_384,   # 1.809, scalar fmadd
           cyc_per_op_fft_plainc=20_400/(5*512*9),
           cyc_per_op_dot_f32=977/(2*512)),
    short="RPi5", libs="none linked beyond libc/libm; no USB stack",
    intlib="plain C auto-vectorised to `sdot`",
    f32lib="plain C, scalar `fmadd` (strict IEEE)",
    per_bin=6,                       # ESTIMATE, as on every tier
    mem_binding=False,
    mem_note="SD card 59.5 GB (55.2 GB free) / 8 GB LPDDR4X — not binding",
    # The other tiers' ratio says "SIMD, not the narrower type". Here the cause
    # is one level up and the boilerplate would be wrong: the compiler
    # vectorised the int8 reduction and was FORBIDDEN from vectorising the f32
    # one, because float addition is not associative and -ffast-math is declined.
    dtype_note="compiler vectorised int8 (`sdot`) and could not vectorise the "
               "f32 reduction under strict IEEE — a toolchain result, not a "
               "datatype or microarchitecture one")

PLATFORMS = [F411, H723, S3, RPI5]

# ----------------------------------------------------------------------------
# Algorithm cost models, built FROM the measured kernels
# ----------------------------------------------------------------------------
def classical(p, variant):
    """Per-frame cycles for a 257-bin classical enhancer at 16 kHz.

    THIS IS THE v1 PREDICTOR, KEPT VERBATIM. It is what the recorded
    fit/no-fit predictions were generated from (2026-09-12) and it is
    FALSIFIED in its proportions on every tier (2026-10-14/16): the per-bin
    constant is 2.9-18x low, there is no noise-tracker term, and no
    sample-conversion term. It is retained so the predictions still regenerate
    exactly as recorded; the quotable classical rows come from
    classical_measured() below.

    Common path (all three): window, rFFT, magnitude, per-bin gain rule, apply
    gain, iRFFT, overlap-add. The gain rule is what separates them, and on an MCU
    it is dominated by transcendentals -- which is exactly the cost TinyLSTMs'
    inference-only model never sees.
    """
    k = p.k
    c = {}
    c['window']      = k['mult512']
    c['rfft']        = k['rfft512']
    c['magnitude']   = k['mag257']
    c['apply_gain']  = k['mult512']        # complex mul, same shape as mult
    c['irfft']       = k['rfft512']
    c['overlap_add'] = k['mult512'] // 2   # 256-pt add
    # Per-bin arithmetic that is not a named kernel: divides, max, MAC updates.
    # 20 cyc/bin on M4, 12 on M7 -- f32 divide dominates and is not pipelined.
    per_bin = p.per_bin
    if variant == 'specsub':
        c['gain_rule'] = BINS * per_bin + k['sqrtf257']
    elif variant == 'wiener':
        c['gain_rule'] = BINS * per_bin * 2
    elif variant == 'mmse_stsa':
        # Ephraim-Malah: exp(-v/2) and log for the decision-directed a priori
        # SNR, plus a sqrt. Bessel I0/I1 are assumed table-approximated, which is
        # the standard implementation -- a direct series would cost far more.
        c['gain_rule'] = (BINS * per_bin * 2 + k['expf257']
                          + k['logf257'] + k['sqrtf257'])
    return c

# ----------------------------------------------------------------------------
# Classical arm, MEASURED stage constants (cycles per 16 ms hop)
#
# From the stage-stamp builds of the real port on each tier: DWT between the
# stages of cdsp_process() on the STM32s and CCOUNT on the S3 (2026-10-14,
# benchmark_{f411,h723,s3}/RESULTS.md), CLOCK_MONOTONIC on the RPi5 (2026-10-16,
# benchmark_rpi5/RESULTS.md; ns x 2.4 at the verified 2.4 GHz). Single utterance
# p232_001, 110 hops, sums agree with STATS to <1 %. Three terms the v1 model
# did not have at all: the noise tracker, the per-bin loop at its real cost,
# and the output conversion (overlap-add, clamp, lrintf, emit).
# ----------------------------------------------------------------------------
STAGES = {
    'F411': dict(window=8_835,  fwd=38_997, mag=3_377, mcra=25_217, vad=52_674,
                 gain=dict(specsub=24_821, wiener=27_135, mmse_stsa=130_059),
                 inv=43_219, conv=25_397, fft="CMSIS `arm_rfft_fast_f32`"),
    'H723': dict(window=4_597,  fwd=21_097, mag=2_858, mcra=28_904, vad=47_002,
                 gain=dict(specsub=26_298, wiener=27_194, mmse_stsa=113_837),
                 inv=22_556, conv=19_265, fft="CMSIS `arm_rfft_fast_f32`, DTCM"),
    'S3':   dict(window=18_585, fwd=72_673, mag=3_121, mcra=47_437, vad=105_318,
                 gain=dict(specsub=65_235, wiener=63_805, mmse_stsa=258_430),
                 inv=76_102, conv=37_181, fft="esp-dsp complex `dsps_fft2r_fc32` both ways (no packed real inverse)"),
    'RPi5': dict(window=1_786,  fwd=27_521, mag=617,   mcra=5_218,  vad=8_393,
                 gain=dict(specsub=4_409, wiener=4_718, mmse_stsa=23_086),
                 inv=28_704, conv=3_730, fft="the oracle's plain-C radix-2 (`cdsp_fft_c.c`), no vendor pairing"),
}
# Corpus medians (824 VoiceBank utterances x variant, median over utterances of
# the per-hop median), the figures to QUOTE; the stamp totals above are one
# utterance and MMSE in particular is data-dependent (6-10 % spread).
CORPUS_MEDIAN = {
    'F411': dict(specsub=170_690, wiener=172_742, mmse_stsa=297_663),
    'H723': dict(specsub=126_806, wiener=127_744, mmse_stsa=224_680),
    'S3':   dict(specsub=327_692, wiener=327_162, mmse_stsa=539_885),
    'RPi5': dict(specsub=71_422,  wiener=71_777,  mmse_stsa=92_443),
}

def classical_measured(p, variant, tracker='mcra'):
    """Per-hop cycles from the measured stage constants. Same stage order as
    cdsp_process(); every term is a measurement on the tier named."""
    st = STAGES[p.short]
    return {'shift+window': st['window'], 'forward FFT': st['fwd'],
            '|Y|^2': st['mag'], f'noise tracker ({tracker})': st[tracker],
            'a priori SNR + gain + apply': st['gain'][variant],
            'inverse FFT': st['inv'], 'overlap-add + lrintf + emit': st['conv']}

def measured_section():
    print("\n\n## Classical rows from MEASURED stage constants (2026-10-14 / 2026-10-16)\n")
    print("The v1 predictions above are kept as recorded. These are the rows to quote. "
          "Every term is a stage-stamp measurement of the real port on the tier named; "
          "corpus medians are over 824 utterances.\n")
    print("| stage | " + " | ".join(p.short for p in PLATFORMS) + " |")
    print("|---|" + "---:|"*len(PLATFORMS))
    for key, label in [('window','shift in + window'), ('fwd','forward FFT'),
                       ('mag','\\|Y\\|²'), ('mcra','noise tracker, MCRA'),
                       ('vad','noise tracker, VAD'), ('inv','inverse FFT'),
                       ('conv','overlap-add + `lrintf` + emit')]:
        print(f"| {label} | " + " | ".join(f"{STAGES[p.short][key]:,}" for p in PLATFORMS) + " |")
    for v, label in [('specsub','gain loop, spectral subtraction'),
                     ('wiener','gain loop, Wiener'), ('mmse_stsa','gain loop, MMSE-STSA')]:
        print(f"| {label} | " + " | ".join(f"{STAGES[p.short]['gain'][v]:,}" for p in PLATFORMS) + " |")
    print("| *FFT in use* | " + " | ".join(STAGES[p.short]['fft'] for p in PLATFORMS) + " |")

    print("\n**Per-bin cost of the gain loop (cycles / bin, 257 bins) — the v1 constant "
          "beside it:**\n")
    print("| variant | " + " | ".join(f"{p.short} (v1 {p.per_bin})" for p in PLATFORMS) + " |")
    print("|---|" + "---:|"*len(PLATFORMS))
    for v, label in [('specsub','spectral subtraction'), ('wiener','Wiener'), ('mmse_stsa','MMSE-STSA')]:
        print(f"| {label} | " + " | ".join(
            f"**{STAGES[p.short]['gain'][v]/BINS:.0f}**" for p in PLATFORMS) + " |")

    print("\n**Totals, MCRA tracker: stamp total (one utterance) / corpus median / RTF, "
          "against the v1 prediction:**\n")
    print("| variant | " + " | ".join(f"{p.short} stamp | median | RTF | v1 | ratio" for p in PLATFORMS) + " |")
    print("|---|" + "---:|"*(5*len(PLATFORMS)))
    for v, label in [('specsub','Spectral subtraction'), ('wiener','Wiener'), ('mmse_stsa','MMSE-STSA')]:
        cells = []
        for p in PLATFORMS:
            tot = sum(classical_measured(p, v).values())
            med = CORPUS_MEDIAN[p.short][v]
            v1  = sum(classical(p, v).values())
            cells.append(f"{tot:,} | {med:,} | {med/p.frame_budget():.4f} | {v1:,} | {med/v1:.2f}×")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\n**Where each tier's hop goes (MCRA, spectral subtraction): transforms vs "
          "per-bin loops vs the rest.** This is the finding the v1 model could not see.\n")
    print("| | " + " | ".join(p.short for p in PLATFORMS) + " |")
    print("|---|" + "---:|"*len(PLATFORMS))
    def share(p, keys):
        st = STAGES[p.short]; tot = sum(classical_measured(p,'specsub').values())
        return sum(st[k] if k != 'gain' else st['gain']['specsub'] for k in keys) / tot * 100
    print("| both transforms | " + " | ".join(f"{share(p,['fwd','inv']):.0f} %" for p in PLATFORMS) + " |")
    print("| tracker + gain loop (divide-bound) | " + " | ".join(f"{share(p,['mcra','gain']):.0f} %" for p in PLATFORMS) + " |")
    print("| window, magnitude, conversion | " + " | ".join(f"{share(p,['window','mag','conv']):.0f} %" for p in PLATFORMS) + " |")
    f, pi = STAGES['F411'], STAGES['RPi5']
    print(f"\nOn the per-bin loops the RPi5 is **{(f['mcra']+f['gain']['specsub'])/(pi['mcra']+pi['gain']['specsub']):.1f}×** "
          f"the F411 per cycle; on the transforms only **{(f['fwd']+f['inv'])/(pi['fwd']+pi['inv']):.1f}×**, and "
          f"**{(STAGES['H723']['fwd']+STAGES['H723']['inv'])/(pi['fwd']+pi['inv']):.2f}×** the H723 — because the Pi runs "
          "the oracle's plain-C radix-2 FFT against CMSIS on the STM32s. The Pi's deficit on this arm is a "
          "library, exactly as the F411's was in the probe; the NEON/ACL pairing is where its headroom is.")

RNNOISE_W   = 87_503       # verified, Valin 2018 sec. IV
RNNOISE_NN_OPS   = 175_000 # ops/frame, multiply-add = 2 ops
RNNOISE_FFT_OPS  = 75_000  # IFFT + 2 FFT, ~7.5 Mflops at 100 fps
RNNOISE_PITCH_OPS= 100_000 # ~10 Mflops at 100 fps
RNNOISE_OTHER_OPS= 50_000  # 40 Mflops total - the three above

def rnnoise(p):
    k = p.k
    c = {}
    # NN weights are quantized to int8 (Valin: "no loss of performance"), so the
    # deployed kernel is a q7 matvec. RNNOISE_NN_OPS counts 2 ops per MAC.
    c['nn_int8']  = int((RNNOISE_NN_OPS/2) * k['cyc_per_mac_q7'])
    # 480-point transform: NOT a power of two, so arm_rfft_fast_f32 cannot do it
    # and Xiph's mixed-radix FFT ships with the port. Costed at our PLAIN-C FFT
    # rate, which is the honest proxy -- CMSIS is unavailable here by construction.
    c['fft_480'] = int(RNNOISE_FFT_OPS * k['cyc_per_op_fft_plainc'])
    # Pitch search is autocorrelation: dot-product shaped.
    c['pitch']   = int(RNNOISE_PITCH_OPS * k['cyc_per_op_dot_f32'])
    c['other']   = int(RNNOISE_OTHER_OPS * k['cyc_per_op_dot_f32'])
    return c

# DTLN: VERIFIED 2026-09-16 against Westhausen & Meyer, arXiv:2005.07551v2
# (now on disk as papers/2005.07551v2.pdf) and against the weights the
# authors actually ship. The paper's Table 1 gives "987 K" and its section 2.4
# gives 128 units in each of four LSTM layers, frame 32 ms, shift 8 ms, FFT 512,
# 256 conv filters. Worked out layer by layer that is:
#
#   core 1 (STFT)     LSTM(257->128) 197,632 + LSTM(128->128) 131,584
#                     + Dense(128->257) 33,153                 = 362,369
#   core 2 (learned)  Conv1D 512->256 (no bias) 131,072 + iLN 512
#                     + LSTM(256->128) 197,120 + LSTM(128->128) 131,584
#                     + Dense(128->256) 33,024
#                     + Conv1D 256->512 (no bias) 131,072      = 624,384
#                                                        total = 986,753
#
# and the shipped int8 tflite files confirm it exactly: 983,680 B of INT8 weight
# tensors + 3,073 float32 bias/scale params (4x128x4 LSTM biases + 257 + 256
# dense + 512 iLN) = 986,753. The old 986,241 was short by exactly the instant
# layer norm's 512 gamma/beta -- the reconstruction had missed one layer.
#
# NOTE the paper's own config uses norm_stft=False ("behaviour like in the
# paper", DTLN_model.py:337), i.e. ONE instant layer norm, not two.
DTLN_PARAMS = 986_753
DTLN_NOTE   = ("verified against Westhausen & Meyer arXiv:2005.07551v2 "
               "Table 1 (987 K) and against the shipped int8 weights, which "
               "reproduce it exactly")

# DTLN's NATIVE hop is 8 ms, half the project's locked 16 ms (paper section 2.4:
# "The frame size is 32 ms and the shift 8 ms"). The frame itself is 32 ms / 512
# samples, the SAME as the locked configuration -- only the hop differs, so the
# per-frame cost is unchanged and the frame RATE doubles.
DTLN_NATIVE_HOP_S = 0.008

# The deployable artifact is the tflite flatbuffer, not the bare weight count:
# it carries the graph, per-tensor quantization parameters and the float32
# biases too. MEASURED from the authors' own pretrained models (breizhn/DTLN,
# pretrained_model/), which is what an Arm A "as-shipped" run would flash:
DTLN_TFLITE_INT8_BYTES = 1_027_232    # model_quant_1 377,776 + model_quant_2 649,456
DTLN_TFLITE_F32_BYTES  = 3_975_748    # model_1 1,459,944 + model_2 2,515,804

def dtln(p, dtype='int8'):
    k = p.k
    cpm = k['cyc_per_mac_q7'] if dtype=='int8' else k['cyc_per_mac_f32']
    c = {'nn': int(DTLN_PARAMS * cpm)}
    # DTLN is a hybrid: core 1 works on the STFT magnitude, so it pays the
    # transform cost too -- the exact cost TinyLSTMs excludes.
    c['stft'] = 2*k['rfft512'] + k['mag257'] + 2*k['mult512']
    return c

# ----------------------------------------------------------------------------
# Memory models
# ----------------------------------------------------------------------------
# Weight footprint is exact: one byte per int8 weight, four per float32.
def weight_bytes(params, dtype):
    return params * (1 if dtype == 'int8' else 4)

# Working memory = everything live per frame that is not a weight. These are
# computed from the locked configuration, not guessed.
WM_CLASSICAL = (FRAME*4      # analysis window buffer
              + FRAME*4      # rFFT in/out (packed real)
              + BINS*4       # magnitude
              + BINS*4       # gain
              + BINS*4       # noise estimate
              + BINS*4       # a priori SNR state (decision-directed)
              + HOP*4)       # overlap-add tail
WM_RNNOISE   = (960*4        # 480-pt transform, complex
              + 215*4        # GRU state
              + 1024*4       # pitch history at 12 kHz
              + 42*4*8)      # band energies + feature history
# DTLN: LSTM state (4 layers x 128 units x {h,c}) plus a TFLite Micro tensor
# arena. The arena is the dominant and least predictable term; 40 KB was the
# estimate -- kept here so the v1 predictions re-emit verbatim. MEASURED
# 2026-10-19 (DTLN_DEPLOY below): 63,920 B int8 / 68,512 B 16x8, 42-52 % low.
WM_DTLN      = 4*128*2*4 + 40*1024

def fits(used, avail):
    return used <= avail

def pct(x, of):
    return 100.0*x/of

def report():
    print("# Task 4 — Constraint budget tables (generated by budget_model.py)\n")
    print(f"Locked configuration: {SR/1000:g} kHz, {FRAME}-sample frame "
          f"({FRAME/SR*1000:g} ms), {HOP}-sample hop ({HOP_S*1000:g} ms), "
          f"{BINS} bins. Ops convention: 2 ops per MAC.\n")

    for p in PLATFORMS:
        fb = p.frame_budget()
        print(f"\n## {p.name} @ {p.hz/1e6:g} MHz\n")
        print("| Constraint | Specification | Rationale |")
        print("|---|---|---|")
        print(f"| Compute complexity | **{fb:,} cycles/frame** "
              f"({fb/1e6:.3f} Mcyc) | {HOP_S*1000:g} ms hop at "
              f"{p.hz/1e6:g} MHz, i.e. RTF = 1.0 |")
        q7_rate = (2*65_536/p.k['q7_cycles_256'] if 'q7_cycles_256' in p.k else 2/p.k['cyc_per_mac_q7'])
        print(f"| Sustained throughput | {q7_rate*p.hz/1e6:,.0f} MOps/s int8, "
              f"{2/p.k['cyc_per_mac_f32']*p.hz/1e6:,.0f} MOps/s f32 | measured "
              f"{p.intlib} / {p.f32lib} |")
        if p.mem_binding:
            print(f"| Model size | **{p.free_flash():,} B** of {p.flash//1024} KB flash | "
                  f"after {p.infra_flash:,} B infrastructure ({p.libs}) |")
            print(f"| Working memory | **{p.free_ram():,} B** of {p.ram//1024} KB "
                  f"{p.ram_note} | after {p.infra_ram:,} B infrastructure; "
                  f"bench weight pool reclaimed |")
        else:
            # Printing a free-bytes figure here would invite a fit/no-fit reading
            # of a question that does not arise on this tier.
            print(f"| Model size | **not binding** | {p.mem_note} |")
            print(f"| Working memory | **not binding** | {p.mem_note} |")
        print(f"| Data type | int8 weights preferred | measured "
              f"{p.k['cyc_per_mac_f32']/p.k['cyc_per_mac_q7']:.2f}x over f32 "
              f"on this part, {p.dtype_note} |")
        if not p.mem_binding:
            print()
            print("*Caveats specific to this tier, which must travel with any number "
                  "taken from the row above:* the int8 constant comes from **plain C "
                  "auto-vectorised by the compiler into `sdot`** (confirmed by "
                  "disassembly), not from a vendor kernel, and unlike "
                  "`esp_nn_fully_connected_s8` / `arm_nn_vec_mat_mult_t_s8` it does "
                  "**not** fold in the zero-point and output requantisation — so it is "
                  "not a like-for-like deployment figure. The f32 constant is scalar "
                  "because float addition is not associative and this project declines "
                  "`-ffast-math`. The compute row additionally assumes the "
                  "**`performance` governor, one pinned core, and no thermal "
                  "throttling**, all three verified at run time and recorded in "
                  "`g_plat`; `rfft512` is **estimated** as half the measured complex "
                  "FFT pending a NEON/ACL pairing.")

    # --- per-algorithm cost, measured-kernel derived ------------------------
    print("\n\n## Measured-kernel cost per frame (v1 model — the recorded predictions)\n")
    print("Every line below is built from kernels timed on the actual silicon, "
          "transforms and transcendentals included — not from an inference-only "
          "op count. **The classical rows of this v1 model are superseded by the "
          "measured-stage section at the end** (per-bin constant and tracker term "
          "falsified 2026-10-14/16); they are kept because the predictions were "
          "recorded from them.\n")
    hdr = "| Algorithm | " + " | ".join(
        f"{p.short} cyc | % frame" for p in PLATFORMS) + " |"
    print(hdr); print("|---" * (1+2*len(PLATFORMS)) + "|")
    rows = []
    for variant, label in [('specsub','Spectral subtraction'),
                           ('wiener','Wiener filter'),
                           ('mmse_stsa','MMSE-STSA (Ephraim–Malah)')]:
        cells = []
        for p in PLATFORMS:
            tot = sum(classical(p, variant).values())
            cells.append((tot, pct(tot, p.frame_budget())))
        rows.append((label, cells))
        print(f"| {label} | " + " | ".join(
            f"{t:,} | {q:.1f}%" for t,q in cells) + " |")
    # RNNoise runs on its own 10 ms clock
    cells = []
    for p in PLATFORMS:
        tot = sum(rnnoise(p).values())
        cells.append((tot, pct(tot, p.frame_budget(RNN_HOP_S))))
    print(f"| RNNoise *(48 kHz, 10 ms hop)* | " + " | ".join(
        f"{t:,} | **{q:.1f}%**" for t,q in cells) + " |")
    for dt in ('int8',):
        cells = []
        for p in PLATFORMS:
            tot = sum(dtln(p, dt).values())
            cells.append((tot, pct(tot, p.frame_budget())))
        print(f"| DTLN ({dt}) | " + " | ".join(
            f"{t:,} | **{q:.1f}%**" for t,q in cells) + " |")

    # --- memory ------------------------------------------------------------
    print("\n\n## Memory\n")
    print("| Algorithm | weights | working mem | " +
          " | ".join(f"{p.short} flash | {p.short} RAM" for p in PLATFORMS) + " |")
    print("|---|---:|---:|" + ":--:|"*(2*len(PLATFORMS)))
    def memrow(label, wb, wm):
        cells = []
        for p in PLATFORMS:
            if not p.mem_binding:
                cells += ["n/a", "n/a"]        # never a constraint on this tier
                continue
            cells.append("fit" if fits(wb, p.free_flash()) else "**NO**")
            cells.append("fit" if fits(wm, p.free_ram()) else "**NO**")
        print(f"| {label} | {wb:,} | {wm:,} | " + " | ".join(cells) + " |")
    memrow("Classical (any)", 0, WM_CLASSICAL)
    memrow("RNNoise int8", weight_bytes(RNNOISE_W,'int8'), WM_RNNOISE)
    memrow("RNNoise f32",  weight_bytes(RNNOISE_W,'f32'),  WM_RNNOISE)
    # The DTLN rows use the MEASURED tflite flatbuffers, not params x bytes:
    # what goes in flash is the artifact, graph and biases included.
    memrow("DTLN int8 *(tflite, measured)*", DTLN_TFLITE_INT8_BYTES, WM_DTLN)
    memrow("DTLN f32 *(tflite, measured)*", DTLN_TFLITE_F32_BYTES, WM_DTLN)

    # --- the predictions ---------------------------------------------------
    print("\n\n## Fit / no-fit predictions — recorded before any model is touched\n")
    for p in PLATFORMS:
        print(f"\n**{p.name}**\n")
        fb, fbr = p.frame_budget(), p.frame_budget(RNN_HOP_S)
        for variant, label in [('specsub','Spectral subtraction'),
                               ('wiener','Wiener'),
                               ('mmse_stsa','MMSE-STSA')]:
            tot = sum(classical(p, variant).values())
            print(f"- {label}: **{'FIT' if tot<=fb else 'NO FIT'}** "
                  f"— RTF {tot/fb:.3f}")
        tot = sum(rnnoise(p).values())
        wb  = weight_bytes(RNNOISE_W,'int8')
        if p.mem_binding:
            ok = tot<=fbr and fits(wb,p.free_flash()) and fits(WM_RNNOISE,p.free_ram())
            extra = f", weights {wb:,} B, WM {WM_RNNOISE:,} B"
        else:
            ok, extra = tot<=fbr, ", memory not binding"
        print(f"- RNNoise int8: **{'FIT' if ok else 'NO FIT'}** — RTF {tot/fbr:.3f}"
              + extra)
        tot = sum(dtln(p,'int8').values())
        # The flash test counts BOTH terms. It did not until 2026-09-16, and the
        # consequence was a generated FIT for the H723 sitting directly above a
        # hand-written finding that said it misses by 51 KB: the row tested the
        # model against free flash while the runtime was accounted for in a
        # separate section that never reached a verdict. A generator that emits
        # an input to the verdict instead of the verdict is a trap of the same
        # family as the build switches -- it reads correctly and is wrong.
        wb  = DTLN_TFLITE_INT8_BYTES + TFLM_FLASH_EST
        if p.mem_binding:
            okc, okf, okr = tot<=fb, fits(wb,p.free_flash()), fits(WM_DTLN,p.free_ram())
            why = [n for n,v in (('compute',okc),('flash',okf),('RAM',okr)) if not v]
            tail = (f", model+runtime {wb:,} B vs {p.free_flash():,} B free"
                    + (f" — fails on: {', '.join(why)}" if why else ""))
        else:
            why  = [] if tot<=fb else ['compute']
            tail = ", memory not binding"
        print(f"- DTLN int8: **{'FIT' if not why else 'NO FIT'}** — RTF {tot/fb:.3f}"
              + tail)

# ----------------------------------------------------------------------------
# TFLite Micro runtime: a flash line item that is easy to omit and can flip a
# verdict. MEASURED 2026-09-16, replacing the 60 KB estimate that stood here --
# and the estimate was low by 2.2x.
#
# Method, which is the project's own "build twice and take the delta": link a
# Cortex-M image containing a MicroInterpreter plus a MicroMutableOpResolver
# carrying exactly DTLN's op set, against the same image without it, and take
# text+data. Flags mirror the firmware tiers' Release settings (-O2 -DNDEBUG,
# -ffunction-sections -fdata-sections -Wl,--gc-sections, nano.specs), plus
# TFLM's own required -DTF_LITE_STATIC_MEMORY -DTF_LITE_STRIP_ERROR_STRINGS.
# CMSIS-NN kernels included (OPTIMIZED_KERNEL_DIR=cmsis_nn). gc-sections was
# confirmed to do its job: no unregistered kernel survives in the image.
#
# The op set is READ OUT OF THE SHIPPED FLATBUFFERS, not assumed, and the
# reading is itself a finding: the LSTMs are NOT converted to
# UNIDIRECTIONAL_SEQUENCE_LSTM. They are unrolled into primitives --
# SPLIT / FULLY_CONNECTED / ADD / MUL / LOGISTIC / TANH / PACK / UNPACK /
# STRIDED_SLICE / RESHAPE -- so CMSIS-NN's LSTM kernel is never reached and the
# gate arithmetic lands in FULLY_CONNECTED plus elementwise ops. 16 distinct ops
# in the shipped graphs; 18 once QUANTIZE/DEQUANTIZE are added, which the
# full-integer conversion the MCU tiers require puts at the graph boundary.
TFLM_FLASH_M7_O2 = 135_316   # Cortex-M7 fpv5-d16, -O2, 18 ops  <- the figure used
TFLM_FLASH_M7_OS =  91_492   # same, -Os: 1.49x the old estimate even when squeezed
TFLM_FLASH_M4_O2 = 134_560   # Cortex-M4 fpv4-sp-d16, -O2, 18 ops
TFLM_BSS         =     904   # TFLM's own .bss, arena excluded
TFLM_FLASH_EST   = TFLM_FLASH_M7_O2

# TFLM REJECTS the quantization the authors shipped. Their model_quant_*.tflite
# are dynamic-range (hybrid): float32 activations against int8 weight tensors.
# tensorflow/lite/micro/kernels/fully_connected.cc:59 returns kTfLiteError for
# float32 input with a non-float32 filter -- verified in the source, not assumed.
# So DTLN's Arm A "as-shipped, costs nothing" path does NOT exist on an MCU: it
# needs a full-integer re-quantization with a representative dataset, which is
# what this project's 200-utterance calib subset is for.
DTLN_ARMA_NEEDS_REQUANT = True

def breakdown():
    print("\n\n## Where the cycles actually go\n")
    for p in PLATFORMS:
        fb, fbr = p.frame_budget(), p.frame_budget(RNN_HOP_S)
        print(f"\n**{p.name}** — MMSE-STSA (of {fb:,} cyc/frame):\n")
        c = classical(p, 'mmse_stsa'); t = sum(c.values())
        for k_, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"- `{k_}` {v:,} ({pct(v,t):.1f}% of algorithm)")
        print(f"\n**{p.name}** — RNNoise (of {fbr:,} cyc/frame at 10 ms hop):\n")
        c = rnnoise(p); t = sum(c.values())
        for k_, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"- `{k_}` {v:,} ({pct(v,t):.1f}% of algorithm)")
        dsp = c['fft_480'] + c['pitch'] + c['other']
        print(f"\n  Signal-processing front end = **{pct(dsp,t):.0f}%** of RNNoise; "
              f"the neural network is {pct(c['nn_int8'],t):.0f}%.")

def tflm_check():
    print("\n\n## The S3's flash budget depends on the partition table, not the chip\n")
    wb = DTLN_TFLITE_INT8_BYTES
    dep = S3.free_flash()
    chip = S3_FLASH_CHIP - S3.infra_flash
    print(f"- default IDF partition table (1 MB factory app): {dep:,} B free -> "
          f"DTLN int8 ({wb:,} B) **{'FITS' if wb<=dep else 'DOES NOT FIT'}**")
    print(f"- custom partition table over the 16 MB chip: {chip:,} B free -> "
          f"DTLN int8 **{'FITS' if wb<=chip else 'DOES NOT FIT'}**")
    print(f"- plus TFLite Micro ({TFLM_FLASH_EST:,} B measured): default "
          f"**{'FITS' if wb+TFLM_FLASH_EST<=dep else 'DOES NOT FIT'}**, "
          f"16 MB custom **{'FITS' if wb+TFLM_FLASH_EST<=chip else 'DOES NOT FIT'}**")
    print("\nThis is a configuration choice, not a platform limit, and it must be stated "
          "as such: the S3 is the only tier in the ladder where the flash verdict is set "
          "by the partition table rather than by the silicon.")

    print("\n\n## DTLN's flash verdict, now measured on both terms\n")
    print("Both line items were estimates when this table was first written and both "
          "were low. The model is the MEASURED tflite flatbuffer "
          f"({DTLN_TFLITE_INT8_BYTES:,} B, was {986_241:,} assumed) and the runtime is a "
          f"MEASURED link ({TFLM_FLASH_M7_O2:,} B at -O2, was {60*1024:,} assumed).\n")
    wb = DTLN_TFLITE_INT8_BYTES
    for p in PLATFORMS:
        if not p.mem_binding:
            print(f"\n**{p.name}** — not binding (see the RPi5 note)")
            continue
        free = p.free_flash()
        print(f"\n**{p.name}** — flash free {free:,} B")
        print(f"- DTLN int8 tflite alone: {wb:,} B → "
              f"{'fits, margin ' + format(free-wb, ',') + ' B' if wb<=free else '**does NOT fit, over by ' + format(wb-free, ',') + ' B**'}")
        for lbl, rt in (("-O2, as the firmware tiers are built", TFLM_FLASH_M7_O2),
                        ("-Os, squeezed for size", TFLM_FLASH_M7_OS)):
            tot = wb + rt
            print(f"- plus TFLite Micro ({lbl}): {rt:,} B → {tot:,} B → "
                  f"**{'FITS' if tot<=free else 'DOES NOT FIT'}** "
                  f"({'margin ' if tot<=free else 'over by '}{abs(free-tot):,} B)")

    print("\n**And the as-shipped path does not exist on an MCU at all.** The authors' "
          "`model_quant_*.tflite` are dynamic-range (hybrid) quantized: float32 "
          "activations against int8 weights. TFLite Micro's fully-connected kernel "
          "rejects that pairing outright (`fully_connected.cc:59`), so Arm A on every "
          "MCU tier requires a full-integer re-quantization against a representative "
          "dataset — which is exactly what the 200-utterance calib subset exists for. "
          "An Arm A cell that was costed at zero effort is a conversion job.")

    print("\n\n## DTLN at its NATIVE hop\n")
    print(f"The paper's shift is {DTLN_NATIVE_HOP_S*1000:g} ms, half this project's locked "
          f"{HOP/SR*1000:g} ms. The frame is 32 ms / {FRAME} samples either way, so the "
          "per-frame cost below is unchanged and only the frame RATE doubles.\n")
    for p in PLATFORMS:
        tot = sum(dtln(p, 'int8').values())
        locked = pct(tot, p.frame_budget())
        native = pct(tot, p.frame_budget(DTLN_NATIVE_HOP_S))
        print(f"- **{p.name}**: {tot:,} cyc/frame → RTF {locked/100:.3f} at the locked "
              f"{HOP/SR*1000:g} ms hop, **{native/100:.3f} at the native "
              f"{DTLN_NATIVE_HOP_S*1000:g} ms hop**"
              f"{'  — still fits' if native < 100 else '  — **NO LONGER FITS**'}")

# ----------------------------------------------------------------------------
# DTLN deployable artifacts -- MEASURED 2026-10-19 (dtln/quantize.py,
# dtln/tflm_check.py, dtln/RESULTS.md). The hybrid flatbuffer above is what the
# authors ship and what TFLM refuses; these are the full-integer re-exports
# TFLM accepts, with the arena the MicroInterpreter actually allocated.
# WM_DTLN's 40 KB arena estimate is CLOSED by these: it was low by 42-52 %.
# ----------------------------------------------------------------------------
DTLN_DEPLOY = {
    # tag: (flash bytes, arena bytes, quality verdict vs the float oracle)
    'int8 full-integer, split states': (1_097_712, 27_408 + 36_512,
        'COLLAPSES: PESQ 1.26 / STOI 0.867 / SI-SDR 4.4 dB on VoiceBank, below the noisy input'),
    '16x8 (int16 act., int8 weights)':  (1_149_416, 29_456 + 39_056,
        'TOST-equivalent to float on all four metrics, both corpora, on TFLite Micro '
        '(PESQ -0.059 / -0.051); the deployable Arm A form'),
    # Arm B, first operating point (task 13/14, 2026-10-19): learned-threshold structured
    # pruning at lambda 0.02 from the VoiceBank fine-tune, shrunk to its live units
    # (30/58/0/11 of 128 per LSTM; 336,671 params), 16x8 on TFLite Micro.
    '16x8 PRUNED (Arm B, lambda 0.02)': (428_872, 16_304 + 18_448,
        'vs the fine-tuned float: VoiceBank PESQ -0.12 / STOI -0.006 (equiv.) / SI-SDR -0.67 dB; '
        'DNS -0.31 / -0.024 / -2.7 dB. The first DTLN that FITS the H723'),
    # Two more operating points, 2026-10-20 (penny; training/RESULTS.md): 45/77/0/15 units
    # (380,242 params) and 63/96/22/82 (510,157 params, stage 2's recurrence survives).
    '16x8 PRUNED (Arm B, lambda 0.01)': (478_680, 19_056 + 18_704,
        'float TOST-equivalent to the fine-tune on VoiceBank; 16x8 PESQ -0.146 / SI-SDR -0.503 (outside by 0.003)'),
    '16x8 PRUNED (Arm B, lambda 0.005)': (633_160, 21_952 + 27_696,
        'float equivalent; 16x8 STOI/SI-SDR equivalent, PESQ -0.093 with its CI crossing the margin by 0.001'),
    # 16x8 QAT of the lambda 0.02 model (2026-10-21): same units, gate and learned state clips
    # carried as MINIMUM/MAXIMUM ops (+4,280 B). Equivalent to the PTQ export on every metric.
    '16x8 PRUNED + QAT (Arm B, lambda 0.02)': (433_152, 17_824 + 18_848,
        'vs the PTQ 16x8 of the same model: PESQ -0.004 / SI-SDR +0.13 dB, EQUIVALENT; '
        'vs the fine-tuned float: PESQ -0.128 / SI-SDR -0.54 dB. QAT buys nothing at 16x8'),
    # THE DEPLOYED ARTIFACT (H723 port, 2026-10-21): the lambda 0.02 model re-exported with
    # per-TENSOR FULLY_CONNECTED weights -- CMSIS-NN's s16 kernel refuses per-channel ones
    # (TFLM 64b3de8 fully_connected.cc:115) while the host's reference kernel runs them.
    # Arena: host RecordingMicroAllocator total; the H723 reports 5,748 + 14,348 B used.
    '16x8 PRUNED, per-tensor dense (DEPLOYED, H723)': (396_800, 8_592 + 15_696,
        'vs the per-channel 16x8 export: EQUIVALENT on all four metrics, both corpora '
        '(PESQ -0.001 VoiceBank / -0.008 DNS, SI-SDR -0.05 / -0.07 dB); the artifact in the H723 image'),
}
DTLN_STATE_BYTES_16X8 = 2 * 4 * 128 * 2          # four int16 state tensors per stage, carried by the app

def dtln_deploy_section():
    print("\n\n## DTLN deployable artifacts -- measured 2026-10-19\n")
    print(f"The shipped hybrid ({DTLN_TFLITE_INT8_BYTES:,} B) is refused by TFLite Micro. "
          "Full-integer re-exports against the 200-utterance calibration subset, with the LSTM "
          "states as separate tensors (the authors' packed layout destroys the hidden state under "
          "per-tensor int8):\n")
    print("| artifact | flash | TFLM arena (measured) | quality |")
    print("|---|---:|---:|---|")
    for tag, (fb, ar, q) in DTLN_DEPLOY.items():
        print(f"| DTLN {tag} | {fb:,} | {ar:,} | {q} |")
    print(f"\nThe arena estimate carried since 2026-09-12 was {WM_DTLN:,} B; measured "
          f"{DTLN_DEPLOY['int8 full-integer, split states'][1]:,} B (int8) and "
          f"{DTLN_DEPLOY['16x8 (int16 act., int8 weights)'][1]:,} B (16x8), plus "
          f"{DTLN_STATE_BYTES_16X8:,} B of state the application carries between frames.\n")
    print("Flash verdicts, model alone and with the TFLite Micro runtime measured for the int8 op set "
          f"({TFLM_FLASH_M7_O2:,} B; the int16 kernels add an unmeasured amount):")
    for tag in ('16x8 (int16 act., int8 weights)', '16x8 PRUNED (Arm B, lambda 0.02)',
                '16x8 PRUNED (Arm B, lambda 0.01)', '16x8 PRUNED (Arm B, lambda 0.005)',
                '16x8 PRUNED + QAT (Arm B, lambda 0.02)',
                '16x8 PRUNED, per-tensor dense (DEPLOYED, H723)'):
      fb, ar, _ = DTLN_DEPLOY[tag]
      print(f"\n**DTLN {tag}** -- {fb:,} B, arena {ar:,} B\n")
      for p_ in PLATFORMS:
          if not p_.mem_binding:
              print(f"- **{p_.name}**: not binding"); continue
          free = p_.free_flash(); tot = fb + TFLM_FLASH_M7_O2
          print(f"- **{p_.name}** (free {free:,} B): model alone "
                f"{'fits, margin ' + format(free-fb, ',') if fb <= free else 'over by ' + format(fb-free, ',')} B; "
                f"with TFLM {'fits, margin ' + format(free-tot, ',') if tot <= free else 'over by ' + format(tot-free, ',')} B; "
                f"RAM arena {ar + DTLN_STATE_BYTES_16X8:,} B of {p_.free_ram():,} B free -> "
                f"{'fits' if ar + DTLN_STATE_BYTES_16X8 <= p_.free_ram() else 'NO'}")
      chip = S3_FLASH_CHIP - S3.infra_flash
      print(f"- **{S3.name}, custom 16 MB partition table** (free {chip:,} B): "
            f"{'fits' if fb + TFLM_FLASH_M7_O2 <= chip else 'NO'} with TFLM, margin {chip - fb - TFLM_FLASH_M7_O2:,} B")
    print("\nSo the as-shipped DTLN reaches no MCU tier on flash, and the Arm A quantization that "
          "survives (16x8) changes nothing there. The first Arm B operating point does: learned-threshold "
          "pruning at lambda 0.02 leaves a 428,872 B 16x8 artifact that FITS the H723 and the S3's default "
          "partition with TFLite Micro, and misses the F411 by 88 KB -- at a VoiceBank cost of 0.12 PESQ "
          "and 0.67 dB against the fine-tuned float model. 16-bit activations, which the CMSIS-NN s16 "
          "kernels serve at a cost the probe has not yet measured. The two further points (2026-10-20) "
          "keep the H723 verdict and the S3's default partition -- the largest, lambda 0.005 at 633,160 B, "
          "by 40 KB -- and in float both are TOST-equivalent to the fine-tune on VoiceBank; what keeps "
          "every point outside the margin is the 16x8 export's own 0.05-0.10 PESQ. The 16x8 QAT (2026-10-21) "
          "did not close it: trained and post-training exports of the same pruned model are equivalent on "
          "every metric, and the export cost is the same 0.07-0.10 PESQ either way -- so the MCU candidate "
          "is the lambda 0.02 model at 429-433 KB, 0.12 PESQ under the fine-tuned float, and the gap is "
          "outside the LSTM cells. Deployed on the H723 2026-10-21 as the per-tensor-dense re-export "
          "(396,800 B, equivalent to the per-channel one on every metric): the converter's default "
          "per-channel FULLY_CONNECTED weights are refused by CMSIS-NN's int16 kernel, so the artifact "
          "the host scored could not run on the MCU as exported -- a kernel constraint, not a quality one. "
          "The S3 the same day: the artifact FITS its default partition (28 % free) and runs at RTF 4.6 -- "
          "TFLM has no optimised int16 kernels for the Xtensa and esp-nn has no int16 path, so the "
          "reference C++ costs ~24 cycles/MAC against 0.4 for esp-nn's int8; the S3's DTLN cell is "
          "NO FIT on compute, the H723's the only MCU FIT (RTF 0.39).")


if __name__ == '__main__':
    report()
    breakdown()
    tflm_check()
    measured_section()
    dtln_deploy_section()
