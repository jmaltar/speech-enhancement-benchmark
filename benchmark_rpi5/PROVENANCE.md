# benchmark_rpi5 — provenance and pinned versions

Fourth tier of the ladder (task 3). Created and run on hardware **2026-09-12**, the same
day the RPi5 tier was reinstated.

## Machine

| | |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1 (BCM2712), Cortex-A76 ×4, 8 GB |
| Storage | 59.5 GB SD card (`mmcblk0`), no NVMe |
| OS | Debian 13 (trixie), kernel `6.18.34+rpt-rpi-2712`, aarch64 |
| Compiler | gcc (Debian 14.2.0-19) 14.2.0 |
| Build | cmake 3.31.6, ninja |
| Caches | L1d 64 KB/core, L1i 64 KB/core, L2 512 KB/core, L3 2 MB shared |
| ISA features | `asimd`, **`asimddp`** (SDOT/UDOT), `fphp`, `asimdhp`, `aes`, `crc32`; **no SVE** |

## Relationship to the other three tiers

| file | relationship |
|---|---|
| `bench/bench.h` | **byte-identical** to all three MCU tiers — `cksum 3232819878 6074`, verified |
| `bench/bench_kernels.c` lines 1–241 | **identical source text** to `benchmark_s3/.../bench_kernels.c` lines 1–240, verified by `diff`; the only difference is the `#include` comment block, as the S3 itself differs from the H723 |
| `bench/bench_clock.c` | **replaces** `bench_dwt.c` (Cortex-M) / `bench_ccount.c` (Xtensa). Measurement semantics preserved line for line — same self-test, same 16-sample overhead calibration subtracted, same `+=` guard, same untimed one-shot checksum, same barriers around `fn()`. Diff it against the S3 version when changing it; do not rewrite from memory. |
| `bench/bench_plat.{c,h}` | this tier's measured variables (governor, throttle, core, sched policy, cache geometry, ISA features) |
| USB | **none.** The Pi is a Linux host, not a device. The one-stack TinyUSB decision covers the three MCU tiers only. |

## Why the timing layer is what it is

Measured on 2026-09-12, not assumed:

| method | cost per read | verdict |
|---|---|---|
| `PMCCNTR_EL0` direct | — | **SIGILL.** Blocked from userspace. |
| `perf_event_open` + `read()` | **826 cyc** | true HW cycles, but `dot_f32_512` is only ~1000 cyc |
| `mrs cntvct_el0` | 32 cyc | ticks at only 54 MHz → ~44-cycle quantisation |
| **`clock_gettime(CLOCK_MONOTONIC_RAW)`** | **89 cyc, ±2** | **chosen** |

`kernel.perf_user_access` **exists on this kernel and is set to 1, but does not work**:
the mmap'd perf page still reports `cap_user_rdpmc=0, index=0` for every attribute shape
tried — `PERF_TYPE_HARDWARE` and the named PMU (`armv8_cortex_a76`, type 9), `config1=1`
(the documented user-access request bit), `pinned`, `exclusive`, and all four
`exclude_kernel`/`exclude_hv` combinations.

`CLOCK_MONOTONIC_RAW`, not `CLOCK_MONOTONIC`: the latter is NTP-slewed and this machine
syncs over the network, so a time-server correction could land inside a timed region.

**These are DERIVED cycles** — elapsed time × a pinned frequency — not counted ones. Valid
only while the frequency is pinned, which is why `bench_plat_init()` refuses to run under any
governor but `performance` and latches the firmware throttle word either side of the suite.

## Build flags, and the guard that checks them

`-O2 -g3 -DNDEBUG -mcpu=cortex-a76`.

**Not `-ffast-math`**, deliberately: it would reassociate float arithmetic and forfeit the
cross-platform checksum equivalence that task 22 rests on. This has a large, visible
consequence — see below.

**`-mcpu=cortex-a76`** to match the other tiers, which build with `-mcpu=cortex-m4/m7` and
LX7 tuning respectively.

The guard is **not** a check on a CMake variable. The first version was, and testing it the
way this project requires — by reintroducing the H723 defect — showed **neither the guard nor
the bug fired**:

1. **The H723 defect does not reproduce in a plain native CMake project.** `project()`
   initialises `CMAKE_C_FLAGS_RELEASE` from the compiler module's `_INIT` value into the
   *cache*, but a *normal* variable of the same name set earlier shadows the cache entry.
   CubeMX's generated structure is what made it bite there. **A lesson learned on one
   generator must be re-verified on the next.**
2. Worse: because the `set()` is unconditional, the guard could never fire at all —
   `cmake -DCMAKE_C_FLAGS_RELEASE=-Os` is silently overwritten too. A guard re-reading a
   variable the same file just forced is decoration, the identical error the H723's guard
   made by sitting two lines under its `set()`.

So the check moved to **the artifact**. `-g3` makes GCC record its real command line in
`DW_AT_producer`; `cmake/check_flags.cmake` runs post-build and reads the optimisation level
out of the object that was actually compiled. `NDEBUG` is not in `DW_AT_producer` (it is a
preprocessor define), so it is asserted from inside C by an `#error` in `bench_clock.c`.

**Both verified by reproducing real bugs** (2026-09-12): patching `-O2`→`-Os` fails the
build at the post-build step quoting the producer string; removing `-DNDEBUG` fails
compilation at the `#error`.

## The auto-vectorisation finding — read this before comparing plain-C rows

On AArch64 NEON is mandatory and GCC vectorises at `-O2`. **Confirmed by disassembly**, not
inferred:

| kernel | code generated | rate |
|---|---|---|
| `matvec_s8_128` | **`sdot v31.4s, v30.16b, v29.16b`** | 0.108 cyc/MAC |
| `matvec_f32_128` | `fmadd s31, s30, s29, s31` — **scalar** | 1.81 cyc/MAC |
| `window_f32_512` | `fmul v30.4s, v31.4s, v30.4s` | 0.35 cyc/elem |

The asymmetry is exactly right and follows from the `-ffast-math` decision: **integer
addition is associative**, so GCC may reorder the int8 reduction into SDOT; **float addition
is not**, so the f32 reduction must stay scalar; elementwise ops have no reduction and
vectorise regardless.

**Consequence: the "plain C" rows are NOT like-for-like across tiers.** On the Cortex-Ms and
the LX7 plain C is scalar; here the int8 row is already SIMD. State this wherever the four
plain-C columns appear together. It also means a later NEON/ACL pairing will show little gain
on int8 — not because NEON is worthless, but because the baseline already reached it.
