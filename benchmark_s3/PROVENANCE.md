# benchmark_s3 — pinned versions and how the harness relates to the STM32 tiers

Recorded 2026-09-11. Task 3 reports flash/SRAM remaining after the vendor
libraries are linked, and task 4 turns that into a budget, so these figures are
not reproducible against moving dependencies.

## Toolchain

| item | version |
|---|---|
| ESP-IDF | **v5.4.1** (`4c2820d377`) |
| Xtensa GCC | `esp-14.2.0_20241119` (GCC 14.2.0) |
| IDF Python env | CPython **3.12.13**, supplied by `uv` |

**Note on the Python env.** The pre-existing `idf5.4_py3.12_env` was broken: its
`python3` symlinked to `/usr/bin/python3`, which the system had moved to 3.14,
so its site-packages no longer matched its interpreter. IDF 5.4.1 predates 3.14.
Rebuilt against a `uv`-managed CPython 3.12.13 via a shim directory at
`~/.espressif/pyshim`, consistent with the project's existing use of `uv` for the
pinned eval environment, and without touching the system Python. **Activate with:**

```
export PATH=~/.espressif/pyshim:$PATH && . ~/.espressif/esp-idf/export.sh
```

The old venv is set aside at `idf5.4_py3.12_env.broken`.

## Managed components

| component | version |
|---|---|
| `espressif/esp-dsp` | **1.5.2** |
| `espressif/esp_tinyusb` | **1.7.6~2** |
| `espressif/esp-nn` | **1.3.2** |
| `espressif/tinyusb` | **0.21.0~1** |

**TinyUSB is 0.21.0 — the same upstream version vendored on the F411 and H723.**
That is the one-stack decision actually holding across all three MCU tiers, so
the USB footprint line item is a genuine cross-platform comparison rather than
three different stacks reported as a platform difference.

## Harness relationship to the STM32 tiers

The property that makes a cycle count differing between boards a *platform*
difference rather than a harness difference:

| file | relationship |
|---|---|
| `bench/bench.h` | **byte-identical** to `benchmark_h723/bench/bench.h` — verified by `cksum` (`3232819878 6074`) |
| `bench/bench_kernels.c` lines 1–236 | **identical source text** to the H723's, apart from one `#include` swap (`main.h` → `bench_plat.h`), verified by `diff`. This is the whole plain-C kernel set. |
| `bench/bench_kernels.c` suite | rewritten: plain-C rows only for now; the esp-dsp pairs are the next step |
| `bench/bench_ccount.c` | **rewritten** — there is no DWT here. Xtensa `CCOUNT` via `esp_cpu_get_cycle_count()`. `bench_measure` is structurally identical to the H723's `bench_dwt.c`: same warm-up, same untimed one-shot checksum, same `+=` guard accumulation, same memory barriers around `fn()`, same overhead calibration over 16 iterations, same critical section around every timed region. |
| `bench/bench_report.c` | **rewritten** — no SWO/ITM; results go to the IDF console. The rule it preserves is the one that matters: no I/O inside a timed region. |
| `bench/bench_plat.{c,h}` | **platform-specific**, as on the H723 — this tier's extra measured variables |

**Why `bench_kernels.c` is copied rather than shared.** A single `#if`-guarded
file serving all three would be tidier, but restructuring the STM32 translation
units would move their measurements: the F411 re-link sensitivity bound is
±0.7 %, and splitting or merging a TU changes inlining and code layout. The
property worth having is that the plain-C kernel *source text* is identical,
which `diff` verifies, not that one file is shared.

## Configuration assertions

`main/CMakeLists.txt` fails **configure** if the optimisation level, assertion
setting or CPU frequency drift. This is the direct counterpart of the
`FATAL_ERROR` guard added to `benchmark_h723/CMakeLists.txt` on 2026-09-11, and
it exists because IDF's defaults are exactly the two things that went wrong
there: `-Os` and assertions enabled. `sdkconfig.defaults` alone is not enough —
it is ignored once `sdkconfig` exists, so the assertions are what actually hold.

Verified by reproducing the defect: forcing `CONFIG_COMPILER_OPTIMIZATION_SIZE`
makes configure fail with the expected message. An untested guard is decoration.

## Not yet done

- ~~esp-dsp paired kernels~~ **DONE** — and they found that the vendor matvec
  loses to plain C while a per-row `dsps_dotprod_f32` is 2.71× faster.
- ~~esp-dsp's build-switch audit~~ **DONE** — two traps found
  (`CONFIG_DSP_OPTIMIZATION`, `*_aes3_enabled`) plus a flash one
  (`CONFIG_DSP_MAX_FFT_SIZE`), all asserted.
- ~~ESP-NN INT8 row~~ **DONE** — 0.795 cyc/MAC, 10.1× plain C.
- ~~TinyUSB CDC integration~~ **DONE** — enumerated `303a:4001`, byte-exact
  round trip, 375.3 KiB/s each way.
- **The internal-vs-PSRAM matrix itself.** RESOLVED 2026-09-11: the board is an
  **ESP32-S3R8 with 8 MB embedded octal PSRAM**, now enabled, so the comparison
  is available. Still to do is actually moving the pool: it is a static array in
  the region kept identical to the STM32 tiers, so use a linker fragment rather
  than adding `EXT_RAM_BSS_ATTR` to that declaration.

## Hardware status

**Run on hardware 2026-09-11.** Connect via the USB-C port marked **`COM`**
(CH343 bridge, `1a86:55d3`, enumerates as `/dev/ttyACM0` — it is CDC-ACM, not
`ttyUSB`). The port marked `USB` is the native peripheral and is unused until
TinyUSB is wired. Board: 16 MB flash, 8 MB octal PSRAM, 40 MHz crystal.
