/**
 * bench_clock.c -- Raspberry Pi 5 timing plumbing and the measurement loop.
 *
 * The counterpart of benchmark_h723/bench/bench_dwt.c (Cortex-M DWT) and
 * benchmark_s3/main/bench/bench_ccount.c (Xtensa CCOUNT). This tier has
 * NEITHER, and the reason is worth recording because it decided the design:
 *
 *   - PMCCNTR_EL0, the A76's own cycle counter, traps from userspace
 *     (SIGILL). Linux gates it behind kernel.perf_user_access; on this kernel
 *     (6.18.34+rpt-rpi-2712) the sysctl EXISTS and is set to 1, but the
 *     mmap'd perf page still reports cap_user_rdpmc=0 and index=0 for every
 *     attribute shape tried: PERF_TYPE_HARDWARE and the named PMU type
 *     (armv8_cortex_a76, type 9), config1=1 (the documented user-access
 *     request bit), pinned, exclusive, and every exclude_kernel/exclude_hv
 *     combination. Measured 2026-09-12, not assumed.
 *   - perf_event_open + read() DOES give true hardware cycles, but the syscall
 *     costs ~826 cycles per read. dot_f32_512 costs ~1.5 kcyc on this core, so
 *     the instrument would cost half of what it measures.
 *   - cntvct_el0 is cheap (~32 cyc) but ticks at only 54 MHz -- one tick per
 *     ~44 CPU cycles, which quantises the small kernels into uselessness.
 *
 * So: CLOCK_MONOTONIC_RAW via the vDSO, converted to cycles with the CPU
 * frequency that bench_plat_init() reads back from sysfs and that the bring-up
 * verified against an independent time reference (2.39974 GHz, -0.011 %).
 * Measured cost of an empty timed region: min 43 cyc, p50 89, p99 91, max 91
 * over 20,000 samples -- i.e. ~89 cycles of overhead with a spread of about
 * TWO cycles. The overhead is calibrated and subtracted exactly as the DWT
 * overhead is on the STM32s, and the harness reports the minimum, so the
 * residual error on the cheapest kernel is well under a percent.
 *
 * TWO CONSEQUENCES, both stated rather than hidden:
 *
 *  1. These are DERIVED cycles, not counted ones. They are elapsed time times
 *     a frequency. That conversion is only valid while the frequency is
 *     pinned, which is why bench_plat_init() refuses to run under any governor
 *     but `performance` and records the throttle flag either side of the
 *     suite. main.c additionally re-runs the whole suite under `perf stat` so
 *     the derived total can be checked against true hardware cycles.
 *  2. CLOCK_MONOTONIC_RAW is immune to NTP slewing, which CLOCK_MONOTONIC is
 *     not. On a machine that syncs its clock over the network -- as this one
 *     does -- using the wrong one would let a time-server correction land
 *     inside a timed region.
 *
 * Everything ABOUT the measurement is otherwise unchanged from the other three
 * tiers, because that is what makes a differing cycle count a platform
 * difference rather than a harness difference: the same counter self-test, the
 * same empty-region calibration subtracted, the same O(1) guard accumulated
 * with += , and the same O(n) checksum computed ONCE, untimed, after the
 * warm-up. Do not merge the guard and the checksum back together; on the F411
 * a checksum inside the timed region both inflated the measurement and failed
 * to prevent the dead-code elimination it existed for.
 */
#define _GNU_SOURCE
#include "bench.h"

/* Build-switch assertion, in the same spirit as the STM32s' ARM_MATH_LOOPUNROLL
 * #error and the S3's CONFIG_DSP_OPTIMIZATION one: a switch that changes what
 * is measured must break the BUILD, not the results. NDEBUG cannot be checked
 * from the CMake side by reading a variable we ourselves forced (see the note
 * at the bottom of CMakeLists.txt), and it does not appear in DW_AT_producer,
 * so it is checked here where the compiler cannot lie about it. */
#ifndef NDEBUG
#error "NDEBUG is not defined: assert() would be live inside timed regions."
#endif
#include "bench_plat.h"
#include <string.h>
#include <time.h>

bench_report_t g_bench;

static uint64_t s_hz = 2400000000ull;   /* replaced by bench_dwt_init() */

static inline uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* ns -> cycles at the pinned frequency. Integer throughout: a 1 ms region at
 * 2.4 GHz is 2.4e6 cycles and ns*hz peaks around 2.4e15, far inside uint64. */
static inline uint64_t ns_to_cyc(uint64_t ns)
{
    return (ns * s_hz) / 1000000000ull;
}

int bench_dwt_init(void)
{
    if (g_plat.cpu_hz) s_hz = (uint64_t)g_plat.cpu_hz;

    /* Verify the clock source actually advances, exactly as the STM32 harness
     * verifies CYCCNT and the S3 verifies CCOUNT. A frozen counter reads as
     * impossibly fast rather than as broken, which is the failure this catches. */
    uint64_t a = now_ns();
    for (volatile int i = 0; i < 4096; ++i) { __asm__ __volatile__("nop"); }
    uint64_t b = now_ns();

    g_bench.dwt_ok = (b > a) ? 1u : 0u;
    return (int)g_bench.dwt_ok;
}

uint32_t bench_dwt_overhead(void)
{
    /* Cost of entering and leaving a timed region with an empty body,
     * subtracted from every kernel below. Minimum over 16 attempts -- the
     * minimum is the least-perturbed sample. 16 matches the STM32 and S3
     * harnesses exactly. */
    uint64_t best = ~0ull;
    for (int i = 0; i < 16; ++i) {
        uint64_t t0 = now_ns();
        __asm__ volatile ("" ::: "memory");
        uint64_t t1 = now_ns();
        uint64_t d = ns_to_cyc(t1 - t0);
        if (d < best) best = d;
    }
    return (uint32_t)best;
}

/* Structurally identical to benchmark_s3/main/bench/bench_ccount.c's
 * bench_measure, which is itself structurally identical to the H723's. The
 * ONLY intended differences are the clock read and the absence of a critical
 * section (see the irq_disabled note in bench_kernels.c). Diff this against
 * the S3 version when changing it; do not rewrite it from memory. */
void bench_measure(const char *name, uint32_t n, uint32_t ops,
                   uint32_t elems, uint32_t reps,
                   bench_prep_t prep, bench_fn_t fn, bench_chk_t chk)
{
    if (g_bench.n_results >= BENCH_MAX_KERNELS) return;

    bench_result_t *r = &g_bench.r[g_bench.n_results];
    memset(r, 0, sizeof(*r));
    strncpy(r->name, name, BENCH_NAME_LEN - 1u);
    r->n       = n;
    r->ops     = ops;
    r->elems   = elems;
    r->reps    = reps;
    r->cyc_min = 0xFFFFFFFFu;
    r->cyc_max = 0u;
    r->cyc_sum = 0u;

    /* One untimed warm-up: settles cache fills and branch predictor state so
     * the first timed rep is not systematically slower for reasons unrelated
     * to the kernel. On this tier that means the L1/L2/L3 hierarchy and, on
     * first touch, the page tables -- which is also why main.c calls
     * mlockall() before any of this runs. */
    if (prep) prep();
    r->guard += fn();

    /* Correctness check: one full pass over the output, here and NOT in the
     * timed loop. Every repetition recomputes the same values from the same
     * fixed-seed inputs, so checking once checks all of them. */
    if (chk) r->checksum = chk();

    for (uint32_t i = 0u; i < reps; ++i) {
        if (prep) prep();

        uint64_t t0 = now_ns();
        __asm__ volatile ("" ::: "memory");
        uint32_t g = fn();
        __asm__ volatile ("" ::: "memory");
        uint64_t t1 = now_ns();

        uint64_t d = ns_to_cyc(t1 - t0);
        d = (d > (uint64_t)g_bench.overhead_cyc) ? (d - g_bench.overhead_cyc) : 0u;

        if (d < r->cyc_min) r->cyc_min = (uint32_t)d;
        if (d > r->cyc_max) r->cyc_max = (uint32_t)d;
        r->cyc_sum  += d;
        r->guard    += g;
    }

    r->valid = 1u;
    g_bench.n_results++;
}
