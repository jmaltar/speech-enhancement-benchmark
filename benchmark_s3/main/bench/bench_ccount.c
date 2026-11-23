/**
 * bench_ccount.c -- ESP32-S3 cycle counter plumbing and the measurement loop.
 *
 * This is the one harness file that CANNOT be shared with the STM32 tiers:
 * bench_dwt.c reads the Cortex-M DWT, and there is no DWT here. The Xtensa LX7
 * CCOUNT register is read through esp_cpu_get_cycle_count().
 *
 * Everything ABOUT the measurement is deliberately unchanged, because that is
 * what makes a cycle count differing between boards a platform difference
 * rather than a harness difference:
 *   - the same self-test that the counter actually increments (a frozen counter
 *     reads as impossibly fast, not as broken);
 *   - the same empty-region overhead calibration, subtracted;
 *   - the same interrupt masking around every timed region;
 *   - the same O(1) guard returned by the kernel and the same O(n) checksum
 *     computed ONCE, untimed, after the warm-up. That split is not cosmetic:
 *     on the F411 a checksum inside the timed region both inflated the
 *     measurement and failed to prevent the dead-code elimination it existed
 *     for. Do not merge them back.
 *
 * One genuine platform difference is recorded rather than hidden: CCOUNT is
 * per-core and counts only while that core is running. The suite therefore
 * pins itself to one core and masks interrupts, and g_plat.core_id records
 * which core the numbers belong to.
 */
#include "bench.h"
#include "bench_plat.h"
#include "esp_cpu.h"
#include "esp_attr.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <string.h>

bench_report_t g_bench;

static portMUX_TYPE s_bench_mux = portMUX_INITIALIZER_UNLOCKED;

int bench_dwt_init(void)
{
    /* Verify the counter actually runs, exactly as the STM32 harness does. */
    uint32_t a = esp_cpu_get_cycle_count();
    for (volatile int i = 0; i < 32; ++i) { __asm__ __volatile__("nop"); }
    uint32_t b = esp_cpu_get_cycle_count();

    g_bench.dwt_ok = (b != a) ? 1u : 0u;
    return (int)g_bench.dwt_ok;
}

uint32_t bench_dwt_overhead(void)
{
    /* Cost of entering/leaving a timed region with an empty body. Subtracted
     * from every kernel measurement below. Take the minimum over several
     * attempts: the minimum is the least-perturbed sample.
     * 16 iterations, matching the STM32 harness exactly. */
    uint32_t best = 0xFFFFFFFFu;
    for (int i = 0; i < 16; ++i) {
        taskENTER_CRITICAL(&s_bench_mux);
        uint32_t t0 = esp_cpu_get_cycle_count();
        __asm__ volatile ("" ::: "memory");
        uint32_t t1 = esp_cpu_get_cycle_count();
        taskEXIT_CRITICAL(&s_bench_mux);
        uint32_t d = t1 - t0;
        if (d < best) best = d;
    }
    return best;
}

/* Structurally identical to benchmark_h723/bench/bench_dwt.c's bench_measure.
 * The ONLY intended differences are the cycle-counter read and the critical
 * section primitive. In particular the guard is accumulated with += in both the
 * warm-up and the loop (not XOR), and the memory barriers around fn() are kept:
 * without them the compiler may move the timestamp reads across the kernel. */
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

    /* One untimed warm-up. Settles cache fills and branch predictor state so
     * the first timed rep is not systematically slower for reasons unrelated
     * to the kernel. On this platform that means the flash instruction cache,
     * which matters more here than the ART did on the STM32s. */
    if (prep) prep();
    r->guard += fn();

    /* Correctness check: one full pass over the output, here and NOT in the
     * timed loop. Every repetition recomputes the same values from the same
     * fixed-seed inputs, so checking once checks all of them. */
    if (chk) r->checksum = chk();

    for (uint32_t i = 0u; i < reps; ++i) {
        if (prep) prep();

        taskENTER_CRITICAL(&s_bench_mux);

        uint32_t t0 = esp_cpu_get_cycle_count();
        __asm__ volatile ("" ::: "memory");
        uint32_t g = fn();
        __asm__ volatile ("" ::: "memory");
        uint32_t t1 = esp_cpu_get_cycle_count();

        taskEXIT_CRITICAL(&s_bench_mux);

        uint32_t d = t1 - t0;
        d = (d > g_bench.overhead_cyc) ? (d - g_bench.overhead_cyc) : 0u;

        if (d < r->cyc_min) r->cyc_min = d;
        if (d > r->cyc_max) r->cyc_max = d;
        r->cyc_sum  += (uint64_t)d;
        r->guard    += g;
    }

    r->valid = 1u;
    g_bench.n_results++;
}
