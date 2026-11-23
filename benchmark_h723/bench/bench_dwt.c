/**
 * bench_dwt.c -- DWT cycle counter plumbing and the measurement loop.
 */
#include "bench.h"
#include "main.h"
#include <string.h>

bench_report_t g_bench;

/* On Cortex-M7 the DWT registers sit behind a software lock that must be
 * opened before CYCCNT can be written. On M4 there is no such lock, so this is
 * compiled out -- it is here so the H723 port needs no change. */
#if defined(__CORTEX_M) && (__CORTEX_M == 7U)
  #define BENCH_DWT_UNLOCK()  do { *(volatile uint32_t *)0xE0001FB0u = 0xC5ACCE55u; } while (0)
#else
  #define BENCH_DWT_UNLOCK()  do { } while (0)
#endif

int bench_dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    BENCH_DWT_UNLOCK();
    DWT->CYCCNT = 0u;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;

    /* Verify the counter actually runs. On some parts CYCCNT stays frozen
     * unless a debugger has enabled trace, and a frozen counter yields
     * cyc == 0 for every kernel -- which looks like an implausibly fast
     * result rather than like a failure. Catch it here instead. */
    uint32_t a = DWT->CYCCNT;
    for (volatile int i = 0; i < 32; ++i) { __NOP(); }
    uint32_t b = DWT->CYCCNT;

    g_bench.dwt_ok = (b != a) ? 1u : 0u;
    return (int)g_bench.dwt_ok;
}

uint32_t bench_dwt_overhead(void)
{
    /* Cost of entering/leaving a timed region with an empty body. Subtracted
     * from every kernel measurement below. Take the minimum over several
     * attempts: the minimum is the least-perturbed sample. */
    uint32_t best = 0xFFFFFFFFu;
    for (int i = 0; i < 16; ++i) {
        uint32_t primask = __get_PRIMASK();
        __disable_irq();
        uint32_t t0 = DWT->CYCCNT;
        __asm volatile ("" ::: "memory");
        uint32_t t1 = DWT->CYCCNT;
        __set_PRIMASK(primask);
        uint32_t d = t1 - t0;
        if (d < best) best = d;
    }
    return best;
}

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

    /* One untimed warm-up. It settles the flash ART line fills and any branch
     * predictor state, so the first timed rep is not systematically slower
     * than the rest for reasons that have nothing to do with the kernel. */
    if (prep) prep();
    r->guard += fn();

    /* Correctness check: one full pass over the output, here and NOT in the
     * timed loop below. The warm-up above left the output buffer holding a
     * representative result, and every repetition recomputes the same values
     * from the same fixed-seed inputs, so checking once checks all of them. */
    if (chk) r->checksum = chk();

    for (uint32_t i = 0u; i < reps; ++i) {
        if (prep) prep();

        uint32_t primask = __get_PRIMASK();
        __disable_irq();

        uint32_t t0 = DWT->CYCCNT;
        __asm volatile ("" ::: "memory");
        uint32_t g = fn();
        __asm volatile ("" ::: "memory");
        uint32_t t1 = DWT->CYCCNT;

        __set_PRIMASK(primask);

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
