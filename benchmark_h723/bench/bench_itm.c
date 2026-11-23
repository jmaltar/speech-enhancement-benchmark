/**
 * bench_itm.c -- SWO/ITM reporting. Called once, after every kernel has run.
 *
 * Integer formatting only: newlib-nano omits float printf unless the link is
 * given -u _printf_float, and pulling that in would inflate the flash figure
 * this probe exists to measure. Fractions are carried as scaled integers.
 */
#include "bench.h"
#include "main.h"
#include <stdio.h>

#define ITM_STIM_U8(p)   (*(volatile uint8_t  *)&ITM->PORT[p].u8)
#define ITM_STIM_U32(p)  (*(volatile uint32_t *)&ITM->PORT[p].u32)

static void itm_putc(char c)
{
    /* If no debugger has enabled trace, drop the character rather than spin.
     * A probe that hangs when run standalone is worse than a silent one. */
    if ((ITM->TCR & ITM_TCR_ITMENA_Msk) == 0u) return;
    if ((ITM->TER & 1u) == 0u) return;

    uint32_t guard = 100000u;
    while (ITM_STIM_U32(0) == 0u) {
        if (--guard == 0u) return;
        __NOP();
    }
    ITM_STIM_U8(0) = (uint8_t)c;
}

void bench_itm_puts(const char *s)
{
    while (*s) itm_putc(*s++);
}

void bench_itm_report(void)
{
    char line[128];

    bench_itm_puts("\r\n=== throughput probe ===\r\n");

    if (!g_bench.dwt_ok) {
        bench_itm_puts("DWT CYCCNT not running -- results void\r\n");
        return;
    }

    snprintf(line, sizeof(line),
             "cpu_hz=%lu overhead=%lu cyc  irq_masked=%lu  kernels=%lu\r\n",
             (unsigned long)g_bench.cpu_hz,
             (unsigned long)g_bench.overhead_cyc,
             (unsigned long)g_bench.irq_disabled,
             (unsigned long)g_bench.n_results);
    bench_itm_puts(line);

    bench_itm_puts("name                  min_cyc   max_cyc  mean_cyc   MOps/s  cyc/elem\r\n");

    for (uint32_t i = 0; i < g_bench.n_results; ++i) {
        const bench_result_t *r = &g_bench.r[i];
        if (!r->valid) continue;

        uint32_t mean = (r->reps != 0u)
                      ? (uint32_t)(r->cyc_sum / (uint64_t)r->reps) : 0u;

        /* MOps/s x100, computed from the MINIMUM cycle count: the minimum is
         * the least-perturbed sample and is what a ceiling should be built on.
         * The spread between min and max is reported alongside, because on this
         * project jitter is a result in its own right, not noise to be hidden. */
        unsigned long mops_i = 0u, mops_f = 0u;
        if (r->ops != BENCH_OPS_NONE && r->cyc_min != 0u) {
            uint64_t num = (uint64_t)r->ops * (uint64_t)g_bench.cpu_hz * 100ull;
            uint64_t den = (uint64_t)r->cyc_min * 1000000ull;
            uint64_t x100 = num / den;
            mops_i = (unsigned long)(x100 / 100ull);
            mops_f = (unsigned long)(x100 % 100ull);
        }

        unsigned long cpe_i = 0u, cpe_f = 0u;
        if (r->elems != 0u) {
            uint64_t x100 = ((uint64_t)r->cyc_min * 100ull) / (uint64_t)r->elems;
            cpe_i = (unsigned long)(x100 / 100ull);
            cpe_f = (unsigned long)(x100 % 100ull);
        }

        if (r->ops != BENCH_OPS_NONE) {
            snprintf(line, sizeof(line),
                     "%-20s %8lu  %8lu  %8lu  %5lu.%02lu  %4lu.%02lu\r\n",
                     r->name, (unsigned long)r->cyc_min, (unsigned long)r->cyc_max,
                     (unsigned long)mean, mops_i, mops_f, cpe_i, cpe_f);
        } else {
            snprintf(line, sizeof(line),
                     "%-20s %8lu  %8lu  %8lu       --  %4lu.%02lu\r\n",
                     r->name, (unsigned long)r->cyc_min, (unsigned long)r->cyc_max,
                     (unsigned long)mean, cpe_i, cpe_f);
        }
        bench_itm_puts(line);
    }

    bench_itm_puts("=== end ===\r\n");
}
