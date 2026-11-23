/**
 * bench_report.c -- result readout.
 *
 * The STM32 tiers print over SWO/ITM and are read back over SWD; the S3 prints
 * to the IDF console. Here it is plain stdout, plus a CSV block so the budget
 * model can consume the numbers without anyone retyping them.
 *
 * The harness rule that matters is unchanged and is why this is a separate
 * file: NO I/O INSIDE A TIMED REGION. Every kernel runs, then this prints.
 */
#define _GNU_SOURCE
#include "bench.h"
#include "bench_plat.h"
#include <stdio.h>
#include <inttypes.h>

void bench_report(void)
{
    printf("\n=== benchmark_rpi5 ===\n");
    printf("magic=%08" PRIx32 " done=%" PRIu32 " clock_ok=%" PRIu32
           " cpu_hz=%" PRIu32 " overhead=%" PRIu32 " irq_off=%" PRIu32
           " n=%" PRIu32 "\n",
           g_bench.magic, g_bench.done, g_bench.dwt_ok, g_bench.cpu_hz,
           g_bench.overhead_cyc, g_bench.irq_disabled, g_bench.n_results);

    printf("plat: cpu_hz=%" PRIu32 " (max %" PRIu32 ") gov=%s gov_ok=%" PRIu32
           " core=%" PRIu32 "/%" PRIu32 "\n",
           g_plat.cpu_hz, g_plat.cpu_hz_max, g_plat.governor,
           g_plat.governor_ok, g_plat.core_id, g_plat.n_cpus);
    printf("plat: sched_policy=%" PRIu32 " prio=%" PRIu32 " mlock=%" PRIu32
           " pool=0x%016" PRIx64 "\n",
           g_plat.sched_policy, g_plat.sched_prio, g_plat.mlock_ok,
           g_plat.pool_addr);
    printf("plat: throttled before=0x%08" PRIx32 " after=0x%08" PRIx32
           "  temp %.1f -> %.1f C\n",
           g_plat.throttled_before, g_plat.throttled_after,
           g_plat.temp_milli_before / 1000.0, g_plat.temp_milli_after / 1000.0);
    printf("plat: L1d=%" PRIu32 " L1i=%" PRIu32 " L2=%" PRIu32 " L3=%" PRIu32
           "  asimd=%" PRIu32 " asimddp=%" PRIu32 " sve=%" PRIu32 "\n",
           g_plat.l1d_bytes, g_plat.l1i_bytes, g_plat.l2_bytes, g_plat.l3_bytes,
           g_plat.has_asimd, g_plat.has_asimddp, g_plat.has_sve);

    /* A throttled or unpinned run still produces plausible-looking numbers.
     * Say so loudly here rather than letting the table be read on its own. */
    if (g_plat.throttled_after != 0u || g_plat.throttled_before != 0u)
        printf("*** WARNING: firmware throttle word non-zero -- RESULTS VOID ***\n");
    if (!g_plat.governor_ok)
        printf("*** WARNING: not every CPU is on the performance governor ***\n");
    if (!g_bench.dwt_ok)
        printf("*** WARNING: clock source self-test failed -- RESULTS VOID ***\n");

    printf("\n%-24s %10s %10s %10s %10s %12s %10s\n",
           "kernel", "cyc_min", "cyc_max", "jitter", "ops", "MOps/s", "checksum");
    for (uint32_t i = 0; i < g_bench.n_results; ++i) {
        const bench_result_t *r = &g_bench.r[i];
        double mops = 0.0;
        if (r->ops && r->cyc_min)
            mops = (double)r->ops * (double)g_bench.cpu_hz / (double)r->cyc_min / 1e6;
        printf("%-24s %10" PRIu32 " %10" PRIu32 " %10" PRIu32 " %10" PRIu32
               " %12.1f %10" PRIx32 "\n",
               r->name, r->cyc_min, r->cyc_max, r->cyc_max - r->cyc_min,
               r->ops, mops, r->checksum);
    }

    printf("\n--- CSV ---\n");
    printf("kernel,n,ops,elems,reps,cyc_min,cyc_max,cyc_mean,checksum\n");
    for (uint32_t i = 0; i < g_bench.n_results; ++i) {
        const bench_result_t *r = &g_bench.r[i];
        printf("%s,%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32
               ",%" PRIu32 ",%.1f,%08" PRIx32 "\n",
               r->name, r->n, r->ops, r->elems, r->reps, r->cyc_min, r->cyc_max,
               r->reps ? (double)r->cyc_sum / r->reps : 0.0, r->checksum);
    }
    printf("=== end ===\n");
}
