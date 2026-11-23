/**
 * bench_report.c -- result readout.
 *
 * The STM32 tiers print over SWO/ITM and are also read back over SWD with
 * OpenOCD. The S3 has neither, so results go to the IDF console. The harness
 * rule that matters is unchanged and is the reason this is a separate file:
 * NO I/O INSIDE A TIMED REGION. Every kernel runs, then this prints.
 */
#include "bench.h"
#include "bench_plat.h"
#include <stdio.h>
#include <inttypes.h>

void bench_report(void)
{
    printf("\n=== benchmark_s3 ===\n");
    printf("magic=%08" PRIx32 " done=%" PRIu32 " ccount_ok=%" PRIu32
           " cpu_hz=%" PRIu32 " overhead=%" PRIu32 " irq_off=%" PRIu32
           " n=%" PRIu32 "\n",
           g_bench.magic, g_bench.done, g_bench.dwt_ok, g_bench.cpu_hz,
           g_bench.overhead_cyc, g_bench.irq_disabled, g_bench.n_results);
    printf("plat: cpu_hz=%" PRIu32 " (cfg %" PRIu32 ") psram=%" PRIu32
           " size=%" PRIu32 " pool=0x%08" PRIx32 " internal=%" PRIu32
           " iram_kernels=%" PRIu32 " core=%" PRIu32 "\n",
           g_plat.cpu_hz, g_plat.cpu_hz_configured, g_plat.psram_present,
           g_plat.psram_size, g_plat.pool_addr, g_plat.pool_is_internal,
           g_plat.iram_kernels, g_plat.core_id);
    printf("%-24s %10s %10s %10s %10s\n", "kernel", "cyc_min", "cyc_max", "ops", "checksum");
    for (uint32_t i = 0; i < g_bench.n_results; ++i) {
        const bench_result_t *r = &g_bench.r[i];
        printf("%-24s %10" PRIu32 " %10" PRIu32 " %10" PRIu32 " %10" PRIx32 "\n",
               r->name, r->cyc_min, r->cyc_max, r->ops, r->checksum);
    }
    printf("=== end ===\n");
}
