/**
 * bench_plat.h -- ESP32-S3 platform configuration and its record.
 *
 * Same role as benchmark_h723/bench/bench_plat.h: the knobs that change
 * throughput by large factors on THIS platform, recorded as read back from the
 * hardware rather than as asserted by the #define that asked for them. The H7
 * bring-up established why (the cache bits), and the 2026-09-11 -Os defect
 * established the stronger form: a setting that silently changes what is
 * measured must be observable in the result, not merely configured.
 *
 * The S3's counterparts of the H7's (cache, DTCM-vs-AXI) are:
 *   - CPU frequency: IDF defaults to 160 MHz, not 240.
 *   - WHERE the working set lives: internal SRAM vs octal-SPI PSRAM. This is
 *     the DTCM-vs-AXI decision on this platform and it is worth far more than
 *     a cache setting -- buffers landing in PSRAM collapse throughput.
 *   - WHETHER PSRAM is present at all, which varies by module.
 *   - Instruction cache: code runs XIP from flash, so IRAM_ATTR placement of
 *     hot kernels is the instruction-side variable.
 *   - Which core ran, and what the other was doing (feeds duty-cycle logging).
 */
#ifndef BENCH_PLAT_H
#define BENCH_PLAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BENCH_PLAT_MAGIC 0x53334331u   /* "S3C1" */

/* Build-time knob: place the shared weight pool and signal buffers in internal
 * SRAM (the default, and the only configuration in which the numbers mean
 * anything) or deliberately in PSRAM, which is the comparison row. */
#ifndef BENCH_MEM
#define BENCH_MEM 0        /* 0 = MALLOC_CAP_INTERNAL, 1 = MALLOC_CAP_SPIRAM */
#endif

/* Build-time knob: put the hot kernels in IRAM instead of XIP flash. */
#ifndef BENCH_IRAM_KERNELS
#define BENCH_IRAM_KERNELS 0
#endif

typedef struct {
    uint32_t magic;
    uint32_t cpu_hz;            /* as reported by the RTC/clock driver        */
    uint32_t cpu_hz_configured; /* what sdkconfig asked for                   */
    uint32_t psram_present;     /* 1 if PSRAM was detected at boot            */
    uint32_t psram_size;        /* bytes, 0 if absent                         */
    uint32_t bench_mem_is_psram;/* the knob, as built                         */
    uint32_t pool_addr;         /* WITNESSES where the pool actually landed   */
    uint32_t pool_is_internal;  /* derived from pool_addr, not from the knob  */
    uint32_t iram_kernels;      /* the knob, as built                         */
    uint32_t core_id;           /* which core the suite ran on                */
    uint32_t other_core_idle;   /* what the other core was doing              */
    uint32_t icache_size;       /* from sdkconfig, recorded for the table     */
    uint32_t dcache_size;
    uint32_t free_internal;     /* heap free, internal only, before the suite */
    uint32_t free_psram;
} bench_plat_t;

extern bench_plat_t g_plat;

/* Result readout. bench.h declares bench_itm_report() for the SWO/ITM path the
 * STM32 tiers use; there is no ITM here, so the console printer is declared
 * separately rather than editing the byte-identical bench.h. */
void bench_report(void);

/* Defined in bench_kernels.c: address of the shared weight pool. */
uint32_t bench_pool_addr(void);

void bench_plat_init(void);      /* call FIRST, before the probe runs */
void bench_plat_snapshot(void);  /* record configuration as read back */

#ifdef __cplusplus
}
#endif

#endif /* BENCH_PLAT_H */
