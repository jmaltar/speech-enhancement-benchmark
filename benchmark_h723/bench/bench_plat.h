/**
 * bench_plat.h -- STM32H723 platform configuration and its record.
 *
 * On the F411 the whole measurement configuration fitted in g_bench (clock,
 * interrupt state, ops convention) plus a handful of registers read out over
 * SWD. The H7 adds knobs that change throughput by large factors and that a
 * results table is meaningless without: L1 instruction and data caches, MPU
 * state, and whether the working set lives in DTCM or in AXI SRAM.
 *
 * Those live HERE rather than in bench.h on purpose. bench.h, bench_dwt.c,
 * bench_kernels.c and bench_itm.c are byte-identical copies of the F411's, so
 * that a cycle count differing between the two boards is the platform
 * differing and not the harness. Adding H7 fields to the shared struct would
 * have broken that property for no gain -- the debugger can read two structs
 * as easily as one.
 */
#ifndef BENCH_PLAT_H
#define BENCH_PLAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Build-time knobs. Defaults are the deployed-realistic configuration: caches
 * on, which is how anyone shipping an M7 audio product would run it. The
 * cache-off rows are the interesting comparison, not the default. */
#ifndef BENCH_ICACHE
#define BENCH_ICACHE 1
#endif
#ifndef BENCH_DCACHE
#define BENCH_DCACHE 1
#endif

#define BENCH_PLAT_MAGIC 0x48374331u   /* "H7C1" */

typedef struct {
    uint32_t magic;
    uint32_t sysclk_hz;

    /* configuration as it actually ended up, read back from the core */
    uint32_t icache_on;
    uint32_t dcache_on;
    uint32_t mpu_ctrl;        /* MPU->CTRL: bit0 ENABLE, bit2 PRIVDEFENA     */
    uint32_t bss_in_axi;      /* 1 = harness data in AXI SRAM, 0 = DTCM      */
    uint32_t bss_addr;        /* proves the above rather than asserting it   */

    /* the H7 counterparts of the F411's FLASH_ACR / RCC_CFGR readout */
    uint32_t flash_acr;       /* wait states + ART                            */
    uint32_t pwr_d3cr;        /* VOS field: VOS0 required for 550 MHz         */
    uint32_t rcc_cr;
    uint32_t rcc_pllckselr;
    uint32_t rcc_pll1divr;
    uint32_t rcc_cfgr;
    uint32_t rcc_d1cfgr;
    uint32_t rcc_d2cfgr;
    uint32_t rcc_d3cfgr;

    /* CRS. CubeMX selected RCC_CRS_SYNC_SOURCE_USB2 on a part whose only USB
     * instance is USB1_OTG_HS, and nothing in the HAL validates that, so this
     * is checked empirically instead of argued about:
     *   SYNCOKF set + SYNCMISS clear -> the source is right
     *   SYNCOKF never sets, SYNCMISS/SYNCERR set -> wrong source
     * Only meaningful once the host is sending SOF, i.e. after enumeration,
     * so bench_plat_poll() refreshes it from the main loop. */
    uint32_t crs_cfgr;
    uint32_t crs_isr;
    uint32_t crs_syncok_seen;  /* latched: SYNCOKF was observed at least once */
} bench_plat_t;

extern bench_plat_t g_plat;

/* Applies the cache/MPU configuration. Call FIRST, before the probe runs. */
void bench_plat_init(void);

/* Records the configuration as read back from hardware. Call after the clock
 * tree is up and after bench_plat_init(). */
void bench_plat_snapshot(void);

/* Refreshes the CRS status. Cheap; call from the main loop. */
void bench_plat_poll(void);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_PLAT_H */
