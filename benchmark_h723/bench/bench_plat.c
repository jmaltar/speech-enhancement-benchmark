/**
 * bench_plat.c -- STM32H723 cache / MPU configuration and register readout.
 */
#include "bench_plat.h"
#include "main.h"

bench_plat_t g_plat;

void bench_plat_init(void)
{
    /* --- MPU ---------------------------------------------------------------
     * Explicitly DISABLED, and explicitly so rather than by omission.
     *
     * CubeMX generates an MPU_Config() that runs before HAL_Init() and does
     * HAL_MPU_Disable() followed by HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT)
     * with no regions defined. That call is removed from main.c; this is what
     * replaces it. With no regions, the Cortex-M7 default memory map applies,
     * which already marks 0x20000000-0x3FFFFFFF as Normal write-back
     * write-allocate -- so AXI SRAM is cacheable without any MPU programming,
     * which is what the DTCM-vs-AXI comparison needs.
     *
     * Regions become necessary when a DMA engine shares a buffer with the
     * cached CPU -- task 7's audio path, if it moves off the dwc2 FIFO. That
     * is the place to add them, and it is a measured change, not a tidy-up. */
    ARM_MPU_Disable();

    /* --- caches ------------------------------------------------------------
     * Invalidate before enabling: the caches' contents after reset are
     * architecturally UNKNOWN, and SCB_EnableICache/SCB_EnableDCache in CMSIS
     * already invalidate, but the ordering matters if this is ever called
     * twice (e.g. a soft reset path that does not clear the caches). */
#if BENCH_ICACHE
    SCB_EnableICache();
#else
    SCB_DisableICache();
#endif
#if BENCH_DCACHE
    SCB_EnableDCache();
#else
    SCB_DisableDCache();
#endif
}

void bench_plat_snapshot(void)
{
    g_plat.magic     = BENCH_PLAT_MAGIC;
    g_plat.sysclk_hz = SystemCoreClock;

    /* Read back from the core, not from the #defines that asked for it. The
     * F411 bring-up established the habit the hard way: a configuration that
     * was requested is not a configuration that took. */
    g_plat.icache_on = (SCB->CCR & SCB_CCR_IC_Msk) ? 1u : 0u;
    g_plat.dcache_on = (SCB->CCR & SCB_CCR_DC_Msk) ? 1u : 0u;
    g_plat.mpu_ctrl  = MPU->CTRL;

    /* g_plat itself lives in .bss, and the DTCM-vs-AXI switch relocates .bss
     * wholesale in the linker script, so its own address is a faithful witness
     * to where the harness's data landed -- no need to reach into
     * bench_kernels.c's statics and break the byte-identity of that file. */
    g_plat.bss_addr   = (uint32_t)(uintptr_t)&g_plat;
    g_plat.bss_in_axi = ((g_plat.bss_addr & 0xFF000000u) == 0x24000000u) ? 1u : 0u;

    g_plat.flash_acr     = FLASH->ACR;
    g_plat.pwr_d3cr      = PWR->D3CR;
    g_plat.rcc_cr        = RCC->CR;
    g_plat.rcc_pllckselr = RCC->PLLCKSELR;
    g_plat.rcc_pll1divr  = RCC->PLL1DIVR;
    g_plat.rcc_cfgr      = RCC->CFGR;
    g_plat.rcc_d1cfgr    = RCC->D1CFGR;
    g_plat.rcc_d2cfgr    = RCC->D2CFGR;
    g_plat.rcc_d3cfgr    = RCC->D3CFGR;

    g_plat.crs_cfgr = CRS->CFGR;
    bench_plat_poll();
}

void bench_plat_poll(void)
{
    uint32_t isr = CRS->ISR;
    g_plat.crs_isr = isr;
    if (isr & CRS_ISR_SYNCOKF) {
        g_plat.crs_syncok_seen = 1u;
    }
}
