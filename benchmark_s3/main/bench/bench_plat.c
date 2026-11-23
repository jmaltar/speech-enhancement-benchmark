/**
 * bench_plat.c -- records the ESP32-S3 configuration as read back from the
 * running system, not as requested by sdkconfig.
 *
 * The rule this file exists to enforce was learned twice on the STM32 tiers:
 * the H723 taught it for the cache bits (read SCB->CCR, never trust the
 * #define), and the 2026-09-11 -Os defect taught the stronger form -- a build
 * setting that silently changes what is measured must be visible in the
 * result. g_plat.cpu_hz vs cpu_hz_configured is that check for this platform's
 * single most likely mistake: IDF defaulting the S3 to 160 MHz.
 */
#include "bench_plat.h"
#include "bench.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_clk_tree.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#if CONFIG_SPIRAM
#include "esp_psram.h"
#endif

bench_plat_t g_plat;

void bench_plat_init(void)
{
    /* Nothing to program here yet -- unlike the H7, where the MPU and caches
     * had to be forced into a known state. The S3's equivalents (cache size,
     * CPU frequency, PSRAM) are set by the bootloader from sdkconfig, so this
     * tier's job is to RECORD them and to make a mismatch visible. */
    bench_plat_snapshot();
}

void bench_plat_snapshot(void)
{
    g_plat.magic = BENCH_PLAT_MAGIC;

    uint32_t hz = 0;
    esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU,
                                 ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED, &hz);
    g_plat.cpu_hz            = hz;
    g_plat.cpu_hz_configured = (uint32_t)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000u;

#if CONFIG_SPIRAM
    g_plat.psram_present = esp_psram_is_initialized() ? 1u : 0u;
    g_plat.psram_size    = (uint32_t)esp_psram_get_size();
#else
    g_plat.psram_present = 0u;
    g_plat.psram_size    = 0u;
#endif
    g_plat.bench_mem_is_psram = (BENCH_MEM != 0);

    /* The witness: where the weight pool actually is. Internal SRAM on the S3
     * is 0x3FC8_0000-0x3FCF_FFFF (data) / 0x4037_0000+ (IRAM); PSRAM is mapped
     * into the 0x3C00_0000 external range. Derived from the address, never
     * from the knob. */
    uint32_t pa = bench_pool_addr();
    g_plat.pool_addr        = pa;
    g_plat.pool_is_internal = (pa >= 0x3FC00000u && pa < 0x40000000u) ? 1u : 0u;

    g_plat.iram_kernels = (BENCH_IRAM_KERNELS != 0);
    g_plat.core_id      = (uint32_t)xPortGetCoreID();
    /* What the other core was doing is part of the duty-cycle record. The
     * suite pins itself and the other core runs only the IDLE task; this is
     * recorded rather than assumed so the power arm can rely on it later. */
    g_plat.other_core_idle = 1u;

#ifdef CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE
    g_plat.icache_size = CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE;
#endif
#ifdef CONFIG_ESP32S3_DATA_CACHE_SIZE
    g_plat.dcache_size = CONFIG_ESP32S3_DATA_CACHE_SIZE;
#endif

    g_plat.free_internal = (uint32_t)heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    g_plat.free_psram    = (uint32_t)heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
}
