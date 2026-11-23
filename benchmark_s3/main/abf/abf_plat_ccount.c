/**
 * abf_plat_ccount.c -- ABF1 tick source for the ESP32-S3 (Xtensa LX7).
 *
 * There is no DWT on Xtensa; CCOUNT is read through esp_cpu_get_cycle_count(),
 * exactly as bench_ccount.c does for the probe.
 *
 * ONE GENUINE PLATFORM DIFFERENCE, recorded rather than hidden: CCOUNT is
 * PER-CORE and counts only while that core runs. The harness must therefore be
 * pinned to one core, and which core it was belongs in the run's provenance --
 * the probe records it in g_plat.core_id for the same reason.
 */
#include "abf_cdc.h"
#include "esp_cpu.h"
#include "sdkconfig.h"

uint32_t abf_plat_ticks(void)   { return esp_cpu_get_cycle_count(); }
uint32_t abf_plat_tick_hz(void) { return CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000u; }
