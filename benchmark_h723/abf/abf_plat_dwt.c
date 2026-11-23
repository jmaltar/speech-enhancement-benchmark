/**
 * abf_plat_dwt.c -- ABF1 tick source for the STM32 tiers (Cortex-M4F / M7).
 *
 * One of three swapped files, the same shape as bench_dwt.c / bench_ccount.c /
 * bench_clock.c: identical semantics, one file per platform, the difference
 * confined and recorded. Ticks here are CPU CYCLES, and HELLO says so by
 * reporting tick_hz = SystemCoreClock -- 96,000,000 on the F411 (never the
 * datasheet's 100) and 550,000,000 on the H723.
 *
 * The counter is assumed already running: bench_dwt.c enables TRCENA and
 * CYCCNTENA and self-tests that CYCCNT actually increments, because a frozen
 * counter reads as impossibly fast rather than as broken. Do not re-init it
 * here; call bench_dwt_init() once at startup and let this read it.
 *
 * CYCCNT wraps every 44.7 s at 96 MHz and 7.81 s at 550 MHz. Only differences
 * are ever taken, and every interval of interest is one hop.
 */
#include "abf_cdc.h"

#if defined(STM32H7) || defined(STM32H723xx)
#  include "stm32h7xx.h"
#else
#  include "stm32f4xx.h"
#endif

uint32_t abf_plat_ticks(void)   { return DWT->CYCCNT; }
uint32_t abf_plat_tick_hz(void) { return SystemCoreClock; }
