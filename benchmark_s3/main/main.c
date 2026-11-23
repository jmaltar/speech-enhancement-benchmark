/**
 * benchmark_s3 -- ESP32-S3 throughput probe (task 3, third tier).
 *
 * Ordering mirrors the STM32 tiers deliberately: the probe runs to completion
 * BEFORE any USB/console traffic that could perturb it. On the F411 the USB
 * bring-up is explicitly placed after bench_run_all() because an enumeration
 * storm arriving mid-suite is exactly the I/O-inside-a-measurement the harness
 * rules forbid. Same here.
 */
#include "bench.h"
#include "bench_plat.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#ifdef BENCH_WITH_TINYUSB
#include "usb_glue.h"
#include "abf_cdc.h"
#endif
#ifdef CDSP_STAMP_DWT
extern volatile uint32_t g_cdsp_stage[8], g_cdsp_stage_n;
#endif
#ifdef DTLN_STAMP_DWT
#include "dtln_proc.h"
#endif
#ifndef BENCH_ABF_STACK_BYTES
#define BENCH_ABF_STACK_BYTES 8192
#endif
#ifndef BENCH_WITH_PROBE
/* No probe, no pool: the placement witness bench_plat_snapshot() records
 * reads as 0, which is the truthful value for an image that has none. */
uint32_t bench_pool_addr(void) { return 0; }
#endif

static const char *TAG = "bench_s3";

static void bench_task(void *arg)
{
    (void)arg;
    bench_plat_init();

    if (g_plat.cpu_hz != g_plat.cpu_hz_configured) {
        /* Loud, not silent. This is the S3's version of benchmarking on the
         * HSI, and it is the single most likely way to produce plausible
         * numbers at the wrong clock. */
        ESP_LOGE(TAG, "CPU CLOCK MISMATCH: running %" PRIu32 " Hz, configured %"
                 PRIu32 " Hz -- results are NOT comparable",
                 g_plat.cpu_hz, g_plat.cpu_hz_configured);
    }

#ifdef BENCH_WITH_PROBE
    bench_run_all();
#else
    /* Streaming image: no probe, but the cycle counter self-test still runs
     * so a frozen CCOUNT cannot read as impossibly fast. */
    g_bench.cpu_hz = g_plat.cpu_hz;
    g_bench.done   = bench_dwt_init() ? 1u : 0u;
#endif

#ifdef BENCH_WITH_TINYUSB
    /* AFTER the suite, never before -- see usb_glue.c. Identity first, so the
     * binding is ready for the first service call after enumeration. */
    usb_glue_abf_init(xTaskGetCurrentTaskHandle());
    usb_glue_init();

    /* The ABF1 loop, on THIS core (core 0, pinned above -- CCOUNT is per-core).
     * Event-driven: the wrapper's rx and DTR callbacks notify this task, and a
     * 1 ms timeout covers the one case no callback reports, a TX FIFO draining
     * after backpressure. Never a bare spin: that would starve the idle task
     * and trip the task watchdog 5 s into a host stall. */
#ifdef CDSP_STAMP_DWT
    uint32_t last_n = 0; TickType_t last_print = xTaskGetTickCount();
#endif
    for (;;) {
        abf_cdc_service();
        if (!abf_cdc_busy()) ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1));
        else vTaskDelay(1);
#ifdef CDSP_STAMP_DWT
        /* Stage sums to the console once per run, outside any timed region:
         * this tier has no debugger to read them with (the OTG PHY owns the
         * USB-Serial-JTAG pins). Printed when the count has been stable 1 s. */
        if (g_cdsp_stage_n != last_n && xTaskGetTickCount() - last_print > pdMS_TO_TICKS(1000)) {
            if (g_cdsp_stage_n == last_n) {}
            static uint32_t prev_seen; if (prev_seen == g_cdsp_stage_n) {
                printf("STAGES n=%lu %lu %lu %lu %lu %lu %lu %lu\n", (unsigned long)g_cdsp_stage_n,
                       (unsigned long)g_cdsp_stage[0], (unsigned long)g_cdsp_stage[1], (unsigned long)g_cdsp_stage[2],
                       (unsigned long)g_cdsp_stage[3], (unsigned long)g_cdsp_stage[4], (unsigned long)g_cdsp_stage[5],
                       (unsigned long)g_cdsp_stage[6]);
                last_n = g_cdsp_stage_n; last_print = xTaskGetTickCount();
            }
            prev_seen = g_cdsp_stage_n;
        }
#endif
#ifdef DTLN_STAMP_DWT
        /* Same arrangement for the DTLN processor's stage sums. */
        { static uint32_t d_last, d_prev; static TickType_t d_t;
          if (g_dtln.frames != d_last && xTaskGetTickCount() - d_t > pdMS_TO_TICKS(1000)) {
              if (d_prev == g_dtln.frames) {
                  printf("DTLN_STAGES frames=%lu fft=%lu s1=%lu ifft=%lu s2=%lu ola=%lu total=%lu arena=%lu+%lu rc=%lu"
                         " fc=%lu sc=%lu vec=%lu ref=%lu xcp=%lu pool=%lu/%lu@%08lx\n",
                         (unsigned long)g_dtln.frames, (unsigned long)g_dtln.t_fft, (unsigned long)g_dtln.t_stage1,
                         (unsigned long)g_dtln.t_ifft, (unsigned long)g_dtln.t_stage2, (unsigned long)g_dtln.t_ola,
                         (unsigned long)g_dtln.t_total, (unsigned long)g_dtln.arena1_used, (unsigned long)g_dtln.arena2_used,
                         (unsigned long)g_dtln.setup_rc, (unsigned long)g_dtln.fc_kernel, (unsigned long)g_dtln.fc_selfcheck,
                         (unsigned long)g_dtln.fc_calls_vec, (unsigned long)g_dtln.fc_calls_ref, (unsigned long)g_dtln.fc_x_copies,
                         (unsigned long)g_dtln.fc_pool_used, (unsigned long)g_dtln.fc_pool_size, (unsigned long)g_dtln.fc_pool_addr);
                  d_last = g_dtln.frames; d_t = xTaskGetTickCount();
              }
              d_prev = g_dtln.frames;
          } }
#endif
#ifdef BENCH_RNNOISE_STAMPS
        /* RNNoise stage sums (rnnoise_stamps.h), cumulative since boot, printed
         * when the stamped-frame count has not moved for a full second and
         * nothing is staged -- i.e. after a batch, not between two utterances
         * of one (the host's gap is well under 1 s) and never inside a paced
         * stream. 64-bit values as two 32-bit hex halves so the line does not
         * depend on printf's %llu. */
        { static uint32_t s_printed, s_seen; static TickType_t s_changed;
          extern volatile uint64_t g_rn_stage[8];
          extern volatile uint32_t g_rn_stage_n, g_rn_silent_n;
          uint32_t n = g_rn_stage_n;
          TickType_t now = xTaskGetTickCount();
          if (n != s_seen) { s_seen = n; s_changed = now; }
          else if (n != s_printed && !abf_cdc_busy() && now - s_changed > pdMS_TO_TICKS(1000)) {
              printf("RN_STAGES n=%lu sil=%lu", (unsigned long)n, (unsigned long)g_rn_silent_n);
              for (int i = 0; i < 8; i++) {
                  uint64_t v = g_rn_stage[i];
                  printf(" s%d=%08lx%08lx", i, (unsigned long)(uint32_t)(v >> 32), (unsigned long)(uint32_t)v);
              }
              printf("\n");
              s_printed = n;
          } }
#endif
#ifdef BENCH_WITH_RNNOISE
        /* Heap-placement and stack witness, once per completed run: printed
         * when frames_out has been stable for 1 s and nothing is staged, so
         * it never lands inside a paced stream. */
        { static uint32_t r_last, r_prev; static TickType_t r_t;
          uint32_t fo = abf_cdc_stats()->frames_out;
          if (fo != r_last && !abf_cdc_busy() && xTaskGetTickCount() - r_t > pdMS_TO_TICKS(1000)) {
              if (r_prev == fo) {
                  usb_glue_rnnoise_witness(fo, (uint32_t)uxTaskGetStackHighWaterMark(NULL));
                  r_last = fo; r_t = xTaskGetTickCount();
              }
              r_prev = fo;
          } }
#endif
    }
#else
    vTaskDelete(NULL);
#endif
}

void app_main(void)
{
    /* Pinned to core 0. CCOUNT is per-core and counts only while that core
     * runs, so the suite must not migrate mid-measurement. */
    /* Stack: 8 KB for the probe/classical/DTLN images; the RNNoise image sets
     * BENCH_ABF_STACK_BYTES from its -fstack-usage chain (CMakeLists.txt). */
    xTaskCreatePinnedToCore(bench_task, "bench", BENCH_ABF_STACK_BYTES, NULL, 5, NULL, 0);
}
