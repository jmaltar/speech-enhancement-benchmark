/**
 * usb_glue.c -- TinyUSB CDC device on the ESP32-S3's native USB, and the
 * identity handed to the ABF1 CDC binding.
 *
 * Same role and ordering rule as the STM32 glue: the stack is brought up AFTER
 * the probe (when the probe is built in), on the native USB peripheral (the
 * port silkscreened USB), never the USB-Serial-JTAG controller or the CH343
 * bridge on COM -- those are different hardware and would not be the same
 * measurement as the STM32 tiers.
 *
 * HOW THE BINDING MEETS THE WRAPPER. esp_tinyusb owns tud_task() (its own task,
 * CONFIG_TINYUSB_TASK_AFFINITY_CPU1) and the TinyUSB callbacks. The binding's
 * four transport primitives (abf_cdc_tinyusb.c, byte-identical to the STM32
 * copies) call tud_cdc_* directly from the ABF task on core 0, which TinyUSB
 * supports -- its CDC FIFOs are single-producer/single-consumer. The wrapper's
 * CDC-ACM layer is initialised ONLY so its rx and line-state callbacks can wake
 * the ABF task; its ring buffer is never read, so nothing is drained behind the
 * binding's back. The wake is what keeps the ABF loop event-driven rather than
 * a spin that would starve the idle task and trip the task watchdog.
 *
 * Recorded platform differences: tud_task on the other core (the STM32s have
 * one core), DCD in DMA mode (CONFIG_TINYUSB_MODE_DMA, no cache on internal
 * SRAM so no coherence issue), and the USB interrupt on core 0 because the
 * driver is installed from the core-0 task -- the same core the timed region
 * runs on, as on the STM32s.
 */
#include "usb_glue.h"
#include "tinyusb.h"
#include "tusb_cdc_acm.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_ota_ops.h"
#include "esp_partition.h"
#include "esp_image_format.h"
#include "esp_app_desc.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#include <string.h>
#include <stdio.h>

#include "abf_cdc.h"
#include "abf_proc.h"
#ifdef BENCH_WITH_CDSP
#include "cdsp.h"
#endif
#ifdef BENCH_WITH_DTLN
#include "dtln_proc.h"
#endif
#ifdef BENCH_WITH_RNNOISE
#include "rnnoise_proc.h"
#include "esp_heap_caps.h"
#endif
#if !defined(BENCH_WITH_CDSP) && !defined(BENCH_WITH_RNNOISE)
/* The build-twice twin: binding + bypass + OLA, every other id NAKed. */
static const abf_proc_t *basic_select(const abf_config_t *cfg)
{
    if (cfg->flags & ABF_CF_BYPASS) return abf_proc_bypass();
    if (cfg->algo_id >> ABF_NOISE_SHIFT) return NULL;
    uint32_t algo = cfg->algo_id & ABF_ALGO_MASK;
    if (algo == ABF_ALGO_BYPASS) return abf_proc_bypass();
    if (algo == ABF_ALGO_OLA)    return abf_proc_ola();
    return NULL;
}
#endif

static const char *TAG = "usb_glue";
static TaskHandle_t s_abf_task;

/* ---- descriptors: the same identity scheme as the STM32 tiers ---------------
 * cafe:4201 "ESP32S3 Audio Bench", serial = the base MAC from eFuse, so the
 * host finds the board by id and several boards stay distinguishable. The
 * wrapper's Kconfig defaults (303a:4001 "Espressif Device", serial "123456")
 * are overridden here, not in sdkconfig, so a menuconfig cannot revert it. */
static const tusb_desc_device_t s_dev_desc = {
    .bLength            = sizeof(tusb_desc_device_t),
    .bDescriptorType    = TUSB_DESC_DEVICE,
    .bcdUSB             = 0x0200,
    .bDeviceClass       = TUSB_CLASS_MISC,
    .bDeviceSubClass    = MISC_SUBCLASS_COMMON,
    .bDeviceProtocol    = MISC_PROTOCOL_IAD,
    .bMaxPacketSize0    = CFG_TUD_ENDPOINT0_SIZE,
    .idVendor           = 0xCafe,
    .idProduct          = 0x4201,
    .bcdDevice          = 0x0100,
    .iManufacturer      = 0x01,
    .iProduct           = 0x02,
    .iSerialNumber      = 0x03,
    .bNumConfigurations = 0x01,
};
static char s_serial[13];
static const char *s_strings[] = {
    (const char[]){0x09, 0x04},   /* English (US) */
    "Benchmark",
    "ESP32S3 Audio Bench",
    s_serial,
    "Benchmark CDC",
};

static void wake(void)
{
    if (s_abf_task) xTaskNotifyGive(s_abf_task);
}
static void cdc_rx_cb(int itf, cdcacm_event_t *event)         { (void)itf; (void)event; wake(); }
static void cdc_line_cb(int itf, cdcacm_event_t *event)       { (void)itf; (void)event; wake(); }

void usb_glue_init(void)
{
    uint8_t mac[6] = {0};
    esp_efuse_mac_get_default(mac);
    snprintf(s_serial, sizeof s_serial, "%02X%02X%02X%02X%02X%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    const tinyusb_config_t cfg = {
        .device_descriptor        = &s_dev_desc,
        .string_descriptor        = s_strings,
        .string_descriptor_count  = sizeof s_strings / sizeof s_strings[0],
        .external_phy             = false,  /* internal FS PHY */
        .configuration_descriptor = NULL,   /* wrapper default: one CDC, 64 B bulk */
    };
    ESP_ERROR_CHECK(tinyusb_driver_install(&cfg));

    const tinyusb_config_cdcacm_t acm = {
        .usb_dev                      = TINYUSB_USBDEV_0,
        .cdc_port                     = TINYUSB_CDC_ACM_0,
        .rx_unread_buf_sz             = 64,       /* never read; see the file comment */
        .callback_rx                  = &cdc_rx_cb,
        .callback_rx_wanted_char      = NULL,
        .callback_line_state_changed  = &cdc_line_cb,
        .callback_line_coding_changed = NULL,
    };
    ESP_ERROR_CHECK(tusb_cdc_acm_init(&acm));
    ESP_LOGI(TAG, "TinyUSB CDC up on native USB as cafe:4201, serial %s", s_serial);
}

/* ---- identity ----------------------------------------------------------------
 * HELLO.image_crc32 = CRC-32 of the running app image as it sits in flash,
 * from the partition start over esp_image_get_metadata()'s image_len -- which
 * is the whole .bin idf.py produced, trailing SHA-256 included -- so the host's
 * --check-bin compares like for like, as on the STM32s. Read in 4 KB chunks
 * through the same CRC the framing uses. */
static uint32_t image_crc32(void)
{
    const esp_partition_t *part = esp_ota_get_running_partition();
    if (!part) return 0u;
    esp_partition_pos_t pos = { .offset = part->address, .size = part->size };
    esp_image_metadata_t md;
    if (esp_image_get_metadata(&pos, &md) != ESP_OK) return 0u;
    uint32_t crc = ABF_CRC_INIT, off = 0u, len = md.image_len;
    static uint8_t buf[4096];
    while (off < len) {
        uint32_t n = len - off < sizeof buf ? len - off : (uint32_t)sizeof buf;
        if (esp_partition_read(part, off, buf, n) != ESP_OK) return 0u;
        crc = abf_crc32(crc, buf, n);
        off += n;
    }
    ESP_LOGI(TAG, "image %s @0x%" PRIx32 " len %" PRIu32 " crc32 %08" PRIx32,
             part->label, part->address, len, abf_crc32_final(crc));
    return abf_crc32_final(crc);
}

#ifndef ABF_BUILD_ID
#  define ABF_BUILD_ID "unstamped"
#endif

void usb_glue_abf_init(TaskHandle_t abf_task)
{
    s_abf_task = abf_task;
    abf_cdc_ident_t id = {
        .platform    = "ESP32S3",
        .build_id    = ABF_BUILD_ID,
        .image_crc32 = image_crc32(),
        .cpu_hz      = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000u,
    };
    abf_cdc_init(abf_proc_bypass(), &id);
#if defined(BENCH_WITH_DTLN)
    abf_cdc_set_selector(dtln_select);    /* DTLN for ABF_ALGO_DTLN, else cdsp_select */
#elif defined(BENCH_WITH_RNNOISE)
    abf_cdc_set_selector(rnnoise_select); /* RNNoise for ABF_ALGO_RNNOISE, else cdsp_select / NAK */
    usb_glue_rnnoise_heap_init();
#elif defined(BENCH_WITH_CDSP)
    abf_cdc_set_selector(cdsp_select);    /* bypass / OLA / classical from CONFIG.algo_id */
#else
    abf_cdc_set_selector(basic_select);   /* the twin: bypass / OLA only */
#endif
}

#ifdef BENCH_WITH_RNNOISE
/* ---- RNNoise heap placement: forced internal, then witnessed ----------------
 * rnnoise_create() takes its 18,492 B DenoiseState from malloc(). With
 * CONFIG_SPIRAM_USE_MALLOC the default policy places any request above
 * CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL (16,384 B) in PSRAM FIRST, so the
 * as-shipped code would put the state -- the GRU vectors, the pitch buffer, the
 * analysis/synthesis memories, read and written every hop -- in external RAM
 * without a word, and the number would be a PSRAM number (RESULTS.md: 1.5x on
 * the fastest kernel, jitter up 60-90x). The runtime limit is raised so every
 * malloc tries internal first. That is still only a preference (internal
 * exhausted falls through to PSRAM silently), so the glue records both free
 * sizes before the first CONFIG and prints their deltas after the first run:
 * the PSRAM delta must be 0, and the internal delta is this tier's reading of
 * the 28,820 B the Cortex-M builds measured (mallinfo() is a zero stub on IDF,
 * so g_rnnoise_heap_bytes cannot witness it here). */
static uint32_t s_free0_int, s_free0_ext;

void usb_glue_rnnoise_heap_init(void)
{
    heap_caps_malloc_extmem_enable(0x7FFFFFFFu);   /* limit above any request: internal first */
    s_free0_int = (uint32_t)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    s_free0_ext = (uint32_t)heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "rnnoise: malloc policy internal-first; free internal %" PRIu32 " B, PSRAM %" PRIu32 " B",
             s_free0_int, s_free0_ext);
}

void usb_glue_rnnoise_witness(uint32_t frames_out, uint32_t stack_hw_bytes)
{
    uint32_t fi = (uint32_t)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    uint32_t fe = (uint32_t)heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    /* One line, under the 128-byte UART FIFO, printed only between runs. */
    printf("RNNOISE frames=%lu heap_int_used=%ld heap_psram_used=%ld stack_hw=%lu stack=%u mallinfo=%lu %s\n",
           (unsigned long)frames_out, (long)(s_free0_int - fi), (long)(s_free0_ext - fe),
           (unsigned long)stack_hw_bytes, (unsigned)BENCH_ABF_STACK_BYTES,
           (unsigned long)g_rnnoise_heap_bytes, rnnoise_proc_version());
#ifdef RNNOISE_RAM_WEIGHTS
    /* +W: where the first copied weight array landed (0x3fc..... = internal
     * DRAM) and how many bytes were copied (88,399 expected). */
    { extern volatile uint32_t g_rnnoise_model_where, g_rnnoise_model_bytes;
      printf("RNNOISE_W model_where=%08lx model_bytes=%lu\n",
             (unsigned long)g_rnnoise_model_where, (unsigned long)g_rnnoise_model_bytes); }
#endif
}
#endif
