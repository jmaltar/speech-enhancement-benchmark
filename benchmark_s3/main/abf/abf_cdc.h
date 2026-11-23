/**
 * abf_cdc.h -- the USB CDC binding for the three MCU tiers (task 7).
 *
 * The same ABF1 byte stream the RPi5 receives as a file, delivered over
 * TinyUSB CDC instead. One frame format, two transport bindings; a CDC capture
 * and a file run are diffable against each other, and abf_cdc.c is intended to
 * be BYTE-IDENTICAL on the F411, the H723 and the ESP32-S3, exactly as bench.h
 * is. Verify with cksum before trusting a cross-platform number.
 *
 * Two seams keep it that way, both modelled on bench_dwt.c / bench_ccount.c /
 * bench_clock.c -- identical semantics, one swapped file per platform:
 *
 *   abf_plat_ticks()   the free-running cycle counter: DWT CYCCNT on the
 *                      STM32s, CCOUNT on Xtensa.
 *   abf_cdc_io_*()     the four transport primitives, mapped onto tud_cdc_* by
 *                      abf_cdc_tinyusb.c on hardware and onto a memory pipe by
 *                      the host tests, which is how this binding is proved to
 *                      emit the same stream as the file binding without a board.
 *
 * IT NEVER BLOCKS. abf_cdc_service() is called from the main loop beside
 * tud_task() and always returns promptly: at most one packet is decoded per
 * call, and at most one packet is ever staged for transmission, so a full TX
 * FIFO applies backpressure instead of stalling the core. The placeholder echo
 * it replaces had that property and it is the one property that must survive --
 * ITM stalls on a full trace FIFO and semihosting halts the core outright, and
 * a transport that waits inside a timed region corrupts the measurement it is
 * there to carry.
 */
#ifndef ABF_CDC_H
#define ABF_CDC_H

#include "abf_frame.h"
#include "abf_proc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Largest payload this device will accept. The CDC FIFOs are 1024 B each
 * (CFG_TUD_CDC_RX/TX_BUFSIZE), so a packet must fit one: 1008 B of payload
 * plus 16 B of framing. RNNoise's 960 B hop is the worst case in scope and
 * clears it by 48 B. A CONFIG asking for more is refused with
 * ABF_NAK_HOP_TOO_BIG rather than truncated. */
#define ABF_DEV_MAX_PAYLOAD  1008u

typedef struct {
    const char *platform;      /* "STM32F411" / "STM32H723" / "ESP32S3" */
    const char *build_id;
    uint32_t    image_crc32;
    uint32_t    cpu_hz;
} abf_cdc_ident_t;

/* Transport primitives. Provided by abf_cdc_tinyusb.c on hardware. */
uint32_t abf_cdc_io_read(void *buf, uint32_t n);         /* 0 if nothing ready */
uint32_t abf_cdc_io_write(const void *buf, uint32_t n);  /* bytes accepted      */
void     abf_cdc_io_flush(void);
int      abf_cdc_io_connected(void);

/* Tick source. Provided per platform, like bench_dwt.c / bench_ccount.c. */
uint32_t abf_plat_ticks(void);
uint32_t abf_plat_tick_hz(void);

void abf_cdc_init(const abf_proc_t *proc, const abf_cdc_ident_t *id);

/* Optional: choose the processor from the accepted CONFIG (algo_id, flags),
 * so one image serves every algorithm it carries. Called after the CONFIG is
 * validated and before setup(); returning NULL means "this image does not
 * carry that algorithm" and the CONFIG is NAKed with ABF_NAK_NO_ALGO. Without
 * a selector the proc given to abf_cdc_init() serves every run. */
typedef const abf_proc_t *(*abf_proc_select_fn)(const abf_config_t *cfg);
void abf_cdc_set_selector(abf_proc_select_fn sel);
void abf_cdc_service(void);
int  abf_cdc_busy(void);       /* output still staged */
const abf_stats_t *abf_cdc_stats(void);

#ifdef __cplusplus
}
#endif
#endif /* ABF_CDC_H */
