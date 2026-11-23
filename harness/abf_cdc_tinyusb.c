/**
 * abf_cdc_tinyusb.c -- maps the four ABF1 transport primitives onto TinyUSB CDC.
 *
 * BUILT ON HARDWARE ONLY (it needs tusb.h); the host tests substitute a memory
 * pipe for these four functions, which is how abf_cdc.c is proved to emit the
 * same stream as the file binding without a board. Identical on all three MCU
 * tiers -- TinyUSB 0.21.0 is the same commit on each, which is the whole point
 * of the one-stack decision.
 *
 * The contract abf_cdc.c relies on: WRITE MAY ACCEPT ZERO BYTES. A full TX FIFO
 * is backpressure, not an error and not a reason to wait. tud_cdc_write() can
 * short-write, so the available room is checked first and the caller resumes
 * where it left off.
 */
#include "abf_cdc.h"
#include "tusb.h"

uint32_t abf_cdc_io_read(void *buf, uint32_t n)
{
    if (!tud_cdc_available()) return 0u;
    return tud_cdc_read(buf, n);
}

uint32_t abf_cdc_io_write(const void *buf, uint32_t n)
{
    uint32_t room = tud_cdc_write_available();
    if (room == 0u) return 0u;              /* FIFO full: backpressure */
    if (n > room) n = room;
    return tud_cdc_write(buf, n);
}

void abf_cdc_io_flush(void) { tud_cdc_write_flush(); }

int abf_cdc_io_connected(void) { return tud_cdc_connected() ? 1 : 0; }
