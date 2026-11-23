/* dtln_relocate.c -- the S3's model-placement seam for the DTLN port.
 *
 * On this tier the flatbuffer is .rodata in external octal flash, executed in
 * place through the 64 KB data cache; every weight is touched once per hop, so
 * the 397 KB model streams through the cache from SPI flash each hop. Whether
 * that is what the hop pays is a measured question: -DBENCH_DTLN_MODEL=psram
 * copies the model into the octal PSRAM (also cached, a different bus),
 * -DBENCH_DTLN_MODEL=flash leaves it (the default). Internal SRAM cannot hold it.
 * g_dtln_model_where witnesses where the interpreter actually read it from. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "esp_heap_caps.h"

volatile uint32_t g_dtln_model_where[2];
const uint8_t *dtln_model_relocate(const uint8_t *p, size_t len)
{
    static int k;
#ifdef DTLN_MODEL_PSRAM
    /* The copy keeps p's address mod 16: the PIE build's --align-inplace
     * emission places the flatbuffer at a 4-aligned offset chosen so its
     * stage-2 basis rows are 16-aligned, and a 16-aligned copy would undo
     * that (and the glue would then try to copy 262 KB into its pool). */
    const size_t skew = (size_t)((uintptr_t)p & 15u);
    uint8_t *q = heap_caps_aligned_alloc(16, len + 16, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!q) return NULL;
    memcpy(q + skew, p, len);
    p = q + skew;
#endif
    if (k < 2) g_dtln_model_where[k++] = (uint32_t)(uintptr_t)p;
    return p;
}
