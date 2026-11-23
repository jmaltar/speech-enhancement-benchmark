/**
 * abf_proc.h -- the seam between the ABF1 transport and whatever processes audio.
 *
 * A processor is handed hop_len*channels input samples and must write the same
 * number of output samples. It owns whatever internal state creates the
 * (frame_len - hop_len) output lag the format declares -- including a bypass,
 * which is a pure delay line rather than a copy, so that the geometry is
 * uniform and the host's drop-and-trim is identical for every run.
 */
#ifndef ABF_PROC_H
#define ABF_PROC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct abf_proc_s {
    const char *name;
    void       *ctx;
    /* Called once per hop. Returns 0 on success. */
    int (*process)(void *ctx, const int16_t *in, int16_t *out, uint32_t n);
    /* Optional: called after CONFIG is accepted, before the first hop. */
    int (*setup)(void *ctx, uint32_t rate_hz, uint32_t frame_len,
                 uint32_t hop_len, uint32_t channels);
    void (*teardown)(void *ctx);
} abf_proc_t;

/* Reference processors. Neither is an algorithm; both exist to exercise the
 * binding before tasks 8-11 provide anything real. */

/* Pure delay of (frame_len - hop_len) samples. Output after the host's
 * drop-and-trim must equal the input BIT-EXACTLY -- the sharpest available
 * test of the lag and flush arithmetic, with no float in the path. */
const abf_proc_t *abf_proc_bypass(void);

/* Hann-windowed overlap-add with no spectral step. Exercises the COLA path;
 * reconstruction is exact to int16 rounding. */
const abf_proc_t *abf_proc_ola(void);

#ifdef __cplusplus
}
#endif
#endif /* ABF_PROC_H */
