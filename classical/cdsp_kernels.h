/**
 * cdsp_kernels.h -- the kernel seam under the classical arm (task 8).
 *
 * The algorithms in cdsp.c never call an FFT directly; they call these. The
 * portable reference (cdsp_fft_c.c) is the radix-2 body the throughput probe
 * validated against numpy to float32 round-off, and a port swaps the file for
 * one backed by arm_rfft_fast_f32 / dsps_fft2r_fc32 without touching the
 * algorithm -- the same one-swapped-file pattern as bench_dwt / bench_ccount /
 * bench_clock. Task 15 then measures the port, not the framing.
 */
#ifndef CDSP_KERNELS_H
#define CDSP_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1024 on the host (NOIZEUS 256, VoiceBank/DNS 512, room for 48 kHz). The
 * MCU builds set 512 on the command line: the context is ~46 KB at 1024 and
 * ~23 KB at 512, and the F411 has the latter beside the probe's pool. */
#ifndef CDSP_FFT_MAX
#define CDSP_FFT_MAX 1024u
#endif

typedef struct {
    uint32_t n;                       /* transform length, power of two     */
    float    tw[CDSP_FFT_MAX];        /* n/2 twiddles, interleaved cos,-sin */
    float    work[2u * CDSP_FFT_MAX]; /* interleaved complex scratch        */
} cdsp_fft_t;

/* Returns 0, or -1 if n is not a power of two in [8, CDSP_FFT_MAX]. */
int cdsp_fft_init(cdsp_fft_t *f, uint32_t n);

/* Forward real transform: x[n] -> X, the n/2+1 non-negative-frequency bins as
 * interleaved (re, im), i.e. n+2 floats. Unscaled. */
void cdsp_fft_forward(cdsp_fft_t *f, const float *x, float *X);

/* Inverse: X (n/2+1 bins, Hermitian symmetry implied) -> x[n], scaled 1/n, so
 * inverse(forward(x)) == x to float round-off. */
void cdsp_fft_inverse(cdsp_fft_t *f, const float *X, float *x);

#ifdef __cplusplus
}
#endif
#endif /* CDSP_KERNELS_H */
