/**
 * cdsp_fft_espdsp.c -- the kernel seam backed by esp-dsp (ESP32-S3).
 *
 * esp-dsp 1.5 ships a complex radix-2 transform (dsps_fft2r_fc32, LX7 SIMD
 * path asserted below) and a forward-only real-spectrum unpack, and NO inverse
 * real transform. Rather than hand-roll the packed inverse, both directions
 * use the 512-point COMPLEX transform: forward with a zero imaginary part,
 * inverse by the conjugation identity ifft(X) = conj(fft(conj(X))) / N over
 * the full Hermitian spectrum. Correct by construction and vendor-optimised,
 * but each transform does the work of a 512-point complex FFT where a packed
 * real transform would do a 256-point one -- so this tier's FFT stage is
 * roughly 2x what the probe's espdsp_rfft512 row costs. Stated here and in
 * the results so the stage table is read right; the packed form is a known
 * optimisation left undone. cdsp.c above this file is unmodified.
 */
#include "cdsp_kernels.h"
#include "esp_dsp.h"
#include "sdkconfig.h"
#include <string.h>

#if !defined(CONFIG_DSP_OPTIMIZED)
#  error "CONFIG_DSP_OPTIMIZED unset: esp-dsp would build its ANSI C fallback"
#endif
#if (dsps_fft2r_fc32_aes3_enabled != 1)
#  error "dsps_fft2r_fc32 would not use the ESP32-S3 SIMD path"
#endif

static int s_table_ready;

/* The LX7 SIMD path loads 128 bits at a time and REQUIRES 16-byte alignment;
 * the context's work array is only 4-aligned (it is a float array inside a
 * struct that abf_frame.h aligns to 4), and feeding it to the aes3 transform
 * produced plausible-looking garbage -- tens of thousands of LSB off, with no
 * error returned. Found on hardware 2026-10-14. The probe's own buffer was
 * declared aligned, which is why its checksums never showed it. */
static float s_w[2u * CDSP_FFT_MAX] __attribute__((aligned(16)));

int cdsp_fft_init(cdsp_fft_t *f, uint32_t n)
{
    if (n < 8u || n > CDSP_FFT_MAX || (n & (n - 1u)) != 0u) return -1;
    if (n > (uint32_t)CONFIG_DSP_MAX_FFT_SIZE) return -1;
    if (!s_table_ready) {
        esp_err_t e = dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
        if (e != ESP_OK && e != ESP_ERR_DSP_REINITIALIZED) return -1;
        s_table_ready = 1;
    }
    f->n = n;
    return 0;
}

void cdsp_fft_forward(cdsp_fft_t *f, const float *x, float *X)
{
    const uint32_t n = f->n;
    float *w = s_w;                              /* 2n floats, interleaved, 16-aligned */
    for (uint32_t i = 0; i < n; i++) { w[2u*i] = x[i]; w[2u*i + 1u] = 0.0f; }
    dsps_fft2r_fc32(w, (int)n);
    dsps_bit_rev_fc32(w, (int)n);
    memcpy(X, w, (n + 2u) * sizeof(float));      /* bins 0..n/2, interleaved */
}

void cdsp_fft_inverse(cdsp_fft_t *f, const float *X, float *x)
{
    const uint32_t n = f->n;
    float *w = s_w;
    const float inv_n = 1.0f / (float)n;
    /* conj of the full Hermitian spectrum: bins 0..n/2 as given, n/2+1..n-1
     * mirrored; conj(X[n-k]) = X[k] so the mirror needs no second negation. */
    for (uint32_t k = 0; k <= n / 2u; k++) { w[2u*k] = X[2u*k]; w[2u*k + 1u] = -X[2u*k + 1u]; }
    for (uint32_t k = n / 2u + 1u; k < n; k++) { w[2u*k] = X[2u*(n - k)]; w[2u*k + 1u] = X[2u*(n - k) + 1u]; }
    dsps_fft2r_fc32(w, (int)n);
    dsps_bit_rev_fc32(w, (int)n);
    for (uint32_t i = 0; i < n; i++) x[i] = w[2u*i] * inv_n;   /* real part of conj(.)/n */
}
