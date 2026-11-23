/**
 * cdsp_fft_c.c -- portable reference FFT for the classical arm.
 *
 * The complex radix-2 body is the one from the throughput probe
 * (bench_kernels.c, validated against numpy.fft to float32 round-off), with
 * the twiddle table built for the transform's own length instead of a shared
 * maximum. Real transforms are done as a complex transform with zero imaginary
 * part: twice the work of a packed real FFT, which is exactly the cost the
 * platform ports remove by swapping this file for the vendor real FFT. This
 * file is the correctness oracle, not the fast path.
 */
#include "cdsp_kernels.h"
#include <math.h>
#include <string.h>

int cdsp_fft_init(cdsp_fft_t *f, uint32_t n)
{
    if (n < 8u || n > CDSP_FFT_MAX || (n & (n - 1u)) != 0u) return -1;
    f->n = n;
    for (uint32_t m = 0; m < n / 2u; ++m) {
        double a = -2.0 * 3.14159265358979323846 * (double)m / (double)n;
        f->tw[2u * m]      = (float)cos(a);
        f->tw[2u * m + 1u] = (float)sin(a);
    }
    return 0;
}

/* In-place complex radix-2 DIT, interleaved re/im. Identical body to the probe. */
static void fft_radix2(float *d, const float *tw, uint32_t N)
{
    for (uint32_t i = 1u, j = 0u; i < N; ++i) {
        uint32_t bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = d[2*i], ti = d[2*i+1];
            d[2*i] = d[2*j]; d[2*i+1] = d[2*j+1];
            d[2*j] = tr;     d[2*j+1] = ti;
        }
    }
    for (uint32_t len = 2u; len <= N; len <<= 1) {
        uint32_t step = N / len;                /* twiddle table stride       */
        uint32_t half = len >> 1;
        for (uint32_t i = 0u; i < N; i += len) {
            for (uint32_t k = 0u; k < half; ++k) {
                uint32_t t = k * step;
                float wr = tw[2*t], wi = tw[2*t+1];
                uint32_t a = i + k, b = a + half;
                float xr = d[2*b], xi = d[2*b+1];
                float tr = xr * wr - xi * wi;
                float ti = xr * wi + xi * wr;
                d[2*b]   = d[2*a]   - tr;
                d[2*b+1] = d[2*a+1] - ti;
                d[2*a]   += tr;
                d[2*a+1] += ti;
            }
        }
    }
}

void cdsp_fft_forward(cdsp_fft_t *f, const float *x, float *X)
{
    uint32_t n = f->n;
    for (uint32_t i = 0; i < n; ++i) { f->work[2*i] = x[i]; f->work[2*i+1] = 0.0f; }
    fft_radix2(f->work, f->tw, n);
    memcpy(X, f->work, (n + 2u) * sizeof(float));   /* bins 0..n/2 */
}

void cdsp_fft_inverse(cdsp_fft_t *f, const float *X, float *x)
{
    uint32_t n = f->n;
    /* Rebuild the full Hermitian spectrum, conjugated: ifft(X) = conj(fft(conj(X)))/n. */
    for (uint32_t k = 0; k <= n / 2u; ++k) {
        f->work[2*k]   =  X[2*k];
        f->work[2*k+1] = -X[2*k+1];
    }
    for (uint32_t k = n / 2u + 1u; k < n; ++k) {
        uint32_t m = n - k;
        f->work[2*k]   = X[2*m];          /* conj(conj(X[m])) = X[m] */
        f->work[2*k+1] = X[2*m+1];
    }
    fft_radix2(f->work, f->tw, n);
    float s = 1.0f / (float)n;
    for (uint32_t i = 0; i < n; ++i) x[i] = f->work[2*i] * s;   /* real part; conj drops out */
}
