/**
 * cdsp_fft_cmsis.c -- the kernel seam backed by arm_rfft_fast_f32 (STM32 tiers).
 *
 * Swapped in for cdsp_fft_c.c on the F411 and H723, and nothing else changes:
 * cdsp.c is compiled from ../classical unmodified, so task 15 measures this
 * file plus the toolchain's libm against the host oracle, not the framing.
 *
 * Packing. CMSIS returns the real transform of n points as n floats,
 * [X0.re, X(n/2).re, X1.re, X1.im, ..., X(n/2-1).re, X(n/2-1).im]; the seam's
 * contract is n/2+1 interleaved bins, so the two DC/Nyquist reals are
 * unpacked (forward) and repacked (inverse), their imaginary parts being zero
 * by Hermitian symmetry. CMSIS modifies its source buffer, so the input is
 * copied into the context's work area first. The inverse includes the 1/n
 * scaling, as the contract requires.
 *
 * Sizes whose tables this build does not carry (the F411 links only the
 * 512-point real transform, see CMakeLists.txt) fail in cdsp_fft_init, which
 * setup() turns into ABF_NAK_NO_ALGO: refused, never approximated.
 */
#include "cdsp_kernels.h"
#include "arm_math.h"
#include <string.h>

/* The probe's guard, repeated here because a deployed image may not carry the
 * probe: without ARM_MATH_LOOPUNROLL the hand-built CMSIS-DSP silently yields
 * the portable fallback and this port's cost figure would measure that. */
#ifndef ARM_MATH_LOOPUNROLL
#  error "ARM_MATH_LOOPUNROLL must be defined for the CMSIS-DSP build (see CMakeLists.txt)"
#endif

static arm_rfft_fast_instance_f32 s_inst;   /* one context, like cdsp.c */

int cdsp_fft_init(cdsp_fft_t *f, uint32_t n)
{
    if (n < 8u || n > CDSP_FFT_MAX || (n & (n - 1u)) != 0u) return -1;
    if (arm_rfft_fast_init_f32(&s_inst, (uint16_t)n) != ARM_MATH_SUCCESS) return -1;
    f->n = n;
    return 0;
}

void cdsp_fft_forward(cdsp_fft_t *f, const float *x, float *X)
{
    const uint32_t n = f->n;
    float *src = f->work, *out = f->work + n;
    memcpy(src, x, n * sizeof(float));
    arm_rfft_fast_f32(&s_inst, src, out, 0);
    X[0] = out[0]; X[1] = 0.0f;                         /* DC      */
    memcpy(X + 2, out + 2, (n - 2u) * sizeof(float));   /* bins 1..n/2-1 */
    X[n] = out[1]; X[n + 1u] = 0.0f;                    /* Nyquist */
}

void cdsp_fft_inverse(cdsp_fft_t *f, const float *X, float *x)
{
    const uint32_t n = f->n;
    float *src = f->work;
    src[0] = X[0]; src[1] = X[n];
    memcpy(src + 2, X + 2, (n - 2u) * sizeof(float));
    arm_rfft_fast_f32(&s_inst, src, x, 1);
}
