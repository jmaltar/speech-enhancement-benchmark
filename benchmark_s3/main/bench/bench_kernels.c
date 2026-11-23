/**
 * bench_kernels.c -- the kernels the probe actually times.
 *
 * These are PLAIN C reference implementations. That is deliberate: the pairing
 * "plain C vs CMSIS-DSP on the identical kernel" measures what the vendor
 * library is worth on this core, which is a number the project needs anyway --
 * the structured-vs-unstructured pruning arm rests on the claim that only whole
 * -group removal runs at full speed on CMSIS kernels, and that claim is
 * currently an assertion. Drop the CMSIS variants in beside these (see the
 * CMSIS-DSP note at the bottom) and time both.
 *
 * Shapes are sized to fit F411 SRAM, not to match DTLN or RNNoise layer sizes.
 * A DTLN gate matrix is 512x257 floats = 526 KB and does not fit in a 128 KB
 * part at all -- which is a budget-table finding, not a probe problem. What the
 * probe needs is the achieved RATE, and the rate is measurable on a shape that
 * fits and exercises the same inner loop.
 */
#include "bench.h"
/* ESP32-S3: the STM32 tiers include "main.h" here for CMSIS core headers.
 * Nothing in the plain-C kernels below needs it; the rest of this file is
 * byte-identical to benchmark_h723/bench/bench_kernels.c lines 1-236, which
 * is asserted by PROVENANCE.md rather than assumed. */
#include "bench_plat.h"
#include <math.h>
#include <string.h>

/* --- storage ------------------------------------------------------------
 * One shared pool, carved up per kernel, so the largest f32 and the largest
 * int8 matrix do not have to be resident simultaneously. 64 KB of the F411's
 * 128 KB, leaving room for the USB stack, ring buffers and STFT scratch that
 * task 21 warns will be competing for the same SRAM later.
 * ---------------------------------------------------------------------- */
#define POOL_BYTES   (64u * 1024u)
#define FFT_MAX      512u
#define BINS         257u          /* 512-pt real FFT -> 257 bins            */

static uint8_t s_pool[POOL_BYTES] __attribute__((aligned(8)));

static float  s_x[512];
static float  s_y[512];
static float  s_fft[2u * FFT_MAX];      /* interleaved re,im                 */
static float  s_fft_src[2u * FFT_MAX];  /* pristine copy, for the prep step  */
static float  s_tw[FFT_MAX];            /* N/2 twiddles, interleaved         */
static float  s_mag[BINS];
static float  s_out[BINS];
static int8_t s_xs8[256];
static int32_t s_ys32[256];

/* Deterministic filler. A fixed seed keeps the operand VALUES identical across
 * builds and across platforms, so a cycle count that moves is the port moving,
 * not the data. */
static uint32_t s_rng = 20260909u;
static uint32_t rnd(void) { s_rng = s_rng * 1664525u + 1013904223u; return s_rng; }
static float    rndf(void) { return ((float)(int32_t)(rnd() >> 8) / 8388608.0f) - 1.0f; }

void bench_kernels_init(void)
{
    float *wf = (float *)(void *)s_pool;
    for (uint32_t i = 0; i < POOL_BYTES / sizeof(float); ++i) wf[i] = rndf();

    for (uint32_t i = 0; i < 512u; ++i) { s_x[i] = rndf(); s_y[i] = 0.0f; }
    for (uint32_t i = 0; i < 256u; ++i) { s_xs8[i] = (int8_t)(rnd() & 0x7Fu) - 64; }

    /* Magnitudes are kept strictly positive and away from zero: logf(0) and the
     * subnormal range are both pathologically slow, and timing either would
     * measure an input distribution rather than the kernel. */
    for (uint32_t i = 0; i < BINS; ++i) s_mag[i] = 0.01f + 0.99f * ((float)i / (float)BINS);

    for (uint32_t m = 0; m < FFT_MAX / 2u; ++m) {
        float a = -2.0f * 3.14159265358979f * (float)m / (float)FFT_MAX;
        s_tw[2u * m]      = cosf(a);
        s_tw[2u * m + 1u] = sinf(a);
    }
    for (uint32_t i = 0; i < 2u * FFT_MAX; ++i) s_fft_src[i] = rndf();
}

/* --- radix-2 complex FFT, in place -------------------------------------- */
static void fft_radix2(float *d, uint32_t N)
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
        uint32_t step = FFT_MAX / len;          /* twiddle table stride       */
        uint32_t half = len >> 1;
        for (uint32_t i = 0u; i < N; i += len) {
            for (uint32_t k = 0u; k < half; ++k) {
                uint32_t t = k * step;
                float wr = s_tw[2*t], wi = s_tw[2*t+1];
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

/* The twiddle stride above assumes the table was built for FFT_MAX. For a
 * shorter transform the stride scales, which is why `step` uses FFT_MAX and not
 * N -- one table serves every size up to FFT_MAX. */

static void prep_fft256(void) { memcpy(s_fft, s_fft_src, 2u * 256u * sizeof(float)); }
static void prep_fft512(void) { memcpy(s_fft, s_fft_src, 2u * 512u * sizeof(float)); }

/* --- guards and checksums ------------------------------------------------
 * `f32_bits` is the O(1) DCE tripwire a kernel returns from inside the timed
 * region. `chk_f32`/`chk_s32` are the O(n) correctness checks, and they run
 * only from the verify callbacks, which bench_measure calls untimed.
 * ---------------------------------------------------------------------- */
static inline uint32_t f32_bits(float v)
{
    uint32_t bits; memcpy(&bits, &v, sizeof(bits));
    return bits;
}

static uint32_t chk_f32(const float *p, uint32_t n)
{
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; ++i) acc += p[i];
    return f32_bits(acc);
}

static uint32_t chk_s32(const int32_t *p, uint32_t n)
{
    uint32_t acc = 0u;
    for (uint32_t i = 0; i < n; ++i) acc += (uint32_t)p[i];
    return acc;
}

/* --- kernels ------------------------------------------------------------ */

static uint32_t k_fft256(void) { fft_radix2(s_fft, 256u); return f32_bits(s_fft[0]); }
static uint32_t k_fft512(void) { fft_radix2(s_fft, 512u); return f32_bits(s_fft[0]); }
static uint32_t v_fft256(void) { return chk_f32(s_fft, 2u * 256u); }
static uint32_t v_fft512(void) { return chk_f32(s_fft, 2u * 512u); }

static uint32_t matvec_f32(uint32_t rows, uint32_t cols)
{
    const float *w = (const float *)(const void *)s_pool;
    for (uint32_t r = 0; r < rows; ++r) {
        const float *wr = w + (size_t)r * cols;
        float acc = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) acc += wr[c] * s_x[c];
        s_y[r] = acc;
    }
    return f32_bits(s_y[0]);
}
static uint32_t k_matvec_f32_64(void)  { return matvec_f32(64u, 64u); }
static uint32_t k_matvec_f32_128(void) { return matvec_f32(128u, 128u); }
static uint32_t v_matvec_f32_64(void)  { return chk_f32(s_y, 64u); }
static uint32_t v_matvec_f32_128(void) { return chk_f32(s_y, 128u); }

static uint32_t matvec_s8(uint32_t rows, uint32_t cols)
{
    const int8_t *w = (const int8_t *)(const void *)s_pool;
    for (uint32_t r = 0; r < rows; ++r) {
        const int8_t *wr = w + (size_t)r * cols;
        int32_t acc = 0;
        for (uint32_t c = 0; c < cols; ++c) acc += (int32_t)wr[c] * (int32_t)s_xs8[c];
        s_ys32[r] = acc;
    }
    return (uint32_t)s_ys32[0];
}
static uint32_t k_matvec_s8_128(void) { return matvec_s8(128u, 128u); }
static uint32_t k_matvec_s8_256(void) { return matvec_s8(256u, 256u); }
static uint32_t v_matvec_s8_128(void) { return chk_s32(s_ys32, 128u); }
static uint32_t v_matvec_s8_256(void) { return chk_s32(s_ys32, 256u); }

static float s_dot;
static uint32_t k_dot512(void)
{
    float acc = 0.0f;
    for (uint32_t i = 0; i < 512u; ++i) acc += s_x[i] * s_fft_src[i];
    s_dot = acc;
    return f32_bits(acc);
}
static uint32_t v_dot512(void) { return f32_bits(s_dot); }

/* Analysis window: the 512-point Hann multiply of the locked configuration. */
static uint32_t k_window512(void)
{
    for (uint32_t i = 0; i < 512u; ++i) s_y[i] = s_x[i] * s_fft_src[i];
    return f32_bits(s_y[0]);
}
static uint32_t v_window512(void) { return chk_f32(s_y, 512u); }

/* Bin magnitude -- every classical enhancer starts here. */
static uint32_t k_mag257(void)
{
    for (uint32_t i = 0; i < BINS; ++i) {
        float re = s_fft_src[2*i], im = s_fft_src[2*i+1];
        s_out[i] = sqrtf(re * re + im * im);
    }
    return f32_bits(s_out[0]);
}

/* expf / logf carry the MMSE-STSA gain rule. On a core with no transcendental
 * hardware these routinely cost more than the FFT, which is exactly the sort of
 * thing an inference-only cost model (TinyLSTMs excludes STFT and mel entirely)
 * never sees. */
static uint32_t k_expf257(void)
{
    for (uint32_t i = 0; i < BINS; ++i) s_out[i] = expf(-s_mag[i]);
    return f32_bits(s_out[0]);
}
static uint32_t k_logf257(void)
{
    for (uint32_t i = 0; i < BINS; ++i) s_out[i] = logf(s_mag[i]);
    return f32_bits(s_out[0]);
}
static uint32_t k_sqrtf257(void)
{
    for (uint32_t i = 0; i < BINS; ++i) s_out[i] = sqrtf(s_mag[i]);
    return f32_bits(s_out[0]);
}

/* tanh and the logistic sigmoid are the GRU/LSTM activations; RNNoise and DTLN
 * evaluate them once per unit per frame. */
static uint32_t k_tanh257(void)
{
    for (uint32_t i = 0; i < BINS; ++i) s_out[i] = tanhf(s_mag[i]);
    return f32_bits(s_out[0]);
}

/* mag / expf / logf / sqrtf / tanhf all write s_out, and bench_measure calls
 * the verify immediately after that kernel's own warm-up, so one callback
 * serves all five. */
static uint32_t v_out_bins(void) { return chk_f32(s_out, BINS); }

/* ===========================================================================
 * esp-dsp paired kernels
 *
 * Same rule as the CMSIS pairs on the STM32s: each row has the SAME shape, the
 * SAME operand buffers and the SAME declared op count as its plain-C partner,
 * so the pair differs only by implementation.
 *
 * Build switches, asserted rather than assumed
 * --------------------------------------------
 * CLAUDE.md predicted esp-dsp would have an ARM_MATH_LOOPUNROLL-shaped problem.
 * It has two, and they are Kconfig rather than compiler macros:
 *
 *   CONFIG_DSP_OPTIMIZED   A Kconfig CHOICE between "Optimized" (assembly) and
 *                          "ANSI C" (portable fallback). It defaults to
 *                          Optimized, but flipping it silently substitutes a
 *                          plain-C implementation and reports it as "what
 *                          esp-dsp is worth" -- exactly the failure
 *                          ARM_MATH_LOOPUNROLL caused on the F411.
 *   *_aes3_enabled         Per-function guards selecting the ESP32-S3 SIMD
 *                          path over the generic Xtensa (ae32) one. The aes3
 *                          kernels are the analogue of CMSIS's SMLAD paths and
 *                          are the entire point of the vendor rows.
 *
 * Both are made build-breaking here, per this project's rule: a build switch
 * that changes what is measured must break the build, not the results.
 * ======================================================================== */
#include "dsps_fft2r.h"
#include "dsps_dotprod.h"
#include "dspi_dotprod.h"
#include "dspm_mult.h"
#include "dsps_mul.h"
#include "dsps_sqrt.h"
#include "dsp_types.h"
#include "esp_err.h"

#if !defined(CONFIG_DSP_OPTIMIZED)
#  error "CONFIG_DSP_OPTIMIZED unset: esp-dsp would build its ANSI C fallback"
#endif
#if (dsps_fft2r_fc32_aes3_enabled != 1)
#  error "dsps_fft2r_fc32 would not use the ESP32-S3 SIMD path"
#endif
#if (dspm_mult_f32_aes3_enabled != 1)
#  error "dspm_mult_f32 would not use the ESP32-S3 SIMD path"
#endif
#if (dsps_dotprod_f32_aes3_enabled != 1)
#  error "dsps_dotprod_f32 would not use the ESP32-S3 SIMD path"
#endif

static float   s_dot_dsp;
static int8_t  s_ys8[256];      /* saturated int8 output, esp-dsp s8 row only */

/* O(n) checksum for the int8 output. The shared plain-C region has chk_f32 and
 * chk_s32 but no chk_s8, because no plain-C kernel produces int8. Declared here
 * so that region stays identical to the STM32 tiers. */
static uint32_t chk_s8(const int8_t *p, uint32_t n)
{
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < n; ++i) { h ^= (uint32_t)(uint8_t)p[i]; h *= 16777619u; }
    return h;
}

static void bench_espdsp_init(void)
{
    /* Twiddle table for the largest transform used. CONFIG_DSP_MAX_FFT_SIZE is
     * the counterpart of CMSIS's ARM_TABLE_* trap: it defaults to 4096 and the
     * table is sized from it, so a project needing 512 pays for 4096 unless it
     * says otherwise. Measured as a flash line item, see CLAUDE.md. */
    dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
}

/* --- transforms ---------------------------------------------------------
 * Paired like-for-like with the plain-C radix-2 COMPLEX transform. The F411
 * got this wrong once -- it documented arm_rfft_fast_f32 as the partner for a
 * complex kernel, which would have flattered the vendor by ~2x. dsps_fft2r_fc32
 * is complex-to-complex over N complex points, so it is the correct partner.
 * Bit-reversal is included in the timed region because the plain-C kernel's
 * output is in natural order and the comparison must produce the same thing. */
static uint32_t k_dsp_fft256(void)
{
    dsps_fft2r_fc32(s_fft, 256);
    dsps_bit_rev_fc32(s_fft, 256);
    return f32_bits(s_fft[0]);
}
static uint32_t k_dsp_fft512(void)
{
    dsps_fft2r_fc32(s_fft, 512);
    dsps_bit_rev_fc32(s_fft, 512);
    return f32_bits(s_fft[0]);
}

/* Real 512-point transform: 512 real samples packed as 256 complex, then
 * fft2r + bit-reverse + cplx2reC, which is esp-dsp's documented real-FFT form
 * (examples/fft/main/dsps_fft_main.c). Measured as its own UNPAIRED row on the
 * 2.5*N*log2(N) real convention -- identical treatment to cmsis_rfft_512 -- 
 * because the STFT of the locked configuration is a real transform and it is
 * what the classical and DTLN budget rows actually pay for. */
static uint32_t k_dsp_rfft512(void)
{
    dsps_fft2r_fc32(s_fft, 256);
    dsps_bit_rev_fc32(s_fft, 256);
    dsps_cplx2reC_fc32(s_fft, 256);
    return f32_bits(s_fft[0]);
}
static uint32_t v_dsp_rfft512(void) { return chk_f32(s_fft, 512u); }

/* --- matrix-vector ------------------------------------------------------
 * dspm_mult_f32(A, B, C, m, n, k) with k = 1 is a matrix-vector product. */
static uint32_t k_dsp_matvec_f32_64(void)
{
    dspm_mult_f32((const float *)(const void *)s_pool, s_x, s_y, 64, 64, 1);
    return f32_bits(s_y[0]);
}
static uint32_t k_dsp_matvec_f32_128(void)
{
    dspm_mult_f32((const float *)(const void *)s_pool, s_x, s_y, 128, 128, 1);
    return f32_bits(s_y[0]);
}

/* ===========================================================================
 * ESP-NN -- the INT8 row a deployed model actually executes
 *
 * esp-dsp has no int8 matvec; the closest it offers (dspi_dotprod_s8) loses to
 * plain C. ESP-NN is the CMSIS-NN analogue, and esp_nn_fully_connected_s8 is
 * the direct counterpart of arm_nn_vec_mat_mult_t_s8: a matrix-vector product
 * with the input/filter zero-points and the output requantization folded in,
 * which is what a TFLite-Micro INT8 layer issues.
 *
 * Because it folds in requantization it does MORE work per MAC than the plain-C
 * kernel or arm_mat_vec_mult_q7, so a speedup here is a conservative figure for
 * what INT8 deployment costs -- which is the right direction for a budget.
 *
 * Two build switches, both asserted:
 *   CONFIG_NN_OPTIMIZED  Kconfig choice, ANSI C vs Optimized. Same shape as
 *                        esp-dsp's CONFIG_DSP_OPTIMIZED and CMSIS's
 *                        ARM_MATH_LOOPUNROLL. ARCH_ESP32_S3 is then derived
 *                        inside esp_nn.h from CONFIG_IDF_TARGET_ESP32S3, so the
 *                        assembly path follows automatically -- but without the
 *                        outer switch every kernel silently becomes ANSI C.
 *   CONFIG_NN_SKIP_NUDGE "fast (non-bit-exact) requantization", ~20% faster on
 *                        S3, differs from the TFLite reference by +/-1 LSB.
 *                        Default OFF. Measured BOTH ways -- see CLAUDE.md; this
 *                        is an accuracy-for-throughput trade of exactly the kind
 *                        task 22's equivalence protocol exists to catch.
 * ======================================================================== */
#include "esp_nn.h"

#if !defined(CONFIG_NN_OPTIMIZED)
#  error "CONFIG_NN_OPTIMIZED unset: esp-nn would build its ANSI C reference kernels"
#endif
#if !defined(ARCH_ESP32_S3)
#  error "ARCH_ESP32_S3 unset: esp-nn would not use the S3 assembly path"
#endif

static int32_t s_bias[256];
static int8_t  s_ys8_nn[256];

/* Representative quantization parameters. Values chosen so the requantize path
 * does real work; the kernel's cost does not depend on them, but a degenerate
 * multiplier could let the compiler or the kernel short-circuit. */
#define NN_OUT_MULT   (1 << 30)
#define NN_OUT_SHIFT  (-8)

static uint32_t fc_s8(uint32_t rows, uint32_t cols)
{
    esp_nn_fully_connected_s8(
        s_xs8, 0, (uint16_t)cols,
        (const int8_t *)(const void *)s_pool, 0,
        s_bias, s_ys8_nn, (uint16_t)rows,
        0, NN_OUT_SHIFT, NN_OUT_MULT, -128, 127);
    return (uint32_t)(uint8_t)s_ys8_nn[0];
}
static uint32_t k_nn_fc_s8_128(void) { return fc_s8(128u, 128u); }
static uint32_t k_nn_fc_s8_256(void) { return fc_s8(256u, 256u); }
static uint32_t v_nn_fc_s8_128(void) { return chk_s8(s_ys8_nn, 128u); }
static uint32_t v_nn_fc_s8_256(void) { return chk_s8(s_ys8_nn, 256u); }

/* --- matrix-vector composed from the vendor DOT PRODUCT --------------------
 * dspm_mult_f32 measured 5.15 cyc/MAC on matvec_f32_128 while dsps_dotprod_f32
 * measured 1.69 cyc/MAC on identical arithmetic. dspm_mult is a matrix-MATRIX
 * kernel and k=1 is its degenerate case, so the SIMD it does have is not being
 * exploited for the one shape that matters most here -- a matrix-vector product
 * is what every RNN layer in this project actually issues.
 *
 * This row tests whether composing the vendor dot product per row recovers the
 * dot rate. It is a legitimate deployment strategy, not a trick: it is exactly
 * what a practitioner would write after seeing the two numbers above. */
static uint32_t matvec_dspdot_f32(uint32_t rows, uint32_t cols)
{
    const float *w = (const float *)(const void *)s_pool;
    for (uint32_t r = 0; r < rows; ++r) {
        dsps_dotprod_f32(w + r * cols, s_x, &s_y[r], (int)cols);
    }
    return f32_bits(s_y[0]);
}
static uint32_t k_dsp_mvdot_f32_64(void)  { return matvec_dspdot_f32(64u, 64u); }
static uint32_t k_dsp_mvdot_f32_128(void) { return matvec_dspdot_f32(128u, 128u); }

/* --- int8 matrix-vector --------------------------------------------------
 * esp-dsp has NO int8 matvec. It does have dspi_dotprod_s8_aes3, an int8 SIMD
 * dot product in the image namespace, so the matvec is built from one call per
 * row. That is the closest available analogue of arm_mat_vec_mult_q7 and, like
 * it, returns a SATURATED int8 -- so its checksum is deliberately not
 * comparable with the plain-C int32-accumulator kernel, exactly as on the F411.
 * The deployed INT8 path on this platform is ESP-NN, the CMSIS-NN analogue;
 * this row is the floor, not the costing basis. */
static uint32_t matvec_dsp_s8(uint32_t rows, uint32_t cols)
{
    const int8_t *w = (const int8_t *)(const void *)s_pool;
    image2d_t mi = { .step_x = 1, .step_y = 1, .stride_x = (int)cols,
                     .stride_y = (int)cols, .size_x = (int)cols, .size_y = 1 };
    image2d_t fi = { .data = (void *)s_xs8, .step_x = 1, .step_y = 1,
                     .stride_x = (int)cols, .stride_y = (int)cols,
                     .size_x = (int)cols, .size_y = 1 };
    for (uint32_t r = 0; r < rows; ++r) {
        mi.data = (void *)(w + r * cols);
        dspi_dotprod_s8_aes3(&mi, &fi, &s_ys8[r], (int)cols, 1, 7);
    }
    return (uint32_t)(uint8_t)s_ys8[0];
}
static uint32_t k_dsp_matvec_s8_128(void) { return matvec_dsp_s8(128u, 128u); }
static uint32_t k_dsp_matvec_s8_256(void) { return matvec_dsp_s8(256u, 256u); }
static uint32_t v_dsp_matvec_s8_128(void) { return chk_s8(s_ys8, 128u); }
static uint32_t v_dsp_matvec_s8_256(void) { return chk_s8(s_ys8, 256u); }

/* --- basic math ---------------------------------------------------------- */
static uint32_t k_dsp_dot512(void)
{
    dsps_dotprod_f32(s_x, s_fft_src, &s_dot_dsp, 512);
    return f32_bits(s_dot_dsp);
}
static uint32_t v_dsp_dot512(void) { return f32_bits(s_dot_dsp); }

static uint32_t k_dsp_mult512(void)
{
    dsps_mul_f32(s_x, s_fft_src, s_y, 512, 1, 1, 1);
    return f32_bits(s_y[0]);
}

/* --- sqrt ----------------------------------------------------------------
 * The one esp-dsp row this platform most needs. The LX7 FPU has no sqrt
 * instruction, so libm's sqrtf is a CALL (confirmed by disassembly 2026-09-11)
 * and sqrtf_257 runs at 0.32x the F411 per cycle. dsps_sqrt_f32 is ANSI-only in
 * BOTH Kconfig branches -- there is no SIMD sqrt -- but it is an approximation
 * rather than a libm call, so whether it wins is an open question and exactly
 * the sort this project measures instead of assuming. Note its accuracy is not
 * libm's: the checksum is expected to DIFFER, and that difference is a quality
 * question for the classical arm, not a bug. */
static uint32_t k_dsp_sqrt257(void)
{
    dsps_sqrt_f32(s_mag, s_out, (int)BINS);
    return f32_bits(s_out[0]);
}

/* ===========================================================================
 * Suite -- plain C only, for now.
 *
 * Staged exactly as the F411 was: the 13 plain-C kernels are measured first and
 * on their own, and the vendor-library (esp-dsp) pairs are added as a second
 * step. That ordering is not just convenience -- on the F411 the plain-C run is
 * what revealed that INT8 buys only 1.29x without SIMD, which is the finding
 * that made the vendor pairing a prerequisite for the compression cost model
 * rather than a refinement of it. The same question has to be asked of esp-dsp
 * independently; assuming the CMSIS answer carries over would beg it.
 *
 * The op-count conventions are the ones the shared bench.h fixes, so the
 * declared ops for a given kernel are identical to the STM32 rows by
 * construction.
 * ======================================================================== */
/* Accessor so bench_plat.c can WITNESS where the shared weight pool actually
 * landed, rather than trusting the BENCH_MEM knob that asked for it. Same rule
 * as the H723 reading SCB->CCR back instead of believing its own #define.
 * Defined here, after the shared plain-C region, so lines 1-236 stay identical
 * to the STM32 tiers. */
uint32_t bench_pool_addr(void) { return (uint32_t)(uintptr_t)s_pool; }

#define MAC2(r, c)   (BENCH_OPS_PER_MAC * (uint32_t)(r) * (uint32_t)(c))
/* 5*N*log2(N) is the standard flop count for a radix-2 complex FFT. */
#define FFTOPS(n, l) (5u * (uint32_t)(n) * (uint32_t)(l))
/* A real transform of N real samples is conventionally 2.5*N*log2(N). */
#define RFFTOPS(n, l) ((5u * (uint32_t)(n) * (uint32_t)(l)) / 2u)

void bench_run_all(void)
{
    memset(&g_bench, 0, sizeof(g_bench));
    g_bench.cpu_hz       = g_plat.cpu_hz;   /* recorded by bench_plat_init() */
    g_bench.irq_disabled = 1u;

    if (!bench_dwt_init()) {
        g_bench.magic = BENCH_MAGIC;
        g_bench.done  = 1u;
        return;                     /* results would be meaningless */
    }
    g_bench.overhead_cyc = bench_dwt_overhead();
    bench_kernels_init();
    bench_espdsp_init();

    const uint32_t R = 64u;   /* repetitions per kernel, as on the STM32s */

    bench_measure("fft_c_256",     256u, FFTOPS(256u, 8u),  256u, R, prep_fft256, k_fft256, v_fft256);
    bench_measure("fft_c_512",     512u, FFTOPS(512u, 9u),  512u, R, prep_fft512, k_fft512, v_fft512);
    bench_measure("matvec_f32_64", 64u*64u,   MAC2(64, 64),   64u*64u,   R, NULL, k_matvec_f32_64,  v_matvec_f32_64);
    bench_measure("matvec_f32_128",128u*128u, MAC2(128, 128), 128u*128u, R, NULL, k_matvec_f32_128, v_matvec_f32_128);
    bench_measure("matvec_s8_128", 128u*128u, MAC2(128, 128), 128u*128u, R, NULL, k_matvec_s8_128,  v_matvec_s8_128);
    bench_measure("matvec_s8_256", 256u*256u, MAC2(256, 256), 256u*256u, R, NULL, k_matvec_s8_256,  v_matvec_s8_256);
    bench_measure("dot_f32_512",   512u, MAC2(512, 1),      512u, R, NULL, k_dot512,   v_dot512);
    bench_measure("window_f32_512",512u, 512u,              512u, R, NULL, k_window512, v_window512);
    bench_measure("mag_f32_257",   BINS, BENCH_OPS_NONE,    BINS, R, NULL, k_mag257,   v_out_bins);
    bench_measure("expf_257",      BINS, BENCH_OPS_NONE,    BINS, R, NULL, k_expf257,  v_out_bins);
    bench_measure("logf_257",      BINS, BENCH_OPS_NONE,    BINS, R, NULL, k_logf257,  v_out_bins);
    bench_measure("sqrtf_257",     BINS, BENCH_OPS_NONE,    BINS, R, NULL, k_sqrtf257, v_out_bins);
    bench_measure("tanhf_257",     BINS, BENCH_OPS_NONE,    BINS, R, NULL, k_tanh257,  v_out_bins);

    /* --- esp-dsp pairs --------------------------------------------------- */
    bench_measure("espdsp_fft_256",      256u, FFTOPS(256u, 8u),  256u, R, prep_fft256, k_dsp_fft256, v_fft256);
    bench_measure("espdsp_fft_512",      512u, FFTOPS(512u, 9u),  512u, R, prep_fft512, k_dsp_fft512, v_fft512);
    bench_measure("espdsp_rfft_512",     512u, RFFTOPS(512u, 9u), 512u, R, prep_fft512, k_dsp_rfft512, v_dsp_rfft512);
    bench_measure("espdsp_matvec_f32_64", 64u*64u,   MAC2(64, 64),   64u*64u,   R, NULL, k_dsp_matvec_f32_64,  v_matvec_f32_64);
    bench_measure("espdsp_matvec_f32_128",128u*128u, MAC2(128, 128), 128u*128u, R, NULL, k_dsp_matvec_f32_128, v_matvec_f32_128);
    bench_measure("espdsp_matvec_s8_128", 128u*128u, MAC2(128, 128), 128u*128u, R, NULL, k_dsp_matvec_s8_128,  v_dsp_matvec_s8_128);
    bench_measure("espdsp_matvec_s8_256", 256u*256u, MAC2(256, 256), 256u*256u, R, NULL, k_dsp_matvec_s8_256,  v_dsp_matvec_s8_256);
    bench_measure("espdsp_mvdot_f32_64", 64u*64u,   MAC2(64, 64),   64u*64u,   R, NULL, k_dsp_mvdot_f32_64,  v_matvec_f32_64);
    bench_measure("espdsp_mvdot_f32_128",128u*128u, MAC2(128, 128), 128u*128u, R, NULL, k_dsp_mvdot_f32_128, v_matvec_f32_128);
    bench_measure("espdsp_dot_f32_512",  512u, MAC2(512, 1),      512u, R, NULL, k_dsp_dot512,  v_dsp_dot512);
    bench_measure("espdsp_mult_f32_512", 512u, 512u,              512u, R, NULL, k_dsp_mult512, v_window512);
    bench_measure("espnn_fc_s8_128",  128u*128u, MAC2(128, 128), 128u*128u, R, NULL, k_nn_fc_s8_128, v_nn_fc_s8_128);
    bench_measure("espnn_fc_s8_256",  256u*256u, MAC2(256, 256), 256u*256u, R, NULL, k_nn_fc_s8_256, v_nn_fc_s8_256);
    bench_measure("espdsp_sqrt_257",     BINS, BENCH_OPS_NONE,    BINS, R, NULL, k_dsp_sqrt257, v_out_bins);

    g_bench.magic = BENCH_MAGIC;
    g_bench.done  = 1u;

    bench_report();
}
