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
#include "main.h"
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
 * CMSIS-DSP paired kernels
 *
 * Each row below has the SAME shape, the SAME operand buffers and the SAME
 * declared op count as its plain-C partner above, so the pair differs only by
 * implementation. That difference is the number the project needs: the
 * structured-vs-unstructured pruning arm assumes whole-group removal is what
 * lets CMSIS kernels run at full speed, and the plain-C run showed INT8 buying
 * only 1.29x without SIMD -- so the vendor-library delta is load-bearing for
 * the compression cost model, not a refinement of it.
 *
 * Build switches, and why they are asserted rather than assumed
 * -------------------------------------------------------------
 * Both of these select which code CMSIS-DSP actually compiles, and both fail
 * SILENTLY -- a missing define yields a working, slower library and no
 * diagnostic, i.e. a measurement of the portable fallback reported as "what
 * CMSIS-DSP is worth". Same failure class as the dead-code elision bug the
 * harness design rules already record.
 *
 *   ARM_MATH_DSP         auto-derived in arm_math_types.h from __ARM_FEATURE_DSP,
 *                        which GCC sets for -mcpu=cortex-m4. Selects the
 *                        SMLAD / SXTB16 paths -- the entire point of the q7 row.
 *   ARM_MATH_LOOPUNROLL  NOT auto-defined anywhere in CMSIS-DSP 1.10. It is
 *                        documented in arm_math.h and set by the library's own
 *                        CMake, so a hand-rolled build silently gets the
 *                        non-unrolled f32 kernels. Set in CMakeLists.txt.
 * ========================================================================= */
#include "arm_math.h"
/* arm_math.h does not pull this in: the arm_cfft_sR_f32_len* instances
 * live here, and without it arm_cfft_f32 has no table to be called with. */
#include "arm_const_structs.h"

#if !defined(ARM_MATH_DSP)
#  error "ARM_MATH_DSP unset: CMSIS-DSP would build its portable scalar fallback"
#endif
#if !defined(ARM_MATH_LOOPUNROLL)
#  error "ARM_MATH_LOOPUNROLL unset: CMSIS-DSP f32 kernels would not unroll"
#endif

static arm_matrix_instance_f32    s_m_f32_64, s_m_f32_128;
static arm_matrix_instance_q7     s_m_q7_128, s_m_q7_256;
static arm_rfft_fast_instance_f32 s_rfft512;

static q7_t  s_ys8[256];
static float s_negmag[BINS];   /* see the expf note in the registration block */
static float s_dot_cmsis;

static uint32_t chk_s8(const q7_t *p, uint32_t n)
{
    uint32_t acc = 0u;
    for (uint32_t i = 0; i < n; ++i) acc += (uint32_t)(int32_t)p[i];
    return acc;
}

static void bench_cmsis_init(void)
{
    /* Fields are set directly rather than through arm_mat_init_f32, which lives
     * in its own translation unit and would be the only reason to vendor it --
     * the instance is a plain {numRows, numCols, pData} struct and there is no
     * arm_mat_init_q7 at all, so doing both the same way keeps them symmetric. */
    s_m_f32_64.numRows  = 64u;  s_m_f32_64.numCols  = 64u;
    s_m_f32_64.pData    = (float32_t *)(void *)s_pool;
    s_m_f32_128.numRows = 128u; s_m_f32_128.numCols = 128u;
    s_m_f32_128.pData   = (float32_t *)(void *)s_pool;

    s_m_q7_128.numRows = 128u; s_m_q7_128.numCols = 128u;
    s_m_q7_128.pData   = (q7_t *)(void *)s_pool;
    s_m_q7_256.numRows = 256u; s_m_q7_256.numCols = 256u;
    s_m_q7_256.pData   = (q7_t *)(void *)s_pool;

    arm_rfft_fast_init_f32(&s_rfft512, 512u);

    for (uint32_t i = 0; i < BINS; ++i) s_negmag[i] = -s_mag[i];
}

/* --- transforms ---------------------------------------------------------
 * The plain-C fft_radix2 is a COMPLEX transform over N complex points, so the
 * like-for-like partner is arm_cfft_f32, NOT arm_rfft_fast_f32 -- a real FFT
 * of N real samples is roughly half the work and pairing them would have
 * flattered CMSIS by ~2x. (The note previously at the bottom of this file
 * named arm_rfft_fast_f32 as the partner for fft_c_512; that was wrong and is
 * corrected here.) The real transform is still measured, as its own unpaired
 * row, because the STFT of the locked configuration is what the classical and
 * DTLN paths actually run.
 *
 * arm_cfft_f32 is in-place on s_fft and reuses the plain rows' prep and verify
 * callbacks, so both implementations see byte-identical input. Their output
 * checksums are NOT expected to match bit-for-bit -- different summation order
 * over the same transform -- so the checksum proves the buffer was written,
 * not that the two agree numerically. That agreement is task 22's job.
 * ---------------------------------------------------------------------- */
static uint32_t k_cmsis_cfft256(void)
{
    arm_cfft_f32(&arm_cfft_sR_f32_len256, s_fft, 0u, 1u);
    return f32_bits(s_fft[0]);
}
static uint32_t k_cmsis_cfft512(void)
{
    arm_cfft_f32(&arm_cfft_sR_f32_len512, s_fft, 0u, 1u);
    return f32_bits(s_fft[0]);
}

/* arm_rfft_fast_f32 destroys its input buffer, hence the prep. 512 real in,
 * 512 floats out (packed: X[0] and X[N/2] real, then re/im pairs). */
static void prep_rfft512(void) { memcpy(s_fft, s_fft_src, 512u * sizeof(float)); }
static uint32_t k_cmsis_rfft512(void)
{
    arm_rfft_fast_f32(&s_rfft512, s_fft, s_y, 0u);
    return f32_bits(s_y[0]);
}
static uint32_t v_cmsis_rfft512(void) { return chk_f32(s_y, 512u); }

/* --- matrix-vector ------------------------------------------------------ */
static uint32_t k_cmsis_matvec_f32_64(void)
{
    arm_mat_vec_mult_f32(&s_m_f32_64, s_x, s_y);
    return f32_bits(s_y[0]);
}
static uint32_t k_cmsis_matvec_f32_128(void)
{
    arm_mat_vec_mult_f32(&s_m_f32_128, s_x, s_y);
    return f32_bits(s_y[0]);
}

/* The q7 row is the load-bearing one. arm_mat_vec_mult_q7 unpacks four int8
 * per word with SXTB16 and issues two SMLADs per four MACs, over four rows at
 * a time -- the SIMD path the plain-C 1.29x result says the whole quantization
 * speed argument depends on.
 *
 * One semantic difference to record rather than hide: CMSIS returns a q7
 * vector (accumulator shifted and saturated back to int8), while the plain-C
 * matvec_s8 writes int32 accumulators. That is a per-ROW cost difference, not
 * a per-MAC one, so the MACs/s comparison stands; but the two kernels do not
 * compute the same output type and their checksums are not comparable. A real
 * INT8 deployment would use CMSIS-NN (arm_nn_vec_mat_mult_t_s8), which also
 * folds in the zero-point and requantization -- worth adding as a third row
 * before the compression cost model is finalised.
 * ---------------------------------------------------------------------- */
static uint32_t k_cmsis_matvec_q7_128(void)
{
    arm_mat_vec_mult_q7(&s_m_q7_128, (const q7_t *)s_xs8, s_ys8);
    return (uint32_t)(int32_t)s_ys8[0];
}
static uint32_t k_cmsis_matvec_q7_256(void)
{
    arm_mat_vec_mult_q7(&s_m_q7_256, (const q7_t *)s_xs8, s_ys8);
    return (uint32_t)(int32_t)s_ys8[0];
}
static uint32_t v_cmsis_matvec_q7_128(void) { return chk_s8(s_ys8, 128u); }
static uint32_t v_cmsis_matvec_q7_256(void) { return chk_s8(s_ys8, 256u); }

/* --- basic / complex math ----------------------------------------------- */
static uint32_t k_cmsis_dot512(void)
{
    arm_dot_prod_f32(s_x, s_fft_src, 512u, &s_dot_cmsis);
    return f32_bits(s_dot_cmsis);
}
static uint32_t v_cmsis_dot512(void) { return f32_bits(s_dot_cmsis); }

static uint32_t k_cmsis_mult512(void)
{
    arm_mult_f32(s_x, s_fft_src, s_y, 512u);
    return f32_bits(s_y[0]);
}
static uint32_t k_cmsis_mag257(void)
{
    arm_cmplx_mag_f32(s_fft_src, s_out, BINS);
    return f32_bits(s_out[0]);
}

/* --- vectorised transcendentals -----------------------------------------
 * These are the highest-stakes pair on the F411: the plain-C run put tanhf at
 * 202 cyc/element and expf at 152, each about twice the per-element cost of a
 * 512-point FFT, and they sit on the critical path of the MMSE-STSA gain rule.
 *
 * arm_vexp_f32 reads a pre-negated buffer while the plain expf_257 negates
 * inside its loop, so the pair differs by one VNEG per element (~1 cycle
 * against ~152). Stated rather than corrected, because correcting it would
 * mean editing a plain-C kernel whose number is already published in CLAUDE.md.
 *
 * There is no arm_vtanh_f32 and no arm_vsqrt_f32 in 1.10.0, so tanhf_257 and
 * sqrtf_257 keep no CMSIS partner. sqrtf already compiles to the FPU's
 * VSQRT.F32, which is also all arm_sqrt_f32 does.
 * ---------------------------------------------------------------------- */
static uint32_t k_cmsis_vexp257(void)
{
    arm_vexp_f32(s_negmag, s_out, BINS);
    return f32_bits(s_out[0]);
}
static uint32_t k_cmsis_vlog257(void)
{
    arm_vlog_f32(s_mag, s_out, BINS);
    return f32_bits(s_out[0]);
}

/* --- suite -------------------------------------------------------------- */
#define MAC2(r, c)   (BENCH_OPS_PER_MAC * (uint32_t)(r) * (uint32_t)(c))
/* 5*N*log2(N) is the standard flop count for a radix-2 complex FFT. */
#define FFTOPS(n, l) (5u * (uint32_t)(n) * (uint32_t)(l))
/* A real transform of N real samples is conventionally 2.5*N*log2(N) --
 * half the complex count. Declared separately so the rfft row's MOps/s is
 * not silently computed on the complex convention. */
#define RFFTOPS(n, l) ((5u * (uint32_t)(n) * (uint32_t)(l)) / 2u)

void bench_run_all(void)
{
    memset(&g_bench, 0, sizeof(g_bench));
    g_bench.cpu_hz       = SystemCoreClock;
    g_bench.irq_disabled = 1u;

    if (!bench_dwt_init()) {
        g_bench.magic = BENCH_MAGIC;
        g_bench.done  = 1u;
        return;                     /* results would be meaningless */
    }
    g_bench.overhead_cyc = bench_dwt_overhead();
    bench_kernels_init();
    bench_cmsis_init();

    const uint32_t R = 64u;   /* repetitions per kernel */

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

    /* --- CMSIS-DSP pairs ------------------------------------------------
     * Registered with byte-identical n / ops / elems / reps to their plain-C
     * partners above, and reusing their prep and verify callbacks wherever the
     * output buffer is the same, so the only difference between the two rows
     * of a pair is which implementation ran. cmsis_rfft_512 is the exception:
     * it has no partner and is measured because the STFT of the locked
     * configuration is a real transform.
     * ------------------------------------------------------------------ */
    bench_measure("cmsis_cfft_256",     256u, FFTOPS(256u, 8u),  256u, R, prep_fft256, k_cmsis_cfft256, v_fft256);
    bench_measure("cmsis_cfft_512",     512u, FFTOPS(512u, 9u),  512u, R, prep_fft512, k_cmsis_cfft512, v_fft512);
    bench_measure("cmsis_rfft_512",     512u, RFFTOPS(512u, 9u), 512u, R, prep_rfft512, k_cmsis_rfft512, v_cmsis_rfft512);
    bench_measure("cmsis_matvec_f32_64", 64u*64u,   MAC2(64, 64),   64u*64u,   R, NULL, k_cmsis_matvec_f32_64,  v_matvec_f32_64);
    bench_measure("cmsis_matvec_f32_128",128u*128u, MAC2(128, 128), 128u*128u, R, NULL, k_cmsis_matvec_f32_128, v_matvec_f32_128);
    bench_measure("cmsis_matvec_q7_128", 128u*128u, MAC2(128, 128), 128u*128u, R, NULL, k_cmsis_matvec_q7_128,  v_cmsis_matvec_q7_128);
    bench_measure("cmsis_matvec_q7_256", 256u*256u, MAC2(256, 256), 256u*256u, R, NULL, k_cmsis_matvec_q7_256,  v_cmsis_matvec_q7_256);
    bench_measure("cmsis_dot_f32_512",  512u, MAC2(512, 1),      512u, R, NULL, k_cmsis_dot512,  v_cmsis_dot512);
    bench_measure("cmsis_mult_f32_512", 512u, 512u,              512u, R, NULL, k_cmsis_mult512, v_window512);
    bench_measure("cmsis_mag_f32_257",  BINS, BENCH_OPS_NONE,    BINS, R, NULL, k_cmsis_mag257,  v_out_bins);
    bench_measure("cmsis_vexp_257",     BINS, BENCH_OPS_NONE,    BINS, R, NULL, k_cmsis_vexp257, v_out_bins);
    bench_measure("cmsis_vlog_257",     BINS, BENCH_OPS_NONE,    BINS, R, NULL, k_cmsis_vlog257, v_out_bins);

#ifdef BENCH_PRUNE
    /* Arm B (c), the cost half: bench_prune.c. Runs last because its sparse
     * kernel rebuilds the pool. -DBENCH_PRUNE=ON; off in the recorded probe. */
    { extern void bench_prune_run(uint32_t, uint8_t *, uint32_t, const int8_t *);
      bench_prune_run(R, s_pool, POOL_BYTES, s_xs8); }
#endif

    g_bench.magic = BENCH_MAGIC;
    g_bench.done  = 1u;

    bench_itm_report();
}

/* ---------------------------------------------------------------------------
 * Kernels still without a CMSIS-DSP partner
 *
 *   tanhf_257   no arm_vtanh_f32 in 1.10.0. It is the most expensive kernel on
 *               the board (202 cyc/element) and it is a GRU/LSTM activation,
 *               so a hand-written SIMD or table-based tanh is a real candidate
 *               for the H723 and ESP32-S3 ports.
 *   sqrtf_257   no arm_vsqrt_f32. libm's sqrtf already compiles to VSQRT.F32.
 *   matvec_s8   paired against arm_mat_vec_mult_q7 above, but a deployed INT8
 *               model would call CMSIS-NN's arm_nn_vec_mat_mult_t_s8, which
 *               also applies zero-point and requantization. CMSIS-NN ships in
 *               the same Cube pack (Drivers/CMSIS/NN); add it as a third row
 *               before the compression cost model is finalised.
 *
 * A caveat that outlives this file: RNNoise runs at 48 kHz with a 20 ms frame,
 * i.e. 960 samples and a 480-point transform. 480 is not a power of two, so
 * neither arm_cfft_f32 nor arm_rfft_fast_f32 can do it -- Xiph's own
 * mixed-radix FFT comes along with the RNNoise port. Every transform figure in
 * this file therefore costs the classical and DTLN paths, not the RNNoise one.
 * ------------------------------------------------------------------------- */
