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
/* RPi5: the STM32 tiers include "main.h" here for CMSIS core headers and the
 * S3 swaps in its own note. Nothing in the plain-C kernels below needs either;
 * the rest of this file is identical source text to
 * benchmark_s3/main/bench/bench_kernels.c lines 1-240 (and so to the H723's),
 * which is checked by diff rather than assumed -- see PROVENANCE.md. */
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
 * End of the shared plain-C region. Everything above this line is identical
 * source text on all four platforms and is verified by `diff`, not asserted.
 *
 * NEON / Arm Compute Library pairs belong below, alongside the CMSIS pairs on
 * the STM32s and the esp-dsp/esp-nn pairs on the S3. They are deliberately NOT
 * in this first run: the plain-C rows are what make the four tiers comparable,
 * and this tier has never produced a number at all. Pair it once the baseline
 * exists -- and when pairing the int8 row, the question that matters is whether
 * the kernel reaches SDOT/UDOT (`asimddp`), because on the S3 three int8
 * implementations of the same arithmetic on the same silicon spanned 10.75x.
 * ======================================================================== */

/* Witnesses where the pool actually landed, as on the S3. There is no PSRAM
 * here, but there IS a page-cache/NUMA-free 64-bit address space and a 4-level
 * cache hierarchy, so recording the address costs nothing and makes an
 * alignment or huge-page surprise visible rather than mysterious. */
uintptr_t bench_pool_addr(void) { return (uintptr_t)(void *)s_pool; }

#define MAC2(r, c)   (BENCH_OPS_PER_MAC * (uint32_t)(r) * (uint32_t)(c))
/* 5*N*log2(N) is the standard flop count for a radix-2 complex FFT. */
#define FFTOPS(n, l) (5u * (uint32_t)(n) * (uint32_t)(l))
/* A real transform of N real samples is conventionally 2.5*N*log2(N). */
#define RFFTOPS(n, l) ((5u * (uint32_t)(n) * (uint32_t)(l)) / 2u)

void bench_run_all(void)
{
    memset(&g_bench, 0, sizeof(g_bench));
    g_bench.cpu_hz = g_plat.cpu_hz;     /* recorded by bench_plat_init() */

    /* HONEST PLATFORM DIFFERENCE, recorded rather than papered over: this is a
     * preemptive multitasking OS and interrupts CANNOT be masked from
     * userspace. The STM32 and S3 tiers set PRIMASK / entered a critical
     * section and recorded irq_disabled=1. Here the nearest equivalents are
     * core pinning, SCHED_FIFO and mlockall -- which reduce preemption but do
     * not eliminate it. Reporting irq_disabled=1 would be a lie, so the field
     * stays 0 and g_plat records what was actually done instead. This is also
     * why the minimum of many repetitions matters more on this tier than on
     * the others: the minimum is the least-preempted sample. */
    g_bench.irq_disabled = 0u;

    if (!bench_dwt_init()) {
        g_bench.magic = BENCH_MAGIC;
        g_bench.done  = 1u;
        return;                     /* results would be meaningless */
    }
    g_bench.overhead_cyc = bench_dwt_overhead();
    bench_kernels_init();

    const uint32_t R = 64u;   /* repetitions per kernel, as on the other tiers */

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

    g_bench.magic = BENCH_MAGIC;
    g_bench.done  = 1u;
}
