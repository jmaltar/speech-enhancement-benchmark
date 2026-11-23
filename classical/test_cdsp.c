/**
 * test_cdsp.c -- unit tests for the classical arm. Each test is paired with a
 * specific way the arithmetic could be wrong and still produce plausible
 * audio, because an enhancer that is subtly wrong does not crash: it just
 * scores a little lower and the loss gets blamed on the algorithm.
 */
#include "cdsp.h"
#include "cdsp_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks = 0, g_fails = 0;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_fails++; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t s_rng = 20261012u;
static float rndf(void) { s_rng = s_rng * 1664525u + 1013904223u; return ((float)(s_rng >> 8) / 16777216.0f) * 2.0f - 1.0f; }
static float gauss(void) { float s = 0; for (int i = 0; i < 12; i++) s += rndf() * 0.5f + 0.5f; return s - 6.0f; }

/* ---- 1. FFT against a naive DFT, and round trip ------------------------- */
static void test_fft(void)
{
    static cdsp_fft_t f; static float x[512], X[514], y[512];
    uint32_t sizes[] = { 256u, 512u };
    for (int si = 0; si < 2; si++) {
        uint32_t n = sizes[si];
        CHECK(cdsp_fft_init(&f, n) == 0, "init %u", n);
        for (uint32_t i = 0; i < n; i++) x[i] = rndf();
        cdsp_fft_forward(&f, x, X);
        double worst = 0;
        for (uint32_t k = 0; k <= n / 2u; k++) {
            double re = 0, im = 0;
            for (uint32_t i = 0; i < n; i++) {
                double a = -2.0 * M_PI * (double)k * (double)i / (double)n;
                re += x[i] * cos(a); im += x[i] * sin(a);
            }
            double e = fabs(re - X[2*k]) + fabs(im - X[2*k+1]);
            if (e > worst) worst = e;
        }
        CHECK(worst < 2e-4 * sqrt((double)n), "fft%u vs DFT worst %.3g", n, worst);
        cdsp_fft_inverse(&f, X, y);
        worst = 0;
        for (uint32_t i = 0; i < n; i++) { double e = fabs(y[i] - x[i]); if (e > worst) worst = e; }
        CHECK(worst < 1e-5, "ifft(fft) round trip %u worst %.3g", n, worst);
    }
    CHECK(cdsp_fft_init(&f, 384u) != 0, "non-power-of-two refused");
    CHECK(cdsp_fft_init(&f, 2048u) != 0, "over max refused");
}

/* ---- 2. Bessel functions against scipy.special.i0e / i1e ---------------- */
static void test_bessel(void)
{
    const double xs[] = {0.0, 0.1, 0.5, 1.0, 2.0, 3.75, 5.0, 10.0, 50.0, 500.0};
    const double i0[] = {1.0, 9.071009258e-01, 6.450352704e-01, 4.657596076e-01, 3.085083226e-01,
                         2.144570512e-01, 1.835408126e-01, 1.278333372e-01, 5.656162665e-02, 1.784570650e-02};
    const double i1[] = {0.0, 4.529844681e-02, 1.564208032e-01, 2.079104153e-01, 2.152692892e-01,
                         1.829684209e-01, 1.639722669e-01, 1.212626814e-01, 5.599312389e-02, 1.782785185e-02};
    for (int i = 0; i < 10; i++) {
        double a = cdsp_i0e((float)xs[i]), b = cdsp_i1e((float)xs[i]);
        CHECK(fabs(a - i0[i]) < 1e-5, "i0e(%g) = %.7g, want %.7g", xs[i], a, i0[i]);
        CHECK(fabs(b - i1[i]) < 1e-5, "i1e(%g) = %.7g, want %.7g", xs[i], b, i1[i]);
    }
    /* continuity at the 3.75 branch point */
    CHECK(fabs(cdsp_i0e(3.7499f) - cdsp_i0e(3.7501f)) < 1e-5, "i0e branch continuity");
    CHECK(fabs(cdsp_i1e(3.7499f) - cdsp_i1e(3.7501f)) < 1e-5, "i1e branch continuity");
}

/* ---- 3. Gain rules against scipy-computed values and their limits ------- */
static void test_gains(void)
{
    const double xi[] = {0.001, 0.01, 0.1, 1.0, 1.0, 10.0, 100.0, 1000.0, 0.3162};
    const double ga[] = {0.5,   1.0,  1.0, 1.0, 2.0, 10.0, 100.0, 1000.0, 3.0};
    const double G[]  = {3.962336405e-02, 8.861888447e-02, 2.792173076e-01, 7.742862303e-01,
                         6.409597883e-01, 9.344702651e-01, 9.926021904e-01, 9.992510303e-01,
                         3.338953927e-01};
    for (int i = 0; i < 9; i++) {
        double g = cdsp_gain_mmse((float)xi[i], (float)ga[i]);
        CHECK(fabs(g - G[i]) / G[i] < 2e-4, "mmse(xi=%g, gamma=%g) = %.7g want %.7g", xi[i], ga[i], g, G[i]);
    }
    /* MMSE tends to the Wiener gain at high SNR; is above it at low SNR (the
     * estimator's well-known conservatism against musical noise) */
    CHECK(fabs(cdsp_gain_mmse(1000.0f, 1000.0f) - cdsp_gain_wiener(1000.0f)) < 1e-3, "mmse -> wiener at high SNR");
    CHECK(cdsp_gain_mmse(0.01f, 1.0f) > cdsp_gain_wiener(0.01f), "mmse > wiener at low SNR");
    /* No NaN or negative value anywhere in a wide sweep, and the ESTIMATED
     * AMPLITUDE A = G sqrt(gamma) (in units of sqrt(N)) never exceeds the
     * observed amplitude plus the prior's: the gain itself is unbounded by
     * design -- E[A | Y] stays near sqrt(xi N) as |Y| -> 0, so G = A/|Y|
     * grows -- and a cap on G would be a cap on the estimator. */
    int bad = 0;
    for (float lx = -40; lx <= 40; lx += 1.0f) for (float lg = -40; lg <= 40; lg += 1.0f) {
        float xi = powf(10.0f, lx / 10), gamma = powf(10.0f, lg / 10);
        float v = cdsp_gain_mmse(xi, gamma);
        float amp = v * sqrtf(gamma);
        if (!(v == v) || v < 0.0f || amp > sqrtf(gamma) + sqrtf(xi) + 1e-3f) bad++;
    }
    CHECK(bad == 0, "mmse sweep produced %d bad values", bad);
    CHECK(cdsp_gain_mmse(100.0f, 0.01f) > 1.0f, "gain exceeds 1 when the prior outweighs a weak observation");
    CHECK(cdsp_gain_mmse(1.0f, 0.0f) == cdsp_gain_wiener(1.0f), "gamma = 0 handled");
    /* spectral subtraction: floor, unity at high SNR, zero crossing at gamma = over */
    CHECK(fabs(cdsp_gain_specsub(1e6f, 1.0f, 0.002f) - 1.0f) < 1e-5, "specsub -> 1");
    CHECK(fabs(cdsp_gain_specsub(0.5f, 1.0f, 0.002f) - sqrtf(0.002f)) < 1e-6, "specsub floor");
    CHECK(fabs(cdsp_gain_specsub(2.0f, 1.0f, 0.002f) - sqrtf(0.5f)) < 1e-6, "specsub gamma=2 -> sqrt(1/2)");
    CHECK(fabs(cdsp_gain_specsub(2.0f, 2.0f, 0.002f) - sqrtf(0.002f)) < 1e-6, "over-subtraction factor applied");
    CHECK(fabs(cdsp_gain_wiener(1.0f) - 0.5f) < 1e-7, "wiener(1) = 1/2");
}

/* ---- framing helpers for the processor tests ----------------------------- */
#define MAXN 65536
static int16_t s_in[MAXN], s_out[MAXN + 4096];

/* Drive `n` samples through `proc` exactly as the binding does: whole hops
 * (last zero-padded), then flush hops; return output with the lag dropped and
 * trimmed to n. */
static int drive(const abf_proc_t *proc, uint32_t rate, uint32_t F, uint32_t H, const int16_t *in, uint32_t n, int16_t *out)
{
    if (proc->setup(proc->ctx, rate, F, H, 1u) != 0) return -1;
    uint32_t dh = (n + H - 1u) / H, fh = ((F - H) + H - 1u) / H, lag = F - H;
    static int16_t hop[4096], res[4096]; static int16_t all[MAXN + 8192]; uint32_t got = 0;
    for (uint32_t j = 0; j < dh + fh; j++) {
        memset(hop, 0, H * sizeof *hop);
        if (j < dh) { uint32_t off = j * H, left = n - off; memcpy(hop, in + off, (left < H ? left : H) * sizeof *hop); }
        if (proc->process(proc->ctx, hop, res, H) != 0) return -1;
        memcpy(all + got, res, H * sizeof *res); got += H;
    }
    memcpy(out, all + lag, n * sizeof *out);
    return 0;
}

/* ---- 4. Unity gain reconstructs the input: the framing is COLA ----------- */
static void test_unity(void)
{
    cdsp_params_t p; cdsp_params_default(&p); p.unity_gain = 1;
    uint32_t n = 40241u;                                  /* not a whole number of hops */
    for (uint32_t i = 0; i < n; i++) s_in[i] = (int16_t)lrintf(20000.0f * rndf());
    cdsp_algo_t algos[] = { CDSP_ALGO_SPECSUB, CDSP_ALGO_WIENER, CDSP_ALGO_MMSE };
    for (int a = 0; a < 3; a++) {
        const abf_proc_t *proc = cdsp_proc(algos[a], &p);
        CHECK(drive(proc, 16000u, 512u, 256u, s_in, n, s_out) == 0, "drive %s", proc->name);
        int worst = 0; uint32_t nz = 0;
        for (uint32_t i = 0; i < n; i++) { int d = abs(s_out[i] - s_in[i]); if (d > worst) worst = d; if (d) nz++; }
        CHECK(worst <= 1, "%s unity gain: worst %d LSB, %u nonzero", proc->name, worst, nz);
    }
    /* 8 kHz geometry (NOIZEUS) as well */
    const abf_proc_t *proc = cdsp_proc(CDSP_ALGO_MMSE, &p);
    CHECK(drive(proc, 8000u, 256u, 128u, s_in, 22529u, s_out) == 0, "drive 8 kHz");
    int worst = 0;
    for (uint32_t i = 0; i < 22529u; i++) { int d = abs(s_out[i] - s_in[i]); if (d > worst) worst = d; }
    CHECK(worst <= 1, "8 kHz unity gain worst %d LSB", worst);
    /* setup refuses what it must */
    CHECK(proc->setup(proc->ctx, 16000u, 512u, 256u, 2u) != 0, "stereo refused");
    CHECK(proc->setup(proc->ctx, 16000u, 480u, 240u, 1u) != 0, "non-power-of-two frame refused");
    CHECK(proc->setup(proc->ctx, 16000u, 512u, 200u, 1u) != 0, "hop not dividing frame refused");
}

/* ---- 5. Both noise estimators converge on stationary white noise ---------
 * E|Y_k|^2 for white noise of variance s^2 through a periodic Hann of length F
 * is s^2 * sum(w^2) = s^2 * 3F/8. MCRA is minimum-controlled and is expected
 * to sit a little low; the tolerance is wide enough for that and far too
 * tight for a wrong window sum or a wrong smoothing constant. */
static void test_noise_estimator(void)
{
    int ests[] = { CDSP_NOISE_VAD, CDSP_NOISE_MCRA };
    for (int e = 0; e < 2; e++) {
        cdsp_params_t p; cdsp_params_default(&p); p.noise_est = ests[e];
        const abf_proc_t *proc = cdsp_proc(CDSP_ALGO_WIENER, &p);
        uint32_t n = 48000u; float sigma = 1000.0f;
        s_rng = 7u;
        for (uint32_t i = 0; i < n; i++) s_in[i] = (int16_t)lrintf(sigma * gauss());
        CHECK(drive(proc, 16000u, 512u, 256u, s_in, n, s_out) == 0, "drive noise");
        uint32_t B; const float *N = cdsp_noise_psd(&B);
        double mean = 0; for (uint32_t k = 2; k < B - 2; k++) mean += N[k]; mean /= (B - 4);
        double expect = (double)sigma * sigma * 3.0 * 512.0 / 8.0;
        printf("  %-4s noise PSD / expected = %.3f\n", cdsp_noise_est_name(ests[e]), mean / expect);
        CHECK(fabs(mean / expect - 1.0) < 0.30, "%s noise PSD mean %.4g vs expected %.4g (ratio %.3f)", cdsp_noise_est_name(ests[e]), mean, expect, mean / expect);
        double pi = 0, po = 0; for (uint32_t i = 8000; i < n; i++) { pi += (double)s_in[i] * s_in[i]; po += (double)s_out[i] * s_out[i]; }
        CHECK(10 * log10(po / pi) < -10.0, "%s noise-only attenuation %.1f dB", cdsp_noise_est_name(ests[e]), 10 * log10(po / pi));
    }
}

/* ---- 5b. Speech from the first sample: the VAD tracker must suffer and
 * MCRA must not. Tone bursts that are ON from t = 0 (no leading silence), so
 * the VAD tracker's initial frames are all signal. This is the DNS situation
 * measured on 2026-10-12. */
static void test_no_leading_silence(void)
{
    uint32_t n = 48000u, rate = 16000u;
    static float clean[MAXN];
    s_rng = 99u;
    for (uint32_t i = 0; i < n; i++) {
        float t = (float)i / (float)rate;
        int on = ((int)(t * 5.0f) % 2) == 0;                       /* ON first */
        clean[i] = on ? 4000.0f * (sinf(2 * (float)M_PI * 300 * t) + 0.6f * sinf(2 * (float)M_PI * 1100 * t) + 0.3f * sinf(2 * (float)M_PI * 2500 * t)) : 0.0f;
        s_in[i] = (int16_t)lrintf(clean[i] + 2500.0f * gauss());
    }
    double gain[2];
    int ests[] = { CDSP_NOISE_VAD, CDSP_NOISE_MCRA };
    for (int e = 0; e < 2; e++) {
        cdsp_params_t p; cdsp_params_default(&p); p.noise_est = ests[e];
        const abf_proc_t *proc = cdsp_proc(CDSP_ALGO_WIENER, &p);
        CHECK(drive(proc, rate, 512u, 256u, s_in, n, s_out) == 0, "drive");
        double num = 0, din = 0, dout = 0;
        for (uint32_t i = 0; i < n; i++) { num += (double)clean[i] * clean[i]; double a = s_in[i] - clean[i], b = s_out[i] - clean[i]; din += a * a; dout += b * b; }
        gain[e] = 10 * log10(din / dout);
        printf("  wiener/%-4s no leading silence: SNR gain %+.2f dB\n", cdsp_noise_est_name(ests[e]), gain[e]);
    }
    CHECK(gain[1] > gain[0] + 3.0, "MCRA (%.2f dB) should beat the VAD tracker (%.2f dB) by > 3 dB with no leading silence", gain[1], gain[0]);
    CHECK(gain[1] > 3.0, "MCRA still enhances (%.2f dB) with no leading silence", gain[1]);
}

/* ---- 6. Each algorithm improves the SNR of tone bursts in white noise ---- */
static void test_enhancement(void)
{
    uint32_t n = 48000u, rate = 16000u;
    static float clean[MAXN];
    for (uint32_t i = 0; i < n; i++) {
        float t = (float)i / (float)rate;
        int on = ((int)(t * 5.0f) % 2) == 1;                       /* 100 ms on / 100 ms off, starting off */
        clean[i] = on ? 4000.0f * (sinf(2 * (float)M_PI * 300 * t) + 0.6f * sinf(2 * (float)M_PI * 1100 * t) + 0.3f * sinf(2 * (float)M_PI * 2500 * t)) : 0.0f;
        s_in[i] = (int16_t)lrintf(clean[i] + 2500.0f * gauss());    /* ~0 dB overall */
    }
    double num = 0, den = 0;
    for (uint32_t i = 0; i < n; i++) { num += (double)clean[i] * clean[i]; double e = s_in[i] - clean[i]; den += e * e; }
    double snr_in = 10 * log10(num / den);
    cdsp_algo_t algos[] = { CDSP_ALGO_SPECSUB, CDSP_ALGO_WIENER, CDSP_ALGO_MMSE };
    int ests[] = { CDSP_NOISE_VAD, CDSP_NOISE_MCRA };
    for (int e = 0; e < 2; e++) for (int a = 0; a < 3; a++) {
        cdsp_params_t p; cdsp_params_default(&p); p.noise_est = ests[e];
        const abf_proc_t *proc = cdsp_proc(algos[a], &p);
        CHECK(drive(proc, rate, 512u, 256u, s_in, n, s_out) == 0, "drive %s", proc->name);
        /* score after the first 400 ms so the initial noise estimate does not count */
        double d = 0; num = 0;
        for (uint32_t i = 6400; i < n; i++) { num += (double)clean[i] * clean[i]; double e2 = s_out[i] - clean[i]; d += e2 * e2; }
        double snr_out = 10 * log10(num / d);
        printf("  %-8s/%-4s SNR in %.2f dB -> out %.2f dB  (+%.2f)\n", proc->name, cdsp_noise_est_name(ests[e]), snr_in, snr_out, snr_out - snr_in);
        CHECK(snr_out - snr_in > 3.0, "%s/%s improves SNR by only %.2f dB", proc->name, cdsp_noise_est_name(ests[e]), snr_out - snr_in);
    }
}

int main(void)
{
    test_fft();
    test_bessel();
    test_gains();
    test_unity();
    test_noise_estimator();
    test_no_leading_silence();
    test_enhancement();
    printf("%d checks, %d failed\n", g_checks, g_fails);
    return g_fails ? 1 : 0;
}
