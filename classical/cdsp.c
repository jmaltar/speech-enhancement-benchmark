/**
 * cdsp.c -- classical speech enhancement in portable C (task 8). See cdsp.h.
 *
 * No I/O, no allocation, no dependency beyond libm and the FFT seam. The only
 * float functions used are sqrtf, expf and logf, which the budget already
 * costs per bin on every tier (and which the budget shows need no
 * approximation anywhere -- the ESP32-S3's dsps_sqrt_f32 was declined on
 * 2026-10-12 for exactly that reason: it changes the output and nothing here
 * needs the speed).
 */
#include "cdsp.h"
#include "cdsp_kernels.h"
#include <math.h>
#include <string.h>

#define CDSP_MAX_FRAME CDSP_FFT_MAX
#define CDSP_MAX_BINS  (CDSP_FFT_MAX / 2u + 1u)

void cdsp_params_default(cdsp_params_t *p)
{
    p->noise_est    = CDSP_NOISE_MCRA;
    p->mcra_tau_s_ms = 35.9f;    /* -8 ms / ln 0.8  */
    p->mcra_tau_d_ms = 156.0f;   /* -8 ms / ln 0.95 */
    p->mcra_tau_p_ms = 5.0f;     /* -8 ms / ln 0.2  */
    p->mcra_delta   = 5.0f;
    p->mcra_win_s   = 1.0f;
    p->mcra_bmin    = 2.5f;      /* 1 / 0.407, the minimum's measured bias */
    p->alpha_dd    = 0.98f;
    p->xi_min_db   = -25.0f;
    p->ss_over     = 1.0f;
    p->ss_floor    = 0.002f;
    p->vad_thresh  = 0.15f;
    p->noise_beta  = 0.98f;
    p->init_frames = 6u;
    p->unity_gain  = 0;
}

/* ---- Bessel functions, exponentially scaled -----------------------------
 * Abramowitz & Stegun 9.8.1-9.8.4, |error| < 2e-7 over the whole range. The
 * scaling is what makes the MMSE gain computable for any nu: exp(-nu/2) I(nu/2)
 * never overflows, and the gain tends smoothly to its Wiener limit. */
float cdsp_i0e(float x)
{
    if (x < 3.75f) {
        float t = x / 3.75f, t2 = t * t;
        float i0 = 1.0f + t2 * (3.5156229f + t2 * (3.0899424f + t2 * (1.2067492f
                 + t2 * (0.2659732f + t2 * (0.0360768f + t2 * 0.0045813f)))));
        return i0 * expf(-x);
    } else {
        float t = 3.75f / x;
        float p = 0.39894228f + t * (0.01328592f + t * (0.00225319f + t * (-0.00157565f
                + t * (0.00916281f + t * (-0.02057706f + t * (0.02635537f
                + t * (-0.01647633f + t * 0.00392377f)))))));
        return p / sqrtf(x);
    }
}

float cdsp_i1e(float x)
{
    if (x < 3.75f) {
        float t = x / 3.75f, t2 = t * t;
        float i1 = x * (0.5f + t2 * (0.87890594f + t2 * (0.51498869f + t2 * (0.15084934f
                 + t2 * (0.02658733f + t2 * (0.00301532f + t2 * 0.00032411f))))));
        return i1 * expf(-x);
    } else {
        float t = 3.75f / x;
        float p = 0.39894228f + t * (-0.03988024f + t * (-0.00362018f + t * (0.00163801f
                + t * (-0.01031555f + t * (0.02282967f + t * (-0.02895312f
                + t * (0.01787654f + t * -0.00420059f)))))));
        return p / sqrtf(x);
    }
}

/* ---- gain rules --------------------------------------------------------- */
float cdsp_gain_specsub(float gamma, float over, float floor_pow)
{
    float g2 = (gamma > 0.0f) ? 1.0f - over / gamma : 0.0f;
    if (g2 < floor_pow) g2 = floor_pow;
    return sqrtf(g2);
}

float cdsp_gain_wiener(float xi)
{
    return xi / (1.0f + xi);
}

float cdsp_gain_mmse(float xi, float gamma)
{
    if (gamma < 1e-12f) return cdsp_gain_wiener(xi);      /* |Y| = 0: any gain */
    float nu = xi * gamma / (1.0f + xi);
    float h  = 0.5f * nu;
    /* (sqrt(pi)/2) sqrt(nu)/gamma [(1+nu) e^{-h} I0(h) + nu e^{-h} I1(h)] */
    return 0.886226925f * (sqrtf(nu) / gamma)
         * ((1.0f + nu) * cdsp_i0e(h) + nu * cdsp_i1e(h));
}

/* ---- the processor ------------------------------------------------------ */
typedef struct {
    cdsp_algo_t   algo;
    cdsp_params_t p;
    uint32_t      F, H, B, rate;
    float         xi_min;
    cdsp_fft_t    fft;
    float w[CDSP_MAX_FRAME];        /* periodic Hann                          */
    float buf[CDSP_MAX_FRAME];      /* last F input samples                   */
    float acc[CDSP_MAX_FRAME];      /* overlap-add accumulator                */
    float frame[CDSP_MAX_FRAME];
    float X[CDSP_MAX_FRAME + 2u];   /* B bins, interleaved                    */
    float P[CDSP_MAX_BINS];         /* |Y|^2                                  */
    float noise[CDSP_MAX_BINS];     /* N, noise PSD                           */
    float shat[CDSP_MAX_BINS];      /* |S_hat|^2 of the previous frame        */
    uint32_t nframes;               /* frames processed, including pre-fill   */
    uint32_t ninit;                 /* full frames folded into the init mean  */
    /* MCRA state */
    float S[CDSP_MAX_BINS];         /* smoothed PSD                           */
    float Smin[CDSP_MAX_BINS];      /* running minimum, current window        */
    float Stmp[CDSP_MAX_BINS];      /* running minimum, next window           */
    float pp[CDSP_MAX_BINS];        /* speech presence probability            */
    uint32_t mcra_L;                /* window length in frames                */
    float    a_s, a_d, a_p;         /* per-hop coefficients from the taus     */
    uint32_t mcra_l;                /* frames into the current window         */
    int      mcra_started;
} cdsp_ctx_t;

static cdsp_ctx_t g_ctx;

/* Stage timing for a port, OFF unless the firmware build defines
 * CDSP_STAMP_DWT, and then it reads the Cortex-M DWT cycle counter directly
 * (no HAL, no header) between the stages of cdsp_process(). Sums per stage
 * over every hop, read off the board with the debugger after a run; eight
 * loads and adds per hop, far below the per-hop cost they decompose. Exists
 * because the first F411 run came in 1.5-1.75x over the budget's prediction
 * and "which stage" is not answerable from a single number. Absent from the
 * host build, so the oracle is unchanged. */
#ifdef CDSP_STAMP_DWT
#  if defined(__linux__)
#    include <time.h>
#  endif
volatile uint32_t g_cdsp_stage[8];
volatile uint32_t g_cdsp_stage_n;
static uint32_t s_stamp_prev;
#  if defined(__XTENSA__)
static inline uint32_t cdsp_stamp_now(void)
{ uint32_t c; __asm__ volatile ("rsr.ccount %0" : "=a"(c)); return c; }
#    define CDSP_DWT_CYCCNT (cdsp_stamp_now())            /* Xtensa CCOUNT, per-core */
#  elif defined(__linux__)
/* RPi5: CLOCK_MONOTONIC nanoseconds, truncated to 32 bits -- the per-stage
 * differences are what is summed, and no stage approaches 4 s. Cycles are
 * ns x 2.4 at the verified clock; the harness reports which it printed. */
static inline uint32_t cdsp_stamp_now(void)
{ struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint32_t)((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec); }
#    define CDSP_DWT_CYCCNT (cdsp_stamp_now())
#  else
#    define CDSP_DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#  endif
#  define CDSP_STAMP_BEGIN() (s_stamp_prev = CDSP_DWT_CYCCNT)
#  define CDSP_STAMP(i) do { uint32_t t_ = CDSP_DWT_CYCCNT; \
                            g_cdsp_stage[i] += t_ - s_stamp_prev; s_stamp_prev = t_; } while (0)
#  define CDSP_STAMP_END() (g_cdsp_stage_n++)
#else
#  define CDSP_STAMP_BEGIN() ((void)0)
#  define CDSP_STAMP(i)      ((void)0)
#  define CDSP_STAMP_END()   ((void)0)
#endif

static int cdsp_setup(void *c, uint32_t rate, uint32_t frame, uint32_t hop, uint32_t ch)
{
    cdsp_ctx_t *s = (cdsp_ctx_t *)c;
    if (ch != 1u) return -1;
    if (hop == 0u || frame % hop != 0u || frame > CDSP_MAX_FRAME) return -1;
    if (cdsp_fft_init(&s->fft, frame) != 0) return -1;
    s->F = frame; s->H = hop; s->B = frame / 2u + 1u; s->rate = rate;
    s->xi_min = powf(10.0f, s->p.xi_min_db / 10.0f);
    for (uint32_t i = 0; i < frame; i++)
        s->w[i] = 0.5f - 0.5f * cosf(6.283185307179586f * (float)i / (float)frame);
    memset(s->buf, 0, sizeof s->buf);
    memset(s->acc, 0, sizeof s->acc);
    memset(s->noise, 0, sizeof s->noise);
    memset(s->shat, 0, sizeof s->shat);
    memset(s->S, 0, sizeof s->S); memset(s->Smin, 0, sizeof s->Smin);
    memset(s->Stmp, 0, sizeof s->Stmp); memset(s->pp, 0, sizeof s->pp);
    s->nframes = 0u; s->ninit = 0u;
    s->mcra_L = (uint32_t)(s->p.mcra_win_s * (float)rate / (float)hop + 0.5f);
    if (s->mcra_L < 2u) s->mcra_L = 2u;
    {
        float hop_ms = 1000.0f * (float)hop / (float)rate;
        s->a_s = expf(-hop_ms / s->p.mcra_tau_s_ms);
        s->a_d = expf(-hop_ms / s->p.mcra_tau_d_ms);
        s->a_p = expf(-hop_ms / s->p.mcra_tau_p_ms);
    }
    s->mcra_l = 0u; s->mcra_started = 0;
    return 0;
}

/* MCRA: Cohen & Berdugo 2002, eqs. 4-9 in their numbering, one frame. */
static void mcra_update(cdsp_ctx_t *s)
{
    const uint32_t B = s->B;
    const cdsp_params_t *p = &s->p;
    float Sf[CDSP_MAX_BINS];
    /* frequency smoothing, b = [1/4, 1/2, 1/4] */
    for (uint32_t k = 0; k < B; k++) {
        float lo = (k > 0u) ? s->P[k - 1u] : s->P[k];
        float hi = (k + 1u < B) ? s->P[k + 1u] : s->P[k];
        Sf[k] = 0.25f * lo + 0.5f * s->P[k] + 0.25f * hi;
    }
    if (!s->mcra_started) {
        for (uint32_t k = 0; k < B; k++) {
            s->S[k] = s->Smin[k] = s->Stmp[k] = Sf[k];
            s->noise[k] = s->P[k];
            s->pp[k] = 0.0f;
        }
        s->mcra_started = 1; s->mcra_l = 1u;
        return;
    }
    s->mcra_l++;
    const int reset = (s->mcra_l >= s->mcra_L);
    for (uint32_t k = 0; k < B; k++) {
        float S = s->a_s * s->S[k] + (1.0f - s->a_s) * Sf[k];
        s->S[k] = S;
        if (reset) {
            s->Smin[k] = (s->Stmp[k] < S) ? s->Stmp[k] : S;
            s->Stmp[k] = S;
        } else {
            if (S < s->Smin[k]) s->Smin[k] = S;
            if (S < s->Stmp[k]) s->Stmp[k] = S;
        }
        float smin = (s->Smin[k] > 1e-10f) ? s->Smin[k] : 1e-10f;
        float I = (S / smin > p->mcra_delta) ? 1.0f : 0.0f;
        s->pp[k] = s->a_p * s->pp[k] + (1.0f - s->a_p) * I;
        float ad = s->a_d + (1.0f - s->a_d) * s->pp[k];
        s->noise[k] = ad * s->noise[k] + (1.0f - ad) * s->P[k];
        float cap = p->mcra_bmin * s->Smin[k];
        if (s->noise[k] > cap) s->noise[k] = cap;      /* the recorded deviation */
    }
    if (reset) s->mcra_l = 0u;
}

static int cdsp_process(void *c, const int16_t *in, int16_t *out, uint32_t n)
{
    cdsp_ctx_t *s = (cdsp_ctx_t *)c;
    const uint32_t F = s->F, H = s->H, B = s->B;
    const cdsp_params_t *p = &s->p;
    if (n != H) return -1;
    CDSP_STAMP_BEGIN();

    /* 1. shift one hop in, window, transform */
    memmove(s->buf, s->buf + H, (F - H) * sizeof *s->buf);
    for (uint32_t i = 0; i < H; i++) s->buf[F - H + i] = (float)in[i];
    for (uint32_t i = 0; i < F; i++) s->frame[i] = s->w[i] * s->buf[i];
    CDSP_STAMP(0);                                   /* shift + window      */
    cdsp_fft_forward(&s->fft, s->frame, s->X);
    CDSP_STAMP(1);                                   /* forward FFT         */
    for (uint32_t k = 0; k < B; k++)
        s->P[k] = s->X[2*k] * s->X[2*k] + s->X[2*k+1] * s->X[2*k+1];
    s->nframes++;
    CDSP_STAMP(2);                                   /* |Y|^2               */

    /* 2. noise PSD. The first (F/H - 1) frames straddle the zero pre-fill and
     *    are not representative, so the initial mean starts at the first FULL
     *    frame; after init_frames of those, the VAD takes over. */
    const uint32_t first_full = F / H;              /* 1-based frame index */
    int have_noise;
    if (p->noise_est == CDSP_NOISE_MCRA) {
        if (s->nframes >= first_full) mcra_update(s);
        have_noise = s->mcra_started;
    } else if (s->nframes >= first_full && s->ninit < p->init_frames) {
        s->ninit++;
        float inv = 1.0f / (float)s->ninit;
        for (uint32_t k = 0; k < B; k++)
            s->noise[k] += (s->P[k] - s->noise[k]) * inv;
    } else if (s->ninit >= p->init_frames) {
        /* VAD on the pre-update statistics (Sohn et al. 1999; Hu & Loizou eq. 6) */
        float llr = 0.0f;
        for (uint32_t k = 0; k < B; k++) {
            float N = s->noise[k] > 1e-10f ? s->noise[k] : 1e-10f;
            float gamma = s->P[k] / N;
            float gm1 = gamma - 1.0f; if (gm1 < 0.0f) gm1 = 0.0f;
            float xi = p->alpha_dd * (s->shat[k] / N) + (1.0f - p->alpha_dd) * gm1;
            if (xi < s->xi_min) xi = s->xi_min;
            llr += gamma * xi / (1.0f + xi) - logf(1.0f + xi);
        }
        llr /= (float)B;
        if (llr < p->vad_thresh)
            for (uint32_t k = 0; k < B; k++)
                s->noise[k] = p->noise_beta * s->noise[k] + (1.0f - p->noise_beta) * s->P[k];
    }
    /* before any noise estimate exists (pre-fill frames), pass through */
    if (p->noise_est != CDSP_NOISE_MCRA) have_noise = (s->ninit > 0u);
    CDSP_STAMP(3);                                   /* noise tracker       */

    /* 3. gains */
    for (uint32_t k = 0; k < B; k++) {
        float G;
        if (p->unity_gain || !have_noise) {
            G = 1.0f;
        } else {
            float N = s->noise[k] > 1e-10f ? s->noise[k] : 1e-10f;
            float gamma = s->P[k] / N;
            float gm1 = gamma - 1.0f; if (gm1 < 0.0f) gm1 = 0.0f;
            float xi = p->alpha_dd * (s->shat[k] / N) + (1.0f - p->alpha_dd) * gm1;
            if (xi < s->xi_min) xi = s->xi_min;
            switch (s->algo) {
            case CDSP_ALGO_SPECSUB: G = cdsp_gain_specsub(gamma, p->ss_over, p->ss_floor); break;
            case CDSP_ALGO_WIENER:  G = cdsp_gain_wiener(xi); break;
            default:                G = cdsp_gain_mmse(xi, gamma); break;
            }
        }
        s->shat[k] = G * G * s->P[k];
        s->X[2*k]   *= G;
        s->X[2*k+1] *= G;
    }
    CDSP_STAMP(4);                                   /* a priori SNR + gain */

    /* 4. inverse, overlap-add (analysis Hann only: COLA at 50 %), emit a hop */
    cdsp_fft_inverse(&s->fft, s->X, s->frame);
    CDSP_STAMP(5);                                   /* inverse FFT         */
    for (uint32_t i = 0; i < F; i++) s->acc[i] += s->frame[i];
    for (uint32_t i = 0; i < H; i++) {
        float v = s->acc[i];
        v = (v < -32768.0f) ? -32768.0f : (v > 32767.0f) ? 32767.0f : v;
        out[i] = (int16_t)lrintf(v);
    }
    memmove(s->acc, s->acc + H, (F - H) * sizeof *s->acc);
    memset(s->acc + (F - H), 0, H * sizeof *s->acc);
    CDSP_STAMP(6);                                   /* overlap-add + emit  */
    CDSP_STAMP_END();
    return 0;
}

static abf_proc_t g_proc = { "cdsp", &g_ctx, cdsp_process, cdsp_setup, NULL };

const abf_proc_t *cdsp_proc(cdsp_algo_t algo, const cdsp_params_t *p)
{
    g_ctx.algo = algo;
    if (p) g_ctx.p = *p; else cdsp_params_default(&g_ctx.p);
    g_proc.name = (algo == CDSP_ALGO_SPECSUB) ? "specsub"
                : (algo == CDSP_ALGO_WIENER)  ? "wiener"
                : (algo == CDSP_ALGO_MMSE)    ? "mmse" : "cdsp?";
    return &g_proc;
}

const char *cdsp_noise_est_name(int noise_est)
{
    return noise_est == CDSP_NOISE_MCRA ? "mcra" : noise_est == CDSP_NOISE_VAD ? "vad" : "?";
}

const float *cdsp_mcra_smin(uint32_t *bins)
{
    if (bins) *bins = g_ctx.B;
    return g_ctx.Smin;
}

const float *cdsp_noise_psd(uint32_t *bins)
{
    if (bins) *bins = g_ctx.B;
    return g_ctx.noise;
}
