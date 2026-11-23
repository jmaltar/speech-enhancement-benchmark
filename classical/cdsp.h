/**
 * cdsp.h -- classical speech enhancement in portable C (task 8).
 *
 * Three gain rules behind one STFT, exposed as ABF1 processors so the same
 * code runs through the file binding on the host (where it is the PC float
 * oracle for this arm), the file binding on the RPi5, and the CDC binding on
 * the three MCUs. The framing policy -- zero pre-fill, (frame - hop) output
 * lag, flush hops, drop-and-trim on the host -- is the transport's, not this
 * file's, so the oracle and every port frame identically and task 15 measures
 * the port alone.
 *
 * Algorithms (all magnitude-domain, noisy phase, Hann analysis at 50 %
 * overlap with no synthesis window -- periodic Hann sums to one, so a unity
 * gain reconstructs the input exactly):
 *
 *   SPECSUB  power spectral subtraction with a spectral floor
 *            G^2 = max(1 - over / gamma, floor)            (Berouti 1979 form,
 *            over = 1 is Boll's power subtraction)
 *   WIENER   a priori SNR Wiener filter, decision-directed  (Scalart & Filho
 *            G = xi / (1 + xi)                               1996, "Wiener-as")
 *   MMSE     MMSE short-time spectral amplitude estimator   (Ephraim & Malah
 *            G = (sqrt(pi)/2) (sqrt(nu)/gamma) exp(-nu/2)    1984, eq. 7)
 *                [(1+nu) I0(nu/2) + nu I1(nu/2)],  nu = xi gamma / (1 + xi)
 *
 * where gamma = |Y|^2 / N is the a posteriori SNR and xi the a priori SNR,
 * estimated by the decision-directed rule xi = a |S_prev|^2 / N +
 * (1 - a) max(gamma - 1, 0) with a = 0.98 (Ephraim & Malah 1984, eq. 51).
 *
 * Noise PSD, two trackers, selected by cdsp_params_t.noise_est:
 *
 *   CDSP_NOISE_VAD   initialise from the first `init_frames` full frames, then
 *                    update in frames a log-likelihood-ratio VAD (Sohn, Kim &
 *                    Sung 1999) calls speech-absent -- the rule Hu & Loizou's
 *                    NOIZEUS testbed used for every Fourier-based algorithm
 *                    (Speech Communication 49, 2007, eqs. 6-8):
 *                    mean_k[gamma xi/(1+xi) - log(1+xi)] < eta, eta = 0.15,
 *                    N = beta N + (1 - beta) |Y|^2, beta = 0.98.
 *                    ASSUMES A LEADING SILENCE. Measured 2026-10-12: fine on
 *                    VoiceBank and NOIZEUS, whose utterances start quiet, and
 *                    harmful on DNS, where 93 % of clips start in speech and the
 *                    contaminated initial estimate costs up to 0.10 STOI.
 *   CDSP_NOISE_MCRA  minima-controlled recursive averaging (Cohen & Berdugo,
 *                    IEEE SPL 9(1), 2002; also what Lu & Loizou 2011 used on
 *                    NOIZEUS): frequency-smoothed, time-smoothed |Y|^2 S;
 *                    running minimum S_min over a window of mcra_win_s seconds
 *                    (two-buffer scheme); speech-presence indicator
 *                    S/S_min > mcra_delta, smoothed into a probability p with
 *                    mcra_alpha_p; N updated with the probability-dependent
 *                    rate alpha_d + (1 - alpha_d) p. Needs no leading silence:
 *                    the floor is found at the first inter-word gap.
 *                    ONE DEVIATION from the paper, recorded as such: N is
 *                    clamped from above to mcra_bmin * S_min. Textbook MCRA
 *                    adapts to a FALLING noise level only through its
 *                    probability-gated average, which Cohen notes is slow (and
 *                    IMCRA, 2003, exists to fix); with speech from the first
 *                    sample the initial N sits at speech level in the speech
 *                    bins and was measured still 30x high after 3 s. The clamp
 *                    bounds that delay by the minimum-tracking window. bmin =
 *                    2.5 is 1 / 0.407, the measured ratio S_min / N in
 *                    stationary white noise at the locked framing, so the
 *                    clamp IS the bias-corrected minimum-statistics estimate
 *                    (Martin 2001's construction) and leaves N at 0.84 of the
 *                    true level when nothing is falling -- a 0.8 dB under-
 *                    estimate, typical of minimum trackers and recorded.
 *
 * Every constant is a parameter with its source recorded, so the paper can
 * state them and a reviewer can change them.
 */
#ifndef CDSP_H
#define CDSP_H

#include <stdint.h>
#include "abf_proc.h"
#include "abf_frame.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CDSP_ALGO_SPECSUB = 1,
    CDSP_ALGO_WIENER  = 2,
    CDSP_ALGO_MMSE    = 3
} cdsp_algo_t;

typedef enum {
    CDSP_NOISE_VAD  = 1,
    CDSP_NOISE_MCRA = 2
} cdsp_noise_est_t;

typedef struct {
    int      noise_est;     /* CDSP_NOISE_VAD or CDSP_NOISE_MCRA      MCRA   */
    /* MCRA smoothing is given as TIME CONSTANTS and converted to per-hop
     * coefficients in setup(), alpha = exp(-hop_s / tau). Cohen & Berdugo
     * quote alpha_s = 0.8, alpha_d = 0.95, alpha_p = 0.2 for their hop; the
     * values below are those coefficients expressed at an 8 ms hop, so at
     * this project's 16 ms hop they become 0.64 / 0.90 / 0.04. Carrying the
     * paper's coefficients unchanged to a 2x longer hop doubles every time
     * constant, which was measured to leave the tracker 17x high in a tone
     * bin after 3 s. (The paper is not on disk; its hop is taken as 8 ms
     * from the frame size and overlap it states, and is flagged unverified.) */
    float    mcra_tau_s_ms; /* PSD smoothing time constant            35.9   */
    float    mcra_tau_d_ms; /* noise averaging time constant          156    */
    float    mcra_tau_p_ms; /* presence-probability time constant     5.0    */
    float    mcra_delta;    /* MCRA presence threshold S/S_min        5.0    */
    float    mcra_win_s;    /* MCRA minimum-tracking window, seconds  1.0    */
    float    mcra_bmin;     /* MCRA upper clamp N <= bmin * S_min     (see .c)*/
    float    alpha_dd;      /* decision-directed smoothing a          0.98   */
    float    xi_min_db;     /* a priori SNR floor, dB                 -25    */
    float    ss_over;       /* SPECSUB over-subtraction factor        1.0    */
    float    ss_floor;      /* SPECSUB spectral floor on power        0.002  */
    float    vad_thresh;    /* VAD LLR threshold eta                  0.15   */
    float    noise_beta;    /* noise PSD smoothing, speech-absent     0.98   */
    uint32_t init_frames;   /* full frames averaged for initial N     6      */
    int      unity_gain;    /* test hook: force G = 1 on every bin    0      */
} cdsp_params_t;

void cdsp_params_default(cdsp_params_t *p);

/* Returns the processor for `algo`, configured with `p` (NULL = defaults).
 * ONE instance: the three algorithms share a single static context, which is
 * all an MCU needs and all the file binding drives. Calling this again
 * reconfigures that instance. setup() rejects a frame that is not a power of
 * two in [8, 1024], a hop that does not divide the frame, or channels != 1. */
const abf_proc_t *cdsp_proc(cdsp_algo_t algo, const cdsp_params_t *p);

/* The ABF1 selector (cdsp_select.c): maps CONFIG.algo_id / flags onto bypass,
 * OLA or one of the three gain rules with the default parameters and the
 * tracker named in bits 8-15. NULL for an id this arm does not carry. The
 * same function serves the CDC binding on the MCUs and, through cdsp_wav, the
 * file binding, so host and device agree on what an id means by construction. */
const abf_proc_t *cdsp_select(const abf_config_t *cfg);

/* The gain rules and the Bessel functions, exposed for the unit tests and
 * for anyone who wants the arithmetic without the framing. */
float cdsp_gain_specsub(float gamma, float over, float floor_pow);
float cdsp_gain_wiener(float xi);
float cdsp_gain_mmse(float xi, float gamma);
float cdsp_i0e(float x);     /* exp(-x) I0(x), x >= 0 */
float cdsp_i1e(float x);     /* exp(-x) I1(x), x >= 0 */

/* Noise PSD estimate after the last processed frame (bins = frame/2 + 1),
 * for tests of the estimator. Valid until the next process() call. */
const char *cdsp_noise_est_name(int noise_est);
/* MCRA's tracked minimum S_min, same validity as cdsp_noise_psd(); for tests. */
const float *cdsp_mcra_smin(uint32_t *bins);

const float *cdsp_noise_psd(uint32_t *bins);

#ifdef __cplusplus
}
#endif
#endif /* CDSP_H */
