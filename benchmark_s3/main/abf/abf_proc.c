/**
 * abf_proc.c -- reference processors for the ABF1 harness.
 *
 * Neither is an algorithm. They exist so the binding can be tested before
 * tasks 8-11 provide anything real, and they are deliberately the two that
 * check different halves of the framing: bypass pins the integer geometry with
 * no float in the path, OLA pins COLA.
 */
#include "abf_proc.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

/* Largest frame_len*channels the reference processors accept. 4096 admits
 * the MUSDB tier (44.1 kHz stereo, file binding). The MCU builds set it to the
 * locked 512 on the command line: at 4096 the two contexts cost 56 KB of .bss,
 * which the F411 does not have beside the probe's 64 KB pool. A CONFIG whose
 * frame exceeds it is refused in setup() -- NAK_NO_ALGO -- never truncated. */
#ifndef ABF_PROC_MAX_FRAME
#define ABF_PROC_MAX_FRAME 4096u
#endif

/* ---- bypass: a pure delay line of (frame_len - hop_len) samples ---------- */
typedef struct {
    int16_t  d[ABF_PROC_MAX_FRAME];
    uint32_t lag;           /* samples held back */
} bypass_ctx_t;

static bypass_ctx_t g_bypass;

static int bypass_setup(void *c, uint32_t rate, uint32_t frame, uint32_t hop,
                        uint32_t ch)
{
    bypass_ctx_t *s = (bypass_ctx_t *)c;
    (void)rate;
    s->lag = (frame - hop) * ch;
    if (s->lag > ABF_PROC_MAX_FRAME) return -1;
    memset(s->d, 0, sizeof s->d);
    return 0;
}

static int bypass_process(void *c, const int16_t *in, int16_t *out, uint32_t n)
{
    bypass_ctx_t *s = (bypass_ctx_t *)c;
    uint32_t lag = s->lag;

    if (lag == 0u) { memcpy(out, in, n * sizeof *in); return 0; }

    if (lag >= n) {
        /* Whole output comes from the delay line. */
        memcpy(out, s->d, n * sizeof *out);
        memmove(s->d, s->d + n, (lag - n) * sizeof *s->d);
        memcpy(s->d + (lag - n), in, n * sizeof *in);
    } else {
        memcpy(out, s->d, lag * sizeof *out);
        memcpy(out + lag, in, (n - lag) * sizeof *in);
        memcpy(s->d, in + (n - lag), lag * sizeof *in);
    }
    return 0;
}

static const abf_proc_t g_bypass_proc = {
    "bypass", &g_bypass, bypass_process, bypass_setup, NULL
};
const abf_proc_t *abf_proc_bypass(void) { return &g_bypass_proc; }

/* ---- OLA: Hann analysis window, overlap-add, no spectral step ------------
 *
 * Periodic Hann at 50 % overlap sums to exactly 1, so an analysis window with
 * no synthesis window reconstructs the input -- which is the property the
 * classical arm relies on when it applies a spectral gain and adds back. A
 * synthesis window too would need a 1/sum(w^2) normalisation; that choice
 * belongs to task 8, not to the transport.
 * ------------------------------------------------------------------------- */
typedef struct {
    float    w[ABF_PROC_MAX_FRAME];
    float    buf[ABF_PROC_MAX_FRAME];   /* analysis: last frame_len samples */
    float    acc[ABF_PROC_MAX_FRAME];   /* overlap-add accumulator          */
    uint32_t frame, hop, ch;
} ola_ctx_t;

static ola_ctx_t g_ola;

static int ola_setup(void *c, uint32_t rate, uint32_t frame, uint32_t hop,
                     uint32_t ch)
{
    ola_ctx_t *s = (ola_ctx_t *)c;
    (void)rate;
    if (ch != 1u) return -1;                 /* mono only, deliberately */
    if (frame > ABF_PROC_MAX_FRAME) return -1;
    if (hop == 0u || frame % hop != 0u) return -1;
    s->frame = frame; s->hop = hop; s->ch = ch;
    for (uint32_t i = 0; i < frame; i++)
        s->w[i] = 0.5f - 0.5f * cosf(6.283185307179586f * (float)i / (float)frame);
    memset(s->buf, 0, sizeof s->buf);
    memset(s->acc, 0, sizeof s->acc);
    return 0;
}

static int ola_process(void *c, const int16_t *in, int16_t *out, uint32_t n)
{
    ola_ctx_t *s = (ola_ctx_t *)c;
    uint32_t F = s->frame, H = s->hop;
    if (n != H) return -1;

    /* shift in one hop */
    memmove(s->buf, s->buf + H, (F - H) * sizeof *s->buf);
    for (uint32_t i = 0; i < H; i++) s->buf[F - H + i] = (float)in[i];

    /* window, (no transform), overlap-add */
    for (uint32_t i = 0; i < F; i++) s->acc[i] += s->w[i] * s->buf[i];

    for (uint32_t i = 0; i < H; i++) {
        float v = s->acc[i];
        v = (v < -32768.0f) ? -32768.0f : (v > 32767.0f) ? 32767.0f : v;
        out[i] = (int16_t)lrintf(v);
    }
    memmove(s->acc, s->acc + H, (F - H) * sizeof *s->acc);
    memset(s->acc + (F - H), 0, H * sizeof *s->acc);
    return 0;
}

static const abf_proc_t g_ola_proc = {
    "ola", &g_ola, ola_process, ola_setup, NULL
};
const abf_proc_t *abf_proc_ola(void) { return &g_ola_proc; }
