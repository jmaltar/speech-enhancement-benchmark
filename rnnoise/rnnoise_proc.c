/* rnnoise_proc.c -- see rnnoise_proc.h. */
#include "rnnoise_proc.h"
#include "rnnoise.h"
#include <math.h>
#include <string.h>
#ifdef RNNOISE_PROC_MCU
#include <malloc.h>
/* What rnnoise_create() and the 960-point kiss_fft twiddles took from newlib's
 * heap: DenoiseState + three GRU state vectors + kiss_fft_state, twiddles and
 * bit-reversal table. The linker reserves _Min_Heap_Size for it; this is the
 * number that reservation is checked against (benchmark_f411/RESULTS.md). */
volatile uint32_t g_rnnoise_heap_bytes;
#else
#include <stdio.h>
#endif

typedef struct {
    DenoiseState *st;
    float in[RNNOISE_HOP], out[RNNOISE_HOP];
} ctx_t;

static ctx_t g_ctx;

static int rn_setup(void *vctx, uint32_t rate_hz, uint32_t frame_len, uint32_t hop_len, uint32_t channels)
{
    ctx_t *c = (ctx_t *)vctx;
    if (rate_hz != RNNOISE_RATE_HZ || frame_len != RNNOISE_FRAME || hop_len != RNNOISE_HOP || channels != 1u) {
#ifndef RNNOISE_PROC_MCU
        fprintf(stderr, "rnnoise: geometry %u Hz %u/%u ch %u refused; the weights fix 48000 Hz, 960/480, mono\n",
                rate_hz, frame_len, hop_len, channels);
#endif
        return -1;                        /* the binding NAKs with ABF_NAK_NO_ALGO */
    }
    if (c->st) rnnoise_destroy(c->st);
#ifdef RNNOISE_V01
    c->st = rnnoise_create();
#else
    c->st = rnnoise_create(NULL);          /* NULL = the built-in model (rnn_data.c) */
#endif
#ifdef RNNOISE_PROC_MCU
    g_rnnoise_heap_bytes = (uint32_t)mallinfo().uordblks;
#endif
    return c->st ? 0 : -1;
}

static int rn_process(void *vctx, const int16_t *in, int16_t *out, uint32_t n)
{
    ctx_t *c = (ctx_t *)vctx;
    if (n != RNNOISE_HOP || !c->st) return -1;
    for (uint32_t i = 0; i < n; i++) c->in[i] = (float)in[i];          /* int16 scale, as the library expects */
    (void)rnnoise_process_frame(c->st, c->out, c->in);                /* VAD probability discarded */
    for (uint32_t i = 0; i < n; i++) {
        float v = c->out[i];
        v = (v < -32768.0f) ? -32768.0f : (v > 32767.0f) ? 32767.0f : v;
        out[i] = (int16_t)lrintf(v);
    }
    return 0;
}

static void rn_teardown(void *vctx)
{
    ctx_t *c = (ctx_t *)vctx;
    if (c->st) { rnnoise_destroy(c->st); c->st = NULL; }
}

static const abf_proc_t g_proc = { "rnnoise", &g_ctx, rn_process, rn_setup, rn_teardown };

const abf_proc_t *rnnoise_proc(void) { return &g_proc; }

const char *rnnoise_proc_version(void)
{
#ifdef RNNOISE_V01
    return "v0.1";
#elif defined(RNNOISE_NN8)
    return "v0.1.1+N";                 /* the GRUs on CMSIS-NN: nn8/rnn_nn8.h */
#else
    return "v0.1.1";
#endif
}
