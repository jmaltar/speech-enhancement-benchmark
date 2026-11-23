/* dtln_fc_pie_core.c -- see dtln_fc_pie.h. Plain C; compiled on the host for
 * the bit-exactness test and on the S3 for the kernel. */
#include "dtln_fc_pie.h"
#include <string.h>

int64_t dtln_fc_pie_dot_c(const int16_t *x, const int8_t *w, int n)
{
    int64_t acc = 0;
    for (int i = 0; i < n; i++)
        acc += (int32_t)w[i] * (int32_t)x[i];      /* as the reference: int32 product, int64 sum */
    return acc;
}

int32_t dtln_fc_pie_mbqm64(int64_t x, int32_t mult, int shift)
{
    /* tensorflow/lite/micro/kernels/internal/common.cc, the int64 overload;
     * identical under TFLITE_SINGLE_ROUNDING and without it. */
    const int32_t rm = (mult < 0x7FFF0000) ? ((mult + (1 << 15)) >> 16) : 0x7FFF;
    const int total_shift = 15 - shift;
    x = x * (int64_t)rm + ((int64_t)1 << (total_shift - 1));
    return (int32_t)(x >> total_shift);
}

#if DTLN_FC_PIE_ASM
#define DOT16(x, w, n) dtln_pie_dot_s16s8((x), (w), (n))
#else
#define DOT16(x, w, n) dtln_fc_pie_dot_c((x), (w), (n))
#endif

void dtln_fc_pie_matvec(const dtln_fc_pie_mv_t *m, const int16_t *x, int16_t *out)
{
    const int n16  = m->depth & ~15;
    const int tail = m->depth - n16;
    const int8_t *w = m->w;
    for (int o = 0; o < m->rows; o++, w += m->w_stride) {
        int64_t acc = n16 ? DOT16(x, w, n16) : 0;
        if (tail) acc += dtln_fc_pie_dot_c(x + n16, w + n16, tail);
        if (m->bias) acc += m->bias[o];
        const int k = m->per_channel ? o : 0;
        int32_t y = dtln_fc_pie_mbqm64(acc, m->mult[k], m->shift[k]);
        y += m->out_offset;
        if (y < m->act_min) y = m->act_min;
        if (y > m->act_max) y = m->act_max;
        out[o] = (int16_t)y;
    }
}

/* Fixed-seed LCG so the check is the same on every run and every build. */
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }

int dtln_fc_pie_selfcheck(void)
{
    static __attribute__((aligned(16))) int16_t x[512];
    static __attribute__((aligned(16))) int8_t  w[512];
    uint32_t s = 0x20261022u;
    for (int i = 0; i < 512; i++) x[i] = (int16_t)(lcg(&s) >> 16);
    for (int i = 0; i < 512; i++) w[i] = (int8_t)(lcg(&s) >> 24);
    /* every vector length the model's rows reduce to: 512, 256 (stage 2),
     * 256 of 257, 16 of 30, 48 of 58 (stage 1), and the 32/16 remainders */
    static const int lens[] = { 512, 256, 48, 32, 16, 64, 496 };
    for (unsigned c = 0; c < sizeof lens / sizeof lens[0]; c++)
        if (DOT16(x, w, lens[c]) != dtln_fc_pie_dot_c(x, w, lens[c])) return (int)c + 1;
    /* extremes: all-min inputs against all-min weights, the largest |acc| the
     * 40-bit accumulator can see at depth 512 (2^22 * 512 = 2^31) */
    for (int i = 0; i < 512; i++) { x[i] = -32768; w[i] = -128; }
    if (DOT16(x, w, 512) != dtln_fc_pie_dot_c(x, w, 512)) return 100;
    for (int i = 0; i < 512; i++) w[i] = 127;
    if (DOT16(x, w, 512) != dtln_fc_pie_dot_c(x, w, 512)) return 101;
    return 0;
}
