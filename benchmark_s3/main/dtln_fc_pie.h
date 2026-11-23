/**
 * dtln_fc_pie.h -- an int16-activation / int8-weight matrix-vector kernel for
 * the ESP32-S3 (Xtensa LX7 + PIE SIMD), written 2026-10-22 to replace TFLite
 * Micro's portable C++ reference on the two ops that carry DTLN's MACs:
 * FULLY_CONNECTED (per-tensor, stage 1 and the mask Dense) and the 1x1 CONV_2D
 * pair (per-channel, the stage-2 encoder/decoder basis, 262,144 of the model's
 * ~335 k MACs per hop). esp-nn has no int16 path; this is the one the tier
 * lacked (benchmark_s3/RESULTS.md, "DTLN on the ESP32-S3", item 3).
 *
 * Arithmetic contract -- bit-identical to TFLM 64b3de8's reference kernels
 * (tensorflow/lite/micro/kernels/internal/reference/integer_ops/
 * {fully_connected,conv}.h, int16 input, int8 filter, int64 bias):
 *
 *   acc  = sum_d  w[o][d] * x[d]            exact (int64; the PIE path sums
 *                                            into the 40-bit ACCX, |acc| <=
 *                                            depth * 2^22 <= 2^31 for depth
 *                                            <= 512, so nothing saturates)
 *   acc += bias[o]                           int64, if a bias tensor exists
 *   y    = MultiplyByQuantizedMultiplier(acc, mult[o], shift[o])   (int64 form:
 *          rm = mult < 0x7FFF0000 ? (mult + 2^15) >> 16 : 0x7FFF;
 *          y  = (acc * rm + 2^(14 - shift)) >> (15 - shift))
 *   y   += output_offset                     FC only; the int16 conv reference
 *                                            adds none (and both are 0 here)
 *   y    = clamp(y, act_min, act_max);  out[o] = (int16) y
 *
 * Zero points: TFLite's 16x8 scheme is symmetric, so input and filter offsets
 * are 0; the glue verifies that at Prepare and routes anything else to the
 * reference kernel, counted in g_dtln.fc_calls_ref (a number the board run
 * must report as 0 for the measurement to be the kernel's).
 *
 * Three layers, so the arithmetic is testable where the assembly is not:
 *   dtln_fc_pie.S        the PIE dot product (target only, verified on the
 *                        board by dtln_fc_pie_selfcheck() at setup)
 *   dtln_fc_pie_core.c   the matvec + epilogue in C, calling either the
 *                        assembly or its own C dot product (DTLN_FC_PIE_ASM)
 *   dtln_fc_pie_tflm.cc  the TFLM registrations, padded weight copies and
 *                        the x-alignment scratch
 * benchmark_s3/host_test/ builds the core with DTLN_FC_PIE_ASM=0 against the
 * reference templates from the same TFLM tree and asserts bit-identity.
 */
#ifndef DTLN_FC_PIE_H
#define DTLN_FC_PIE_H
#include <stdint.h>
#include <stddef.h>

#ifndef DTLN_FC_PIE_ASM
#error "DTLN_FC_PIE_ASM must be defined as 0 (C dot product) or 1 (PIE assembly): a build that does not say which it measures is not a measurement"
#endif
#if DTLN_FC_PIE_ASM && !defined(__XTENSA__)
#error "DTLN_FC_PIE_ASM=1 is the ESP32-S3 PIE kernel and cannot be built for this target"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The PIE dot product: n must be a multiple of 16 (0 allowed), x and w
 * 16-byte aligned. Returns the exact 40-bit ACCX sum as int64. */
int64_t dtln_pie_dot_s16s8(const int16_t *x, const int8_t *w, int n);
/* The same sum in plain C (any n, any alignment). */
int64_t dtln_fc_pie_dot_c(const int16_t *x, const int8_t *w, int n);
/* TFLM's int64 MultiplyByQuantizedMultiplier, transcribed. */
int32_t dtln_fc_pie_mbqm64(int64_t x, int32_t mult, int shift);

typedef struct {
    int             rows;        /* outputs                                  */
    int             depth;       /* accumulation length                      */
    const int8_t   *w;           /* row-major, row o at w + o * w_stride     */
    int             w_stride;    /* bytes; a multiple of 16 when ASM is used */
    const int64_t  *bias;        /* NULL if none                             */
    const int32_t  *mult;        /* per-channel if per_channel, else [0]     */
    const int32_t  *shift;
    int             per_channel;
    int32_t         out_offset;
    int32_t         act_min, act_max;
} dtln_fc_pie_mv_t;

/* out[o] for every row; x must be 16-byte aligned when ASM is used. */
void dtln_fc_pie_matvec(const dtln_fc_pie_mv_t *m, const int16_t *x, int16_t *out);

/* Fixed-seed check that the selected dot product (assembly on the board)
 * agrees with the C one on the shapes the model uses. 0 = identical; else the
 * 1-based case that differed. Run once at setup, never in a timed region. */
int dtln_fc_pie_selfcheck(void);

#ifdef __cplusplus
}
/* TFLM registrations (dtln_fc_pie_tflm.cc). C++ linkage whatever block this
 * header is included from: dtln_proc.cc includes it inside its extern "C"
 * region, which would otherwise rename these to C symbols the glue never
 * defines (found by the host link, 2026-10-22). */
#include "tensorflow/lite/micro/micro_common.h"
extern "C++" {
TFLMRegistration dtln_fc_pie_register_fully_connected(void);
TFLMRegistration dtln_fc_pie_register_conv2d(void);
}
#endif
#endif /* DTLN_FC_PIE_H */
