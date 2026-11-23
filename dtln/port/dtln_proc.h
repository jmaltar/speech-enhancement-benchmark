/**
 * dtln_proc.h -- DTLN (two-stage LSTM mask + post-filter) as an ABF1 processor,
 * 16x8 on TFLite Micro: int16 activations and states, int8 weights.
 *
 * The processing is dtln/oracle.py's `enhance()` and `DtlnTflm`, written in C++
 * against the same runtime: a 512-sample rectangular frame at the model's own
 * 128-sample hop (8 ms -- DTLN's trained geometry, a documented deviation from
 * the locked 256), rFFT -> magnitude -> stage 1 (mask) -> masked spectrum ->
 * irFFT -> stage 2 -> rectangular overlap-add, the LSTM states carried across
 * frames in float and requantized at the boundary exactly as the oracle does
 * (each state has its own input and output scale). CONFIG must say 16 kHz,
 * frame 512, hop 128, mono; anything else is refused (NAK_NO_ALGO), never
 * approximated.
 *
 * The model is the image's: two flatbuffers compiled in as dtln_model_1/2 by
 * dtln/port/gen_model_c.py, their SHA-256 in the generated file. Input and
 * output tensors are wired by name from the flatbuffer the way the oracle
 * wires them from LiteRT (rank-3 main tensor; states sorted by name on the
 * input side and by ":N" position on the output side), so the port is not
 * tied to one pruning point's sizes.
 */
#ifndef DTLN_PROC_H
#define DTLN_PROC_H

#include <stdint.h>
#include <stddef.h>
#include "abf_proc.h"
#include "abf_frame.h"

#ifdef __cplusplus
extern "C" {
#endif

const abf_proc_t *dtln_proc(void);

/* Selector for an image that carries DTLN beside the classical arm: DTLN for
 * ABF_ALGO_DTLN (noise bits must be zero), cdsp_select() for everything else. */
const abf_proc_t *dtln_select(const abf_config_t *cfg);

/* Witness block, readable over SWD like g_bench: what the runtime actually
 * allocated and whether setup succeeded, so the arena figures in the budget
 * are measured on the silicon rather than on the host. */
typedef struct {
    uint32_t magic;            /* 'DTL1' once setup has run                  */
    uint32_t setup_rc;         /* 0 ok; else the step that failed (1..9)     */
    uint32_t arena1_used;      /* bytes, interpreter 1 (stage 1)             */
    uint32_t arena2_used;      /* bytes, interpreter 2 (stage 2)             */
    uint32_t arena1_size;      /* bytes provisioned                          */
    uint32_t arena2_size;
    uint32_t n_state1;         /* state tensors per stage, as wired          */
    uint32_t n_state2;
    uint32_t frames;           /* frames processed since setup               */
    uint32_t model1_len;
    uint32_t model2_len;
#ifdef DTLN_STAMP_DWT
    /* Stage sums in device ticks over all frames since setup (stamp build
     * only, -DBENCH_DTLN_STAMPS=ON; the measurement image carries none). */
    uint32_t t_fft, t_stage1, t_ifft, t_stage2, t_ola, t_total;
#endif
    /* Which matvec kernel ran FULLY_CONNECTED / 1x1 CONV_2D (2026-10-22;
     * benchmark_s3/main/dtln_fc_pie.h): 0 = the tree's reference C++,
     * 1 = PIE assembly, 2 = the same glue with its C dot product. The rest
     * witness that the fast path was actually taken: fc_selfcheck must be 0,
     * fc_calls_ref 0 and fc_x_copies 0 for a run to be the kernel's number.
     * Appended last so earlier fields keep their offsets. */
    uint32_t fc_kernel, fc_selfcheck, fc_pool_used, fc_pool_size, fc_pool_addr;
    uint32_t fc_calls_vec, fc_calls_ref, fc_x_copies;
} dtln_stats_t;
extern volatile dtln_stats_t g_dtln;

#ifdef __cplusplus
}
#endif
#endif /* DTLN_PROC_H */
