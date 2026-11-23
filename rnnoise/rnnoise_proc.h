/**
 * rnnoise_proc.h -- Xiph RNNoise as an ABF1 processor (task 9).
 *
 * RNNoise's geometry is fixed by its weights: 48 kHz, 480-sample (10 ms)
 * frames analysed in a 960-sample window with 50 % overlap. Its API is one
 * 480-sample frame in, one out, and frame_synthesis() emits the overlap-add
 * of the previous window's second half with the current window's first half,
 * so the output lags the input by exactly 480 samples -- frame_len - hop_len
 * for CONFIG frame 960 / hop 480, which is what this processor requires and
 * what the host's drop-and-trim then removes. The DenoiseState's zeroed
 * analysis/synthesis memories are the zero pre-fill. No framing lives here;
 * rnnoise_process_frame() is called once per hop and nothing else is done to
 * the samples beyond the int16 <-> float conversion at the int16 scale the
 * library expects (its own demo feeds raw shorts as floats).
 *
 * Two vendored releases, chosen at link time: v0.1.1 (the as-shipped release;
 * identical weights to v0.1 plus the 2017-10-10 gain-decay limiter and
 * maintenance fixes) and v0.1 (the paper-era tree, 2017-09-19). Build with
 * -DRNNOISE_V01 for the latter, whose rnnoise_create() takes no model argument.
 */
#ifndef RNNOISE_PROC_H
#define RNNOISE_PROC_H

#include "abf_proc.h"
#include "abf_frame.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RNNOISE_RATE_HZ   48000u
#define RNNOISE_FRAME     960u     /* CONFIG.frame_len: analysis window */
#define RNNOISE_HOP       480u     /* CONFIG.hop_len:   rnnoise frame   */

/* algo_id: ABF_ALGO_RNNOISE in the abf_frame.h registry, entered 2026-10-22
 * with the F411 port (it was a provisional 4 for the host-only runs; 4 went
 * to DTLN on 2026-10-21, so RNNoise is 5 -- the output WAVs carry no id). */

const abf_proc_t *rnnoise_proc(void);
const char *rnnoise_proc_version(void);   /* "v0.1.1", "v0.1" or "v0.1.1+N" */

/* CONFIG.algo_id -> processor for an image carrying RNNoise beside the
 * classical arm: rnnoise_proc() for ABF_ALGO_RNNOISE (noise bits must be
 * zero), cdsp_select() for everything else. rnnoise_select.c; firmware only. */
const abf_proc_t *rnnoise_select(const abf_config_t *cfg);

/* Firmware builds define RNNOISE_PROC_MCU: no stdio in the refusal path, and
 * a heap witness (below) for the linker's reservation to be sized from. */
#ifdef RNNOISE_PROC_MCU
extern volatile uint32_t g_rnnoise_heap_bytes;   /* newlib heap in use after setup */
#endif

#ifdef __cplusplus
}
#endif
#endif
