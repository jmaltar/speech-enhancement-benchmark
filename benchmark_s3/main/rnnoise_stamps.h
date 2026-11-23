/* Per-stage CCOUNT stamps for the S3 RNNoise image -- the F411/H723 stamped
 * build (benchmark_f411/rnnoise_stamps.h, 2026-10-30) ported 2026-11-02. ON
 * only in a build configured with -DBENCH_RNNOISE_STAMPS=1, which compiles
 * rnnoise_denoise_stamped.c (Xiph's denoise.c plus RN_STAMP lines and nothing
 * else; byte-identical to the F411's copy, both asserted at configure time)
 * with this header force-included. The recorded measurement image never sees
 * it.
 *
 * The ONE difference from the Cortex-M header is the counter: this tier has no
 * DWT, so the stamps read the Xtensa CCOUNT special register (per-core; the
 * ABF task is pinned to core 0, so every stamp of a frame is read on the core
 * that runs it). Same storage, same macro names, same stage boundaries, so
 * rnnoise_stamps.c and the stamped denoise.c are the F411's files unchanged.
 * Sums are printed to the UART console (CH343) once a run has been quiet for
 * 1 s (main.c) -- this tier has no debugger to read them with while TinyUSB
 * owns the PHY. They are zeroed by reset, not by the host.
 *
 * Stages, in frame order (identical to the Cortex-M builds):
 *   0 high-pass biquad + analysis window + 960-point forward transform + band energies
 *   1 pitch downsample + pitch search + doubling removal
 *   2 pitch-window + 960-point forward transform of the pitch buffer + band energies / correlation
 *   3 the remaining features: log band energies, two DCTs, cepstral deltas, spectral variability
 *     (a silent frame leaves after the log loop; g_rn_silent_n counts those)
 *   4 compute_rnn -- the GRU network (skipped on silent frames)
 *   5 pitch filter + gain smoothing + band-gain interpolation + apply (skipped on silent frames)
 *   6 synthesis: 960-point inverse transform + window + overlap-add
 *   7 unused */
#ifndef RNNOISE_STAMPS_H
#define RNNOISE_STAMPS_H
#include <stdint.h>
extern volatile uint64_t g_rn_stage[8];
extern volatile uint32_t g_rn_stage_n;     /* frames stamped */
extern volatile uint32_t g_rn_silent_n;    /* frames that took the silence exit */
extern uint32_t g_rn_stamp_prev;
static inline uint32_t rn_ccount(void)
{ uint32_t c; __asm__ volatile ("rsr.ccount %0" : "=a"(c)); return c; }
#define RN_DWT_CYCCNT (rn_ccount())
#define RN_STAMP_BEGIN() (g_rn_stamp_prev = RN_DWT_CYCCNT)
#define RN_STAMP(i) do { uint32_t t_ = RN_DWT_CYCCNT; \
                         g_rn_stage[i] += (uint64_t)(t_ - g_rn_stamp_prev); g_rn_stamp_prev = t_; } while (0)
#define RN_STAMP_END() (g_rn_stage_n++)
#define RN_STAMP_SILENT() (g_rn_silent_n++)
#endif
