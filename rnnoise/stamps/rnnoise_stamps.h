/* Per-stage stamps for the RPi5's RNNoise run tool -- the F411/H723 stamped
 * build (benchmark_f411/rnnoise_stamps.h, 2026-10-30) ported 2026-11-02, the
 * twin of the S3's. Built only as rnnoise_run_stamp (rnnoise/CMakeLists.txt),
 * which compiles stamps/rnnoise_denoise_stamped.c (Xiph's denoise.c plus
 * RN_STAMP lines and nothing else; byte-identical to the F411's copy, pinned
 * by SHA-256 and diffed against the vendored denoise.c at configure time) with
 * this header force-included. The measurement tool rnnoise_run never sees it.
 *
 * The ONE difference from the Cortex-M header is the counter: userspace on the
 * Pi has no cycle counter, so the stamps read CLOCK_MONOTONIC in nanoseconds,
 * truncated to 32 bits (the per-stage differences are what is summed and no
 * stage approaches 4 s) -- the same time base as this tier's classical stamps
 * (classical/cdsp.c) and its per-hop figures. Cycles are ns x 2.4 at the
 * verified 2.4 GHz clock. Each run tool process is one utterance; the sums are
 * printed at exit by cdsp_run.c and collected per utterance by
 * abf_batch_file.py --stages-log.
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
#include <time.h>
extern volatile uint64_t g_rn_stage[8];
extern volatile uint32_t g_rn_stage_n;     /* frames stamped */
extern volatile uint32_t g_rn_silent_n;    /* frames that took the silence exit */
extern uint32_t g_rn_stamp_prev;
static inline uint32_t rn_now_ns(void)
{ struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint32_t)((uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec); }
#define RN_DWT_CYCCNT (rn_now_ns())
#define RN_STAMP_BEGIN() (g_rn_stamp_prev = RN_DWT_CYCCNT)
#define RN_STAMP(i) do { uint32_t t_ = RN_DWT_CYCCNT; \
                         g_rn_stage[i] += (uint64_t)(t_ - g_rn_stamp_prev); g_rn_stamp_prev = t_; } while (0)
#define RN_STAMP_END() (g_rn_stage_n++)
#define RN_STAMP_SILENT() (g_rn_silent_n++)
#endif
