/* Per-stage DWT stamps for the RNNoise image -- the RNNoise twin of the
 * classical arm's CDSP_STAMP_DWT (classical/cdsp.c). ON only in a build
 * configured with -DBENCH_RNNOISE_STAMPS=ON, which compiles
 * rnnoise_denoise_stamped.c (Xiph's denoise.c plus RN_STAMP lines and nothing
 * else -- the configure step diffs the two and refuses any other divergence)
 * with this header force-included. The recorded measurement images never see
 * it. Reads the Cortex-M DWT cycle counter directly between the stages of
 * rnnoise_process_frame(); seven 64-bit adds per hop, far below the ~3 M
 * cycles they decompose. Sums are read off the board with the debugger after
 * a run (read_rnnoise_stamps.sh); they are zeroed by reset, not by the host.
 *
 * Stages, in frame order:
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
#define RN_DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define RN_STAMP_BEGIN() (g_rn_stamp_prev = RN_DWT_CYCCNT)
#define RN_STAMP(i) do { uint32_t t_ = RN_DWT_CYCCNT; \
                         g_rn_stage[i] += (uint64_t)(t_ - g_rn_stamp_prev); g_rn_stamp_prev = t_; } while (0)
#define RN_STAMP_END() (g_rn_stage_n++)
#define RN_STAMP_SILENT() (g_rn_silent_n++)
#endif
