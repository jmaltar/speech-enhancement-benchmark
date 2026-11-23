/* rnnoise_select.c -- CONFIG.algo_id -> processor for an image carrying
 * RNNoise. Everything that is not RNNoise goes to cdsp_select, so the registry
 * semantics (bypass, OLA, noise-tracker bits) stay in one place -- the same
 * shape as dtln/port/dtln_select.c.
 *
 * RNNOISE_SELECT_NO_CDSP: an image WITHOUT the classical arm. On the F411,
 * RNNoise's as-shipped stack (51 KB, two 960-point complex buffers and the
 * feature arrays in one inlined frame) plus its 29 KB heap do not fit beside
 * the classical context (23,708 B) in 128 KB: the link overflows by 1,504 B
 * at the measured reservations (benchmark_f411/RESULTS.md, 2026-10-22). Such
 * an image keeps bypass and OLA, so the 48 kHz framing anchor is still in it;
 * a classical id is NAKed with ABF_NAK_NO_ALGO like any id the image lacks. */
#include "rnnoise_proc.h"

#ifdef RNNOISE_SELECT_NO_CDSP
static const abf_proc_t *fallback(const abf_config_t *cfg)
{
    if (cfg->flags & ABF_CF_BYPASS) return abf_proc_bypass();
    if (cfg->algo_id >> ABF_NOISE_SHIFT) return NULL;      /* no tracker bits */
    uint32_t algo = cfg->algo_id & ABF_ALGO_MASK;
    if (algo == ABF_ALGO_BYPASS) return abf_proc_bypass();
    if (algo == ABF_ALGO_OLA)    return abf_proc_ola();
    return NULL;
}
#else
#include "cdsp.h"
#define fallback cdsp_select
#endif

const abf_proc_t *rnnoise_select(const abf_config_t *cfg)
{
    if (!(cfg->flags & ABF_CF_BYPASS) &&
        (cfg->algo_id & ABF_ALGO_MASK) == ABF_ALGO_RNNOISE) {
        if (cfg->algo_id >> ABF_NOISE_SHIFT) return NULL;   /* no tracker here */
        return rnnoise_proc();
    }
    return fallback(cfg);
}
