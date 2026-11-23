/**
 * cdsp_select.c -- CONFIG.algo_id -> processor, for every binding.
 *
 * The registry is in abf_frame.h. Its classical values are asserted equal to
 * cdsp_algo_t / cdsp_noise_est_t here, so the two can never drift apart
 * silently (a build switch that changes what is measured must break the
 * build). Parameters are the defaults: a run that needs others is a different
 * variant and gets a different image or a different tool, deliberately --
 * there is no parameter channel on the wire and none is wanted.
 */
#include "cdsp.h"
#include "abf_frame.h"
#include "abf_proc.h"

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(ABF_ALGO_SPECSUB == CDSP_ALGO_SPECSUB, "registry/cdsp_algo_t drift");
_Static_assert(ABF_ALGO_WIENER  == CDSP_ALGO_WIENER,  "registry/cdsp_algo_t drift");
_Static_assert(ABF_ALGO_MMSE    == CDSP_ALGO_MMSE,    "registry/cdsp_algo_t drift");
_Static_assert(ABF_NOISE_VAD    == CDSP_NOISE_VAD,    "registry/cdsp_noise_est_t drift");
_Static_assert(ABF_NOISE_MCRA   == CDSP_NOISE_MCRA,   "registry/cdsp_noise_est_t drift");
#endif

const abf_proc_t *cdsp_select(const abf_config_t *cfg)
{
    if (cfg->flags & ABF_CF_BYPASS) return abf_proc_bypass();

    uint32_t algo  = cfg->algo_id & ABF_ALGO_MASK;
    uint32_t noise = (cfg->algo_id >> ABF_NOISE_SHIFT) & ABF_NOISE_MASK;
    if (cfg->algo_id >> 16) return NULL;                 /* reserved bits set */

    if (algo == ABF_ALGO_BYPASS) return noise ? NULL : abf_proc_bypass();
    if (algo == ABF_ALGO_OLA)    return noise ? NULL : abf_proc_ola();

    cdsp_params_t p;
    cdsp_params_default(&p);
    if      (noise == ABF_NOISE_VAD)  p.noise_est = CDSP_NOISE_VAD;
    else if (noise == ABF_NOISE_MCRA) p.noise_est = CDSP_NOISE_MCRA;
    else if (noise != ABF_NOISE_DEFAULT) return NULL;

    switch (algo) {
    case ABF_ALGO_SPECSUB: return cdsp_proc(CDSP_ALGO_SPECSUB, &p);
    case ABF_ALGO_WIENER:  return cdsp_proc(CDSP_ALGO_WIENER,  &p);
    case ABF_ALGO_MMSE:    return cdsp_proc(CDSP_ALGO_MMSE,    &p);
    default:               return NULL;
    }
}
