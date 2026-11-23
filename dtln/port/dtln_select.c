/* dtln_select.c -- CONFIG.algo_id -> processor for an image carrying DTLN
 * beside the classical arm. Everything that is not DTLN goes to cdsp_select,
 * so the registry semantics (bypass, OLA, noise-tracker bits) stay in one
 * place. */
#include "dtln_proc.h"
#include "cdsp.h"

const abf_proc_t *dtln_select(const abf_config_t *cfg)
{
    if (!(cfg->flags & ABF_CF_BYPASS) &&
        (cfg->algo_id & ABF_ALGO_MASK) == ABF_ALGO_DTLN) {
        if (cfg->algo_id >> ABF_NOISE_SHIFT) return NULL;   /* no tracker here */
        return dtln_proc();
    }
    return cdsp_select(cfg);
}
