/**
 * cdsp_wav.c -- WAV in, WAV out, through the ABF1 FILE BINDING (task 8 tool).
 *
 *   cdsp_wav --in noisy.wav --out enhanced.wav --proc specsub|wiener|mmse|bypass|ola
 *            [--noise mcra|vad] [--frame N --hop N]
 *
 * This is deliberately NOT a loop that calls the processor on hops. The stream
 * is built, run through abf_file_run() and drop-and-trimmed by abf_wav_run()
 * in the harness (factored out of this file 2026-10-19 so the RNNoise oracle
 * shares it), so the PC float oracle for this arm frames exactly as every port
 * does, by construction -- the requirement task 7 placed on tasks 8-10.
 *
 * Default frame/hop are the locked 32 ms / 16 ms IN TIME: 512/256 at 16 kHz,
 * 256/128 at 8 kHz (NOIZEUS). 16-bit PCM mono only, refused otherwise.
 */
#include "cdsp.h"
#include "abf_wav.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void)
{
    fprintf(stderr, "usage: cdsp_wav --in IN.wav --out OUT.wav --proc specsub|wiener|mmse|bypass|ola\n"
                    "                [--noise mcra|vad] [--frame N --hop N]\n");
}

int main(int argc, char **argv)
{
    const char *inp = NULL, *outp = NULL, *which = "mmse", *noise = "mcra";
    uint32_t frame = 0, hop = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) inp = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else if (!strcmp(argv[i], "--proc") && i + 1 < argc) which = argv[++i];
        else if (!strcmp(argv[i], "--frame") && i + 1 < argc) frame = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hop") && i + 1 < argc) hop = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noise") && i + 1 < argc) noise = argv[++i];
        else { usage(); return 2; }
    }
    if (!inp || !outp) { usage(); return 2; }

    cdsp_params_t prm; cdsp_params_default(&prm);
    if      (!strcmp(noise, "mcra")) prm.noise_est = CDSP_NOISE_MCRA;
    else if (!strcmp(noise, "vad"))  prm.noise_est = CDSP_NOISE_VAD;
    else { usage(); return 2; }
    const abf_proc_t *proc;
    if      (!strcmp(which, "specsub")) proc = cdsp_proc(CDSP_ALGO_SPECSUB, &prm);
    else if (!strcmp(which, "wiener"))  proc = cdsp_proc(CDSP_ALGO_WIENER, &prm);
    else if (!strcmp(which, "mmse"))    proc = cdsp_proc(CDSP_ALGO_MMSE, &prm);
    else if (!strcmp(which, "bypass"))  proc = abf_proc_bypass();
    else if (!strcmp(which, "ola"))     proc = abf_proc_ola();
    else { usage(); return 2; }

    /* Stamp algo_id as the registry defines it, so a file run and a CDC run of
     * the same variant carry the same id (the device selects from it). */
    uint32_t a = !strcmp(which, "specsub") ? ABF_ALGO_SPECSUB
               : !strcmp(which, "wiener")  ? ABF_ALGO_WIENER
               : !strcmp(which, "mmse")    ? ABF_ALGO_MMSE
               : !strcmp(which, "ola")     ? ABF_ALGO_OLA : ABF_ALGO_BYPASS;
    uint32_t ne = (a >= 1u && a <= 3u) ? (!strcmp(noise, "vad") ? ABF_NOISE_VAD : ABF_NOISE_MCRA) : 0u;

    abf_wav_opts_t o = { frame, hop, a | (ne << ABF_NOISE_SHIFT), which, "cdsp_wav" };
    abf_wav_result_t res; char err[256];
    if (abf_wav_run(inp, outp, proc, &o, &res, err, sizeof err) != 0) { fprintf(stderr, "%s\n", err); return 1; }

    printf("%s: %s/%s %u Hz frame %u hop %u  in %u samples  hops %u+%u  overruns %u  out %s\n",
           inp, proc->name, noise, res.rate_hz, res.frame_len, res.hop_len, res.total_samples,
           res.data_hops, res.flush_hops, res.rep.overruns, outp);
    return 0;
}
