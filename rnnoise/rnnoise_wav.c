/**
 * rnnoise_wav.c -- 48 kHz WAV in, 48 kHz WAV out, through the ABF1 FILE BINDING.
 *
 *   rnnoise_wav --in noisy48k.wav --out enhanced48k.wav [--proc rnnoise|bypass]
 *
 * The task-9 host oracle. Geometry is RNNoise's own, 960/480 at 48 kHz, and
 * the framing (stream build, binding, drop-and-trim) is abf_wav_run(), shared
 * with the classical oracle. Output stays at 48 kHz: the locked 577-tap
 * downsample to 16 kHz belongs to the METRIC path (eval_harness/audio.py) and
 * is applied there to the reference and the degraded signal alike.
 *
 * `bypass` is the pure-delay reference processor at the same geometry -- the
 * framing regression anchor for this arm, bit-identical to its input after
 * drop-and-trim, and after the metric path's downsample its scores are the
 * refcheck S4 cell (our-filtered noisy vs our-filtered clean).
 */
#include "rnnoise_proc.h"
#ifdef RNNOISE_NN8
#include "rnn_nn8.h"
#endif
#include "abf_wav.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void)
{
    fprintf(stderr, "usage: rnnoise_wav --in IN48k.wav --out OUT48k.wav [--proc rnnoise|bypass]\n");
}

int main(int argc, char **argv)
{
    const char *inp = NULL, *outp = NULL, *which = "rnnoise";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) inp = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else if (!strcmp(argv[i], "--proc") && i + 1 < argc) which = argv[++i];
        else { usage(); return 2; }
    }
    if (!inp || !outp) { usage(); return 2; }
    const abf_proc_t *proc; uint32_t algo;
    if      (!strcmp(which, "rnnoise")) { proc = rnnoise_proc();    algo = ABF_ALGO_RNNOISE; }
    else if (!strcmp(which, "bypass"))  { proc = abf_proc_bypass(); algo = ABF_ALGO_BYPASS; }
    else { usage(); return 2; }

    char variant[24]; snprintf(variant, sizeof variant, "%s-%s", which, rnnoise_proc_version());
    abf_wav_opts_t o = { RNNOISE_FRAME, RNNOISE_HOP, algo, variant, "rnnoise_wav" };
    abf_wav_result_t res; char err[256];
    if (abf_wav_run(inp, outp, proc, &o, &res, err, sizeof err) != 0) { fprintf(stderr, "%s\n", err); return 1; }
    if (res.rate_hz != RNNOISE_RATE_HZ) { fprintf(stderr, "%s: %u Hz; RNNoise needs 48000\n", inp, res.rate_hz); return 1; }
    printf("%s: %s %s %u Hz frame %u hop %u  in %u samples  hops %u+%u  out %s\n",
           inp, proc->name, rnnoise_proc_version(), res.rate_hz, res.frame_len, res.hop_len,
           res.total_samples, res.data_hops, res.flush_hops, outp);
#ifdef RNNOISE_NN8
    /* The +N variant's two counters, so a corpus run can total them: a layer
     * lookup failure is a wrong build (exit 3); a saturated int16 output is a
     * data event the variant's write-up must count rather than assume away. */
    fprintf(stderr, "nn8: errors=%lu saturations=%lu\n",
            (unsigned long)rnnoise_nn8_errors, (unsigned long)rnnoise_nn8_saturations);
    if (rnnoise_nn8_errors) return 3;
#endif
    return 0;
}
