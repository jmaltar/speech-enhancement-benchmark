/**
 * dtln_wav_host.cc -- the DTLN port's processor (dtln/port/dtln_proc.cc) on
 * the host, WAV in / WAV out through the ABF1 file binding, at the port's
 * geometry (512/128 at 16 kHz). Built twice by run_wav.sh: with TFLite
 * Micro's reference CONV_2D / FULLY_CONNECTED registrations, and with the
 * S3 PIE glue (dtln_fc_pie_tflm.cc) over its C dot product. The two must be
 * bit-identical on real utterances; against the Python TFLM oracle both
 * differ by the float32 cdsp_fft_c transform (the board's own gap).
 *
 *   dtln_wav_host --in noisy16k.wav --out enhanced16k.wav
 */
#include "dtln_proc.h"
#include "abf_wav.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *inp = NULL, *outp = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc) inp = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else { fprintf(stderr, "usage: dtln_wav_host --in IN.wav --out OUT.wav\n"); return 2; }
    }
    if (!inp || !outp) { fprintf(stderr, "usage: dtln_wav_host --in IN.wav --out OUT.wav\n"); return 2; }
    abf_wav_opts_t o = { 512u, 128u, ABF_ALGO_DTLN, "dtln-i16-host", "dtln_wav_host" };
    abf_wav_result_t res; char err[256];
    if (abf_wav_run(inp, outp, dtln_proc(), &o, &res, err, sizeof err) != 0) {
        fprintf(stderr, "%s (setup_rc=%u)\n", err, (unsigned)g_dtln.setup_rc); return 1;
    }
    printf("%s: %u Hz frame %u hop %u  in %u samples  hops %u+%u  fc_kernel=%u vec=%u ref=%u xcopies=%u pool=%u B  out %s\n",
           inp, res.rate_hz, res.frame_len, res.hop_len, res.total_samples, res.data_hops, res.flush_hops,
           (unsigned)g_dtln.fc_kernel, (unsigned)g_dtln.fc_calls_vec, (unsigned)g_dtln.fc_calls_ref,
           (unsigned)g_dtln.fc_x_copies, (unsigned)g_dtln.fc_pool_used, outp);
    return 0;
}
