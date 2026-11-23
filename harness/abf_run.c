/**
 * abf_run.c -- CLI for the ABF1 file binding (the RPi5 transport).
 *
 *   abf_run --in u.in.abf --out u.out.abf [--proc bypass|ola] [--measure] [--force]
 *
 * --measure makes the run refuse unless the platform conditions are verified:
 * every online CPU on `performance` and the throttle word clear, the same
 * conditions bench_plat_init() enforces for the probe. Without --measure the
 * run is functional only and is stamped VOID, so a number can never be quoted
 * from a run whose conditions nobody checked.
 */
#define _POSIX_C_SOURCE 200809L
#include "abf_file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void)
{
    fprintf(stderr,
        "usage: abf_run --in FILE --out FILE [--proc bypass|ola]\n"
        "               [--measure] [--force] [--platform NAME] [--build ID]\n");
}

int main(int argc, char **argv)
{
    const char *inp = NULL, *outp = NULL, *which = "bypass";
    abf_file_opts_t o = { "RPI5", "unknown", 0, 0 };

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc)            inp = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)      outp = argv[++i];
        else if (!strcmp(argv[i], "--proc") && i + 1 < argc)     which = argv[++i];
        else if (!strcmp(argv[i], "--platform") && i + 1 < argc) o.platform = argv[++i];
        else if (!strcmp(argv[i], "--build") && i + 1 < argc)    o.build_id = argv[++i];
        else if (!strcmp(argv[i], "--measure"))                  o.measure = 1;
        else if (!strcmp(argv[i], "--force"))                    o.force = 1;
        else { usage(); return 2; }
    }
    if (!inp || !outp) { usage(); return 2; }

    const abf_proc_t *proc = !strcmp(which, "ola") ? abf_proc_ola()
                                                   : abf_proc_bypass();

    char err[256] = {0}, why[200] = {0};
    int okc = abf_platform_ok(why, sizeof why);
    printf("platform: %s\n", why);

    uint8_t *in = NULL; size_t inlen = 0;
    if (abf_slurp(inp, &in, &inlen, err, sizeof err)) {
        fprintf(stderr, "abf_run: %s\n", err); return 1;
    }
    printf("input: %s, %zu bytes (resident before the first timestamp)\n",
           inp, inlen);

    uint8_t *out = NULL; size_t outlen = 0;
    abf_file_report_t rep;
    if (abf_file_run(in, inlen, &out, &outlen, proc, &o, &rep,
                     err, sizeof err)) {
        fprintf(stderr, "abf_run: %s\n", err);
        free(in); free(out); return 1;
    }

    FILE *f = fopen(outp, "wb");
    if (!f || fwrite(out, 1, outlen, f) != outlen) {
        fprintf(stderr, "abf_run: cannot write %s\n", outp);
        if (f) fclose(f); free(in); free(out); return 1;
    }
    fclose(f);

    double mean = rep.frames_out ? (double)rep.proc_ticks_sum / rep.frames_out : 0.0;
    printf("proc      : %s\n", proc->name);
    printf("paced     : %s\n", rep.paced ? "yes (deadline-released)"
                                         : "NO -- batch pass, jitter not comparable");
    printf("frames    : %u in, %u out, %llu samples\n", rep.frames_in,
           rep.frames_out, (unsigned long long)rep.samples_out);
    printf("overruns  : %u\n", rep.overruns);
    printf("proc ticks: min %u  max %u  mean %.1f  (ns; tick_hz = 1e9)\n",
           rep.proc_ticks_min, rep.proc_ticks_max, mean);
    printf("integrity : %u resyncs, %u crc errors\n", rep.resyncs, rep.crc_errors);
    printf("output    : %s, %zu bytes\n", outp, outlen);
    printf("STATUS    : %s\n", rep.void_run
        ? "VOID -- conditions unverified or --force; do not quote these numbers"
        : "measurement-grade");
    if (okc < 0 && !rep.void_run) printf("  (note: %s)\n", why);

    free(in); free(out);
    return 0;
}
