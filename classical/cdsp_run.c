/**
 * cdsp_run.c -- the RPi5 tier's run tool for the classical arm (task 8 / 16).
 *
 *   cdsp_run --in u.in.abf --out u.out.abf [--measure] [--force] [--core N]
 *            [--build ID] [--platform NAME]
 *
 * The same role abf_cdc_service() plays on the MCUs: take an ABF1 stream, pick
 * the processor from CONFIG.algo_id through cdsp_select() -- the one registry
 * every binding uses -- and run it through the file binding. Nothing about the
 * algorithm is chosen on the command line, so a .in.abf made by abf_host.py is
 * the identical experiment on every tier.
 *
 * The operating conditions the probe records are set and WITNESSED here, not
 * assumed from the shell that launched us (trap 5: a probe that reports state
 * rather than capability): pin to one core, ask for SCHED_FIFO and mlockall,
 * and print what each call actually returned. A number from a run whose
 * conditions were not verified is stamped VOID by the binding itself.
 */
#define _GNU_SOURCE
#include "abf_file.h"
#include "cdsp.h"
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

const abf_proc_t *cdsp_select(const abf_config_t *cfg);

static void usage(void)
{
    fprintf(stderr,
        "usage: cdsp_run --in FILE --out FILE [--measure] [--force] [--core N]\n"
        "                [--platform NAME] [--build ID] [--rtprio P]\n");
}

static long read_long(const char *path)
{
    FILE *f = fopen(path, "r"); long v = -1;
    if (f) { if (fscanf(f, "%ld", &v) != 1) v = -1; fclose(f); }
    return v;
}

int main(int argc, char **argv)
{
    const char *inp = NULL, *outp = NULL;
    abf_file_opts_t o = { "RPI5", "unknown", 0, 0 };
    unsigned core = 2; int rtprio = 50;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--in") && i + 1 < argc)            inp = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)      outp = argv[++i];
        else if (!strcmp(argv[i], "--platform") && i + 1 < argc) o.platform = argv[++i];
        else if (!strcmp(argv[i], "--build") && i + 1 < argc)    o.build_id = argv[++i];
        else if (!strcmp(argv[i], "--core") && i + 1 < argc)     core = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rtprio") && i + 1 < argc)   rtprio = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--measure"))                  o.measure = 1;
        else if (!strcmp(argv[i], "--force"))                    o.force = 1;
        else { usage(); return 2; }
    }
    if (!inp || !outp) { usage(); return 2; }

    /* ---- operating conditions: set, then witness ----------------------- */
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    int aff_ok = sched_setaffinity(0, sizeof set, &set) == 0;
    struct sched_param sp; memset(&sp, 0, sizeof sp); sp.sched_priority = rtprio;
    int fifo_ok = rtprio > 0 && sched_setscheduler(0, SCHED_FIFO, &sp) == 0;
    int mlock_ok = mlockall(MCL_CURRENT | MCL_FUTURE) == 0;
    int pol = sched_getscheduler(0);
    char p[160];
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%u/cpufreq/scaling_cur_freq", core);
    long cur_khz = read_long(p);
    snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%u/cpufreq/cpuinfo_max_freq", core);
    long max_khz = read_long(p);
    printf("conditions: core=%u affinity=%s sched_policy=%d(%s) rtprio=%d mlockall=%s "
           "cur_khz=%ld max_khz=%ld\n",
           core, aff_ok ? "ok" : "FAILED", pol,
           pol == SCHED_FIFO ? "SCHED_FIFO" : pol == SCHED_OTHER ? "SCHED_OTHER" : "other",
           fifo_ok ? rtprio : 0, mlock_ok ? "ok" : "FAILED", cur_khz, max_khz);

    char err[256] = {0}, why[200] = {0};
    abf_platform_ok(why, sizeof why);
    printf("platform  : %s\n", why);

    uint8_t *in = NULL; size_t inlen = 0;
    if (abf_slurp(inp, &in, &inlen, err, sizeof err)) {
        fprintf(stderr, "cdsp_run: %s\n", err); return 1;
    }

    /* ---- select the processor from the stream's own CONFIG ------------- */
    abf_decoder_t dec; abf_decoder_init(&dec);
    abf_packet_t pkt; int got = 0; size_t used = 0;
    while (used < inlen && !got) used += abf_decode(&dec, in + used, inlen - used, &pkt, &got);
    if (!got || pkt.type != ABF_T_CONFIG || pkt.payload_len != ABF_CONFIG_BYTES) {
        fprintf(stderr, "cdsp_run: stream does not start with CONFIG\n"); free(in); return 1;
    }
    abf_config_t cfg; memcpy(&cfg, pkt.payload, sizeof cfg);
    const abf_proc_t *proc = cdsp_select(&cfg);
    if (!proc) {
        fprintf(stderr, "cdsp_run: algo_id 0x%08x not in the registry\n", cfg.algo_id);
        free(in); return 1;
    }
    printf("config    : algo_id=0x%04x variant='%.*s' run='%.*s' %u Hz frame %u hop %u flags 0x%04x\n",
           cfg.algo_id, ABF_VARIANT_LEN, cfg.variant_id, ABF_RUN_ID_LEN, cfg.run_id,
           cfg.rate_hz, cfg.frame_len, cfg.hop_len, cfg.flags);

    uint8_t *out = NULL; size_t outlen = 0;
    abf_file_report_t rep;
    if (abf_file_run(in, inlen, &out, &outlen, proc, &o, &rep, err, sizeof err)) {
        fprintf(stderr, "cdsp_run: %s\n", err); free(in); free(out); return 1;
    }
    FILE *f = fopen(outp, "wb");
    if (!f || fwrite(out, 1, outlen, f) != outlen) {
        fprintf(stderr, "cdsp_run: cannot write %s\n", outp);
        if (f) fclose(f);
        free(in); free(out);
        return 1;
    }
    fclose(f);

    double mean = rep.frames_out ? (double)rep.proc_ticks_sum / rep.frames_out : 0.0;
    printf("proc      : %s\n", proc->name);
    printf("paced     : %s\n", rep.paced ? "yes (deadline-released)"
                                         : "NO -- batch pass, jitter not comparable");
    printf("frames    : %u in, %u out\n", rep.frames_in, rep.frames_out);
    printf("overruns  : %u\n", rep.overruns);
    printf("proc ns   : min %u  max %u  mean %.1f\n", rep.proc_ticks_min, rep.proc_ticks_max, mean);
#ifdef CDSP_STAMP_DWT
    {
        extern volatile uint32_t g_cdsp_stage[8], g_cdsp_stage_n;
        static const char *nm[7] = { "shift+window", "forward FFT", "|Y|^2",
            "noise tracker", "snr+gain+apply", "inverse FFT", "ola+lrintf+emit" };
        uint64_t tot = 0;
        for (int i = 0; i < 7; i++) tot += g_cdsp_stage[i];
        printf("stages    : %u hops, ns per hop:", g_cdsp_stage_n);
        for (int i = 0; i < 7; i++)
            printf(" %s=%.0f", nm[i], g_cdsp_stage_n ? (double)g_cdsp_stage[i] / g_cdsp_stage_n : 0.0);
        printf(" total=%.0f\n", g_cdsp_stage_n ? (double)tot / g_cdsp_stage_n : 0.0);
    }
#endif
#ifdef BENCH_RNNOISE_STAMPS
    {   /* rnnoise_run_stamp only (rnnoise/stamps/rnnoise_stamps.h): the RNNoise
         * stage sums of this run, CLOCK_MONOTONIC ns, collected per utterance by
         * abf_batch_file.py --stages-log. Printed after the run, untimed. */
        extern volatile uint64_t g_rn_stage[8];
        extern volatile uint32_t g_rn_stage_n, g_rn_silent_n;
        printf("rn_stages : frames=%u silent=%u ns", g_rn_stage_n, g_rn_silent_n);
        for (int i = 0; i < 8; i++) printf(" s%d=%llu", i, (unsigned long long)g_rn_stage[i]);
        printf("\n");
    }
#endif
    printf("STATUS    : %s\n", rep.void_run
        ? "VOID -- conditions unverified or --force; do not quote these numbers"
        : "measurement-grade");
    free(in); free(out);
    return 0;
}
