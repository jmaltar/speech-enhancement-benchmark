/**
 * main.c -- benchmark_rpi5 entry point (project task 3, fourth tier).
 *
 * Order matters and mirrors the other three tiers: establish and RECORD the
 * machine state first, refuse to measure if it is wrong, then run the suite
 * with no I/O in it, then print.
 */
#define _GNU_SOURCE
#include "bench.h"
#include "bench_plat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void bench_report(void);

int main(int argc, char **argv)
{
    unsigned core  = 2u;    /* not core 0: it takes most of the IRQ load     */
    int      force = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "--core=", 7)) core = (unsigned)atoi(argv[i] + 7);
        else if (!strcmp(argv[i], "--force")) force = 1;
        else if (!strcmp(argv[i], "--help")) {
            printf("usage: %s [--core=N] [--force]\n"
                   "  --core=N  pin the suite to CPU N (default 2)\n"
                   "  --force   measure even if the machine state is wrong\n"
                   "            (prints the numbers ANYWAY MARKED VOID -- for\n"
                   "             debugging the harness, never for results)\n",
                   argv[0]);
            return 0;
        }
    }

    int ok = bench_plat_init(core);
    if (!ok && !force) {
        fprintf(stderr,
            "refusing to measure: machine state is wrong.\n"
            "  governor_ok=%u (want 1, every CPU on `performance`)\n"
            "  cpu_hz=%u max=%u (want equal)\n"
            "  throttled=0x%08x (want 0)\n"
            "Fix with:\n"
            "  for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; "
            "do echo performance | sudo tee $c >/dev/null; done\n"
            "Then re-run. Use --force only to debug the harness itself.\n",
            g_plat.governor_ok, g_plat.cpu_hz, g_plat.cpu_hz_max,
            g_plat.throttled_before);
        return 2;
    }

    bench_run_all();
    g_plat.pool_addr = (uint64_t)bench_pool_addr();
    bench_plat_finish();
    bench_report();

    if (!ok) printf("*** --force was used: MACHINE STATE WAS WRONG, RESULTS VOID ***\n");
    return 0;
}
