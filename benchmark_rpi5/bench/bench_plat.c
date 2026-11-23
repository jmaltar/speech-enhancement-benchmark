/**
 * bench_plat.c -- read the machine's state back, never trust what asked for it.
 *
 * The rule this file exists to enforce is the one the H723 bring-up produced
 * and the S3 pre-empted: a setting that silently changes what is measured must
 * be OBSERVABLE IN THE RESULT, not merely configured. On the STM32s that meant
 * reading SCB->CCR instead of believing -DBENCH_DCACHE; here it means reading
 * sysfs and the firmware mailbox instead of believing that somebody ran the
 * right shell command earlier.
 */
#define _GNU_SOURCE
#include "bench_plat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include <unistd.h>

bench_plat_t g_plat;

static long read_long(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

static int read_str(const char *path, char *out, size_t n)
{
    FILE *f = fopen(path, "r");
    if (!f) { out[0] = 0; return 0; }
    if (!fgets(out, (int)n, f)) { out[0] = 0; fclose(f); return 0; }
    fclose(f);
    out[strcspn(out, "\n")] = 0;
    return 1;
}

/* The firmware throttle word, via vcgencmd. Bit 0 = under-voltage now, bit 1 =
 * ARM frequency capped now, bit 2 = currently throttled, bit 3 = soft temp
 * limit; bits 16-19 are the same conditions "since boot". Anything non-zero
 * means the numbers from this run are suspect, and the sticky bits mean a run
 * that finished clean can still have been compromised earlier. */
static uint32_t read_throttled(void)
{
    FILE *p = popen("vcgencmd get_throttled 2>/dev/null", "r");
    if (!p) return 0xFFFFFFFFu;
    unsigned v = 0xFFFFFFFFu;
    if (fscanf(p, "throttled=0x%x", &v) != 1) v = 0xFFFFFFFFu;
    pclose(p);
    return (uint32_t)v;
}

static uint32_t cache_bytes(int index)
{
    char path[128], buf[32];
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
    if (!read_str(path, buf, sizeof buf)) return 0;
    unsigned long v = strtoul(buf, NULL, 10);
    if (strchr(buf, 'K')) v *= 1024ul;
    else if (strchr(buf, 'M')) v *= 1024ul * 1024ul;
    return (uint32_t)v;
}

int bench_plat_init(unsigned core)
{
    memset(&g_plat, 0, sizeof g_plat);
    g_plat.magic = BENCH_PLAT_MAGIC;
    g_plat.n_cpus = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);

    /* --- pin to one core, as the S3 pins to core 0 ---------------------- */
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof set, &set) == 0) g_plat.core_id = core;
    else                                             g_plat.core_id = 0xFFFFFFFFu;

    /* --- reduce preemption: SCHED_FIFO and locked pages ------------------ */
    struct sched_param sp = { .sched_priority = 50 };
    sched_setscheduler(0, SCHED_FIFO, &sp);       /* may fail without privs  */
    g_plat.sched_policy = (uint32_t)sched_getscheduler(0);
    struct sched_param got;
    if (sched_getparam(0, &got) == 0) g_plat.sched_prio = (uint32_t)got.sched_priority;
    g_plat.mlock_ok = (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) ? 1u : 0u;

    /* --- frequency and governor: the `ondemand` trap -------------------- */
    char path[160];
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%u/cpufreq/scaling_cur_freq", core);
    long khz = read_long(path);
    g_plat.cpu_hz = (khz > 0) ? (uint32_t)(khz * 1000L) : 0u;
    snprintf(path, sizeof path,
             "/sys/devices/system/cpu/cpu%u/cpufreq/cpuinfo_max_freq", core);
    long mkhz = read_long(path);
    g_plat.cpu_hz_max = (mkhz > 0) ? (uint32_t)(mkhz * 1000L) : 0u;

    read_str("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
             g_plat.governor, sizeof g_plat.governor);

    /* EVERY online CPU, not just ours: the suite is pinned, but a neighbour
     * ramping up and down shares the package power and thermal budget. */
    g_plat.governor_ok = 1u;
    for (unsigned c = 0; c < g_plat.n_cpus; ++c) {
        char g[24];
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%u/cpufreq/scaling_governor", c);
        if (!read_str(path, g, sizeof g) || strcmp(g, "performance") != 0) {
            g_plat.governor_ok = 0u;
        }
    }

    /* --- thermal and throttle, latched before the suite ------------------ */
    g_plat.throttled_before  = read_throttled();
    long t = read_long("/sys/class/thermal/thermal_zone0/temp");
    g_plat.temp_milli_before = (t > 0) ? (uint32_t)t : 0u;

    /* --- cache geometry, for the budget table --------------------------- */
    g_plat.l1d_bytes = cache_bytes(0);
    g_plat.l1i_bytes = cache_bytes(1);
    g_plat.l2_bytes  = cache_bytes(2);
    g_plat.l3_bytes  = cache_bytes(3);

    /* --- ISA features. asimddp is the int8 lever on this core ----------- */
    unsigned long hw = getauxval(AT_HWCAP);
    g_plat.has_asimd   = (hw & HWCAP_ASIMD)   ? 1u : 0u;
    g_plat.has_asimddp = (hw & HWCAP_ASIMDDP) ? 1u : 0u;
#ifdef HWCAP_SVE
    g_plat.has_sve     = (hw & HWCAP_SVE)     ? 1u : 0u;
#endif

    /* Refuse rather than produce numbers that look fine. The clock must be at
     * its maximum AND every governor pinned AND nothing currently throttling. */
    int ok = 1;
    if (!g_plat.governor_ok) ok = 0;
    if (g_plat.cpu_hz == 0u || g_plat.cpu_hz != g_plat.cpu_hz_max) ok = 0;
    if (g_plat.throttled_before != 0u) ok = 0;
    return ok;
}

void bench_plat_finish(void)
{
    g_plat.throttled_after = read_throttled();
    long t = read_long("/sys/class/thermal/thermal_zone0/temp");
    g_plat.temp_milli_after = (t > 0) ? (uint32_t)t : 0u;
}
