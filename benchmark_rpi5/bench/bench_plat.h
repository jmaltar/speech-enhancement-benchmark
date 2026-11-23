/**
 * bench_plat.h -- Raspberry Pi 5 platform configuration and its record.
 *
 * Same role as the H723's and S3's: the knobs that change throughput by large
 * factors on THIS platform, recorded as read back from the machine rather than
 * as asserted by the #define or the shell command that asked for them.
 *
 * This tier's counterparts of the H7's (cache, DTCM-vs-AXI) and the S3's
 * (240-vs-160 MHz, internal-SRAM-vs-PSRAM) are:
 *   - CPU frequency, and specifically the GOVERNOR. The Pi arrived running at
 *     1.70 GHz while configured for 2.40 -- the `ondemand` governor idles the
 *     cores down. That is this platform's version of benchmarking on the HSI
 *     or at the S3's default 160 MHz, and it is the reason cpu_hz and
 *     governor are both recorded and the suite REFUSES to run under anything
 *     but `performance`.
 *   - THERMAL THROTTLING, which has no analogue on the MCUs at all. A
 *     throttled run still produces plausible-looking numbers. The firmware
 *     throttle word is latched either side of the suite and a non-zero value
 *     voids the results the way dwt_ok=0 does.
 *   - WHICH CORE ran, as on the S3 -- four cores and a preemptive scheduler.
 *   - Scheduling policy and memory locking, which are this tier's (weaker)
 *     substitute for masking interrupts. See the irq_disabled note in
 *     bench_kernels.c: they reduce preemption, they do not remove it.
 */
#ifndef BENCH_PLAT_H
#define BENCH_PLAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BENCH_PLAT_MAGIC 0x50493531u   /* "PI51" */

typedef struct {
    uint32_t magic;
    uint32_t cpu_hz;             /* scaling_cur_freq at run time, Hz          */
    uint32_t cpu_hz_max;         /* cpuinfo_max_freq, Hz                      */
    uint32_t governor_ok;        /* 1 iff every online CPU is `performance`   */
    char     governor[24];       /* cpu0's, as read                           */
    uint32_t core_id;            /* which core the suite was pinned to        */
    uint32_t n_cpus;
    uint32_t sched_policy;       /* SCHED_FIFO etc., as read back             */
    uint32_t sched_prio;
    uint32_t mlock_ok;           /* mlockall() succeeded                      */
    uint32_t throttled_before;   /* firmware throttle word, latched           */
    uint32_t throttled_after;
    uint32_t temp_milli_before;  /* thermal_zone0, millidegrees C             */
    uint32_t temp_milli_after;
    uint32_t l1d_bytes, l1i_bytes, l2_bytes, l3_bytes;
    uint32_t has_asimd;          /* NEON                                      */
    uint32_t has_asimddp;        /* SDOT/UDOT -- the int8 lever on this core  */
    uint32_t has_sve;            /* expected 0 on Cortex-A76                  */
    uint64_t pool_addr;          /* witnesses where the weight pool landed    */
} bench_plat_t;

extern bench_plat_t g_plat;

/* Returns 1 if the machine is in a state whose numbers are worth keeping:
 * every CPU on `performance`, not currently throttled, clock at max. Returns 0
 * and fills g_plat anyway otherwise, so the caller can report WHY it refused
 * rather than printing numbers that look fine. */
/* Defined in bench_kernels.c, below the shared plain-C region, so that region
 * stays diff-identical to the other tiers. Witnesses where the pool landed. */
uintptr_t bench_pool_addr(void);

int  bench_plat_init(unsigned core);
void bench_plat_finish(void);     /* latches the after-suite throttle/temp    */

#ifdef __cplusplus
}
#endif
#endif /* BENCH_PLAT_H */
