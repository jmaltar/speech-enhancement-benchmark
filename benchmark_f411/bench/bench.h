/**
 * bench.h -- cycle-accurate kernel throughput probe (project task 3).
 *
 * Produces, per kernel: min / max / mean cycles over `reps` invocations, and a
 * derived MOps/s figure. The results live in one RAM struct (`g_bench`) that the
 * debugger reads after the run, and are optionally echoed over SWO/ITM.
 *
 * Two rules this harness exists to enforce:
 *
 *  1. NOTHING does I/O inside a timed region. ITM stalls when the trace FIFO
 *     fills and semihosting halts the core outright; either one silently
 *     corrupts a measurement. Printing happens after every kernel has run.
 *
 *  2. The measurement configuration is part of the result. CPU clock,
 *     compiler flags, interrupt state and the ops-counting convention are all
 *     recorded in the struct, because a MOps/s figure without them cannot be
 *     compared against anything -- which is precisely the gap in TinyLSTMs'
 *     155 MOps/s STM32F746VE number.
 */
#ifndef BENCH_H
#define BENCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Ops-counting convention.
 *
 * A multiply-accumulate is counted as TWO ops (one multiply, one add). This is
 * the usual flop convention and it MUST match whatever convention the
 * constraint budget table uses -- getting it wrong is a clean factor of two on
 * every comparison against published figures. TinyLSTMs does not state its
 * convention, so record ours and say so when citing their number.
 * ------------------------------------------------------------------------- */
#define BENCH_OPS_PER_MAC   2u

/* A kernel that declares ops == 0 is reported as cycles-per-element only.
 * Transcendentals (expf, logf, sqrtf) have no meaningful "op" count, and
 * folding them into a MOps/s average would flatter or wreck it arbitrarily. */
#define BENCH_OPS_NONE      0u

/* 13 plain-C kernels + 12 CMSIS-DSP pairs = 25. Headroom left for the CMSIS-NN
 * int8 rows and the H723 port rather than re-tuning this on every addition. */
#define BENCH_MAX_KERNELS   32u
#define BENCH_MAGIC         0x42454E31u   /* "BEN1" */
#define BENCH_NAME_LEN      24u

typedef struct {
    char     name[BENCH_NAME_LEN];
    uint32_t n;          /* problem size (points, or rows*cols for matvec)   */
    uint32_t ops;        /* declared ops per single invocation, or 0         */
    uint32_t elems;      /* elements per invocation (for cycles/element)     */
    uint32_t reps;       /* invocations actually timed                       */
    uint32_t cyc_min;
    uint32_t cyc_max;
    uint64_t cyc_sum;
    uint32_t checksum;   /* full output checksum -- computed ONCE, UNTIMED   */
    uint32_t guard;      /* O(1) value returned per rep; the DCE tripwire    */
    uint8_t  valid;
} bench_result_t;

typedef struct {
    uint32_t magic;          /* BENCH_MAGIC once populated                   */
    uint32_t done;           /* 1 after the full suite has run               */
    uint32_t dwt_ok;         /* 0 => CYCCNT never incremented; results void  */
    uint32_t cpu_hz;         /* SystemCoreClock at run time                  */
    uint32_t overhead_cyc;   /* measured cost of an empty measurement        */
    uint32_t irq_disabled;   /* 1 => timed regions ran with PRIMASK set      */
    uint32_t n_results;
    bench_result_t r[BENCH_MAX_KERNELS];
} bench_report_t;

extern bench_report_t g_bench;

/* --- cycle counter ------------------------------------------------------ */
int      bench_dwt_init(void);      /* returns 1 if CYCCNT verified running  */
uint32_t bench_dwt_overhead(void);  /* cycles consumed by an empty measure   */

/* --- measurement -------------------------------------------------------- */
typedef uint32_t (*bench_fn_t)(void);   /* the kernel; returns an O(1) guard */
typedef void     (*bench_prep_t)(void); /* re-seeds inputs, never timed      */
typedef uint32_t (*bench_chk_t)(void);  /* full output checksum, never timed */

/**
 * Time `fn` over `reps` invocations with interrupts masked.
 *
 * Stopping -O2 from deleting a pure computation and checking that the
 * computation was correct are two different jobs, and only the first one may
 * touch a timed region:
 *
 *   `fn`  must return an O(1) value derived from its output -- one element is
 *         enough. The harness accumulates it into `guard`, which makes the
 *         kernel observable and so unremovable, at a cost of a cycle or two.
 *
 *   `chk` (may be NULL) returns a checksum over the WHOLE output and is called
 *         exactly once, after the untimed warm-up. It is never timed.
 *
 * These were previously one and the same: each kernel ended with a full O(n)
 * checksum pass that fell inside the timed window. That pass costs ~8 cycles
 * per element on a Cortex-M4F, which is the same order as the cheap kernels
 * themselves -- it inflated window_f32_512 by ~50%%, and sqrtf/mag_257 by
 * ~25%%, while leaving the matvec and FFT rows almost untouched. A cost model
 * must not be paying for its own self-test.
 */
/* `prep` (may be NULL) runs before each timed repetition and OUTSIDE the timed
 * window. In-place kernels such as the FFT need it: without re-seeding, every
 * repetition after the first transforms the previous repetition's output, and
 * the values escalate until the FPU is grinding through infinities and NaNs
 * instead of the signal the kernel is supposed to be costed on. */
void bench_measure(const char *name, uint32_t n, uint32_t ops,
                   uint32_t elems, uint32_t reps,
                   bench_prep_t prep, bench_fn_t fn, bench_chk_t chk);

/* --- kernels ------------------------------------------------------------ */
void bench_kernels_init(void);      /* fills weight/signal buffers           */
void bench_run_all(void);           /* the whole suite, then reports         */

/* --- SWO/ITM output (never called from inside a timed region) ----------- */
void bench_itm_puts(const char *s);
void bench_itm_report(void);

#ifdef __cplusplus
}
#endif
#endif /* BENCH_H */
