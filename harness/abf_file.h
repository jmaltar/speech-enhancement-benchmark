/**
 * abf_file.h -- the RPi5 file binding for the ABF1 harness (task 7).
 *
 * The Pi streams no USB: it exposes no device controller, and a Linux gadget
 * stack measured beside bare-metal TinyUSB would be the "three stacks read as a
 * platform difference" the one-stack rule exists to prevent. So it is fed the
 * SAME byte stream as files staged over SCP. A CDC capture and a file run are
 * therefore diffable against each other.
 *
 * THIS FILE IS TRANSPORT, NOT DSP. It hands `hop_len` samples to a callback and
 * takes `hop_len` back; the STFT, the overlap-add and the lag they create live
 * behind that callback. Anything else would put framing policy in two places.
 *
 * Three things it does that a naive read-the-file loop would get wrong:
 *
 *  1. THE WHOLE INPUT IS RESIDENT BEFORE THE FIRST TIMESTAMP. SCP time enters
 *     no result, and neither does the first read -- otherwise the opening pass
 *     times the SD card rather than the algorithm.
 *
 *  2. READING AS FAST AS THE FILE ALLOWS IS A BATCH PASS, NOT A STREAM. With
 *     ABF_CF_FRAME_PACED each hop is released against a real monotonic
 *     deadline, so an overrun means on this tier what it means on an MCU. A
 *     batch pass yields a legitimate RTF but NOT the MCUs' jitter, and the two
 *     must never share a column.
 *
 *  3. MEASUREMENT IS OPT-IN-VERIFIED, NOT DEFAULT-ASSUMED. abf_file_run()
 *     refuses to produce measurement-grade output unless the platform
 *     conditions were checked. A harness that silently "passes" its own
 *     preconditions on a laptop is the probe-reports-state-not-capability trap
 *     that cost this project a correction on the Pi already.
 */
#ifndef ABF_FILE_H
#define ABF_FILE_H

#include "abf_frame.h"
#include "abf_proc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *platform;     /* goes into HELLO; "RPI5" on the real tier */
    const char *build_id;     /* git short hash or build stamp            */
    int         measure;      /* 1 = refuse unless conditions verified    */
    int         force;        /* 1 = run anyway; output is stamped VOID   */
} abf_file_opts_t;

typedef struct {
    uint32_t frames_in, frames_out, overruns, resyncs, crc_errors;
    uint32_t proc_ticks_min, proc_ticks_max;
    uint64_t proc_ticks_sum;
    uint64_t samples_out;
    int      paced;
    int      void_run;        /* conditions unverified or --force used    */
} abf_file_report_t;

/* Runs one stream end to end. `in` is a complete .in.abf image already in
 * memory; `*out` is malloc'd and must be freed by the caller. Returns 0, or
 * negative with a message in `err`. */
int abf_file_run(const uint8_t *in, size_t in_len,
                 uint8_t **out, size_t *out_len,
                 const abf_proc_t *proc, const abf_file_opts_t *opts,
                 abf_file_report_t *rep, char *err, size_t errlen);

/* Reads a whole file into memory, which is also what primes the page cache.
 * Returns 0 or negative; *buf is malloc'd. */
int abf_slurp(const char *path, uint8_t **buf, size_t *len,
              char *err, size_t errlen);

/* Checks that every online CPU is on `performance`, that the throttle word is
 * clear, and that the clock is at its maximum -- the same conditions
 * bench_plat_init() enforces for the RPi5 probe. Returns 1 if verified, 0 if
 * the checks could not be made (not a Pi / files absent), negative if the
 * machine is demonstrably in the wrong state, with the reason in `why`. */
int abf_platform_ok(char *why, size_t whylen);

#ifdef __cplusplus
}
#endif
#endif /* ABF_FILE_H */
