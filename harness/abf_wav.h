/**
 * abf_wav.h -- WAV in, WAV out, through the ABF1 FILE BINDING.
 *
 * The host float oracles (classical, RNNoise) share this one path so that
 * every oracle frames exactly as every port does, by construction -- the
 * requirement task 7 placed on tasks 8-10. It builds the same CONFIG / DATA /
 * FLUSH stream abf_host.py builds, runs it through abf_file_run() (the binding
 * verified on the RPi5) and performs the host's half of the FLUSH contract with
 * the geometry helpers in abf_frame.h: drop abf_lag_samples(), trim to
 * total_samples. Factored out of classical/cdsp_wav.c on 2026-10-19, unchanged
 * in behaviour (verified bit-identical against the task-8 outputs on disk).
 *
 * 16-bit PCM mono only; anything else is refused, never converted.
 */
#ifndef ABF_WAV_H
#define ABF_WAV_H

#include "abf_file.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Minimal RIFF reader: walks chunks, requires PCM 16-bit mono. *pcm is
 * malloc'd. Returns 0, or -1 with a message on stderr. */
int abf_wav_read(const char *path, uint8_t **pcm, uint32_t *nsamp, uint32_t *rate);

/* 44-byte header + data. Returns 0 or -1. */
int abf_wav_write(const char *path, const uint8_t *pcm, uint32_t nsamp, uint32_t rate);

typedef struct {
    uint32_t    frame_len;   /* 0 = 32 ms in time at the file's rate */
    uint32_t    hop_len;     /* 0 = frame_len / 2                     */
    uint32_t    algo_id;     /* stamped into CONFIG, see the registry */
    const char *variant_id;  /* human-readable identity, <= 23 chars  */
    const char *run_id;      /* tool name, <= 23 chars                */
} abf_wav_opts_t;

typedef struct {
    uint32_t rate_hz, frame_len, hop_len, total_samples, data_hops, flush_hops;
    abf_file_report_t rep;
} abf_wav_result_t;

/* One file end to end. Returns 0, or negative with a message in err. */
int abf_wav_run(const char *in_path, const char *out_path, const abf_proc_t *proc,
                const abf_wav_opts_t *opts, abf_wav_result_t *res,
                char *err, size_t errlen);

#ifdef __cplusplus
}
#endif
#endif
