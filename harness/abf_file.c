#define _POSIX_C_SOURCE 200809L
#include "abf_file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>

/* ---- output buffer -------------------------------------------------------- */
typedef struct { uint8_t *p; size_t len, cap; } obuf_t;

static int ob_reserve(obuf_t *o, size_t n)
{
    if (o->len + n <= o->cap) return 0;
    size_t cap = o->cap ? o->cap * 2 : 65536;
    while (cap < o->len + n) cap *= 2;
    uint8_t *q = (uint8_t *)realloc(o->p, cap);
    if (!q) return -1;
    o->p = q; o->cap = cap;
    return 0;
}

static int ob_pkt(obuf_t *o, uint8_t type, uint32_t seq,
                  const void *payload, uint16_t plen)
{
    if (ob_reserve(o, (size_t)ABF_OVERHEAD_BYTES + plen)) return -1;
    int n = abf_encode(o->p + o->len, o->cap - o->len, type, seq, payload, plen);
    if (n < 0) return -1;
    o->len += (size_t)n;
    return 0;
}

/* ---- clock ---------------------------------------------------------------
 * Ticks are NANOSECONDS here, and HELLO says so via tick_hz. The Pi 5 kernel
 * refuses userspace PMU access, so there is no cycle counter to report; calling
 * this field "cycles" because the MCUs do would make the row a lie. Truncated
 * to 32 bits exactly as the MCU counters are, with the same rule: take
 * differences, never absolute values. u32 nanoseconds wrap every 4.29 s, and
 * every interval of interest is one hop.
 * ------------------------------------------------------------------------- */
static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* ---- platform conditions -------------------------------------------------- */
int abf_platform_ok(char *why, size_t whylen)
{
    DIR *d = opendir("/sys/devices/system/cpu/cpufreq");
    if (!d) { snprintf(why, whylen, "no cpufreq sysfs: not a Pi, or conditions unverifiable"); return 0; }
    closedir(d);

    int checked = 0, bad = 0;
    char path[160], buf[64];
    for (int cpu = 0; cpu < 64; cpu++) {
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpu);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        if (fgets(buf, sizeof buf, f)) {
            checked++;
            buf[strcspn(buf, "\n")] = 0;
            if (strcmp(buf, "performance") != 0) {
                snprintf(why, whylen, "cpu%d governor is '%s', not 'performance'", cpu, buf);
                bad = 1;
            }
        }
        fclose(f);
        if (bad) return -1;
    }
    if (!checked) { snprintf(why, whylen, "no scaling_governor files: conditions unverifiable"); return 0; }

    /* vcgencmd is Pi-only; its absence means we are not on the tier at all. */
    FILE *p = popen("vcgencmd get_throttled 2>/dev/null", "r");
    if (p) {
        char t[64] = {0};
        if (fgets(t, sizeof t, p) && strstr(t, "throttled=")) {
            unsigned long v = strtoul(strstr(t, "=") + 1, NULL, 0);
            if (v != 0ul) {
                snprintf(why, whylen, "throttled=0x%lx before the run", v);
                pclose(p); return -1;
            }
            pclose(p);
            snprintf(why, whylen, "governor performance on %d cpus, throttled=0x0", checked);
            return 1;
        }
        pclose(p);
    }
    snprintf(why, whylen, "governor performance on %d cpus, but no vcgencmd: not the RPi5 tier", checked);
    return 0;
}

int abf_slurp(const char *path, uint8_t **buf, size_t *len,
              char *err, size_t errlen)
{
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "cannot open %s", path); return -1; }
    if (fseek(f, 0, SEEK_END)) { fclose(f); snprintf(err, errlen, "seek failed"); return -1; }
    long n = ftell(f);
    if (n < 0) { fclose(f); snprintf(err, errlen, "tell failed"); return -1; }
    rewind(f);
    uint8_t *p = (uint8_t *)malloc((size_t)n ? (size_t)n : 1u);
    if (!p) { fclose(f); snprintf(err, errlen, "out of memory"); return -1; }
    /* One read of the whole file: this IS the page-cache priming. Nothing is
     * timed until it has returned, so SCP and the SD card stay out of every
     * result. */
    if (n && fread(p, 1, (size_t)n, f) != (size_t)n) {
        free(p); fclose(f); snprintf(err, errlen, "short read"); return -1;
    }
    fclose(f);
    *buf = p; *len = (size_t)n;
    return 0;
}

/* ---- the run -------------------------------------------------------------- */
int abf_file_run(const uint8_t *in, size_t in_len,
                 uint8_t **out_buf, size_t *out_len,
                 const abf_proc_t *proc, const abf_file_opts_t *opts,
                 abf_file_report_t *rep, char *err, size_t errlen)
{
    obuf_t out = {0};
    abf_decoder_t dec;
    abf_packet_t pkt;
    abf_config_t cfg;
    abf_stats_t st;
    int configured = 0, flushed = 0, rc = -1;
    uint32_t hop_samples = 0, hop_bytes = 0, out_seq = 0;
    uint64_t deadline = 0, hop_ns = 0;
    int16_t inbuf[ABF_MAX_PAYLOAD / 2], outbuf[ABF_MAX_PAYLOAD / 2];
    uint8_t body[ABF_MAX_PAYLOAD];

    memset(&cfg, 0, sizeof cfg);
    memset(&st, 0, sizeof st);
    memset(rep, 0, sizeof *rep);
    st.proc_ticks_min = 0xFFFFFFFFu;
    abf_decoder_init(&dec);

    /* Conditions first: a run that cannot be verified is not a measurement. */
    char why[160] = {0};
    int okc = abf_platform_ok(why, sizeof why);
    if (opts->measure && okc != 1) {
        if (!opts->force) {
            snprintf(err, errlen,
                     "refusing to measure: %s (use force to run anyway; output is VOID)", why);
            return -1;
        }
        rep->void_run = 1;
    }
    if (okc < 0 && !opts->force) rep->void_run = 1;
    if (!opts->measure) rep->void_run = 1;

    /* HELLO goes out first, so an .out.abf is self-describing. */
    {
        abf_hello_t h;
        memset(&h, 0, sizeof h);
        h.abf_version   = ABF_VERSION;
        snprintf(h.platform, sizeof h.platform, "%s",
                 opts->platform ? opts->platform : "HOST");
        snprintf(h.build_id, sizeof h.build_id, "%s",
                 opts->build_id ? opts->build_id : "unknown");
        h.image_crc32   = 0u;              /* no firmware image on this tier */
        h.cpu_hz        = 0u;
        h.tick_hz       = 1000000000u;     /* nanoseconds -- see now_ns()    */
        h.max_hop_bytes = ABF_MAX_PAYLOAD;
        memcpy(body, &h, sizeof h);
        if (ob_pkt(&out, ABF_T_HELLO, 0, body, ABF_HELLO_BYTES)) goto oom;
    }

    while (in_len) {
        int got;
        size_t c = abf_decode(&dec, in, in_len, &pkt, &got);
        in += c; in_len -= c;
        if (!got) { if (c == 0) break; else continue; }

        switch (pkt.type) {
        case ABF_T_CONFIG: {
            if (pkt.payload_len != ABF_CONFIG_BYTES) {
                snprintf(err, errlen, "CONFIG is %u bytes, want %u",
                         pkt.payload_len, ABF_CONFIG_BYTES);
                goto done;
            }
            memcpy(&cfg, pkt.payload, sizeof cfg);

            uint32_t nak = 0;
            if (cfg.sample_fmt != ABF_FMT_S16LE)                 nak = ABF_NAK_FORMAT;
            else if (cfg.channels == 0u)                         nak = ABF_NAK_CHANNELS;
            else if (!abf_geometry_ok(cfg.frame_len, cfg.hop_len)) nak = ABF_NAK_FRAME_HOP;
            else if (cfg.rate_hz == 0u)                          nak = ABF_NAK_RATE;
            else {
                hop_bytes = abf_hop_bytes(cfg.hop_len, cfg.channels, cfg.sample_fmt);
                if (hop_bytes == 0u || hop_bytes > ABF_MAX_PAYLOAD)
                    nak = ABF_NAK_HOP_TOO_BIG;
            }
            if (!nak && proc->setup &&
                proc->setup(proc->ctx, cfg.rate_hz, cfg.frame_len,
                            cfg.hop_len, cfg.channels) != 0)
                nak = ABF_NAK_NO_ALGO;

            if (nak) {
                uint8_t r[4]; abf_put_u32(r, nak);
                ob_pkt(&out, ABF_T_CONFIG_NAK, pkt.seq, r, 4);
                snprintf(err, errlen, "CONFIG refused, reason %u", nak);
                goto done;
            }
            hop_samples = cfg.hop_len * cfg.channels;
            hop_ns = (uint64_t)cfg.hop_len * 1000000000ull / cfg.rate_hz;
            rep->paced = (cfg.flags & ABF_CF_FRAME_PACED) ? 1 : 0;
            configured = 1;
            /* Echo the config as accepted, so the output stream stands on
             * its own and records what the device agreed to. */
            memcpy(body, &cfg, sizeof cfg);
            if (ob_pkt(&out, ABF_T_CONFIG_ACK, pkt.seq, body, ABF_CONFIG_BYTES))
                goto oom;
            break;
        }

        case ABF_T_DATA: {
            if (!configured) { snprintf(err, errlen, "DATA before CONFIG"); goto done; }
            if (pkt.payload_len != hop_bytes) {
                snprintf(err, errlen, "DATA %u bytes, want %u", pkt.payload_len, hop_bytes);
                goto done;
            }
            memcpy(inbuf, pkt.payload, hop_bytes);

            /* Pacing. Reading as fast as the file allows is a batch pass; with
             * ABF_CF_FRAME_PACED each hop is released on a real deadline so an
             * overrun here means what it means on an MCU. */
            if (rep->paced) {
                if (deadline == 0ull) deadline = now_ns() + hop_ns;
                else {
                    struct timespec ts;
                    ts.tv_sec  = (time_t)(deadline / 1000000000ull);
                    ts.tv_nsec = (long)(deadline % 1000000000ull);
                    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);
                }
            }

            uint64_t t_rx = now_ns();
            uint64_t t0 = now_ns();
            if (proc->process(proc->ctx, inbuf, outbuf, hop_samples) != 0) {
                snprintf(err, errlen, "processor failed at seq %u", pkt.seq);
                goto done;
            }
            uint64_t t1 = now_ns();
            uint32_t dt = (uint32_t)(t1 - t0);
            if (dt < st.proc_ticks_min) st.proc_ticks_min = dt;
            if (dt > st.proc_ticks_max) st.proc_ticks_max = dt;
            st.proc_ticks_sum += dt;

            if (rep->paced) {
                deadline += hop_ns;
                if (t1 > deadline) st.overruns++;
            }

            abf_timing_t tm;
            tm.t_rx_done    = (uint32_t)t_rx;
            tm.t_proc_start = (uint32_t)t0;
            tm.t_proc_end   = (uint32_t)t1;
            tm.t_tx_start   = (uint32_t)now_ns();
            tm.overruns     = st.overruns;
            tm.reserved     = 0u;
            memcpy(body, &tm, sizeof tm);
            memcpy(body + ABF_TIMING_BYTES, outbuf, hop_bytes);
            if (ob_pkt(&out, ABF_T_RESULT, pkt.seq, body,
                       (uint16_t)(ABF_TIMING_BYTES + hop_bytes))) goto oom;
            out_seq = pkt.seq + 1u;
            st.frames_in++; st.frames_out++;
            st.samples_out += hop_samples;
            break;
        }

        case ABF_T_FLUSH: {
            if (!configured) { snprintf(err, errlen, "FLUSH before CONFIG"); goto done; }
            /* Emit the tail: inject (frame_len - hop_len) samples of silence,
             * i.e. exactly abf_flush_hops() more hops. Nothing is discarded --
             * see FRAME_FORMAT.md section 8. */
            uint32_t extra = abf_flush_hops(cfg.frame_len, cfg.hop_len);
            memset(inbuf, 0, hop_bytes);
            for (uint32_t k = 0; k < extra; k++) {
                uint64_t t0 = now_ns();
                if (proc->process(proc->ctx, inbuf, outbuf, hop_samples) != 0) {
                    snprintf(err, errlen, "processor failed during flush"); goto done;
                }
                uint64_t t1 = now_ns();
                uint32_t dt = (uint32_t)(t1 - t0);
                if (dt < st.proc_ticks_min) st.proc_ticks_min = dt;
                if (dt > st.proc_ticks_max) st.proc_ticks_max = dt;
                st.proc_ticks_sum += dt;

                abf_timing_t tm;
                tm.t_rx_done = (uint32_t)t0; tm.t_proc_start = (uint32_t)t0;
                tm.t_proc_end = (uint32_t)t1; tm.t_tx_start = (uint32_t)now_ns();
                tm.overruns = st.overruns; tm.reserved = 0u;
                memcpy(body, &tm, sizeof tm);
                memcpy(body + ABF_TIMING_BYTES, outbuf, hop_bytes);
                if (ob_pkt(&out, ABF_T_RESULT, out_seq++, body,
                           (uint16_t)(ABF_TIMING_BYTES + hop_bytes))) goto oom;
                st.frames_out++; st.samples_out += hop_samples;
            }
            flushed = 1;
            break;
        }

        default:
            /* HELLO/RESULT/STATS arriving here means the stream is a device
             * transcript, not an input. Refuse rather than interpret. */
            snprintf(err, errlen, "unexpected type 0x%02X in an input stream", pkt.type);
            goto done;
        }
    }

    if (!configured) { snprintf(err, errlen, "stream carried no CONFIG"); goto done; }
    if (!flushed)    { snprintf(err, errlen, "stream ended without FLUSH"); goto done; }

    if (st.proc_ticks_min == 0xFFFFFFFFu) st.proc_ticks_min = 0u;
    st.resyncs = dec.resyncs;
    st.crc_errors = dec.crc_errors;
    memcpy(body, &st, sizeof st);
    if (ob_pkt(&out, ABF_T_STATS, out_seq, body, ABF_STATS_BYTES)) goto oom;
    rc = 0;

done:
    rep->frames_in = st.frames_in; rep->frames_out = st.frames_out;
    rep->overruns = st.overruns; rep->resyncs = dec.resyncs;
    rep->crc_errors = dec.crc_errors;
    rep->proc_ticks_min = st.proc_ticks_min; rep->proc_ticks_max = st.proc_ticks_max;
    rep->proc_ticks_sum = st.proc_ticks_sum; rep->samples_out = st.samples_out;
    if (proc->teardown) proc->teardown(proc->ctx);
    *out_buf = out.p; *out_len = out.len;
    return rc;
oom:
    snprintf(err, errlen, "out of memory");
    free(out.p); *out_buf = NULL; *out_len = 0;
    return -1;
}
