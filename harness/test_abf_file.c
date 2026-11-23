/**
 * test_abf_file.c -- tests for the RPi5 file binding.
 *
 * The one that matters is BYPASS: a pure delay line, so after the host's
 * drop-and-trim the output must equal the input BIT-EXACTLY. No float, no
 * tolerance, no judgement -- if the lag or flush arithmetic is wrong by one
 * sample, this fails. It is the end-to-end version of the integer identities
 * test_abf.c checks in the abstract.
 */
#include "abf_file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; \
    printf("  FAIL %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); \
    printf("\n"); } } while (0)

#define RATE 16000u
#define FRAME 512u
#define HOP   256u

/* Build a complete .in.abf image for `n` samples of `sig`. */
static uint8_t *build_stream(const int16_t *sig, uint32_t n, uint32_t frame,
                             uint32_t hop, uint16_t flags, size_t *len)
{
    uint32_t dh = abf_data_hops(n, hop);
    size_t cap = (size_t)(dh + 4u) * (hop * 2u + ABF_OVERHEAD_BYTES) + 4096u;
    uint8_t *b = (uint8_t *)calloc(cap, 1);
    size_t off = 0;

    abf_config_t c;
    memset(&c, 0, sizeof c);
    c.rate_hz = RATE; c.frame_len = (uint16_t)frame; c.hop_len = (uint16_t)hop;
    c.channels = 1; c.sample_fmt = ABF_FMT_S16LE; c.flags = flags;
    c.total_samples = n;
    snprintf(c.variant_id, sizeof c.variant_id, "%s", "test");
    snprintf(c.run_id, sizeof c.run_id, "%s", "run-test-0001");
    uint8_t body[ABF_MAX_PAYLOAD];
    memcpy(body, &c, sizeof c);
    off += (size_t)abf_encode(b + off, cap - off, ABF_T_CONFIG, 0,
                              body, ABF_CONFIG_BYTES);

    int16_t hopbuf[ABF_MAX_PAYLOAD / 2];
    for (uint32_t j = 0; j < dh; j++) {
        memset(hopbuf, 0, hop * 2u);                 /* last hop zero-padded */
        uint32_t take = (j * hop + hop <= n) ? hop : (n - j * hop);
        memcpy(hopbuf, sig + j * hop, take * 2u);
        off += (size_t)abf_encode(b + off, cap - off, ABF_T_DATA, j,
                                  hopbuf, (uint16_t)(hop * 2u));
    }
    off += (size_t)abf_encode(b + off, cap - off, ABF_T_FLUSH, dh, NULL, 0);
    *len = off;
    return b;
}

/* Pull every RESULT payload out of an .out.abf, concatenated. */
static int16_t *collect(const uint8_t *out, size_t len, uint32_t *nout,
                        int *n_results, abf_stats_t *stats, int *saw_hello)
{
    static int16_t acc[1 << 20];
    abf_decoder_t d; abf_packet_t p;
    abf_decoder_init(&d);
    *nout = 0; *n_results = 0; *saw_hello = 0;
    memset(stats, 0, sizeof *stats);
    while (len) {
        int got; size_t c = abf_decode(&d, out, len, &p, &got);
        out += c; len -= c;
        if (!got) { if (c == 0) break; else continue; }
        if (p.type == ABF_T_HELLO) *saw_hello = 1;
        else if (p.type == ABF_T_RESULT) {
            uint32_t ns = (uint32_t)(p.payload_len - ABF_TIMING_BYTES) / 2u;
            memcpy(acc + *nout, p.payload + ABF_TIMING_BYTES, ns * 2u);
            *nout += ns; (*n_results)++;
        } else if (p.type == ABF_T_STATS) memcpy(stats, p.payload, sizeof *stats);
    }
    return acc;
}

static int run(const uint8_t *in, size_t inlen, const abf_proc_t *proc,
               uint8_t **out, size_t *outlen, abf_file_report_t *rep, char *err)
{
    abf_file_opts_t o = { "TEST", "testbuild", 0, 0 };
    return abf_file_run(in, inlen, out, outlen, proc, &o, rep, err, 256);
}

int main(void)
{
    static int16_t sig[40241];
    char err[256];
    uint8_t *out = NULL; size_t outlen = 0;
    abf_file_report_t rep; abf_stats_t stats;

    for (size_t i = 0; i < sizeof sig / sizeof sig[0]; i++)
        sig[i] = (int16_t)(((int)(i * 2654435761u) >> 13) % 20000 - 10000);

    /* ---- 1. BYPASS is bit-exact after drop-and-trim ---------------------- */
    {
        const uint32_t N = 40241;              /* not a multiple of the hop */
        size_t inlen; uint8_t *in = build_stream(sig, N, FRAME, HOP, 0, &inlen);
        int rc = run(in, inlen, abf_proc_bypass(), &out, &outlen, &rep, err);
        CHECK(rc == 0, "bypass run failed: %s", err);

        uint32_t nout; int nres, hello;
        int16_t *y = collect(out, outlen, &nout, &nres, &stats, &hello);
        uint32_t lag = abf_lag_samples(FRAME, HOP);
        uint32_t dh  = abf_data_hops(N, HOP);
        uint32_t fh  = abf_flush_hops(FRAME, HOP);

        CHECK(hello == 1, "no HELLO in the output stream");
        CHECK((uint32_t)nres == dh + fh, "%d RESULTs, want %u", nres, dh + fh);
        CHECK(nout == (dh + fh) * HOP, "%u samples, want %u", nout, (dh+fh)*HOP);
        CHECK(nout - lag >= N, "post-lag %u < N %u", nout - lag, N);

        int bad = -1;
        for (uint32_t i = 0; i < N; i++)
            if (y[lag + i] != sig[i]) { bad = (int)i; break; }
        CHECK(bad < 0, "bypass differs at sample %d (%d vs %d)",
              bad, bad >= 0 ? y[lag + bad] : 0, bad >= 0 ? sig[bad] : 0);
        CHECK(stats.frames_in == dh, "stats.frames_in = %u, want %u",
              stats.frames_in, dh);
        CHECK(stats.frames_out == dh + fh, "stats.frames_out = %u", stats.frames_out);
        CHECK(stats.resyncs == 0 && stats.crc_errors == 0, "unclean stream");
        free(in); free(out); out = NULL;
    }

    /* ---- 2. the same at 75 % overlap ------------------------------------- */
    {
        const uint32_t N = 19812, F = 512, H = 128;
        size_t inlen; uint8_t *in = build_stream(sig, N, F, H, 0, &inlen);
        int rc = run(in, inlen, abf_proc_bypass(), &out, &outlen, &rep, err);
        CHECK(rc == 0, "75%% run failed: %s", err);
        uint32_t nout; int nres, hello;
        int16_t *y = collect(out, outlen, &nout, &nres, &stats, &hello);
        uint32_t lag = abf_lag_samples(F, H);
        CHECK((uint32_t)nres == abf_data_hops(N,H) + abf_flush_hops(F,H),
              "75%%: %d RESULTs", nres);
        int bad = -1;
        for (uint32_t i = 0; i < N; i++)
            if (y[lag + i] != sig[i]) { bad = (int)i; break; }
        CHECK(bad < 0, "75%% bypass differs at %d", bad);
        free(in); free(out); out = NULL;
    }

    /* ---- 3. OLA reconstructs through COLA -------------------------------- */
    {
        const uint32_t N = 12345;
        size_t inlen; uint8_t *in = build_stream(sig, N, FRAME, HOP, 0, &inlen);
        int rc = run(in, inlen, abf_proc_ola(), &out, &outlen, &rep, err);
        CHECK(rc == 0, "ola run failed: %s", err);
        uint32_t nout; int nres, hello;
        int16_t *y = collect(out, outlen, &nout, &nres, &stats, &hello);
        uint32_t lag = abf_lag_samples(FRAME, HOP);
        /* The first frame's leading half has no partner frame, so skip one
         * frame at each end -- that region is the pad, not signal. */
        int worst = 0; double sse = 0.0;
        for (uint32_t i = FRAME; i + FRAME < N; i++) {
            int e = abs((int)y[lag + i] - (int)sig[i]);
            if (e > worst) worst = e;
            sse += (double)e * e;
        }
        CHECK(worst <= 1, "OLA reconstruction off by %d LSB (want <= 1)", worst);
        printf("  OLA reconstruction: worst %d LSB, rms %.4f LSB\n",
               worst, sqrt(sse / (N - 2.0 * FRAME)));
        free(in); free(out); out = NULL;
    }

    /* ---- 4. illegal geometry is NAK'd, not approximated ------------------ */
    {
        const uint32_t N = 4096;
        size_t inlen; uint8_t *in = build_stream(sig, N, 500, 300, 0, &inlen);
        int rc = run(in, inlen, abf_proc_bypass(), &out, &outlen, &rep, err);
        CHECK(rc != 0, "illegal 500/300 geometry was accepted");
        abf_decoder_t d; abf_packet_t p; abf_decoder_init(&d);
        int nak = 0; const uint8_t *q = out; size_t l = outlen;
        while (l) { int g; size_t c = abf_decode(&d, q, l, &p, &g); q += c; l -= c;
                    if (g && p.type == ABF_T_CONFIG_NAK)
                        nak = (int)abf_get_u32(p.payload);
                    else if (!g && c == 0) break; }
        CHECK(nak == ABF_NAK_FRAME_HOP, "NAK reason %d, want %d",
              nak, ABF_NAK_FRAME_HOP);
        free(in); free(out); out = NULL;
    }

    /* ---- 5. a truncated stream is refused, not half-processed ------------ */
    {
        const uint32_t N = 4096;
        size_t inlen; uint8_t *in = build_stream(sig, N, FRAME, HOP, 0, &inlen);
        int rc = run(in, inlen - 20u, abf_proc_bypass(), &out, &outlen, &rep, err);
        CHECK(rc != 0, "stream without FLUSH was accepted");
        CHECK(strstr(err, "FLUSH") != NULL, "wrong error: %s", err);
        free(in); free(out); out = NULL;
    }

    /* ---- 6. a device transcript is not an input stream ------------------- */
    {
        uint8_t t[256];
        int n = abf_encode(t, sizeof t, ABF_T_RESULT, 0, t + 128, 32);
        int rc = run(t, (size_t)n, abf_proc_bypass(), &out, &outlen, &rep, err);
        CHECK(rc != 0, "a RESULT was accepted as input");
        free(out); out = NULL;
    }

    /* ---- 7. measurement mode refuses unverified conditions --------------- */
    {
        const uint32_t N = 4096;
        size_t inlen; uint8_t *in = build_stream(sig, N, FRAME, HOP, 0, &inlen);
        abf_file_opts_t o = { "TEST", "testbuild", 1, 0 };   /* measure, no force */
        char why[200];
        int okc = abf_platform_ok(why, sizeof why);
        int rc = abf_file_run(in, inlen, &out, &outlen, abf_proc_bypass(),
                              &o, &rep, err, sizeof err);
        if (okc == 1) {
            CHECK(rc == 0, "verified conditions still refused: %s", err);
            CHECK(rep.void_run == 0, "verified run marked VOID");
            printf("  platform conditions: VERIFIED (%s)\n", why);
        } else {
            CHECK(rc != 0, "measurement ran with unverified conditions");
            CHECK(strstr(err, "refusing to measure") != NULL, "wrong error: %s", err);
            printf("  platform conditions: not verifiable here (%s) -- refusal works\n", why);
        }
        free(in); free(out); out = NULL;
    }

    printf("%d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
