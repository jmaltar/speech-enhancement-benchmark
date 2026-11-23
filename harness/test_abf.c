/**
 * test_abf.c -- round-trip and failure-mode tests for the ABF1 reference codec.
 *
 * Host-side only. Run before either transport binding is written, for the same
 * reason the eval harness was validated against the published "Noisy" row
 * before any algorithm existed: a framing bug found here costs minutes, and the
 * same bug found through a CDC link costs a day of blaming the USB stack.
 */
#include "abf_frame.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; \
    printf("  FAIL %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); \
    printf("\n"); } } while (0)

static uint8_t pkt[ABF_MAX_PACKET * 4];
static abf_decoder_t dec;

/* Feed a whole buffer, collecting packets. Returns how many were delivered. */
static int drain(const uint8_t *p, size_t n, abf_packet_t *out, int max)
{
    int cnt = 0;
    while (n) {
        int got; abf_packet_t q;
        size_t c = abf_decode(&dec, p, n, &q, &got);
        p += c; n -= c;
        if (got) { if (cnt < max) out[cnt] = q; cnt++; }
        else if (c == 0) break;              /* no progress: input exhausted */
    }
    return cnt;
}

int main(void)
{
    /* Zero-initialised, and every dereference below is guarded on delivery:
     * a test that segfaults when the codec regresses reports a crash where it
     * should report which check failed. */
    abf_packet_t got[8] = {{0,0,0,0,NULL}};

    /* ---- 1. CRC known-answer vector -------------------------------------- */
    {
        uint32_t c = abf_crc32_final(abf_crc32(ABF_CRC_INIT, "123456789", 9));
        CHECK(c == 0xCBF43926u, "CRC32(\"123456789\") = %08X, want CBF43926", c);
    }

    /* ---- 2. round trip, every message type ------------------------------- */
    {
        const uint8_t types[] = { ABF_T_HELLO, ABF_T_CONFIG, ABF_T_CONFIG_ACK,
            ABF_T_CONFIG_NAK, ABF_T_DATA, ABF_T_RESULT, ABF_T_FLUSH,
            ABF_T_STATS, ABF_T_ERROR, ABF_T_PING, ABF_T_PONG };
        const uint16_t lens[] = { ABF_HELLO_BYTES, ABF_CONFIG_BYTES, 0, 4,
            512, ABF_TIMING_BYTES + 512, 0, ABF_STATS_BYTES, 4, 0, 0 };
        uint8_t body[ABF_MAX_PAYLOAD];
        for (size_t i = 0; i < sizeof(body); i++) body[i] = (uint8_t)(i * 7 + 3);

        for (size_t i = 0; i < sizeof(types); i++) {
            int n = abf_encode(pkt, sizeof pkt, types[i], (uint32_t)i * 11u,
                               body, lens[i]);
            CHECK(n == (int)(ABF_OVERHEAD_BYTES + lens[i]),
                  "type %02X encode returned %d", types[i], n);
            abf_decoder_init(&dec);
            int k = drain(pkt, (size_t)n, got, 8);
            CHECK(k == 1, "type %02X decoded %d packets", types[i], k);
            if (k == 1) {
                CHECK(got[0].type == types[i], "type mismatch");
                CHECK(got[0].seq == (uint32_t)i * 11u, "seq mismatch");
                CHECK(got[0].payload_len == lens[i], "len mismatch");
                CHECK(lens[i] == 0 ||
                      (got[0].payload_len == lens[i] && got[0].payload &&
                       memcmp(got[0].payload, body, lens[i]) == 0),
                      "payload mismatch on type %02X", types[i]);
            }
        }
    }

    /* ---- 3. body structs survive memcpy round trip ------------------------ */
    {
        abf_config_t c = {0}, back = {0};
        c.rate_hz = 16000; c.frame_len = 512; c.hop_len = 256;
        c.channels = 1; c.sample_fmt = ABF_FMT_S16LE;
        c.flags = ABF_CF_FRAME_PACED; c.algo_id = 0;
        strcpy(c.variant_id, "dtln-int8-prune50");
        strcpy(c.run_id, "run-20260912-0001");

        uint8_t body[ABF_CONFIG_BYTES];
        memcpy(body, &c, sizeof c);
        int n = abf_encode(pkt, sizeof pkt, ABF_T_CONFIG, 0, body, sizeof body);
        abf_decoder_init(&dec);
        int k = drain(pkt, (size_t)n, got, 8);
        CHECK(k == 1, "config decoded %d", k);
        if (k == 1 && got[0].payload) memcpy(&back, got[0].payload, sizeof back);
        CHECK(back.rate_hz == 16000 && back.frame_len == 512 &&
              back.hop_len == 256 && back.channels == 1 &&
              back.flags == ABF_CF_FRAME_PACED, "config fields corrupted");
        CHECK(strcmp(back.variant_id, "dtln-int8-prune50") == 0, "variant lost");
        CHECK(strcmp(back.run_id, "run-20260912-0001") == 0, "run_id lost");
    }

    /* ---- 4. byte-at-a-time feed: the streaming property ------------------- */
    {
        int n = abf_encode(pkt, sizeof pkt, ABF_T_DATA, 42, pkt + 2048, 512);
        abf_decoder_init(&dec);
        int delivered = 0;
        for (int i = 0; i < n; i++) {
            int g; abf_packet_t q;
            abf_decode(&dec, pkt + i, 1, &q, &g);
            if (g) { delivered++; CHECK(q.seq == 42, "seq after byte feed"); }
        }
        CHECK(delivered == 1, "byte-at-a-time delivered %d", delivered);
        CHECK(dec.resyncs == 0, "byte feed caused %u resyncs", dec.resyncs);
    }

    /* ---- 5. several packets in one chunk --------------------------------- */
    {
        size_t off = 0;
        for (uint32_t i = 0; i < 4; i++)
            off += (size_t)abf_encode(pkt + off, sizeof pkt - off,
                                      ABF_T_DATA, i, pkt + 3000, 256);
        abf_decoder_init(&dec);
        int k = drain(pkt, off, got, 8);
        CHECK(k == 4, "back-to-back delivered %d of 4", k);
        CHECK(got[3].seq == 3, "last seq = %u", got[3].seq);
        CHECK(dec.resyncs == 0 && dec.crc_errors == 0, "clean stream unclean");
    }

    /* ---- 6. garbage prefix: recover, and count ONE resync ----------------- */
    {
        uint8_t junk[71];
        for (size_t i = 0; i < sizeof junk; i++) junk[i] = (uint8_t)(i * 13 + 1);
        memcpy(pkt, junk, sizeof junk);
        int n = abf_encode(pkt + sizeof junk, sizeof pkt - sizeof junk,
                           ABF_T_DATA, 7, pkt + 3000, 256);
        abf_decoder_init(&dec);
        int k = drain(pkt, sizeof junk + (size_t)n, got, 8);
        CHECK(k == 1, "after garbage delivered %d", k);
        CHECK(got[0].seq == 7, "seq after garbage");
        CHECK(dec.resyncs == 1, "resyncs = %u, want 1 event", dec.resyncs);
    }

    /* ---- 7. corrupted payload: not delivered, counted -------------------- */
    {
        int n = abf_encode(pkt, sizeof pkt, ABF_T_DATA, 9, pkt + 3000, 256);
        pkt[ABF_HDR_BYTES + 100] ^= 0x01u;
        abf_decoder_init(&dec);
        int k = drain(pkt, (size_t)n, got, 8);
        CHECK(k == 0, "corrupted packet was delivered");
        CHECK(dec.crc_errors == 1, "crc_errors = %u, want 1", dec.crc_errors);
    }

    /* ---- 8. truncated packet: no false positive -------------------------- */
    {
        int n = abf_encode(pkt, sizeof pkt, ABF_T_DATA, 11, pkt + 3000, 256);
        abf_decoder_init(&dec);
        int k = drain(pkt, (size_t)n - 3u, got, 8);
        CHECK(k == 0, "truncated packet delivered");
        CHECK(dec.crc_errors == 0, "truncation miscounted as CRC error");
    }

    /* ---- 9. magic appearing INSIDE a payload must not desync ------------- */
    {
        uint8_t body[256];
        memset(body, 0xA5, sizeof body);
        abf_put_u32(body + 64, ABF_MAGIC);      /* a plausible-looking lure  */
        body[68] = ABF_T_DATA; body[69] = (uint8_t)ABF_VERSION;
        abf_put_u16(body + 70, 256);
        int n = abf_encode(pkt, sizeof pkt, ABF_T_DATA, 13, body, sizeof body);
        abf_decoder_init(&dec);
        int k = drain(pkt, (size_t)n, got, 8);
        CHECK(k == 1, "embedded magic broke the stream (%d packets)", k);
        CHECK(dec.resyncs == 0, "embedded magic caused %u resyncs", dec.resyncs);
        CHECK(k == 1 && got[0].payload &&
              memcmp(got[0].payload, body, sizeof body) == 0,
              "payload with embedded magic corrupted");
    }

    /* ---- 10. coincidental magic in GARBAGE, with a nonsense header -------
     * Distinct from test 9, which proves an embedded magic inside a valid
     * payload is never examined. Here the decoder IS hunting, and the magic is
     * real -- only the fields behind it are absurd. Without the plausibility
     * check the decoder would believe a garbage length and swallow the head of
     * the genuine packet behind it, which shows up as a CRC error rather than
     * as the clean rejection it should be.
     *
     * Added after a mutation test: deleting abf_header_plausible() failed no
     * test in the first version of this suite. */
    {
        uint8_t junk[96];
        for (size_t i = 0; i < sizeof junk; i++) junk[i] = (uint8_t)(i * 5 + 2);
        abf_put_u32(junk + 40, ABF_MAGIC);
        junk[44] = ABF_T_DATA;
        junk[45] = 0xFFu;                    /* version nobody will ever have */
        abf_put_u16(junk + 46, 2044);        /* plausible length, bogus packet */
        abf_put_u32(junk + 48, 0xDEADBEEFu);

        memcpy(pkt, junk, sizeof junk);
        int n = abf_encode(pkt + sizeof junk, sizeof pkt - sizeof junk,
                           ABF_T_DATA, 21, pkt + 3000, 256);
        abf_decoder_init(&dec);
        int k = drain(pkt, sizeof junk + (size_t)n, got, 8);
        CHECK(k == 1, "false magic in garbage lost the packet (%d delivered)", k);
        CHECK(k == 1 && got[0].seq == 21, "wrong packet recovered");
        CHECK(dec.crc_errors == 0,
              "false header swallowed real bytes: %u CRC errors", dec.crc_errors);
    }

    /* ---- 11. encoder argument checks ------------------------------------- */
    {
        CHECK(abf_encode(pkt, sizeof pkt, ABF_T_DATA, 0, pkt,
                         ABF_MAX_PAYLOAD + 4u) == ABF_E_TOO_BIG, "too-big");
        CHECK(abf_encode(pkt, sizeof pkt, ABF_T_DATA, 0, pkt, 6) == ABF_E_ALIGN,
              "misaligned length accepted");
        CHECK(abf_encode(pkt, 8, ABF_T_DATA, 0, pkt, 256) == ABF_E_SPACE,
              "overflow not caught");
        CHECK(abf_encode(pkt, sizeof pkt, ABF_T_DATA, 0, NULL, 4) == ABF_E_ARG,
              "null payload accepted");
    }

    /* ---- 12. largest packet in scope ------------------------------------- */
    {
        int n = abf_encode(pkt, sizeof pkt, ABF_T_DATA, 99, pkt + 3000,
                           ABF_MAX_PAYLOAD);
        CHECK(n == (int)ABF_MAX_PACKET, "max packet = %d, want %u",
              n, ABF_MAX_PACKET);
        abf_decoder_init(&dec);
        int k = drain(pkt, (size_t)n, got, 8);
        CHECK(k == 1 && got[0].payload_len == ABF_MAX_PAYLOAD, "max packet lost");
    }

    /* ---- 13. stream geometry: the FLUSH decision, as arithmetic ----------
     * Output lags input by (frame_len - hop_len); FLUSH makes up exactly that
     * much and no more. The identity that matters is the last CHECK: after
     * dropping the lag, what remains is hop*data_hops samples, which is always
     * >= total_samples, so trimming yields the original length exactly. If that
     * ever fails, device output stops being comparable with the float oracle. */
    {
        struct { uint32_t frame, hop, expect_lag, expect_flush; } g[] = {
            { 512, 256, 256, 1 },   /* locked primary, 50 % overlap */
            { 256, 128, 128, 1 },   /* NOIZEUS at 8 kHz             */
            { 960, 480, 480, 1 },   /* RNNoise at 48 kHz            */
            { 512, 128, 384, 3 },   /* DTLN native hop, 75 % overlap */
            /* ILLEGAL ratio -- refused by abf_geometry_ok() and never seen in
             * a real run. Present only to pin the rounding-up in
             * abf_flush_hops(), which every legal config leaves unexercised
             * because frame_len is always a whole number of hops. A mutation
             * test found that gap: floor and ceil agree on all four rows
             * above, so rounding down failed nothing. */
            { 500, 300, 200, 1 },
        };
        for (size_t i = 0; i < sizeof g / sizeof g[0]; i++) {
            uint32_t lag = abf_lag_samples(g[i].frame, g[i].hop);
            uint32_t fl  = abf_flush_hops(g[i].frame, g[i].hop);
            CHECK(lag == g[i].expect_lag, "lag %u/%u = %u, want %u",
                  g[i].frame, g[i].hop, lag, g[i].expect_lag);
            CHECK(fl == g[i].expect_flush, "flush hops %u/%u = %u, want %u",
                  g[i].frame, g[i].hop, fl, g[i].expect_flush);
            CHECK(fl * g[i].hop >= lag, "FLUSH underfills the tail");

            /* The length identity holds only for a LEGAL geometry; the
             * illegal row above is present solely to pin the rounding. */
            if (!abf_geometry_ok(g[i].frame, g[i].hop)) continue;

            /* Real utterance lengths, including the 99 %% that are not a whole
             * number of hops, and a 1-sample degenerate case. */
            const uint32_t lens[] = { 1, 255, 256, 257, 19812, 40241 };
            for (size_t j = 0; j < sizeof lens / sizeof lens[0]; j++) {
                uint32_t N  = lens[j];
                uint32_t dh = abf_data_hops(N, g[i].hop);
                uint32_t out = (dh + fl) * g[i].hop;
                CHECK(out >= lag + N, "N=%u: %u out < lag %u + N", N, out, lag);
                CHECK(out - lag == dh * g[i].hop,
                      "N=%u: post-lag length %u != %u", N, out - lag,
                      dh * g[i].hop);
                CHECK(out - lag >= N, "N=%u: trim would extend, not trim", N);
            }
        }
    }

    /* ---- 13b. the COLA precondition is enforced, not assumed ------------- */
    {
        CHECK(abf_geometry_ok(512, 256), "512/256 rejected");
        CHECK(abf_geometry_ok(512, 128), "512/128 rejected");
        CHECK(abf_geometry_ok(960, 480), "960/480 rejected");
        CHECK(abf_geometry_ok(256, 256), "no-overlap 256/256 rejected");
        CHECK(!abf_geometry_ok(500, 300), "non-integer ratio accepted");
        CHECK(!abf_geometry_ok(512, 0),   "zero hop accepted");
        CHECK(!abf_geometry_ok(128, 256), "hop larger than frame accepted");
    }

    /* ---- 14. CONFIG carries the true length ------------------------------ */
    {
        abf_config_t c = {0}, back = {0};
        c.rate_hz = 16000; c.frame_len = 512; c.hop_len = 256;
        c.channels = 1; c.sample_fmt = ABF_FMT_S16LE;
        c.total_samples = 40241;          /* not a multiple of the hop */
        uint8_t body[ABF_CONFIG_BYTES];
        memcpy(body, &c, sizeof c);
        int n = abf_encode(pkt, sizeof pkt, ABF_T_CONFIG, 0, body, sizeof body);
        CHECK(n == (int)(ABF_OVERHEAD_BYTES + ABF_CONFIG_BYTES),
              "config packet = %d", n);
        abf_decoder_init(&dec);
        int k = drain(pkt, (size_t)n, got, 8);
        CHECK(k == 1 && got[0].payload, "config with length not decoded");
        if (k == 1 && got[0].payload) memcpy(&back, got[0].payload, sizeof back);
        CHECK(back.total_samples == 40241, "total_samples = %u",
              back.total_samples);
    }

    /* ---- 15. emit a CRC vector for the host-side cross-check ------------- */
    {
        uint8_t body[512];
        for (size_t i = 0; i < sizeof body; i++) body[i] = (uint8_t)(i & 0xFF);
        int n = abf_encode(pkt, sizeof pkt, ABF_T_DATA, 0x01020304u,
                           body, sizeof body);
        printf("VECTOR bytes=%d crc=%08X\n", n,
               abf_get_u32(pkt + ABF_HDR_BYTES + 512));
        printf("VECTOR hdr=");
        for (unsigned i = 0; i < ABF_HDR_BYTES; i++) printf("%02x", pkt[i]);
        printf("\n");
    }

    printf("%d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
