/**
 * test_abf_cdc.c -- the CDC binding, proved against the file binding on the host.
 *
 * The claim the format rests on is that ONE frame format with two transport
 * bindings produces ONE byte stream -- "a CDC capture and a file run are
 * diffable against each other". This test is that claim, executed: the same
 * .in.abf is driven through both bindings and the outputs compared packet by
 * packet, everything but the timestamps, which are the only thing that may
 * legitimately differ.
 *
 * The transport primitives are mocked over memory pipes, and deliberately
 * mocked NASTILY: input arrives in 1-, 7- and 64-byte dribbles so packets
 * straddle service calls, and the TX pipe accepts only a few bytes at a time so
 * backpressure is exercised rather than assumed. A binding that only works when
 * a whole packet is available at once would pass a kinder mock and fail on
 * hardware.
 */
#include "abf_cdc.h"
#include "abf_file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; \
    printf("  FAIL %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); \
    printf("\n"); } } while (0)

/* ---- mocked transport ----------------------------------------------------- */
static const uint8_t *rx_src; static size_t rx_len, rx_pos;
static uint8_t tx_sink[1 << 20]; static size_t tx_pos;
static uint32_t rx_grant;      /* bytes the mock will hand over per read  */
static uint32_t tx_grant;      /* bytes the mock will accept per write    */
static uint32_t fake_ticks;

uint32_t abf_cdc_io_read(void *buf, uint32_t n)
{
    uint32_t avail = (uint32_t)(rx_len - rx_pos);
    uint32_t give = n < avail ? n : avail;
    if (give > rx_grant) give = rx_grant;
    memcpy(buf, rx_src + rx_pos, give);
    rx_pos += give;
    return give;
}
/* A real CDC FIFO says "no room" by accepting zero bytes, and a binding must
 * treat that as backpressure rather than as a write that happened. A mock that
 * always accepts something cannot tell the two apart -- with an always-accepts
 * mock, removing the one-packet-in-flight guard from abf_cdc_service() changed
 * nothing and the suite passed. So: every third write is refused outright. */
static uint32_t tx_calls;
uint32_t abf_cdc_io_write(const void *buf, uint32_t n)
{
    if (++tx_calls % 3u == 0u) return 0u;          /* FIFO full */
    uint32_t take = n < tx_grant ? n : tx_grant;
    memcpy(tx_sink + tx_pos, buf, take);
    tx_pos += take;
    return take;
}
void abf_cdc_io_flush(void) {}
int  abf_cdc_io_connected(void) { return 1; }
uint32_t abf_plat_ticks(void) { return fake_ticks += 97u; }
uint32_t abf_plat_tick_hz(void) { return 96000000u; }

/* ---- compare two streams, ignoring the timing block ----------------------- */
typedef struct { uint8_t type; uint32_t seq, plen; const uint8_t *pl; } rec_t;

/* Payloads are COPIED into a caller-supplied arena. A decoded packet's payload
 * points into the decoder's own buffer and is valid only until the next
 * abf_decode() call -- an earlier version of this test stored those pointers,
 * so every record aliased the last packet (and, once collect() returned, a dead
 * stack frame). It compared equal and proved nothing. The header says this in
 * as many words; the test still got it wrong. */
static int collect(const uint8_t *b, size_t n, rec_t *out, int max,
                   uint8_t *arena, size_t arena_cap)
{
    abf_decoder_t d; abf_packet_t p; abf_decoder_init(&d);
    int k = 0; size_t used = 0;
    while (n) {
        int got; size_t c = abf_decode(&d, b, n, &p, &got);
        b += c; n -= c;
        if (got) {
            if (k < max && used + p.payload_len <= arena_cap) {
                out[k].type = p.type; out[k].seq = p.seq;
                out[k].plen = p.payload_len;
                out[k].pl = arena + used;
                if (p.payload_len) memcpy(arena + used, p.payload, p.payload_len);
                used += p.payload_len;
            }
            k++;
        }
        else if (c == 0) break;
    }
    return k;
}

int main(void)
{
    static int16_t sig[40241];
    static rec_t a[4096], b[4096];
    static uint8_t a_store[1 << 20];
    static uint8_t arena_a[1 << 20], arena_b[1 << 20];
    char err[256];

    for (size_t i = 0; i < sizeof sig / sizeof sig[0]; i++)
        sig[i] = (int16_t)(((int)(i * 2654435761u) >> 13) % 20000 - 10000);

    /* Build the input stream once; both bindings get exactly these bytes. */
    const uint32_t N = 40241, FRAME = 512, HOP = 256;
    size_t inlen;
    uint8_t *in;
    {
        uint32_t dh = abf_data_hops(N, HOP);
        size_t cap = (size_t)(dh + 4) * (HOP * 2 + ABF_OVERHEAD_BYTES) + 4096;
        in = (uint8_t *)calloc(cap, 1);
        size_t off = 0;
        abf_config_t c; memset(&c, 0, sizeof c);
        c.rate_hz = 16000; c.frame_len = (uint16_t)FRAME; c.hop_len = (uint16_t)HOP;
        c.channels = 1; c.sample_fmt = ABF_FMT_S16LE; c.total_samples = N;
        uint8_t body[ABF_MAX_PAYLOAD];
        memcpy(body, &c, sizeof c);
        off += (size_t)abf_encode(in + off, cap - off, ABF_T_CONFIG, 0, body, ABF_CONFIG_BYTES);
        int16_t hb[ABF_MAX_PAYLOAD / 2];
        for (uint32_t j = 0; j < dh; j++) {
            memset(hb, 0, HOP * 2);
            uint32_t take = (j * HOP + HOP <= N) ? HOP : (N - j * HOP);
            memcpy(hb, sig + j * HOP, take * 2);
            off += (size_t)abf_encode(in + off, cap - off, ABF_T_DATA, j, hb, (uint16_t)(HOP * 2));
        }
        off += (size_t)abf_encode(in + off, cap - off, ABF_T_FLUSH, dh, NULL, 0);
        inlen = off;
    }

    /* ---- reference: the file binding ------------------------------------- */
    uint8_t *fout = NULL; size_t foutlen = 0;
    abf_file_report_t rep;
    abf_file_opts_t fo = { "TEST", "b", 0, 0 };
    CHECK(abf_file_run(in, inlen, &fout, &foutlen, abf_proc_bypass(), &fo,
                       &rep, err, sizeof err) == 0, "file binding: %s", err);
    memcpy(a_store, fout, foutlen);
    int na = collect(a_store, foutlen, a, 4096, arena_a, sizeof arena_a);

    /* ---- the CDC binding, through a hostile mock -------------------------- */
    const uint32_t dribbles[] = { 1, 7, 64 };
    for (size_t di = 0; di < 3; di++) {
        rx_src = in; rx_len = inlen; rx_pos = 0; tx_pos = 0; fake_ticks = 0; tx_calls = 0;
        rx_grant = dribbles[di];
        tx_grant = (di == 0) ? 3u : (di == 1) ? 17u : 64u;  /* partial writes */

        abf_cdc_ident_t id = { "STM32F411", "testbuild", 0xDEADBEEFu, 96000000u };
        abf_cdc_init(abf_proc_bypass(), &id);

        /* Bounded: a binding that blocks or stalls fails by exhausting this,
         * which is a clearer signal than a hung test. */
        long guard = 40000000L;
        while (guard-- > 0) {
            size_t before_rx = rx_pos, before_tx = tx_pos;
            abf_cdc_service();
            if (rx_pos == before_rx && tx_pos == before_tx && !abf_cdc_busy()
                && rx_pos == rx_len) break;
        }
        CHECK(guard > 0, "dribble %u: binding did not terminate", dribbles[di]);

        int nb = collect(tx_sink, tx_pos, b, 4096, arena_b, sizeof arena_b);
        CHECK(nb == na, "dribble %u: %d packets vs file binding's %d",
              dribbles[di], nb, na);

        /* The file binding emits HELLO too, so the sequences should align 1:1. */
        int mism = 0, tmism = 0;
        int n = (na < nb) ? na : nb;
        for (int i = 0; i < n; i++) {
            if (a[i].type != b[i].type || a[i].seq != b[i].seq
                || a[i].plen != b[i].plen) { mism++; continue; }
            if (a[i].type == ABF_T_RESULT) {
                /* Everything but the 24-byte timing block must match exactly. */
                if (memcmp(a[i].pl + ABF_TIMING_BYTES, b[i].pl + ABF_TIMING_BYTES,
                           a[i].plen - ABF_TIMING_BYTES) != 0) tmism++;
            } else if (a[i].type == ABF_T_CONFIG_ACK) {
                if (memcmp(a[i].pl, b[i].pl, a[i].plen) != 0) mism++;
            }
        }
        CHECK(mism == 0, "dribble %u: %d header/ack mismatches", dribbles[di], mism);
        CHECK(tmism == 0, "dribble %u: %d RESULT payloads differ", dribbles[di], tmism);

        /* And the audio itself is still bit-exact through the CDC path. */
        {
            static int16_t y[1 << 20];
            uint32_t ny = 0;
            for (int i = 0; i < nb; i++)
                if (b[i].type == ABF_T_RESULT) {
                    uint32_t ns = (b[i].plen - ABF_TIMING_BYTES) / 2;
                    memcpy(y + ny, b[i].pl + ABF_TIMING_BYTES, ns * 2);
                    ny += ns;
                }
            uint32_t lag = abf_lag_samples(FRAME, HOP);
            int bad = -1;
            for (uint32_t i = 0; i < N; i++)
                if (y[lag + i] != sig[i]) { bad = (int)i; break; }
            CHECK(bad < 0, "dribble %u: CDC audio differs at sample %d",
                  dribbles[di], bad);
        }
        printf("  dribble rx=%2u tx=%2u : %d packets, streams agree\n",
               dribbles[di], tx_grant, nb);
    }

    /* ---- a hop too big for the FIFO is refused, not truncated ------------- */
    {
        uint8_t s[256]; size_t off = 0;
        abf_config_t c; memset(&c, 0, sizeof c);
        c.rate_hz = 44100; c.frame_len = 1024; c.hop_len = 512;
        c.channels = 2; c.sample_fmt = ABF_FMT_S16LE;  /* 2048 B hop */
        uint8_t body[ABF_MAX_PAYLOAD]; memcpy(body, &c, sizeof c);
        off += (size_t)abf_encode(s, sizeof s, ABF_T_CONFIG, 0, body, ABF_CONFIG_BYTES);
        rx_src = s; rx_len = off; rx_pos = 0; tx_pos = 0;
        rx_grant = 64; tx_grant = 64;
        abf_cdc_ident_t id = { "STM32F411", "t", 0, 96000000u };
        abf_cdc_init(abf_proc_bypass(), &id);
        for (int i = 0; i < 200; i++) abf_cdc_service();
        int nb = collect(tx_sink, tx_pos, b, 4096, arena_b, sizeof arena_b);
        int nak = -1;
        for (int i = 0; i < nb; i++)
            if (b[i].type == ABF_T_CONFIG_NAK) nak = (int)abf_get_u32(b[i].pl);
        CHECK(nak == (int)ABF_NAK_HOP_TOO_BIG,
              "2048 B hop: NAK %d, want %u", nak, ABF_NAK_HOP_TOO_BIG);
    }

    /* ---- PING is answered without a config ------------------------------- */
    {
        uint8_t s[64];
        int n = abf_encode(s, sizeof s, ABF_T_PING, 77, NULL, 0);
        rx_src = s; rx_len = (size_t)n; rx_pos = 0; tx_pos = 0;
        rx_grant = 64; tx_grant = 64;
        abf_cdc_ident_t id = { "STM32F411", "t", 0, 96000000u };
        abf_cdc_init(abf_proc_bypass(), &id);
        for (int i = 0; i < 50; i++) abf_cdc_service();
        int nb = collect(tx_sink, tx_pos, b, 4096, arena_b, sizeof arena_b);
        int pong = 0;
        for (int i = 0; i < nb; i++)
            if (b[i].type == ABF_T_PONG && b[i].seq == 77) pong = 1;
        CHECK(pong, "PING was not answered with a matching PONG");
    }

    free(in); free(fout);
    printf("%d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
