/**
 * abf_cdc.c -- USB CDC binding for the MCU tiers. See abf_cdc.h.
 *
 * INTENDED TO BE BYTE-IDENTICAL ON ALL THREE MCUs. No allocation, no I/O calls
 * that can wait, no floating point, and every buffer static so the footprint is
 * a budget line item rather than a surprise.
 */
#include "abf_cdc.h"
#include <string.h>

#define RXCHUNK 64u    /* one full-speed bulk packet per service call */

typedef enum {
    ST_IDLE = 0,       /* waiting for CONFIG                          */
    ST_RUN,            /* streaming DATA                              */
    ST_FLUSH,          /* emitting the tail                           */
    ST_STATS,          /* STATS still to send                         */
    ST_DONE
} state_t;

static struct {
    const abf_proc_t *proc;
    abf_proc_select_fn sel;
    abf_cdc_ident_t   id;
    abf_decoder_t     dec;
    abf_config_t      cfg;
    abf_stats_t       st;

    /* ONE staging buffer, 4-aligned, and every packet is built directly in it.
     * The obvious shape -- assemble a body, then let abf_encode() copy it in --
     * costs three more kilobyte buffers (a body, an input copy and an output
     * copy) on a part whose effective working budget is 105,008 B. Payloads are
     * read in place out of the decoder, which abf_frame.h aligns for exactly
     * this reason, and results are written in place here. */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    _Alignas(4)
#endif
    uint8_t  tx[ABF_OVERHEAD_BYTES + ABF_DEV_MAX_PAYLOAD];
    uint32_t tx_len, tx_sent;

    /* Bytes read out of the FIFO but not yet consumed by the decoder. A read
     * can straddle a packet boundary, and once the bytes are out of the FIFO
     * they are ours to keep -- an earlier version acted on the first packet in
     * a chunk and dropped the remainder, which a 1-byte-at-a-time mock could
     * not see and a 64-byte one lost half the stream to. */
    uint8_t  pend[RXCHUNK];
    uint32_t pend_len, pend_pos;

    uint32_t hop_bytes, hop_samples, hop_ticks;
    uint32_t out_seq, flush_left;
    /* Decoder counters at CONFIG accept, so STATS reports the RUN's resyncs
     * and CRC errors rather than the connection's running total. */
    uint32_t resyncs0, crc_errors0;
    state_t  state;
    uint8_t  hello_sent;
} g;

/* ---- staging ------------------------------------------------------------- */
static int tx_pending(void) { return g.tx_len > g.tx_sent; }

/* Where a packet's payload is assembled, in the staging buffer itself. */
static uint8_t *tx_payload(void) { return g.tx + ABF_HDR_BYTES; }

/* Seal a packet whose payload is already in place. */
static int seal(uint8_t type, uint32_t seq, uint16_t plen)
{
    if (tx_pending()) return -1;                 /* caller must drain first */
    int n = abf_encode_inplace(g.tx, sizeof g.tx, type, seq, plen);
    if (n < 0) return -1;
    g.tx_len = (uint32_t)n;
    g.tx_sent = 0;
    return 0;
}

static int stage(uint8_t type, uint32_t seq, const void *payload, uint16_t plen)
{
    if (tx_pending()) return -1;
    int n = abf_encode(g.tx, sizeof g.tx, type, seq, payload, plen);
    if (n < 0) return -1;
    g.tx_len = (uint32_t)n;
    g.tx_sent = 0;
    return 0;
}

/* Push whatever the TX FIFO will take. A full FIFO is backpressure, not a
 * reason to wait: we return and the main loop calls us again after tud_task(). */
static void drain(void)
{
    while (tx_pending()) {
        uint32_t n = abf_cdc_io_write(g.tx + g.tx_sent, g.tx_len - g.tx_sent);
        if (n == 0u) return;
        g.tx_sent += n;
    }
    g.tx_len = g.tx_sent = 0;
    abf_cdc_io_flush();
}

/* ---- one processed hop into the staging buffer ---------------------------- */
static void emit_hop(const int16_t *in, uint32_t seq, uint32_t t_rx)
{
    /* Output lands directly in the outgoing packet, after the timing block.
     * tx is 4-aligned and both ABF_HDR_BYTES and ABF_TIMING_BYTES are multiples
     * of 4, so this pointer is safe to use as int16_t* on Xtensa. */
    int16_t *out = (int16_t *)(void *)(tx_payload() + ABF_TIMING_BYTES);

    uint32_t t0 = abf_plat_ticks();
    if (g.proc->process(g.proc->ctx, in, out, g.hop_samples) != 0) {
        uint8_t r[4];
        abf_put_u32(r, 0u);
        stage(ABF_T_ERROR, seq, r, 4);
        g.state = ST_DONE;
        return;
    }
    uint32_t t1 = abf_plat_ticks();
    uint32_t dt = t1 - t0;                  /* unsigned: wraparound is fine */

    if (dt < g.st.proc_ticks_min) g.st.proc_ticks_min = dt;
    if (dt > g.st.proc_ticks_max) g.st.proc_ticks_max = dt;
    g.st.proc_ticks_sum += dt;

    /* An MCU has no scheduler, so the only way a frame can miss real time is
     * by taking longer than its own hop. The Pi's file binding additionally
     * tests completion against a release deadline, because there preemption
     * can miss the deadline even when the compute fits -- the counters mean
     * "this frame missed real time by the tightest test available on this
     * platform", which is not the same test on both. Said plainly rather than
     * left to look identical. */
    if (g.hop_ticks && dt > g.hop_ticks) g.st.overruns++;

    abf_timing_t tm;
    tm.t_rx_done    = t_rx;
    tm.t_proc_start = t0;
    tm.t_proc_end   = t1;
    tm.t_tx_start   = abf_plat_ticks();
    tm.overruns     = g.st.overruns;
    tm.reserved     = 0u;
    memcpy(tx_payload(), &tm, sizeof tm);
    seal(ABF_T_RESULT, seq, (uint16_t)(ABF_TIMING_BYTES + g.hop_bytes));

    g.st.frames_out++;
    g.st.samples_out += g.hop_samples;
}

/* ---- packet handling ------------------------------------------------------ */
static void on_config(const abf_packet_t *p)
{
    uint32_t nak = 0u;

    if (p->payload_len != ABF_CONFIG_BYTES) nak = ABF_NAK_VERSION;
    else {
        memcpy(&g.cfg, p->payload, sizeof g.cfg);
        if (g.cfg.sample_fmt != ABF_FMT_S16LE)                      nak = ABF_NAK_FORMAT;
        else if (g.cfg.channels == 0u)                              nak = ABF_NAK_CHANNELS;
        else if (!abf_geometry_ok(g.cfg.frame_len, g.cfg.hop_len))  nak = ABF_NAK_FRAME_HOP;
        else if (g.cfg.rate_hz == 0u)                               nak = ABF_NAK_RATE;
        else {
            uint32_t hb = abf_hop_bytes(g.cfg.hop_len, g.cfg.channels,
                                        g.cfg.sample_fmt);
            /* The FIFO cap, refused rather than truncated. */
            if (hb == 0u || hb + ABF_TIMING_BYTES > ABF_DEV_MAX_PAYLOAD)
                nak = ABF_NAK_HOP_TOO_BIG;
            else g.hop_bytes = hb;
        }
    }
    if (!nak && g.sel) {
        const abf_proc_t *p = g.sel(&g.cfg);
        if (p) g.proc = p; else nak = ABF_NAK_NO_ALGO;
    }
    if (!nak && g.proc->setup &&
        g.proc->setup(g.proc->ctx, g.cfg.rate_hz, g.cfg.frame_len,
                      g.cfg.hop_len, g.cfg.channels) != 0)
        nak = ABF_NAK_NO_ALGO;

    if (nak) {
        uint8_t r[4];
        abf_put_u32(r, nak);
        stage(ABF_T_CONFIG_NAK, p->seq, r, 4);
        g.state = ST_DONE;
        return;
    }

    g.hop_samples = (uint32_t)g.cfg.hop_len * g.cfg.channels;
    g.hop_ticks   = (uint32_t)(((uint64_t)g.cfg.hop_len * abf_plat_tick_hz())
                               / g.cfg.rate_hz);
    g.out_seq     = 0u;
    /* A CONFIG starts a run, and a session carries many of them -- one per
     * utterance in the matrix -- so the aggregate is per run, not per boot.
     * Proved necessary on hardware: the mock drove one run per process and
     * never saw STATS accumulate across two. */
    memset(&g.st, 0, sizeof g.st);
    g.st.proc_ticks_min = 0xFFFFFFFFu;
    g.resyncs0    = g.dec.resyncs;
    g.crc_errors0 = g.dec.crc_errors;
    g.state       = ST_RUN;
    /* Echo the config as accepted: it makes a capture self-describing and it
     * reports what the device agreed to, not what the host believes it sent. */
    memcpy(tx_payload(), &g.cfg, sizeof g.cfg);
    seal(ABF_T_CONFIG_ACK, p->seq, ABF_CONFIG_BYTES);
}

static void on_packet(const abf_packet_t *p, uint32_t t_rx)
{
    switch (p->type) {
    case ABF_T_PING:
        stage(ABF_T_PONG, p->seq, NULL, 0);
        break;

    case ABF_T_CONFIG:
        on_config(p);
        break;

    case ABF_T_DATA:
        if (g.state != ST_RUN || p->payload_len != g.hop_bytes) {
            uint8_t r[4];
            abf_put_u32(r, 0u);
            stage(ABF_T_ERROR, p->seq, r, 4);
            g.state = ST_DONE;
            break;
        }
        /* Read the payload in place: abf_decoder_t.buf is 4-aligned and the
         * payload starts at a multiple of 4, so no copy is needed and none is
         * made. It stays valid until the next abf_decode() call, which is after
         * emit_hop() returns. */
        g.st.frames_in++;
        emit_hop((const int16_t *)(const void *)p->payload, p->seq, t_rx);
        g.out_seq = p->seq + 1u;
        break;

    case ABF_T_FLUSH:
        if (g.state != ST_RUN) { g.state = ST_DONE; break; }
        g.flush_left = abf_flush_hops(g.cfg.frame_len, g.cfg.hop_len);
        g.state = g.flush_left ? ST_FLUSH : ST_STATS;
        break;

    default:
        /* A device-to-host type arriving here means the peer is confused.
         * Refuse rather than interpret. */
        g.state = ST_DONE;
        break;
    }
}

/* ---- public --------------------------------------------------------------- */
void abf_cdc_init(const abf_proc_t *proc, const abf_cdc_ident_t *id)
{
    memset(&g, 0, sizeof g);
    g.proc = proc;
    g.id = *id;
    g.st.proc_ticks_min = 0xFFFFFFFFu;
    abf_decoder_init(&g.dec);
    g.state = ST_IDLE;
}

void abf_cdc_set_selector(abf_proc_select_fn sel) { g.sel = sel; }
int abf_cdc_busy(void) { return tx_pending(); }
const abf_stats_t *abf_cdc_stats(void) { return &g.st; }

void abf_cdc_service(void)
{
    if (!abf_cdc_io_connected()) {
        /* DTR is down: the host closed the port, or has not opened it yet.
         * Anything in flight belongs to a run that is now void, and the next
         * open must be self-describing from its first byte exactly as the
         * first one was -- so HELLO is re-armed per attach, not per boot. The
         * host driver relies on it: it drops DTR, flushes stale input, raises
         * DTR and expects a fresh HELLO. Found on hardware, where "attach"
         * happens once per run rather than once per process. */
        g.hello_sent = 0u;
        g.tx_len = g.tx_sent = 0u;
        g.pend_len = g.pend_pos = 0u;
        g.state = ST_IDLE;
        abf_decoder_init(&g.dec);
        return;
    }

    drain();
    if (tx_pending()) return;          /* one packet in flight at a time */

    /* HELLO on first service after enumeration, so a capture is self-describing
     * from its first byte. */
    if (!g.hello_sent) {
        abf_hello_t h;
        memset(&h, 0, sizeof h);
        h.abf_version = ABF_VERSION;
        if (g.id.platform)
            strncpy(h.platform, g.id.platform, sizeof h.platform - 1u);
        if (g.id.build_id)
            strncpy(h.build_id, g.id.build_id, sizeof h.build_id - 1u);
        h.image_crc32   = g.id.image_crc32;
        h.cpu_hz        = g.id.cpu_hz;
        h.tick_hz       = abf_plat_tick_hz();
        h.max_hop_bytes = ABF_DEV_MAX_PAYLOAD - ABF_TIMING_BYTES;
        memcpy(tx_payload(), &h, sizeof h);
        if (seal(ABF_T_HELLO, 0, ABF_HELLO_BYTES) == 0)
            g.hello_sent = 1u;
        return;
    }

    if (g.state == ST_FLUSH) {
        /* The flush hops are silence. One zeroed hop is all that is ever
         * needed, and it is const, so it costs .rodata rather than .bss. */
        static const int16_t zeros[ABF_DEV_MAX_PAYLOAD / 2];
        emit_hop(zeros, g.out_seq++, abf_plat_ticks());
        if (--g.flush_left == 0u) g.state = ST_STATS;
        return;
    }

    if (g.state == ST_STATS) {
        if (g.st.proc_ticks_min == 0xFFFFFFFFu) g.st.proc_ticks_min = 0u;
        g.st.resyncs    = g.dec.resyncs    - g.resyncs0;
        g.st.crc_errors = g.dec.crc_errors - g.crc_errors0;
        memcpy(tx_payload(), &g.st, sizeof g.st);
        seal(ABF_T_STATS, g.out_seq, ABF_STATS_BYTES);
        g.state = ST_DONE;
        return;
    }

    /* Refill only when the retained bytes are exhausted: at most one bulk
     * packet of input per call, and at most one decoded packet acted on, so the
     * main loop keeps turning and tud_task() runs. Anything left over is kept
     * and resumed on the next call rather than discarded. */
    if (g.pend_pos >= g.pend_len) {
        g.pend_len = abf_cdc_io_read(g.pend, RXCHUNK);
        g.pend_pos = 0u;
        if (g.pend_len == 0u) return;
    }

    while (g.pend_pos < g.pend_len) {
        int got;
        abf_packet_t pk;
        size_t c = abf_decode(&g.dec, g.pend + g.pend_pos,
                              g.pend_len - g.pend_pos, &pk, &got);
        g.pend_pos += (uint32_t)c;
        if (got) {
            on_packet(&pk, abf_plat_ticks());
            return;                    /* resume from pend on the next call */
        }
        if (c == 0u) break;
    }
}
