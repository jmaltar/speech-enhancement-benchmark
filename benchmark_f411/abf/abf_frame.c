/**
 * abf_frame.c -- reference ABF1 codec: CRC-32, packet encode, streaming decode.
 *
 * Shared by both transport bindings and by the host driver, so that a framing
 * bug is one bug rather than three. No allocation, no blocking, no I/O: the
 * decoder owns a fixed buffer and is fed whatever bytes a transport happened to
 * deliver. That is what lets the device service it from the main loop without
 * the never-block property being an aspiration.
 *
 * Resync is deliberate and counted. USB CDC is hardware-checked and the file
 * binding is reliable, so a CRC failure or a false magic means OUR framing is
 * wrong, not the wire. `resyncs` and `crc_errors` belong in ABF_STATS and must
 * be zero in any run whose numbers are quoted.
 */
#include "abf_frame.h"
#include <string.h>

/* -------------------------------------------------------------------------
 * CRC-32 (IEEE 802.3), nibble-table.
 *
 * The table size is a measured trade, not a default. At the locked
 * configuration the harness moves ~33 KB/s in each direction, so ~66 KB/s is
 * CRC'd. A bitwise implementation costs ~64 ops/byte, which on the F411 at
 * ~0.25 ops/cycle is ~8 % of a 96 MHz core -- unacceptable inside a path whose
 * whole purpose is measuring how much CPU the algorithm uses. This 16-entry
 * table costs 64 B of .rodata and ~8 cycles/byte, i.e. ~0.55 % of the F411.
 * A 256-entry table would halve that again for 1 KB of flash; not worth it
 * until something says otherwise.
 * ------------------------------------------------------------------------- */
static const uint32_t abf_crc_tab[16] = {
    0x00000000u, 0x1DB71064u, 0x3B6E20C8u, 0x26D930ACu,
    0x76DC4190u, 0x6B6B51F4u, 0x4DB26158u, 0x5005713Cu,
    0xEDB88320u, 0xF00F9344u, 0xD6D6A3E8u, 0xCB61B38Cu,
    0x9B64C2B0u, 0x86D3D2D4u, 0xA00AE278u, 0xBDBDF21Cu
};

uint32_t abf_crc32(uint32_t crc, const void *buf, size_t len)
{
    const uint8_t *p = (const uint8_t *)buf;
    while (len--) {
        crc ^= *p++;
        crc = (crc >> 4) ^ abf_crc_tab[crc & 0x0Fu];
        crc = (crc >> 4) ^ abf_crc_tab[crc & 0x0Fu];
    }
    return crc;
}

uint32_t abf_crc32_final(uint32_t crc) { return crc ^ 0xFFFFFFFFu; }

/* -------------------------------------------------------------------------
 * Encode
 * ------------------------------------------------------------------------- */
int abf_encode(uint8_t *out, size_t out_cap, uint8_t type, uint32_t seq,
               const void *payload, uint16_t payload_len)
{
    size_t total = (size_t)ABF_HDR_BYTES + payload_len + ABF_CRC_BYTES;

    if (payload_len > ABF_MAX_PAYLOAD)  return ABF_E_TOO_BIG;
    if (payload_len % 4u != 0u)         return ABF_E_ALIGN;
    if (out_cap < total)                return ABF_E_SPACE;
    if (payload_len && !payload)        return ABF_E_ARG;

    abf_put_u32(out + 0, ABF_MAGIC);
    out[4] = type;
    out[5] = (uint8_t)ABF_VERSION;
    abf_put_u16(out + 6, payload_len);
    abf_put_u32(out + 8, seq);
    if (payload_len) memcpy(out + ABF_HDR_BYTES, payload, payload_len);

    uint32_t crc = abf_crc32(ABF_CRC_INIT, out, ABF_HDR_BYTES + payload_len);
    abf_put_u32(out + ABF_HDR_BYTES + payload_len, abf_crc32_final(crc));

    return (int)total;
}

int abf_encode_inplace(uint8_t *out, size_t out_cap, uint8_t type, uint32_t seq,
                       uint16_t payload_len)
{
    size_t total = (size_t)ABF_HDR_BYTES + payload_len + ABF_CRC_BYTES;

    if (payload_len > ABF_MAX_PAYLOAD)  return ABF_E_TOO_BIG;
    if (payload_len % 4u != 0u)         return ABF_E_ALIGN;
    if (out_cap < total)                return ABF_E_SPACE;

    abf_put_u32(out + 0, ABF_MAGIC);
    out[4] = type;
    out[5] = (uint8_t)ABF_VERSION;
    abf_put_u16(out + 6, payload_len);
    abf_put_u32(out + 8, seq);
    /* payload is already at out + ABF_HDR_BYTES -- deliberately not copied */

    uint32_t crc = abf_crc32(ABF_CRC_INIT, out, ABF_HDR_BYTES + payload_len);
    abf_put_u32(out + ABF_HDR_BYTES + payload_len, abf_crc32_final(crc));
    return (int)total;
}

/* -------------------------------------------------------------------------
 * Streaming decode
 *
 * Contract: consumes from `in` until it either completes one packet (stops
 * there, *got = 1) or exhausts the input (*got = 0). Returns bytes consumed.
 * Call in a loop; the packet's payload points into the decoder's own buffer
 * and stays valid until the next call.
 * ------------------------------------------------------------------------- */
void abf_decoder_init(abf_decoder_t *d)
{
    memset(d, 0, sizeof(*d));
}

/* Drop the first byte and re-examine what is left: the magic we matched was
 * coincidental data, or the packet behind it failed its CRC. One recovery
 * EVENT is counted, not one per discarded byte -- fifty bytes of garbage is
 * one resync, which is what a reader of ABF_STATS needs to know. */
static void abf_shift1(abf_decoder_t *d)
{
    if (d->have > 1u) memmove(d->buf, d->buf + 1, d->have - 1u);
    if (d->have) d->have--;
    if (!d->lost) { d->lost = 1u; d->resyncs++; }
}

static int abf_header_plausible(const uint8_t *b)
{
    uint16_t len = abf_get_u16(b + 6);
    if (b[5] != (uint8_t)ABF_VERSION)    return 0;
    if (len > ABF_MAX_PAYLOAD)           return 0;
    if (len % 4u != 0u)                  return 0;
    if (b[4] == 0u || b[4] > ABF_T_PONG) return 0;
    return 1;
}

/* Copy at most `upto - have` bytes out of the caller's chunk. */
static void abf_fill(abf_decoder_t *d, const uint8_t *in, size_t in_len,
                     size_t *used, size_t upto)
{
    if (d->have >= upto || *used >= in_len) return;
    size_t need = upto - d->have;
    size_t n    = in_len - *used;
    if (n > need) n = need;
    memcpy(d->buf + d->have, in + *used, n);
    d->have += (uint32_t)n;
    *used   += n;
}

size_t abf_decode(abf_decoder_t *d, const uint8_t *in, size_t in_len,
                  abf_packet_t *pkt, int *got)
{
    size_t used = 0;
    *got = 0;

    for (;;) {
        /* 1. A header, before anything is believed about a length. */
        abf_fill(d, in, in_len, &used, ABF_HDR_BYTES);
        if (d->have < ABF_HDR_BYTES) return used;

        /* 2. Is it really a header, or did payload bytes spell the magic? */
        if (abf_get_u32(d->buf) != ABF_MAGIC || !abf_header_plausible(d->buf)) {
            abf_shift1(d);
            continue;
        }

        /* 3. The whole packet. */
        uint16_t plen  = abf_get_u16(d->buf + 6);
        size_t   total = (size_t)ABF_HDR_BYTES + plen + ABF_CRC_BYTES;
        abf_fill(d, in, in_len, &used, total);
        if (d->have < total) return used;

        /* 4. Integrity. A failure here means our framing is wrong, not the
         *    wire -- CDC is hardware-checked and files do not bit-rot in
         *    transit -- so it is counted and the scan resumes one byte on. */
        uint32_t want = abf_get_u32(d->buf + ABF_HDR_BYTES + plen);
        uint32_t have = abf_crc32_final(
            abf_crc32(ABF_CRC_INIT, d->buf, ABF_HDR_BYTES + plen));
        if (want != have) {
            d->crc_errors++;
            abf_shift1(d);
            continue;
        }

        pkt->type        = d->buf[4];
        pkt->version     = d->buf[5];
        pkt->payload_len = plen;
        pkt->seq         = abf_get_u32(d->buf + 8);
        pkt->payload     = plen ? (d->buf + ABF_HDR_BYTES) : NULL;

        d->packets++;
        d->lost = 0u;
        d->have = 0u;      /* payload stays readable until the next call */
        *got = 1;
        return used;
    }
}
