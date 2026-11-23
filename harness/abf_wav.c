/* abf_wav.c -- see abf_wav.h. The body is classical/cdsp_wav.c's, moved. */
#include "abf_wav.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

int abf_wav_read(const char *path, uint8_t **pcm, uint32_t *nsamp, uint32_t *rate)
{
    uint8_t *buf; size_t len; char err[128];
    if (abf_slurp(path, &buf, &len, err, sizeof err) != 0) { fprintf(stderr, "%s: %s\n", path, err); return -1; }
    if (len < 12 || memcmp(buf, "RIFF", 4) || memcmp(buf + 8, "WAVE", 4)) { fprintf(stderr, "%s: not RIFF/WAVE\n", path); free(buf); return -1; }
    size_t i = 12; int have_fmt = 0; uint16_t fmt = 0, ch = 0, bits = 0; uint32_t sr = 0;
    while (i + 8 <= len) {
        const uint8_t *ck = buf + i; uint32_t cl = rd32(ck + 4); size_t body = i + 8;
        if (!memcmp(ck, "fmt ", 4) && cl >= 16) {
            fmt = rd16(buf + body); ch = rd16(buf + body + 2); sr = rd32(buf + body + 4); bits = rd16(buf + body + 14);
            have_fmt = 1;
        } else if (!memcmp(ck, "data", 4)) {
            if (!have_fmt) { fprintf(stderr, "%s: data before fmt\n", path); free(buf); return -1; }
            if (fmt != 1 || ch != 1 || bits != 16) { fprintf(stderr, "%s: need PCM16 mono (fmt %u ch %u bits %u)\n", path, fmt, ch, bits); free(buf); return -1; }
            if (body + cl > len) cl = (uint32_t)(len - body);
            *nsamp = cl / 2u; *rate = sr;
            *pcm = (uint8_t *)malloc(cl ? cl : 1u); memcpy(*pcm, buf + body, cl);
            free(buf); return 0;
        }
        i = body + cl + (cl & 1u);
    }
    fprintf(stderr, "%s: no data chunk\n", path); free(buf); return -1;
}

int abf_wav_write(const char *path, const uint8_t *pcm, uint32_t nsamp, uint32_t rate)
{
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); return -1; }
    uint32_t dlen = nsamp * 2u; uint8_t h[44];
    memcpy(h, "RIFF", 4); uint32_t rl = 36u + dlen; memcpy(h + 4, &rl, 4); memcpy(h + 8, "WAVE", 4);
    memcpy(h + 12, "fmt ", 4); uint32_t fl = 16; memcpy(h + 16, &fl, 4);
    uint16_t one = 1; memcpy(h + 20, &one, 2); memcpy(h + 22, &one, 2);
    memcpy(h + 24, &rate, 4); uint32_t br = rate * 2u; memcpy(h + 28, &br, 4);
    uint16_t ba = 2, bits = 16; memcpy(h + 32, &ba, 2); memcpy(h + 34, &bits, 2);
    memcpy(h + 36, "data", 4); memcpy(h + 40, &dlen, 4);
    int ok = fwrite(h, 1, 44, f) == 44 && (dlen == 0 || fwrite(pcm, 1, dlen, f) == dlen);
    fclose(f); return ok ? 0 : -1;
}

#define FAIL(...) do { snprintf(err, errlen, __VA_ARGS__); rc = -1; goto done; } while (0)

int abf_wav_run(const char *in_path, const char *out_path, const abf_proc_t *proc,
                const abf_wav_opts_t *opts, abf_wav_result_t *res, char *err, size_t errlen)
{
    int rc = 0;
    uint8_t *pcm = NULL, *in = NULL, *out = NULL, *samples = NULL, *hopbuf = NULL;
    uint32_t n, rate;
    if (abf_wav_read(in_path, &pcm, &n, &rate) != 0) { snprintf(err, errlen, "read %s", in_path); return -1; }
    uint32_t frame = opts->frame_len ? opts->frame_len : rate * 32u / 1000u;   /* 32 ms in time */
    uint32_t hop = opts->hop_len ? opts->hop_len : frame / 2u;                 /* 16 ms         */
    if (!abf_geometry_ok(frame, hop)) FAIL("bad geometry %u/%u", frame, hop);

    /* ---- build the input stream exactly as abf_host.py make does ---- */
    uint32_t hb = abf_hop_bytes(hop, 1u, ABF_FMT_S16LE);
    uint32_t dh = abf_data_hops(n, hop);
    size_t cap = (size_t)(dh + 2u) * (hb + ABF_OVERHEAD_BYTES) + 128u;
    in = (uint8_t *)malloc(cap); size_t inlen = 0;
    abf_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.rate_hz = rate; cfg.frame_len = (uint16_t)frame; cfg.hop_len = (uint16_t)hop;
    cfg.channels = 1; cfg.sample_fmt = ABF_FMT_S16LE; cfg.flags = 0; cfg.total_samples = n;
    cfg.algo_id = opts->algo_id;
    strncpy(cfg.variant_id, opts->variant_id ? opts->variant_id : "", ABF_VARIANT_LEN - 1);
    strncpy(cfg.run_id, opts->run_id ? opts->run_id : "abf_wav", ABF_RUN_ID_LEN - 1);
    int r = abf_encode(in + inlen, cap - inlen, ABF_T_CONFIG, 0, &cfg, ABF_CONFIG_BYTES);
    if (r < 0) FAIL("encode CONFIG %d", r);
    inlen += (size_t)r;
    hopbuf = (uint8_t *)calloc(hb, 1);
    for (uint32_t j = 0; j < dh; j++) {
        memset(hopbuf, 0, hb);
        size_t off = (size_t)j * hb, left = (size_t)n * 2u - off;
        memcpy(hopbuf, pcm + off, left < hb ? left : hb);           /* last hop zero-padded */
        r = abf_encode(in + inlen, cap - inlen, ABF_T_DATA, j, hopbuf, (uint16_t)hb);
        if (r < 0) FAIL("encode DATA %d", r);
        inlen += (size_t)r;
    }
    r = abf_encode(in + inlen, cap - inlen, ABF_T_FLUSH, dh, NULL, 0);
    if (r < 0) FAIL("encode FLUSH %d", r);
    inlen += (size_t)r;

    /* ---- run it through the binding ---- */
    abf_file_opts_t o = { "HOST", opts->run_id ? opts->run_id : "abf_wav", 0, 0 };
    size_t outlen = 0; char ferr[256];
    if (abf_file_run(in, inlen, &out, &outlen, proc, &o, &res->rep, ferr, sizeof ferr) != 0)
        FAIL("abf_file_run: %s", ferr);

    /* ---- host half of FLUSH: collect RESULTs, drop the lag, trim ---- */
    uint32_t lag = abf_lag_samples(frame, hop);
    uint32_t fh = abf_flush_hops(frame, hop);
    uint32_t expect = (dh + fh) * hop;
    samples = (uint8_t *)malloc((size_t)expect * 2u + hb); size_t got_bytes = 0; uint32_t nres = 0;
    abf_decoder_t dec; abf_decoder_init(&dec);
    const uint8_t *p = out; size_t left = outlen;
    while (left) {
        abf_packet_t pkt; int got;
        size_t c = abf_decode(&dec, p, left, &pkt, &got);
        p += c; left -= c;
        if (!got) continue;
        if (pkt.type == ABF_T_CONFIG_NAK) FAIL("device refused CONFIG (reason %u)", pkt.payload_len >= 4 ? rd32(pkt.payload) : 0u);
        if (pkt.type == ABF_T_ERROR) FAIL("device ERROR");
        if (pkt.type == ABF_T_RESULT && pkt.payload_len > ABF_TIMING_BYTES) {
            size_t sb = pkt.payload_len - ABF_TIMING_BYTES;
            if (got_bytes + sb <= (size_t)expect * 2u + hb) { memcpy(samples + got_bytes, pkt.payload + ABF_TIMING_BYTES, sb); got_bytes += sb; }
            nres++;
        }
    }
    if (nres != dh + fh) FAIL("RESULT count %u, expected %u", nres, dh + fh);
    size_t L = (size_t)lag * 2u, want = (size_t)n * 2u;
    if (got_bytes < L + want) FAIL("short output: %zu bytes, need %zu", got_bytes, L + want);
    if (abf_wav_write(out_path, samples + L, n, rate) != 0) FAIL("write %s", out_path);
    res->rate_hz = rate; res->frame_len = frame; res->hop_len = hop;
    res->total_samples = n; res->data_hops = dh; res->flush_hops = fh;
done:
    free(pcm); free(in); free(out); free(samples); free(hopbuf);
    return rc;
}
