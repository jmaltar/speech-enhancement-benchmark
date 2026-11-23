/**
 * abf_frame.h -- ABF1 wire format for the audio streaming harness (task 7).
 *
 * ONE frame format, TWO transport bindings: USB CDC on the three MCU tiers,
 * files staged over SCP on the RPi5, which exposes no USB device controller.
 * The byte stream is identical in both cases, so a CDC capture and a file run
 * are diffable against each other and one host-side parser serves both.
 *
 * THIS FILE MUST BE BYTE-IDENTICAL ON EVERY PLATFORM, exactly as bench.h is.
 * Verify with cksum before trusting a cross-platform comparison; a format that
 * has drifted between tiers turns a protocol bug into a platform difference.
 *
 * Six rules this format exists to enforce:
 *
 *  1. THE RUN CONFIGURATION TRAVELS WITH THE RUN. Rate, frame, hop, channels,
 *     sample format, algorithm and compression variant are all declared in
 *     ABF_CONFIG, never compiled in. RNNoise alone is 48 kHz / 20 ms / 10 ms
 *     against the locked 16 kHz / 32 ms / 16 ms, and DTLN's native 8 ms shift
 *     is still undecided -- a hardcoded harness would need rewriting for the
 *     second algorithm, long before compression ever touched it.
 *
 *  2. A CONFIGURATION THE DEVICE CANNOT HONOUR IS REFUSED, NOT APPROXIMATED.
 *     ABF_CONFIG_NAK carries a reason code. Same discipline as the firmware
 *     build guards: a setting that changes what is measured must fail loudly
 *     rather than silently produce numbers.
 *
 *  3. TIMESTAMPS ARE DEVICE-SIDE AND IN TICKS, NOT MICROSECONDS. Each result
 *     carries four raw counter samples, and HELLO declares tick_hz. The MCUs
 *     tick at the CPU clock (DWT CYCCNT / Xtensa CCOUNT); the Pi 5 refuses
 *     userspace PMU access, so its bench_clock.c layer ticks in nanoseconds.
 *     Calling the field "cycles" would make the Pi's row a lie.
 *
 *  4. JITTER IS MEASURED DEVICE-SIDE AND IS THEREFORE UNCONTAMINATED BY THE
 *     TRANSPORT. ABF_STATS reports min/max/sum of processing ticks, computed
 *     where the work happens. Host round-trip timing measures the transport,
 *     which is a different quantity and is decomposed using ABF_PING.
 *
 *  5. THE TAIL IS EMITTED, NOT DISCARDED, AND THE ALIGNMENT IS DECLARED
 *     RATHER THAN IMPLIED. An STFT cannot emit an output hop until every
 *     overlapping frame has contributed, so output lags input by exactly
 *     (frame_len - hop_len) samples. The device pre-fills its analysis buffer
 *     with zeros and FLUSH injects (frame_len - hop_len) more, producing
 *     (frame_len/hop_len - 1) further RESULTs. The host drops the first
 *     (frame_len - hop_len) output samples and trims to CONFIG.total_samples,
 *     leaving output sample-aligned with input and exactly as long.
 *
 *     Discarding the tail instead would lose 16 ms of every utterance -- 0.64 %
 *     of the mean VoiceBank test utterance and 1.29 % of the shortest, at the
 *     end, where word-final energy lives. Worse, it would make device output a
 *     different LENGTH from the PC float reference, and comparing those two is
 *     the entire reason device output exists (task 15). eval_harness raises on
 *     a length mismatch rather than trimming silently, so this would not even
 *     fail quietly.
 *
 *  6. EVERY RUN IS IDENTIFIABLE AFTER THE FACT. HELLO carries platform,
 *     build_id and image_crc32; CONFIG carries run_id and variant_id. The
 *     F411 alone shows +-0.7 % re-link sensitivity, so "which build produced
 *     this" is a measurement input, not bookkeeping.
 */
#ifndef ABF_FRAME_H
#define ABF_FRAME_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABF1 is little-endian on the wire and every target here is little-endian
 * (Cortex-M, Xtensa LX7, Cortex-A76). Break the build rather than emit
 * byte-swapped packets that would decode as plausible garbage. All four
 * toolchains are GCC, so __BYTE_ORDER__ is available. */
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
#  if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#    error "ABF1 is little-endian; this target is not."
#  endif
#endif

/* -------------------------------------------------------------------------
 * Packet layout
 *
 *   +0   u32  magic        'A','B','F','1' in wire order
 *   +4   u8   type         ABF_T_*
 *   +5   u8   version      ABF_VERSION
 *   +6   u16  payload_len  bytes, multiple of 4, <= ABF_MAX_PAYLOAD
 *   +8   u32  seq
 *   +12  u8   payload[payload_len]
 *   +12+N u32 crc32        over bytes [0, 12+N)
 *
 * 16 bytes of overhead. The CRC is a TRAILER, not a header field, so a sender
 * can stream the payload out while computing it and never needs to buffer a
 * whole packet twice.
 *
 * All multi-byte fields are little-endian. All four platforms are LE
 * (Cortex-M, Xtensa LX7, Cortex-A76), which is asserted below rather than
 * assumed.
 *
 * NEVER CAST A RECEIVE BUFFER TO A STRUCT POINTER: Xtensa penalises unaligned
 * 32-bit loads and a packet payload lands wherever the transport put it.
 * `memcpy` into or out of a body struct IS sanctioned -- it handles alignment,
 * and the bodies are _Static_assert'd to carry no padding on a little-endian
 * target, which is the only kind this format runs on. Scalar header fields go
 * through the byte-wise helpers at the bottom of this file.
 * ------------------------------------------------------------------------- */

#define ABF_MAGIC           0x31464241u   /* "ABF1" read as LE u32 */
#define ABF_VERSION         1u
#define ABF_HDR_BYTES       12u
#define ABF_CRC_BYTES       4u
#define ABF_OVERHEAD_BYTES  (ABF_HDR_BYTES + ABF_CRC_BYTES)

/* 2048 admits 44.1 kHz stereo int16 at a 512-sample hop (the MUSDB tier, file
 * binding only). The CDC bindings are tighter -- see ABF_NAK_HOP_TOO_BIG. */
#define ABF_MAX_PAYLOAD     2048u
#define ABF_MAX_PACKET      (ABF_MAX_PAYLOAD + ABF_OVERHEAD_BYTES)

/* -------------------------------------------------------------------------
 * Message types. Direction is fixed per type; a device that receives a
 * device-to-host type has desynchronised and must resync, not interpret it.
 * ------------------------------------------------------------------------- */
#define ABF_T_HELLO         0x01u  /* dev -> host : identity and capabilities */
#define ABF_T_CONFIG        0x02u  /* host -> dev : run parameters            */
#define ABF_T_CONFIG_ACK    0x03u  /* dev -> host : accepted; ECHOES the config
                                    * back as a 72-byte body. Two reasons: it
                                    * makes an .out.abf self-describing, so the
                                    * host can do its drop-and-trim from the
                                    * stream alone; and it reports what the
                                    * device ACCEPTED rather than what the host
                                    * believes it sent -- the protocol version
                                    * of never trusting the #define that asked
                                    * for it. */
#define ABF_T_CONFIG_NAK    0x04u  /* dev -> host : refused, with a reason    */
#define ABF_T_DATA          0x05u  /* host -> dev : one hop of input samples  */
#define ABF_T_RESULT        0x06u  /* dev -> host : one hop out + timing      */
#define ABF_T_FLUSH         0x07u  /* host -> dev : end of stream, drain tail */
/* FLUSH EMITS THE TAIL; NOTHING IS DISCARDED -- decided 2026-09-12, see below */
#define ABF_T_STATS         0x08u  /* dev -> host : end-of-run aggregate      */
#define ABF_T_ERROR         0x09u  /* dev -> host : fault, run is void        */
#define ABF_T_PING          0x0Au  /* host -> dev : transport probe, no work  */
#define ABF_T_PONG          0x0Bu  /* dev -> host : immediate reply to PING   */

/* Sample formats. int16 is the wire default: it is what the locked 16 kHz mono
 * configuration costs 31.25 KiB/s to carry, and what the ports actually run.
 * f32 exists for oracle comparison runs, not for deployment measurement. */
#define ABF_FMT_S16LE       1u
#define ABF_FMT_F32LE       2u

/* CONFIG flags */
#define ABF_CF_FRAME_PACED  0x0001u  /* device must honour the hop deadline and
                                      * count overruns. REQUIRED for any run
                                      * whose jitter is reported: a batch pass
                                      * over a file has no deadline, so its
                                      * spread is not the MCUs' quantity. */
#define ABF_CF_BYPASS       0x0002u  /* passthrough, no algorithm. Measures the
                                      * full path minus compute, which is what
                                      * separates transport cost from kernel
                                      * cost in the end-to-end latency figure. */

/* algo_id registry -- assigned 2026-10-14 with the first algorithm port.
 * Bits 0-7 name the algorithm, bits 8-15 the noise tracker (classical arm
 * only; 0 = the algorithm's default), bits 16-31 are reserved. variant_id
 * remains the human-readable identity ("specsub-mcra", "dtln-int8-prune50").
 * A device selects its processor from this at CONFIG time, so ONE image
 * serves every algorithm it carries and the run configuration genuinely
 * travels with the run (rule 1); an id the image does not carry is NAKed
 * with ABF_NAK_NO_ALGO. The classical values equal cdsp_algo_t /
 * cdsp_noise_est_t in classical/cdsp.h, asserted there. */
#define ABF_ALGO_BYPASS     0u      /* abf_proc_bypass: pure delay, no algorithm */
#define ABF_ALGO_SPECSUB    1u      /* classical: spectral subtraction           */
#define ABF_ALGO_WIENER     2u      /* classical: a priori SNR Wiener            */
#define ABF_ALGO_MMSE       3u      /* classical: MMSE-STSA                      */
#define ABF_ALGO_DTLN       4u      /* DTLN, 16x8 on TFLite Micro (2026-10-21);
                                     * the artifact is the image's, named by
                                     * variant_id; noise bits must be 0       */
#define ABF_ALGO_RNNOISE    5u      /* Xiph RNNoise as shipped (2026-10-22): the
                                     * weights fix 48 kHz, frame 960 / hop 480,
                                     * mono; the release is named in variant_id;
                                     * noise bits must be 0                   */
#define ABF_ALGO_OLA        0xFEu   /* abf_proc_ola: COLA check, no algorithm    */
#define ABF_ALGO_MASK       0xFFu
#define ABF_NOISE_SHIFT     8u
#define ABF_NOISE_DEFAULT   0u      /* the algorithm's default tracker (MCRA)    */
#define ABF_NOISE_VAD       1u
#define ABF_NOISE_MCRA      2u
#define ABF_NOISE_MASK      0xFFu

/* CONFIG_NAK reasons. Refusing beats approximating -- see rule 2. */
#define ABF_NAK_VERSION       1u
#define ABF_NAK_RATE          2u
#define ABF_NAK_FRAME_HOP     3u   /* frame < hop, or COLA violated           */
#define ABF_NAK_CHANNELS      4u
#define ABF_NAK_FORMAT        5u
#define ABF_NAK_HOP_TOO_BIG   6u   /* hop bytes + overhead > transport buffer */
#define ABF_NAK_NO_MEMORY     7u
#define ABF_NAK_NO_ALGO       8u
#define ABF_NAK_BUSY          9u

#define ABF_PLATFORM_LEN    16u
#define ABF_BUILD_ID_LEN    24u
#define ABF_VARIANT_LEN     24u
#define ABF_RUN_ID_LEN      24u

/* -------------------------------------------------------------------------
 * Payload bodies. Sizes are fixed and 4-aligned so the decoder never has to
 * negotiate a length; a body whose declared payload_len does not match is a
 * protocol error, not something to parse defensively.
 * ------------------------------------------------------------------------- */

/* ABF_T_HELLO -- 64 bytes. Sent unsolicited on attach and again on demand. */
typedef struct {
    uint32_t abf_version;
    char     platform[ABF_PLATFORM_LEN];   /* "STM32F411", "RPI5", ...       */
    char     build_id[ABF_BUILD_ID_LEN];   /* git short hash or build stamp  */
    uint32_t image_crc32;                  /* 0 where not applicable (Pi)    */
    uint32_t cpu_hz;
    uint32_t tick_hz;                      /* rate of the RESULT timestamps  */
    uint32_t max_hop_bytes;                /* what this binding can carry    */
    uint32_t reserved;
} abf_hello_t;

/* ABF_T_CONFIG -- 64 bytes. */
typedef struct {
    uint32_t rate_hz;
    uint16_t frame_len;      /* analysis window, samples per channel         */
    uint16_t hop_len;        /* samples per channel in each DATA packet      */
    uint8_t  channels;
    uint8_t  sample_fmt;
    uint16_t flags;
    uint32_t algo_id;
    char     variant_id[ABF_VARIANT_LEN];  /* "dtln-int8-prune50"            */
    char     run_id[ABF_RUN_ID_LEN];       /* host-generated, unique per run */
    uint32_t total_samples;                /* valid input samples per channel;
                                            * 0 = unknown / continuous stream */
    uint32_t reserved;
} abf_config_t;

/* ABF_T_RESULT timing block -- 24 bytes, followed immediately by the output
 * samples. All four stamps are RAW counter samples in device ticks; take
 * differences with unsigned wraparound arithmetic. Every interval of interest
 * is one hop, which is orders below 2^32 even at 2.4 GHz. */
typedef struct {
    uint32_t t_rx_done;      /* last byte of the matching DATA packet in     */
    uint32_t t_proc_start;   /* algorithm entered                            */
    uint32_t t_proc_end;     /* algorithm returned                           */
    uint32_t t_tx_start;     /* first byte of this RESULT queued             */
    uint32_t overruns;       /* cumulative, non-zero means deadlines missed  */
    uint32_t reserved;
} abf_timing_t;

/* ABF_T_STATS -- 48 bytes. The jitter result, computed where the work happened
 * and therefore free of transport noise. */
typedef struct {
    uint32_t frames_in;
    uint32_t frames_out;
    uint32_t overruns;
    uint32_t crc_errors;
    uint32_t resyncs;        /* a resync is a DEFECT here, not routine       */
    uint32_t proc_ticks_min;
    uint32_t proc_ticks_max;
    uint32_t reserved;
    uint64_t proc_ticks_sum;
    uint64_t samples_out;
} abf_stats_t;

#define ABF_HELLO_BYTES     64u
#define ABF_CONFIG_BYTES    72u
#define ABF_TIMING_BYTES    24u
#define ABF_STATS_BYTES     48u

/* -------------------------------------------------------------------------
 * Codec (abf_frame.c) -- shared by both transport bindings and the host
 * driver, so that a framing bug is one bug rather than three. No allocation,
 * no blocking, no I/O.
 * ------------------------------------------------------------------------- */

#define ABF_E_TOO_BIG   (-1)   /* payload_len > ABF_MAX_PAYLOAD            */
#define ABF_E_ALIGN     (-2)   /* payload_len not a multiple of 4          */
#define ABF_E_SPACE     (-3)   /* output buffer too small                  */
#define ABF_E_ARG       (-4)   /* null payload with non-zero length        */

/* One decoded packet. `payload` points into the decoder's own buffer and is
 * valid until the next abf_decode() call on that decoder. */
typedef struct {
    uint8_t        type;
    uint8_t        version;
    uint16_t       payload_len;
    uint32_t       seq;
    const uint8_t *payload;      /* NULL when payload_len == 0 */
} abf_packet_t;

typedef struct {
    /* 4-aligned so a payload, which starts at buf + ABF_HDR_BYTES (also a
     * multiple of 4), can be read in place as 16- or 32-bit samples. Without
     * this a device would have to copy every payload into an aligned buffer
     * before touching it -- a kilobyte of .bss on a part where SRAM is
     * budgeted -- because Xtensa penalises unaligned loads. */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    _Alignas(4)
#endif
    uint8_t  buf[ABF_MAX_PACKET];
    uint32_t have;
    uint32_t packets;
    uint32_t resyncs;            /* recovery EVENTS, not discarded bytes   */
    uint32_t crc_errors;
    uint8_t  lost;               /* currently hunting for a magic          */
} abf_decoder_t;

/* Writes header + payload + CRC into `out`. Returns total bytes, or ABF_E_*. */
int abf_encode(uint8_t *out, size_t out_cap, uint8_t type, uint32_t seq,
               const void *payload, uint16_t payload_len);

void abf_decoder_init(abf_decoder_t *d);

/* As abf_encode(), but the payload is ALREADY in place at out + ABF_HDR_BYTES.
 * Lets a memory-tight sender build a packet once instead of assembling a body
 * and then copying it in -- worth a kilobyte of .bss on the F411. */
int abf_encode_inplace(uint8_t *out, size_t out_cap, uint8_t type, uint32_t seq,
                       uint16_t payload_len);

/* Consumes from `in` until it completes one packet (stops there, *got = 1) or
 * exhausts the input (*got = 0). Returns bytes consumed; call in a loop:
 *
 *   while (n) {
 *       int got; size_t c = abf_decode(&d, p, n, &pkt, &got);
 *       p += c; n -= c;
 *       if (got) handle(&pkt);
 *   }
 */
size_t abf_decode(abf_decoder_t *d, const uint8_t *in, size_t in_len,
                  abf_packet_t *pkt, int *got);

/* -------------------------------------------------------------------------
 * CRC-32, IEEE 802.3: reflected, poly 0xEDB88320, init 0xFFFFFFFF, final XOR
 * 0xFFFFFFFF. Identical to zlib.crc32, so host-side verification is one call
 * and needs no reimplementation to disagree with.
 * ------------------------------------------------------------------------- */
uint32_t abf_crc32(uint32_t crc, const void *buf, size_t len);
uint32_t abf_crc32_final(uint32_t crc);
#define ABF_CRC_INIT  0xFFFFFFFFu

/* -------------------------------------------------------------------------
 * Byte-wise field access. See the layout note above for why these exist.
 * ------------------------------------------------------------------------- */
static inline void abf_put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v); p[1] = (uint8_t)(v >> 8);
}
static inline void abf_put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v);       p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static inline uint16_t abf_get_u16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
static inline uint32_t abf_get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* -------------------------------------------------------------------------
 * Stream geometry. One definition, used by the device, the host driver and the
 * float oracle alike -- if the oracle framed differently from the port, the
 * task 15 comparison would measure the framing rather than the port.
 * ------------------------------------------------------------------------- */

/* COLA, and the precondition the rest of this arithmetic leans on: frame_len
 * must be a whole number of hops. Hann at 50 %% needs hop = frame/2, at 75 %%
 * frame/4; a non-integer ratio breaks COLA, so a CONFIG carrying one is
 * refused with ABF_NAK_FRAME_HOP rather than approximated. The rounding-up in
 * abf_flush_hops() below is therefore unreachable in a legal run -- it is kept
 * so the arithmetic stays correct for anyone who later relaxes this, and test
 * 13 pins it with an illegal ratio for exactly that reason. */
static inline int abf_geometry_ok(uint32_t frame_len, uint32_t hop_len) {
    return hop_len != 0u && frame_len >= hop_len && (frame_len % hop_len) == 0u;
}

/* Output lags input by this many samples, per channel. */
static inline uint32_t abf_lag_samples(uint32_t frame_len, uint32_t hop_len) {
    return (frame_len > hop_len) ? (frame_len - hop_len) : 0u;
}

/* Extra RESULT packets FLUSH produces after the last DATA. */
static inline uint32_t abf_flush_hops(uint32_t frame_len, uint32_t hop_len) {
    return hop_len ? (abf_lag_samples(frame_len, hop_len) + hop_len - 1u) / hop_len
                   : 0u;
}

/* DATA packets the host must send for `total_samples` of real input: always
 * whole hops, the last one zero-padded. 99 % of VoiceBank test utterances are
 * not a whole number of hops, so this is the normal case, not an edge case. */
static inline uint32_t abf_data_hops(uint32_t total_samples, uint32_t hop_len) {
    return hop_len ? (total_samples + hop_len - 1u) / hop_len : 0u;
}

/* Bytes one DATA or RESULT payload carries, excluding the RESULT timing block.
 * Returns 0 for an unknown format, which callers must treat as a NAK. */
static inline uint32_t abf_hop_bytes(uint32_t hop_len, uint32_t channels,
                                     uint32_t sample_fmt) {
    uint32_t w = (sample_fmt == ABF_FMT_S16LE) ? 2u
               : (sample_fmt == ABF_FMT_F32LE) ? 4u : 0u;
    return w ? hop_len * channels * w : 0u;
}

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(abf_hello_t)  == ABF_HELLO_BYTES,  "HELLO wire size");
_Static_assert(sizeof(abf_config_t) == ABF_CONFIG_BYTES, "CONFIG wire size");
_Static_assert(sizeof(abf_timing_t) == ABF_TIMING_BYTES, "RESULT timing wire size");
_Static_assert(sizeof(abf_stats_t)  == ABF_STATS_BYTES,  "STATS wire size");
_Static_assert(ABF_MAX_PAYLOAD % 4u == 0u, "payloads are 4-aligned");
_Static_assert(ABF_HDR_BYTES % 4u == 0u, "payload starts 4-aligned");
#endif

#ifdef __cplusplus
}
#endif
#endif /* ABF_FRAME_H */
