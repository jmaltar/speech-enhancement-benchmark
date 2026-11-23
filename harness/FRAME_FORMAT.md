# ABF1 — the audio-bench frame format (task 7)

**One frame format, two transport bindings.** USB CDC on the three MCU tiers; files staged
over SCP on the RPi5, which exposes no USB device controller. The byte stream is identical in
both cases, so one host-side parser serves both and a CDC capture is diffable against a file
run.

**Every wire size in this document is asserted by the compiler, not by arithmetic in prose.**
`abf_frame.h` carries `_Static_assert`s on all four body sizes and an `#error` on a big-endian
target; the first draft of `HELLO` was declared 64 bytes while its fields summed to 60, and the
assertion is what caught it.

`abf_frame.h` is the normative definition and **must be byte-identical on every platform**,
exactly as `bench.h` is — `cksum 1970303221 19335` at the time of writing. Verify it before
trusting any cross-platform number: a format that has drifted between tiers turns a protocol
bug into a platform difference.

---

## 1. Packet layout

```
 +0   u32  magic        'A','B','F','1'   (0x31464241 read as little-endian u32)
 +4   u8   type
 +5   u8   version      = 1
 +6   u16  payload_len  bytes, multiple of 4, <= 2048
 +8   u32  seq
 +12  u8   payload[payload_len]
+12+N u32  crc32        over bytes [0, 12+N)
```

**16 bytes of overhead** — 12-byte header, 4-byte trailer.

Three choices worth their reasons:

- **The CRC is a trailer, not a header field.** A sender streams the payload out while
  computing the CRC and never buffers a whole packet twice. On the F411 that matters: the CDC
  FIFO is 1,024 B and the weight pool already owns 64 KB of a 128 KB part.
- **All multi-byte fields are little-endian**, which all four platforms are (Cortex-M,
  Xtensa LX7, Cortex-A76). Asserted in the header rather than assumed.
- **Encode and decode byte-wise**, through the helpers in `abf_frame.h`. Never cast a receive
  buffer to a struct: Xtensa penalises unaligned 32-bit loads, and relying on struct padding to
  match a wire layout is precisely the correct-looking assumption this project keeps catching
  in other forms.

**CRC-32 is IEEE 802.3** — reflected, poly `0xEDB88320`, init/final `0xFFFFFFFF`. Identical to
`zlib.crc32`, so host-side verification is one call and has nothing to reimplement incorrectly.

On a bad magic or CRC the receiver scans forward byte-by-byte for the next magic and counts the
event. **A resync is a defect here, not routine operation** — USB CDC is already
hardware-checked and the file binding is reliable, so the CRC exists to catch *our own framing
bugs*, and `ABF_STATS.resyncs` must be zero in any run whose numbers are quoted.

## 2. Message types

| type | dir | body | purpose |
|---|---|---|---|
| `HELLO` 0x01 | dev → host | 64 B | identity and capabilities, sent on attach |
| `CONFIG` 0x02 | host → dev | 64 B | run parameters |
| `CONFIG_ACK` 0x03 | dev → host | 72 B | accepted — **echoes the config back** |
| `CONFIG_NAK` 0x04 | dev → host | 4 B | refused, with a reason code |
| `DATA` 0x05 | host → dev | hop | one hop of input samples |
| `RESULT` 0x06 | dev → host | 24 B + hop | timing block, then one hop of output |
| `FLUSH` 0x07 | host → dev | 0 | end of stream, drain the tail |
| `STATS` 0x08 | dev → host | 48 B | end-of-run aggregate |
| `ERROR` 0x09 | dev → host | 4 B | fault; the run is void |
| `PING` 0x0A | host → dev | 0 | transport probe, no work done |
| `PONG` 0x0B | dev → host | 0 | immediate reply |

Direction is fixed per type. A device that receives a device-to-host type has desynchronised
and must resync rather than interpret it.

## 3. The run configuration travels with the run

`ABF_CONFIG` carries `rate_hz`, `frame_len`, `hop_len`, `channels`, `sample_fmt`, `flags`,
`algo_id`, `variant_id[24]` and `run_id[24]`. **None of these is ever compiled in.**

This is not future-proofing. **RNNoise forces it before any compression work does:** it is
hardwired to 48 kHz / 20 ms / 10 ms against the locked 16 kHz / 32 ms / 16 ms, so a harness
built against the locked configuration would need rewriting for the project's *second*
algorithm. NOIZEUS adds 8 kHz / 256 / 128, and DTLN's native 8 ms shift is still undecided.

Configurations in scope:

| arm | rate | frame | hop | ch | payload | packet | rate on the wire | headroom vs 375 KiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primary (locked) | 16 k | 512 | 256 | 1 | 512 B | 528 B | **32.2 KiB/s** | **11.6×** |
| NOIZEUS | 8 k | 256 | 128 | 1 | 256 B | 272 B | 16.6 KiB/s | 22.6× |
| RNNoise | 48 k | 960 | 480 | 1 | 960 B | 976 B | **95.3 KiB/s** | **3.9×** |
| DTLN at native hop | 16 k | 512 | 128 | 1 | 256 B | 272 B | 33.2 KiB/s | 11.3× |
| MUSDB (file only) | 44.1 k | 1024 | 512 | 2 | 2048 B | 2064 B | 173.6 KiB/s | 2.2× |

Framing overhead is **3.0 %** at the primary configuration. Every payload is a multiple of
4 bytes at every configuration in scope, so no padding rule is needed.

**Note what excludes MUSDB from the CDC bindings: packet size, not bandwidth.** At 173.6 KiB/s
it would still fit the measured link with 2.2× to spare; its 2,064-byte packet simply exceeds
the 1,024-byte FIFO. That distinction matters, because it means the separation tier is
file-only for a buffer reason that could be revisited, not for a throughput reason that could
not. The measured CDC figures — 376
KiB/s on the F411, 375.3 KiB/s on the S3 — are floors bounded by host round-trip latency, so
the headroom column is conservative.

**The CDC bindings cap the hop, and the cap is a budget line item.** `CFG_TUD_CDC_RX/TX_BUFSIZE`
is 1,024 B each, so a packet must be ≤ 1,024 B and a payload ≤ 1,008 B. RNNoise's 976 B fits
with 48 B to spare; MUSDB's 2,064 B does not, which is why the separation tier is file-binding
only. Enlarging the FIFO is possible and costs SRAM inside the F411's 105,008 B working budget
— so it is a budget decision, not a protocol one.

**A configuration the device cannot honour is refused, not approximated.** `CONFIG_NAK` carries
one of: `VERSION`, `RATE`, `FRAME_HOP` (frame < hop, or COLA violated), `CHANNELS`, `FORMAT`,
`HOP_TOO_BIG`, `NO_MEMORY`, `NO_ALGO`, `BUSY`. Same discipline as the firmware build guards: a
setting that changes what is measured must fail loudly rather than quietly produce numbers.

**Two flags matter to the measurements:**

- `ABF_CF_FRAME_PACED` — the device must honour the hop deadline and count overruns.
  **Required for any run whose jitter is reported.** A batch pass over a file has no deadline,
  so its spread is not the quantity the MCUs report; this flag is what makes the Pi's jitter
  comparable instead of merely present.
- `ABF_CF_BYPASS` — passthrough, no algorithm. Measures the whole path minus compute.

## 4. Timing, and what each number means

`RESULT` carries four raw counter samples plus a cumulative overrun count:

```
t_rx_done      last byte of the matching DATA packet in
t_proc_start   algorithm entered
t_proc_end     algorithm returned
t_tx_start     first byte of this RESULT queued
```

**These are ticks, not microseconds, and not cycles.** `HELLO.tick_hz` declares the rate. On
the MCUs a tick is a CPU cycle (DWT `CYCCNT` / Xtensa `CCOUNT`); **the Pi 5 refuses userspace
PMU access**, so its `bench_clock.c` layer ticks in nanoseconds. Calling the field "cycles"
would make the Pi's row a lie — the same reason `irq_disabled` is honestly recorded as 0 there.

Take differences with unsigned wraparound arithmetic. A u32 counter wraps in 44.7 s at 96 MHz
but **1.79 s at 2.4 GHz**; every interval of interest is one hop (16 ms = 38.4 Mcyc on the Pi),
orders below 2³², so differences are unambiguous and absolute values are not.

**Decomposition.** With `PING`/`PONG` giving a transport round-trip at zero compute, and
`BYPASS` giving the full path at zero compute:

| quantity | how |
|---|---|
| **algorithm latency** (the reported metric) | `(t_proc_end − t_proc_start) / tick_hz` |
| device-internal latency | `(t_tx_start − t_rx_done) / tick_hz` — includes queueing |
| transport round trip | median `PONG` RTT on the host clock |
| **end-to-end latency** | host `rx(RESULT n) − tx(DATA n)` |
| consistency check | end-to-end ≈ transport RTT + device-internal |

Host and device clocks are never synchronised and no attempt is made to synchronise them; every
device-side interval is a difference of two device stamps, and every host-side interval a
difference of two host stamps.

**One inherent term belongs in the write-up and in no measurement:** a 50 %-overlap STFT cannot
emit anything until `frame_len` samples have arrived, so `(frame_len − hop_len) / rate` —
16 ms at the locked configuration — is *algorithmic* delay, identical on all four platforms.
Report it separately from anything either the implementation or the transport causes.

**Jitter is measured device-side.** `STATS` reports `proc_ticks_min/max/sum` computed where the
work happens, so the reported spread is free of transport noise. That mirrors the probe, which
retains min/max/mean because jitter is a result on this project rather than noise.

## 5. Session sequence

```
dev  -> HELLO                       on attach (file binding: first record written)
host -> CONFIG
dev  -> CONFIG_ACK | CONFIG_NAK
host -> PING * N                    transport baseline, before any data
dev  -> PONG * N
host -> DATA(seq=0), DATA(seq=1), ...
dev  -> RESULT(seq=0), RESULT(seq=1), ...
host -> FLUSH
dev  -> RESULT(tail...) , STATS
```

`seq` pairs a `RESULT` to its `DATA` and increments from 0 per run. The device services one
packet at a time and **must never block**: the existing `usb_glue_service()` placeholder is to
be replaced wholesale, keeping that property.

## 6. The file binding

The same byte stream, written to a file:

- **input** `<run>.in.abf` — `CONFIG`, `DATA*`, `FLUSH`
- **output** `<run>.out.abf` — `HELLO`, `CONFIG_ACK`, `RESULT*`, `STATS`

**`CONFIG_ACK` echoes the accepted config**, which is what makes an `.out.abf` stand on its own:
the host needs `frame_len`, `hop_len` and `total_samples` to perform its drop-and-trim, and a
bare acknowledgement would have forced it to carry that state out of band. It also reports what
the device *accepted* rather than what the host believes it sent — the protocol version of this
project's rule against trusting the `#define` that asked for a setting. *(Found by running the
first end-to-end round trip, not by reading the spec.)*

So a stream captured from an MCU over CDC can be replayed into the Pi binding and the outputs
compared directly. That is a free strengthening of **task 15**: whole output files diffed
against the PC float oracle, rather than a reassembled stream.

**Two traps the binding must handle, both already recorded against this tier:**

1. **SCP time enters no result, and neither does the first read.** Prime the page cache or
   stage the working set on `tmpfs` — 2.7 GB against 7.4 GiB free — or the first pass times the
   SD card rather than the algorithm.
2. **Reading as fast as the file allows is a batch pass, not a stream.** With
   `ABF_CF_FRAME_PACED` set, the reader releases each `DATA` against a real monotonic deadline
   (`clock_nanosleep`, `TIMER_ABSTIME`), so overruns mean what they mean on an MCU.

## 7. Reference codec

`abf_frame.c` is the one implementation, shared by both transport bindings and the host driver,
so that a framing bug is one bug rather than three. No allocation, no blocking, no I/O — the
decoder owns a fixed buffer and is fed whatever bytes a transport happened to deliver, which is
what lets the device service it from the main loop without the never-block property being an
aspiration.

**CRC table size is a measured trade, not a default.** At the locked configuration ~66 KB/s is
CRC'd across both directions. A bitwise implementation costs ~64 ops/byte, which on the F411 at
~0.25 ops/cycle is **~8 % of a 96 MHz core** — unacceptable inside a path whose whole purpose is
measuring how much CPU the algorithm uses. The 16-entry nibble table costs 64 B of `.rodata` and
~8 cycles/byte, i.e. **~0.55 %**. A 256-entry table halves that again for 1 KB of flash; not
worth it until something says otherwise.

**Resync is deliberate and counted, once per recovery event rather than per discarded byte** —
fifty bytes of garbage is one resync, which is what a reader of `STATS` needs to know.

### Tests, and the mutation table

`test_abf.c` covers 96 checks: round trip on all eleven message types, body structs through
`memcpy`, byte-at-a-time feed, back-to-back packets, garbage prefix, corrupted payload,
truncation, a magic embedded in a valid payload, a coincidental magic in garbage, encoder
argument rejection, and the largest packet in scope. `check_vector.py` confirms the C CRC is
byte-for-byte `zlib.crc32` — two independent implementations agreeing, since one would prove
nothing. `ctest` runs both.

**The suite was itself tested by breaking the codec**, per this project's rule that an untested
guard is decoration. Twelve mutations, all caught:

| mutation | caught by |
|---|---|
| CRC table entry corrupted | known-answer vector + zlib |
| CRC second nibble pass dropped | known-answer vector + zlib |
| CRC final XOR dropped | known-answer vector + zlib |
| resync counted per byte | garbage-prefix test |
| encoder alignment / size check removed | argument tests |
| decoder skips CRC verification | corruption test |
| **header plausibility check removed** | **nothing — gap found and closed** |
| seq read from the wrong offset | 15 checks |
| u16 / u32 byte order swapped | 25 / 29 checks |

**One real gap was found this way.** Deleting `abf_header_plausible()` failed no test in the
first version of the suite: the garbage-prefix case used junk containing no magic, so the magic
compare alone rejected it. Test 10 now plants a *real* magic in garbage with an absurd version
behind it — and under the mutation the decoder believes the bogus length, swallows the head of
the genuine packet behind it, and delivers **nothing**. The check is load-bearing, which the
tests did not know until they were attacked.

## 8. `FLUSH` emits the tail — SETTLED 2026-09-12

**Nothing is discarded, and the alignment is declared rather than implied.**

An STFT cannot emit an output hop until every overlapping frame has contributed to it, so
**output lags input by exactly `frame_len − hop_len` samples**. The mechanism:

1. The device pre-fills its analysis buffer with zeros — equivalent to a leading pad of
   `frame_len − hop_len`.
2. Strict 1:1 — every `DATA` gets one `RESULT` with the same `seq`, emitted as soon as that
   arrival completes an output hop. This keeps the main loop trivial and makes each `RESULT`'s
   timing block unambiguously the work done on receiving *that* packet.
3. **`FLUSH` injects `frame_len − hop_len` samples of silence**, producing
   `abf_flush_hops()` = `(frame_len/hop_len − 1)` further `RESULT`s, `seq` continuing past the
   last `DATA`.
4. The host **drops the first `frame_len − hop_len` output samples** and **trims to
   `CONFIG.total_samples`**.

Output is then sample-aligned with input and exactly as long. At the locked configuration the
lag is 256 samples and `FLUSH` produces one extra `RESULT`.

### Why emit rather than discard

Discarding the tail loses 16 ms of every utterance — **0.64 % of the mean VoiceBank test
utterance and 1.29 % of the shortest**, at the end, where word-final energy lives. That alone
would be a poor trade.

**The decisive reason is different.** Discarding makes device output a *different length* from
the PC float reference, and comparing those two is the entire reason device output exists
(task 15). `eval_harness` aligns lengths and **raises** on a real mismatch rather than trimming
silently, so this would not even fail quietly — it would fail on every utterance. And it is not
simpler: the leading-pad handling is needed either way, so discarding buys an asymmetric policy
for nothing.

**Consequent requirement on the float oracle (tasks 9, 10):** it must implement this identical
policy — same zero pre-fill, same flush, same drop-and-trim. If the oracle framed differently
from the port, task 15 would be measuring the framing rather than the port. `abf_lag_samples()`,
`abf_flush_hops()` and `abf_data_hops()` are in the shared header so that one definition serves
the device, the host driver and the oracle alike.

### Two preconditions this leans on

- **`frame_len` must be a whole number of hops** (`abf_geometry_ok()`). Hann at 50 % needs
  hop = frame/2, at 75 % frame/4; a non-integer ratio breaks COLA. A `CONFIG` carrying one is
  refused with `ABF_NAK_FRAME_HOP`, not approximated.
- **`DATA` packets are always whole hops; the host zero-pads the last one** and records the real
  length in `CONFIG.total_samples` (0 = unknown / continuous). This is the normal case, not an
  edge case: **only 17 of 1,648 VoiceBank test utterances are an exact multiple of 256
  samples**, so 99 % need padding and trimming whatever `FLUSH` does. That fact is what put
  `total_samples` in `CONFIG` — it grew to 72 bytes for it.

### One honest caveat

The flush hops process a frame whose tail is zeros, so the spectral estimate there is computed
on partly-silent input. It is the standard offline behaviour, it is bit-identical on host and
device, and it therefore confounds nothing — but it does differ slightly from what an endless
stream would produce at that instant, and a reader is entitled to know that.

### What the tests pin

Test 13 checks `lag`, `flush_hops` and `data_hops` across all four in-scope configurations and
six utterance lengths including 1 sample and two real VoiceBank lengths, asserting the identity
that matters: **after dropping the lag, what remains is `hop × data_hops` samples, always
≥ `total_samples`, so trimming yields the original length exactly.** Test 13b pins the COLA
precondition.

**A mutation test found a gap here too.** Rounding `flush_hops` *down* instead of up failed
nothing, because `frame_len` is a whole number of hops in every legal configuration, so floor
and ceil agree everywhere. The suite now carries one deliberately illegal row (500/300) whose
only job is to exercise that rounding — guarded out of the length-identity checks, which hold
only for legal geometry.

### The binding itself

`abf_file.c` is **transport, not DSP**: it hands `hop_len` samples to an `abf_proc_t` callback
and takes `hop_len` back. The STFT, the overlap-add and the lag they create live behind that
callback, because framing policy in two places is framing policy that will diverge.

Three things it does that a naive read-the-file loop gets wrong:

1. **The whole input is resident before the first timestamp** — `abf_slurp()` reads the file in
   one call, which is also what primes the page cache. SCP time and the first read enter no
   result.
2. **Pacing is real.** With `ABF_CF_FRAME_PACED` each hop is released against a monotonic
   deadline via `clock_nanosleep(TIMER_ABSTIME)`, and overruns are counted against it. Without
   the flag the run is a batch pass, and the CLI says so in as many words.
3. **Measurement is opt-in-verified, not default-assumed.** `abf_file_run()` refuses to produce
   measurement-grade output unless `abf_platform_ok()` confirms every online CPU is on
   `performance` and the throttle word is clear — the conditions `bench_plat_init()` enforces
   for the probe. `--force` runs anyway and stamps the output **VOID**. A harness that silently
   passes its own preconditions on a laptop is the probe-reports-state-not-capability trap that
   already cost this project a correction on this very tier.

**Ticks are nanoseconds here** (`tick_hz = 1e9` in `HELLO`), truncated to 32 bits exactly as the
MCU counters are and subject to the same rule: take differences, never absolute values. u32
nanoseconds wrap every 4.29 s; a hop is 16 ms.

### Reference processors, and what they prove

Neither is an algorithm; they exist so the binding can be tested before tasks 8–11 provide
anything real, and they check different halves of the framing.

- **`bypass`** is a pure delay line of `frame_len − hop_len` samples, *not* a copy — so the
  geometry is uniform and the host's drop-and-trim is identical for every run. After
  drop-and-trim its output must equal the input **bit-exactly**: no float, no tolerance, no
  judgement. One sample of error in the lag or flush arithmetic fails it.
- **`ola`** is Hann-windowed overlap-add with no spectral step, exercising COLA. Measured
  reconstruction error: **worst 0 LSB, rms 0.0000 LSB.**

Verified end to end on the host — WAV → `.in.abf` → binding → `.out.abf` → WAV — **bit-identical
for both processors** over 40,241 samples (158 data hops + 1 flush hop, a length deliberately
not a multiple of the hop). `abf_host.py` is the host half: `make` builds a stream from a WAV,
`take` performs the drop-and-trim and reports the error against a reference.

**Verified on the Pi itself, 2026-09-12**: staged over SCP, `abf_platform_ok()` reported *governor performance on 4 cpus, throttled=0x0*, the run came back **measurement-grade** with **0 overruns** over 158 paced hops, and the round trip was bit-identical for both processors.

## 9. The CDC binding (the three MCU tiers)

`abf_cdc.c` is the device half, **intended to be byte-identical on the F411, the H723 and the
ESP32-S3** (`cksum 186182664 13032` since 2026-10-14, the selector hook; was `427672475 11462`) — verify it before trusting a cross-platform number. Two seams keep
it so, the same shape as `bench_dwt.c` / `bench_ccount.c` / `bench_clock.c`:

| seam | F411 / H723 | ESP32-S3 | host tests |
|---|---|---|---|
| tick source | `abf_plat_dwt.c` (DWT `CYCCNT`) | `abf_plat_ccount.c` (Xtensa `CCOUNT`) | fake counter |
| transport | `abf_cdc_tinyusb.c` → `tud_cdc_*` | same file, same TinyUSB 0.21.0 | memory pipe |

**It never blocks.** `abf_cdc_service()` is called from the main loop beside `tud_task()`: at
most one bulk packet of input is read per call, at most one decoded packet is acted on, and at
most one packet is ever staged for transmission, so a full TX FIFO applies backpressure instead
of stalling the core. That property is the one thing that had to survive from the placeholder
echo it replaces — ITM stalls on a full trace FIFO and semihosting halts the core outright, and
a transport that waits inside a timed region corrupts the measurement it exists to carry.

**One `.bss` buffer, not four.** Every packet is built directly in the staging buffer:
`abf_encode_inplace()` seals a payload already in position, results are written straight into the
outgoing packet after the timing block, and inputs are read in place out of the decoder — which
`abf_frame.h` aligns to 4 bytes for exactly this reason, since Xtensa penalises unaligned loads.
The obvious shape (assemble a body, let `abf_encode()` copy it in) costs three more kilobyte
buffers. Measured, cross-compiled at the project's Release settings:

| target | flash (text + rodata) | `.bss` | share of the tier's working RAM |
|---|---:|---:|---:|
| F411 (Cortex-M4F) | 3,364 B | 3,352 B | **3.1 pct** of 105,008 B |
| H723 (Cortex-M7) | 3,356 B | 3,352 B | **3.2 pct** of 104,756 B |

Flash is not binding against 476 KB / 996 KB free. The `.bss` is dominated by the 2,064-byte
decoder buffer and the 1,024-byte staging buffer; the one hop of silence the flush needs is
`const`, so it costs flash rather than SRAM.

**`overruns` does not mean quite the same thing on both bindings, and that is said rather than
smoothed over.** An MCU has no scheduler, so the only way a frame misses real time is by taking
longer than its own hop: the CDC binding counts `proc_ticks > hop_ticks`. The Pi's file binding
additionally tests completion against the release deadline, because there preemption can miss it
even when the compute fits. Both mean *"this frame missed real time by the tightest test
available on that platform"* — which is not the same test.

### Proved against the file binding, with no board

The claim this format rests on is that one format with two bindings produces one stream. That
claim is executed as a test: the same `.in.abf` is driven through both bindings and the outputs
compared packet by packet — everything except the 24-byte timing block, which is the only thing
that may legitimately differ. **162 packets, identical, and the audio still bit-exact through the
CDC path.**

The mock is deliberately hostile, and each piece of hostility earned its place by catching
something:

- **Input arrives in 1-, 7- and 64-byte dribbles**, so packets straddle service calls. This
  caught a real defect: the binding read a 64-byte chunk, acted on the first complete packet in
  it and **dropped the remainder** — a comment claimed the leftover stayed in the FIFO, but it
  had already been read out. At 64-byte reads half the stream was lost. Unconsumed bytes are now
  retained in a 64-byte `pend` buffer and resumed on the next call.
- **Every third write is refused outright**, because a real CDC FIFO signals "no room" by
  accepting zero bytes. With an always-accepts mock, deleting the one-packet-in-flight guard
  changed nothing and the suite passed; with this one it fails in seven places.
- The test itself had a bug worth recording: `collect()` stored `abf_packet_t.payload` pointers,
  which alias the decoder's own buffer and are valid only until the next call. Every record
  aliased the last packet, the comparison passed, and it proved nothing. Payloads are copied into
  an arena now. **The header documents this lifetime in as many words and the test got it wrong
  anyway** — which is the argument for checking that a comparison can actually fail.

## 10. Still to decide, and still to verify

- *(Assigned 2026-10-14.)* **The `algo_id` registry** is in `abf_frame.h`: bits 0–7 the algorithm
  (0 bypass, 1 spectral subtraction, 2 Wiener, 3 MMSE-STSA, 0xFE OLA), bits 8–15 the noise tracker
  for the classical arm (0 default = MCRA, 1 VAD, 2 MCRA), 16–31 reserved. A device selects its
  processor from it at CONFIG time (`abf_cdc_set_selector`, `classical/cdsp_select.c`, which
  `_Static_assert`s the values against `cdsp_algo_t`), so one image carries every algorithm it has
  and an id it lacks is NAKed `NO_ALGO`. `abf_host.py make --algo --noise` and `cdsp_wav` stamp
  it identically. RNNoise and DTLN get ids when their tasks land. `variant_id` stays the
  human-readable identity.
- **Whether the MCU CDC FIFOs grow beyond 1,024 B.** Only RNNoise comes close (976 B), and it
  fits, so the answer is currently no — but it is the reason the hop cap exists.
- *(Closed on the F411, 2026-10-14.)* **The CDC binding runs on hardware.** Brought up on the F411
  first, where the echo had proved the pipe: bit-identical round trips paced and unpaced, OLA exact to
  0 LSB, 0 overruns / 0 resyncs over seven runs on one boot, and a host that stops reading for 2 s
  stalls the device cleanly with nothing lost. Two defects the memory mock could not see were fixed in
  `abf_cdc.c` (HELLO once per boot rather than per attach; STATS accumulating across runs), the host
  half is `abf_host.py run`, and HELLO's `image_crc32` is checked against the `.bin` before any run.
  Full record in `benchmark_f411/RESULTS.md`; the H723 followed the same day with no change to the
  binding (`benchmark_h723/RESULTS.md`), and the S3 after it through `esp_tinyusb` with the four
  primitives untouched (`benchmark_s3/RESULTS.md`): the wrapper's CDC layer is initialised only as a
  wake source and never read. **All three MCU tiers run the binding.**
- **`chrt` needs privilege on the Pi.** The probe records `sched_policy=1 (SCHED_FIFO) prio=50`,
  but `chrt -f 50` fails as the login user and `sudo` there wants a password. Measurement runs
  are currently SCHED_OTHER with core pinning only. Resolve before the matrix, since it changes
  the preemption exposure the jitter figures carry.
