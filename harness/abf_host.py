#!/usr/bin/env python3
"""Host-side ABF1 tooling -- the other end of both transport bindings.

One parser serves the CDC capture and the file run alike, which is the whole
point of giving the Pi the same byte stream rather than a second format.

  abf_host.py make  IN.wav  OUT.in.abf  [--frame 512 --hop 256 --paced]
  abf_host.py run   IN.in.abf OUT.out.abf [--port auto --pings 20 --json T.json
                                           --check-bin IMAGE.bin]
  abf_host.py take  OUT.out.abf  OUT.wav  --ref IN.wav
  abf_host.py batch MANIFEST.csv OUTDIR --corpus voicebank16k --algo specsub
                    --noise mcra [--ref-dir enhanced/specsub-mcra/voicebank16k]

`batch` is the driver over the manifest: one run per utterance of one corpus,
one variant, per-run STATS and host timings to a CSV, and -- with --ref-dir --
every output compared sample for sample against the host oracle's WAV. This is
task 15's protocol executed over a whole corpus.

`run` is the CDC transport's host half: it drives an .in.abf through a device
over /dev/ttyACM* and records the device's byte stream verbatim as .out.abf, so
`take` reads a CDC capture and a file-binding run identically. It also records
the host-side timestamps the end-to-end latency decomposition needs
(FRAME_FORMAT.md section 4), which the device cannot see.

`take` performs the host's half of the FLUSH contract: drop the first
(frame - hop) output samples, trim to CONFIG.total_samples. With --ref it also
reports the reconstruction error against the original.

Standard library only, deliberately: the eval environment is a frozen artifact
and this file must run from either interpreter on the bench.
"""
import argparse
import math
import fcntl
import glob
import json
import os
import select
import struct
import sys
import termios
import time
import wave
import zlib

MAGIC = b"ABF1"
VERSION = 1
T = dict(HELLO=1, CONFIG=2, CONFIG_ACK=3, CONFIG_NAK=4, DATA=5, RESULT=6,
         FLUSH=7, STATS=8, ERROR=9, PING=10, PONG=11)
TNAME = {v: k for k, v in T.items()}
TIMING_BYTES = 24
# algo_id registry, abf_frame.h: bits 0-7 algorithm, 8-15 noise tracker.
ALGO = dict(bypass=0, specsub=1, wiener=2, mmse=3, dtln=4, rnnoise=5, ola=0xFE)
NOISE = dict(default=0, vad=1, mcra=2)
CONFIG_FMT = "<IHHBBHI24s24sII"          # 72 bytes
HELLO_FMT = "<I16s24sIIIII"              # 64 bytes
STATS_FMT = "<IIIIIIIIQQ"                # 48 bytes


def pack(type_, seq, payload=b""):
    if len(payload) % 4:
        raise ValueError("payloads are 4-aligned")
    hdr = MAGIC + bytes([type_, VERSION]) + struct.pack("<HI", len(payload), seq)
    return hdr + payload + struct.pack("<I", zlib.crc32(hdr + payload) & 0xFFFFFFFF)


def unpack(buf):
    """Yield (type, seq, payload). Resyncs on a bad magic or CRC, like the C."""
    i, n = 0, len(buf)
    while i + 12 <= n:
        if buf[i:i + 4] != MAGIC:
            i += 1
            continue
        typ, ver = buf[i + 4], buf[i + 5]
        plen, seq = struct.unpack_from("<HI", buf, i + 6)
        if ver != VERSION or plen % 4 or plen > 2048 or not (0 < typ <= T["PONG"]):
            i += 1
            continue
        end = i + 12 + plen
        if end + 4 > n:
            break
        want = struct.unpack_from("<I", buf, end)[0]
        if (zlib.crc32(buf[i:end]) & 0xFFFFFFFF) != want:
            i += 1
            continue
        yield typ, seq, buf[i + 12:end]
        i = end + 4


def lag(frame, hop):
    return frame - hop


def flush_hops(frame, hop):
    return (lag(frame, hop) + hop - 1) // hop


def data_hops(total, hop):
    return (total + hop - 1) // hop


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            sys.exit("only 16-bit PCM")
        return w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())


def write_wav(path, rate, ch, body):
    with wave.open(path, "wb") as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(body)


# ---------------------------------------------------------------------------
# make: WAV -> .in.abf
# ---------------------------------------------------------------------------
def build_stream(rate, ch, raw, frame, hop, paced, algo_id, variant, run_id):
    """The input stream for one run: CONFIG, whole hops of DATA (last one
    zero-padded), FLUSH. Returns (bytes, total_samples, data_hops)."""
    n = len(raw) // 2 // ch
    if frame % hop:
        sys.exit("frame must be a whole number of hops (COLA)")
    flags = 0x0001 if paced else 0
    cfg = struct.pack(CONFIG_FMT, rate, frame, hop, ch, 1, flags, algo_id,
                      variant.encode()[:24], run_id.encode()[:24], n, 0)
    out = [pack(T["CONFIG"], 0, cfg)]
    hb = hop * ch * 2
    dh = data_hops(n, hop)
    for j in range(dh):
        chunk = raw[j * hb:(j + 1) * hb]
        out.append(pack(T["DATA"], j, chunk.ljust(hb, b"\0")))
    out.append(pack(T["FLUSH"], dh))
    return b"".join(out), n, dh


def variant_name(algo, noise):
    if ALGO[algo] in (0, 0xFE):
        return algo
    if algo == "dtln":
        # no noise tracker; the image names its own artifact in HELLO/RESULTS
        return "dtln-i16"
    if algo == "rnnoise":
        # no noise tracker; the firmware carries Xiph v0.1.1 (the host oracle's
        # release -- the two vendored releases differ at the sample level)
        return "rnnoise-v0.1.1"
    
    return f"{algo}-{noise if noise != 'default' else 'mcra'}"


def cmd_make(a):
    rate, ch, raw = read_wav(a.wav)
    algo = ALGO[a.algo] | (NOISE[a.noise] << 8)
    variant = a.variant if a.variant != "none" else variant_name(a.algo, a.noise)
    stream, n, dh = build_stream(rate, ch, raw, a.frame, a.hop, a.paced, algo, variant, a.run)
    open(a.out, "wb").write(stream)
    print(f"{a.out}: rate={rate} ch={ch} n={n} hops={dh} "
          f"flush={flush_hops(a.frame, a.hop)} paced={a.paced} "
          f"algo_id=0x{algo:04x} variant={variant}")


# ---------------------------------------------------------------------------
# take: .out.abf -> WAV (the host's half of FLUSH)
# ---------------------------------------------------------------------------
def reassemble(buf):
    """Parse a device stream. Returns a dict with cfg fields, the sample-aligned
    output body after drop-and-trim, stats, hello, and the RESULT count; or an
    'error' key. Raises nothing; the caller decides what a short stream means."""
    r = dict(cfg=None, hello=None, stats=None, nres=0, nak=None, err=None)
    samples = bytearray()
    for typ, seq, pl in unpack(buf):
        if typ == T["HELLO"]:
            r["hello"] = struct.unpack(HELLO_FMT, pl)
        elif typ in (T["CONFIG"], T["CONFIG_ACK"]) and len(pl) == 72:
            r["cfg"] = struct.unpack(CONFIG_FMT, pl)
        elif typ == T["CONFIG_NAK"]:
            r["nak"] = struct.unpack("<I", pl)[0]
        elif typ == T["RESULT"]:
            samples += pl[TIMING_BYTES:]
            r["nres"] += 1
        elif typ == T["STATS"]:
            r["stats"] = struct.unpack(STATS_FMT, pl)
        elif typ == T["ERROR"]:
            r["err"] = seq
    r["samples"] = bytes(samples)
    if r["cfg"] is not None:
        cfg = r["cfg"]
        rate, frame, hop, ch, total = cfg[0], cfg[1], cfg[2], cfg[3], cfg[9]
        L = lag(frame, hop) * ch * 2
        r["body"] = r["samples"][L:][:total * ch * 2]
        r.update(rate=rate, frame=frame, hop=hop, ch=ch, total=total)
    return r


def compare_pcm(body, ref):
    """max |diff| in LSB, count of differing samples, NMSE in dB; or None on a
    length mismatch (never silently trimmed -- trap 15)."""
    if len(body) != len(ref):
        return None
    a = memoryview(body).cast("h"); b = memoryview(ref).cast("h")
    mx = nz = 0; sd = sr = 0.0
    for x, y in zip(a, b):
        d = x - y
        if d:
            nz += 1
            if d < 0: d = -d
            if d > mx: mx = d
            sd += float(d) * d
        sr += float(y) * y
    nmse = 10 * math.log10(sd / sr) if sd and sr else (float("-inf") if not sd else float("nan"))
    return mx, nz, nmse


def cmd_take(a):
    buf = open(a.abf, "rb").read()
    r = reassemble(buf)
    if r["nak"] is not None:
        sys.exit(f"device refused CONFIG, reason {r['nak']}")
    if not r["samples"]:
        sys.exit("no RESULT packets")
    if r["cfg"] is None:
        if not a.frame:
            sys.exit("output stream carries no CONFIG; pass --frame/--hop/--total")
        rate, frame, hop, ch, total = a.rate, a.frame, a.hop, 1, a.total
        L = lag(frame, hop) * ch * 2
        body = r["samples"][L:][:total * ch * 2]
    else:
        rate, frame, hop, ch, total = r["rate"], r["frame"], r["hop"], r["ch"], r["total"]
        body = r["body"]
    write_wav(a.wav, rate, ch, body)

    hello, stats = r["hello"], r["stats"]
    if hello:
        print(f"device   : {hello[1].rstrip(bytes(1)).decode()} "
              f"build {hello[2].rstrip(bytes(1)).decode()} tick_hz={hello[5]}")
    print(f"results  : {r['nres']}  (expected {data_hops(total, hop)} + "
          f"{flush_hops(frame, hop)})")
    print(f"output   : {a.wav}, {len(body)//2//ch} samples (total_samples={total})")
    if stats:
        mean = stats[8] / stats[1] if stats[1] else 0
        print(f"stats    : in={stats[0]} out={stats[1]} overruns={stats[2]} "
              f"resyncs={stats[4]} crc_err={stats[3]}")
        print(f"proc     : min={stats[5]} max={stats[6]} mean={mean:.1f} ticks")
    # A short stream must FAIL here. An earlier version trimmed the reference
    # to the output's length before comparing, so a run truncated by a stalled
    # transport printed identical=True -- a check that cannot say no (trap 15,
    # again). Found on hardware the first time a stream was cut short.
    if total and len(body) != total * ch * 2:
        print(f"LENGTH MISMATCH: output {len(body)//2//ch} samples, "
              f"CONFIG.total_samples={total} -- stream is incomplete")
        return 1
    if a.ref:
        _, _, raw = read_wav(a.ref)
        c = compare_pcm(body, raw[:len(body)] if len(raw) >= len(body) else raw)
        if c is None:
            print(f"LENGTH MISMATCH: {len(body)//2} vs ref {len(raw)//2}")
            return 1
        mx, nz, nmse = c
        print(f"vs ref   : identical={nz == 0} worst={mx} LSB  nonzero={nz}")
    return 0


# ---------------------------------------------------------------------------
# run: the CDC transport, host half
# ---------------------------------------------------------------------------
ABF_VID, ABF_PID_PREFIX = "cafe", "4"        # usb_descriptors.c: 0x4001 F411, 0x4101 H723, 0x42xx S3


def find_port(spec):
    """Resolve --port. 'auto' walks sysfs for the benchmark's VID:PID rather
    than trusting a ttyACM number, which moves on every replug and on this
    bench already collides with a micro:bit and the ST-Link's own VCP."""
    if spec and spec != "auto":
        return spec
    hits = []
    for dev in sorted(glob.glob("/sys/class/tty/ttyACM*")):
        iface = os.path.realpath(os.path.join(dev, "device"))
        usbdev = os.path.dirname(iface)
        try:
            vid = open(os.path.join(usbdev, "idVendor")).read().strip()
            pid = open(os.path.join(usbdev, "idProduct")).read().strip()
            serial = open(os.path.join(usbdev, "serial")).read().strip()
        except OSError:
            continue
        if vid == ABF_VID and pid.startswith(ABF_PID_PREFIX):
            hits.append(("/dev/" + os.path.basename(dev), serial))
    if not hits:
        sys.exit(f"no ABF device ({ABF_VID}:{ABF_PID_PREFIX}xx) on the bus; pass --port")
    if len(hits) > 1:
        sys.exit("several ABF devices: " + ", ".join(f"{d} (serial {s})" for d, s in hits)
                 + " -- pass --port")
    return hits[0][0]


def open_port(path):
    """Raw, non-blocking, and a FRESH ATTACH: DTR is dropped, stale input is
    discarded, then DTR is raised. The device re-arms HELLO on every attach, so
    a run always starts self-describing -- and the flush can never eat the
    HELLO, because the HELLO is only sent after DTR comes back up."""
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    a = termios.tcgetattr(fd)
    a[0] = a[1] = a[3] = 0                               # iflag, oflag, lflag
    a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL  # cflag
    a[6][termios.VMIN] = 0
    a[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, a)
    dtr = struct.pack("I", termios.TIOCM_DTR)
    fcntl.ioctl(fd, termios.TIOCMBIC, dtr)
    time.sleep(0.05)
    termios.tcflush(fd, termios.TCIOFLUSH)
    fcntl.ioctl(fd, termios.TIOCMBIS, dtr)
    return fd


class Stream:
    """Incremental ABF1 parser over whatever the transport delivers, with the
    same resync rule as unpack() and the C decoder."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, b):
        self.buf += b
        out, i, n = [], 0, len(self.buf)
        while i + 12 <= n:
            if self.buf[i:i + 4] != MAGIC:
                i += 1
                continue
            typ, ver = self.buf[i + 4], self.buf[i + 5]
            plen, seq = struct.unpack_from("<HI", self.buf, i + 6)
            if ver != VERSION or plen % 4 or plen > 2048 or not (0 < typ <= T["PONG"]):
                i += 1
                continue
            end = i + 12 + plen
            if end + 4 > n:
                break
            want = struct.unpack_from("<I", self.buf, end)[0]
            if (zlib.crc32(self.buf[i:end]) & 0xFFFFFFFF) != want:
                i += 1
                continue
            out.append((typ, seq, bytes(self.buf[i + 12:end])))
            i = end + 4
        del self.buf[:i]
        return out


class Session:
    """One port, one select() loop. Reads and writes are interleaved so the
    device's backpressure (a RESULT staged behind a full TX FIFO stops it
    draining RX) can never deadlock against a host that is still writing."""

    def __init__(self, fd):
        self.fd = fd
        self.stream = Stream()
        self.rx_all = bytearray()        # the .out.abf, verbatim
        self.events = []                 # (typ, seq, payload, t_rx_ns)
        self.consumed = 0
        self.txq = []                    # [bytes, offset, tag]
        self.tx_first = {}               # tag -> ns the first byte was handed over
        self.tx_last = {}                # tag -> ns the last byte was handed over

    def send(self, pkt, tag=None):
        self.txq.append([pkt, 0, tag])

    def pending(self):
        return bool(self.txq)

    def step(self, timeout):
        """One select() round. Returns True if anything moved."""
        w = [self.fd] if self.txq else []
        r, w, _ = select.select([self.fd], w, [], timeout)
        moved = False
        if r:
            try:
                b = os.read(self.fd, 4096)
            except BlockingIOError:
                b = b""
            if b:
                t = time.perf_counter_ns()
                self.rx_all += b
                for typ, seq, pl in self.stream.feed(b):
                    self.events.append((typ, seq, pl, t))
                moved = True
        if w and self.txq:
            pkt, off, tag = self.txq[0]
            try:
                n = os.write(self.fd, pkt[off:])
            except BlockingIOError:
                n = 0
            if n:
                t = time.perf_counter_ns()
                if off == 0 and tag is not None:
                    self.tx_first[tag] = t
                off += n
                if off >= len(pkt):
                    if tag is not None:
                        self.tx_last[tag] = t
                    self.txq.pop(0)
                else:
                    self.txq[0][1] = off
                moved = True
        return moved

    def wait(self, types, timeout, seq=None):
        """Pump until a packet of one of `types` (and seq, if given) arrives.
        Returns the event or None on timeout. Keeps writing meanwhile."""
        deadline = time.perf_counter() + timeout
        while True:
            while self.consumed < len(self.events):
                ev = self.events[self.consumed]
                self.consumed += 1
                if ev[0] in types and (seq is None or ev[1] == seq):
                    return ev
            left = deadline - time.perf_counter()
            if left <= 0:
                return None
            self.step(min(left, 0.05))


def _q(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * (len(xs) - 1)))] if xs else float("nan")


class RunError(Exception):
    pass


def run_stream(stream_in, port, pings=20, timeout=5.0, check_bin=None, log=print):
    """Drive one .in.abf through the device on `port`. Returns the report dict
    (rx_all holds the device's bytes verbatim). Raises RunError with the stream
    so far attached as .rx_all, so a batch can record the failure and go on."""
    pkts = list(unpack(stream_in))
    cfgs = [p for p in pkts if p[0] == T["CONFIG"]]
    if len(cfgs) != 1:
        raise RunError(f"expected exactly one CONFIG, found {len(cfgs)}")
    cfg = struct.unpack(CONFIG_FMT, cfgs[0][2])
    rate, frame, hop, ch, fmt, flags, total = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[9]
    paced = bool(flags & 0x0001)
    hop_s = hop / rate
    data = [p for p in pkts if p[0] == T["DATA"]]
    if [p[1] for p in data] != list(range(len(data))):
        raise RunError("DATA seq numbers are not 0..N-1")

    fd = open_port(port)
    S = Session(fd)
    report = {"port": port, "paced": paced, "rate_hz": rate, "frame": frame, "hop": hop,
              "channels": ch, "total_samples": total, "algo_id": cfg[6],
              "variant": cfg[7].rstrip(b"\0").decode(errors="replace")}

    def fail(msg):
        os.close(fd)
        e = RunError(msg); e.rx_all = bytes(S.rx_all); raise e

    # 1. HELLO on attach.
    ev = S.wait({T["HELLO"]}, timeout)
    hello = None
    if ev is None:
        log("warning: no HELLO within timeout (device re-arms HELLO on DTR; "
            "is this a pre-2026-10-14 image?)")
    else:
        h = struct.unpack(HELLO_FMT, ev[2])
        hello = dict(abf_version=h[0], platform=h[1].rstrip(b"\0").decode(),
                     build_id=h[2].rstrip(b"\0").decode(), image_crc32=h[3],
                     cpu_hz=h[4], tick_hz=h[5], max_hop_bytes=h[6])
        log(f"device   : {hello['platform']} build '{hello['build_id']}' "
            f"image_crc32={hello['image_crc32']:08x} cpu_hz={hello['cpu_hz']} "
            f"tick_hz={hello['tick_hz']} max_hop={hello['max_hop_bytes']}")
        if check_bin:
            # verify_image at the protocol level: the stream names its image.
            want = zlib.crc32(open(check_bin, "rb").read()) & 0xFFFFFFFF
            if want != hello["image_crc32"]:
                fail(f"IMAGE MISMATCH: device reports {hello['image_crc32']:08x}, "
                     f"{check_bin} is {want:08x} -- refusing to run against an "
                     f"image that is not the one in the tree")
            log(f"image    : verified against {check_bin}")
    report["hello"] = hello

    # 2. PING * N, sequentially: transport round trip at zero compute.
    rtts = []
    for i in range(pings):
        S.send(pack(T["PING"], 1000 + i), tag=("ping", i))
        ev = S.wait({T["PONG"]}, timeout, seq=1000 + i)
        if ev is None:
            fail(f"PING {i}: no PONG within {timeout}s")
        rtts.append(ev[3] - S.tx_first[("ping", i)])
    if rtts:
        log(f"ping     : n={len(rtts)} rtt min={min(rtts)/1e3:.0f} "
            f"median={_q(rtts, .5)/1e3:.0f} max={max(rtts)/1e3:.0f} us")
    report["ping_rtt_ns"] = rtts

    # 3. CONFIG.
    S.send(pack(T["CONFIG"], cfgs[0][1], cfgs[0][2]))
    ev = S.wait({T["CONFIG_ACK"], T["CONFIG_NAK"]}, timeout)
    if ev is None:
        fail("no CONFIG_ACK/NAK")
    if ev[0] == T["CONFIG_NAK"]:
        fail(f"device refused CONFIG, reason {struct.unpack('<I', ev[2])[0]}")
    acked = struct.unpack(CONFIG_FMT, ev[2])
    if acked[:4] != cfg[:4] or acked[9] != cfg[9]:
        fail("CONFIG_ACK does not echo the CONFIG sent")

    # 4. DATA, paced or not, with reads interleaved throughout.
    t0 = time.perf_counter()
    j = 0
    while j < len(data) or S.pending():
        if j < len(data) and not S.pending():
            if paced:
                left = t0 + j * hop_s - time.perf_counter()
                if left > 0:
                    S.step(left)        # keep reading while waiting
                    continue
            S.send(pack(T["DATA"], j, data[j][2]), tag=j)
            j += 1
        S.step(0.05)
    S.send(pack(T["FLUSH"], len(data)))
    t_send = time.perf_counter() - t0

    # 5. Drain to STATS (or a fault).
    ev = S.wait({T["STATS"], T["ERROR"]}, timeout)
    while S.step(0.05):          # anything still trickling in after STATS
        pass
    if ev is None:
        fail(f"no STATS within {timeout}s after FLUSH")
    if ev[0] == T["ERROR"]:
        fail(f"device ERROR seq={ev[1]}")

    # 6. Account.
    results = {e[1]: e for e in S.events if e[0] == T["RESULT"]}
    st = struct.unpack(STATS_FMT, ev[2])
    tick_hz = hello["tick_hz"] if hello else 0
    e2e, dev_int, proc, per_hop = [], [], [], []
    for seq in range(len(data)):
        r = results.get(seq)
        if r is None:
            continue
        tm = struct.unpack("<IIIIII", r[2][:TIMING_BYTES])
        rx_done, p0, p1, tx0 = tm[0], tm[1], tm[2], tm[3]
        d_int = (tx0 - rx_done) & 0xFFFFFFFF
        d_proc = (p1 - p0) & 0xFFFFFFFF
        e2e.append(r[3] - S.tx_first[seq]); dev_int.append(d_int); proc.append(d_proc)
        per_hop.append(dict(seq=seq, tx_first_ns=S.tx_first[seq], tx_last_ns=S.tx_last[seq],
                            rx_ns=r[3], t_rx_done=rx_done, t_proc_start=p0,
                            t_proc_end=p1, t_tx_start=tx0, overruns=tm[4]))
    nres = sum(1 for e in S.events if e[0] == T["RESULT"])
    log(f"run      : {len(data)} DATA in {t_send*1e3:.0f} ms "
        f"({'paced at ' + f'{hop_s*1e3:.1f} ms/hop' if paced else 'unpaced'}), "
        f"{nres} RESULT (expected {len(data)} + {flush_hops(frame, hop)})")
    log(f"stats    : in={st[0]} out={st[1]} overruns={st[2]} crc_err={st[3]} "
        f"resyncs={st[4]}")
    if tick_hz and dev_int:
        us = 1e6 / tick_hz
        log(f"proc     : min={st[5]} max={st[6]} mean={st[8]/st[1] if st[1] else 0:.1f} "
            f"ticks = {st[5]*us:.1f} / {st[6]*us:.1f} us")
        log(f"dev-int  : rx_done->tx_start min={min(dev_int)*us:.1f} "
            f"median={_q(dev_int,.5)*us:.1f} max={max(dev_int)*us:.1f} us")
    if e2e:
        log(f"e2e      : host tx(DATA)->rx(RESULT) min={min(e2e)/1e3:.0f} "
            f"median={_q(e2e,.5)/1e3:.0f} p95={_q(e2e,.95)/1e3:.0f} max={max(e2e)/1e3:.0f} us")
    report.update(stats=dict(frames_in=st[0], frames_out=st[1], overruns=st[2],
                             crc_errors=st[3], resyncs=st[4], proc_ticks_min=st[5],
                             proc_ticks_max=st[6], proc_ticks_sum=st[8], samples_out=st[9]),
                  send_s=t_send, nres=nres, hops=per_hop,
                  e2e_median_ns=_q(e2e, .5) if e2e else None,
                  proc_data_hops=proc,
                  rx_all=bytes(S.rx_all))
    os.close(fd)
    return report


def cmd_run(a):
    stream_in = open(a.abf, "rb").read()
    try:
        rep = run_stream(stream_in, find_port(a.port), a.pings, a.timeout, a.check_bin)
    except RunError as e:
        if getattr(e, "rx_all", None):
            open(a.out, "wb").write(e.rx_all)
            sys.exit(f"{e} -- stream saved to {a.out}")
        sys.exit(str(e))
    open(a.out, "wb").write(rep.pop("rx_all"))
    print(f"saved    : {a.out}")
    if a.json:
        rep["input"] = a.abf
        json.dump(rep, open(a.json, "w"), indent=1)
    return 0


# ---------------------------------------------------------------------------
# batch: one corpus of the manifest, one variant, through the device
# ---------------------------------------------------------------------------
def cmd_batch(a):
    import csv
    rows = [r for r in csv.DictReader(open(a.manifest))
            if r["corpus"] == a.corpus and r["role"] == a.role]
    if not rows:
        sys.exit(f"no rows for corpus={a.corpus} role={a.role}")
    if a.limit:
        # A head-of-manifest cut is a biased sample (trap 14); a smoke test is
        # the only legitimate use, and it is labelled as such in the CSV name.
        rows = rows[:a.limit]
    algo = ALGO[a.algo] | (NOISE[a.noise] << 8)
    variant = variant_name(a.algo, a.noise)
    port = find_port(a.port)
    os.makedirs(a.outdir, exist_ok=True)
    tag = f"{a.corpus}_{variant}{'_paced' if a.paced else ''}{'_smoke' if a.limit else ''}"
    csv_path = os.path.join(a.outdir, tag + ".csv")
    fields = ["id", "variant", "hops", "nres", "overruns", "crc_errors", "resyncs",
              "proc_min", "proc_max", "proc_mean", "proc_median_data", "e2e_median_us",
              "ref", "max_abs_lsb", "n_diff", "nmse_db", "status"]
    w = csv.DictWriter(open(csv_path, "w", newline=""), fieldnames=fields)
    w.writeheader()
    hello_seen = None
    t_start = time.perf_counter()
    worst = dict(max_abs_lsb=0, n_diff=0, nmse_db=float("-inf"), overruns=0, fails=0)
    for k, r in enumerate(rows, 1):
        uid = r["id"]
        rate, ch, raw = read_wav(r["noisy_path"])
        frame = a.frame or rate * 32 // 1000
        hop = a.hop or frame // 2
        stream, n, dh = build_stream(rate, ch, raw, frame, hop, a.paced, algo, variant, uid[:23])
        row = dict(id=uid, variant=variant, hops=dh, ref="", status="ok")
        try:
            rep = run_stream(stream, port, pings=0, timeout=a.timeout, check_bin=a.check_bin,
                             log=lambda *_: None)
        except RunError as e:
            row["status"] = str(e); worst["fails"] += 1
            w.writerow(row); print(f"[{k}/{len(rows)}] {uid}: FAIL {e}")
            continue
        if hello_seen is None and rep["hello"]:
            hello_seen = rep["hello"]
            print(f"device   : {hello_seen['platform']} build '{hello_seen['build_id']}' "
                  f"image_crc32={hello_seen['image_crc32']:08x} tick_hz={hello_seen['tick_hz']}"
                  + (f"  (verified against {a.check_bin})" if a.check_bin else ""))
        st = rep["stats"]
        pd = rep["proc_data_hops"]
        row.update(nres=rep["nres"], overruns=st["overruns"], crc_errors=st["crc_errors"],
                   resyncs=st["resyncs"], proc_min=st["proc_ticks_min"], proc_max=st["proc_ticks_max"],
                   proc_mean=round(st["proc_ticks_sum"] / st["frames_out"], 1) if st["frames_out"] else "",
                   proc_median_data=_q(pd, .5) if pd else "",
                   e2e_median_us=round(rep["e2e_median_ns"] / 1e3) if rep["e2e_median_ns"] else "")
        worst["overruns"] += st["overruns"]
        out = reassemble(rep["rx_all"])
        body = out.get("body", b"")
        if len(body) != n * ch * 2:
            row["status"] = f"short output {len(body)//2//ch}/{n}"; worst["fails"] += 1
        else:
            if a.keep:
                open(os.path.join(a.outdir, f"{uid}.{variant}.out.abf"), "wb").write(rep["rx_all"])
                write_wav(os.path.join(a.outdir, f"{uid}.{variant}.wav"), rate, ch, body)
            ref_path = None
            if a.ref_dir:
                ref_path = os.path.join(a.ref_dir, uid + ".wav")
            elif ALGO[a.algo] == 0:
                ref_path = r["noisy_path"]          # bypass must return its input
            if ref_path and os.path.exists(ref_path):
                _, _, ref = read_wav(ref_path)
                c = compare_pcm(body, ref)
                row["ref"] = ref_path
                if c is None:
                    row["status"] = f"ref length {len(ref)//2//ch} != {n}"; worst["fails"] += 1
                else:
                    mx, nz, nmse = c
                    row.update(max_abs_lsb=mx, n_diff=nz, nmse_db=round(nmse, 2))
                    worst["max_abs_lsb"] = max(worst["max_abs_lsb"], mx)
                    worst["n_diff"] = max(worst["n_diff"], nz)
                    worst["nmse_db"] = max(worst["nmse_db"], nmse)
            elif ref_path:
                row["status"] = "no ref"
        w.writerow(row)
        if k % a.every == 0 or k == len(rows):
            el = time.perf_counter() - t_start
            print(f"[{k}/{len(rows)}] {el:.0f}s  last {uid}: hops={dh} overruns={st['overruns']} "
                  f"proc_mean={row['proc_mean']} max|d|={row.get('max_abs_lsb','-')} "
                  f"status={row['status']}", flush=True)
    print(f"done     : {len(rows)} runs in {time.perf_counter()-t_start:.0f}s -> {csv_path}")
    print(f"summary  : fails={worst['fails']} total_overruns={worst['overruns']} "
          f"worst max|d|={worst['max_abs_lsb']} LSB  worst n_diff={worst['n_diff']}  "
          f"worst NMSE={worst['nmse_db']:.1f} dB")
    return 1 if worst["fails"] else 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("wav"); m.add_argument("out")
    m.add_argument("--frame", type=int, default=512)
    m.add_argument("--hop", type=int, default=256)
    m.add_argument("--paced", action="store_true")
    m.add_argument("--algo", choices=sorted(ALGO), default="bypass")
    m.add_argument("--noise", choices=sorted(NOISE), default="default")
    m.add_argument("--variant", default="none")
    m.add_argument("--run", default="run-0001")
    m.set_defaults(fn=cmd_make)
    r = sub.add_parser("run"); r.add_argument("abf"); r.add_argument("out")
    r.add_argument("--port", default="auto")
    r.add_argument("--pings", type=int, default=20)
    r.add_argument("--timeout", type=float, default=5.0)
    r.add_argument("--json", help="per-hop host and device timestamps")
    r.add_argument("--check-bin", help="refuse unless HELLO.image_crc32 matches this .bin")
    r.set_defaults(fn=cmd_run)
    b = sub.add_parser("batch"); b.add_argument("manifest"); b.add_argument("outdir")
    b.add_argument("--corpus", default="voicebank16k"); b.add_argument("--role", default="eval")
    b.add_argument("--algo", choices=sorted(ALGO), default="bypass")
    b.add_argument("--noise", choices=sorted(NOISE), default="default")
    b.add_argument("--paced", action="store_true")
    b.add_argument("--frame", type=int, default=0); b.add_argument("--hop", type=int, default=0)
    b.add_argument("--ref-dir", help="host-oracle WAVs named <id>.wav to compare against")
    b.add_argument("--limit", type=int, default=0, help="smoke test only: head of the manifest")
    b.add_argument("--keep", action="store_true", help="keep every .out.abf and .wav")
    b.add_argument("--every", type=int, default=50)
    b.add_argument("--port", default="auto"); b.add_argument("--timeout", type=float, default=10.0)
    b.add_argument("--check-bin")
    b.set_defaults(fn=cmd_batch)
    t = sub.add_parser("take"); t.add_argument("abf"); t.add_argument("wav")
    t.add_argument("--ref"); t.add_argument("--frame", type=int, default=0)
    t.add_argument("--hop", type=int, default=256)
    t.add_argument("--rate", type=int, default=16000)
    t.add_argument("--total", type=int, default=0)
    t.set_defaults(fn=cmd_take)
    a = p.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
