#!/usr/bin/env python3
"""Read the INA226 logger (spare F411) through the debugger's serial port and write calibrated readings to CSV.

Stdlib only, like harness/abf_host.py. Every line the logger sends is kept: data lines become CSV rows, '#' and 'E'
lines are echoed to stderr and written to <out>.log so the banner (chip UID, configuration, guards) travels with the
data.

  python3 ina226_log.py --out idle.csv --seconds 60
  python3 ina226_log.py --gain 1.0123 --shunt-ohm 0.1          # per-module correction from the calibration step

Columns: host_s, seq, t_us (unwrapped), vshunt_raw, vbus_raw, i_mA, v_board_V, p_mW.
VBS is wired to the supply side of the shunt, so v_board = vbus - vshunt; i = vshunt / R x gain.
"""
import argparse
import glob
import os
import re
import select
import signal
import sys
import termios
import time

SHUNT_LSB_V = 2.5e-6
BUS_LSB_V = 1.25e-3


def find_port():
    """The debugger serial port the logger is talking on. Several debuggers may be plugged in, each wired to some
    other board, so this only LISTENS (the logger sends a line at least once a second) and never writes to a port
    that is not known to be the logger's."""
    pats = ["/dev/serial/by-id/*CMSIS-DAP*-if01", "/dev/serial/by-id/*STLink*", "/dev/serial/by-id/*STLINK*"]
    hits = sorted({p for pat in pats for p in glob.glob(pat)})
    found = []
    for p in hits:
        try:
            fd = open_port(p)
        except OSError:
            continue
        got, t0 = b"", time.time()
        while time.time() - t0 < 2.5:
            r, _, _ = select.select([fd], [], [], 0.1)
            if r:
                got += os.read(fd, 4096)
        os.close(fd)
        if re.search(rb"(^|\n)(D,\d+,\d+,-?\d+,\d+|E,[a-z]+,[a-z]+|# ina226-logger)\r?\n", got):
            found.append(p)
    if len(found) != 1:
        sys.exit(f"no single serial port carries the logger (candidates {hits or 'none'}, logger on {found}); "
                 "pass --port")
    return found[0]


def open_port(path):
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
    a = termios.tcgetattr(fd)
    a[0] = 0                                    # iflag: raw
    a[1] = 0                                    # oflag
    a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    a[3] = 0                                    # lflag: no echo, no canonical mode
    a[4] = a[5] = termios.B115200
    a[6][termios.VMIN] = 0
    a[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, a)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port (default: the debugger's, by id)")
    ap.add_argument("--out", default="ina226.csv")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl-C)")
    ap.add_argument("--shunt-ohm", type=float, default=0.1)
    ap.add_argument("--gain", type=float, default=1.0, help="module correction factor from calibration")
    ap.add_argument("--config", help="INA226 configuration to set first, hex (default: the firmware's 0x4527)")
    ap.add_argument("--quiet", action="store_true", help="no per-second summary")
    args = ap.parse_args()

    # SIGTERM ends the run like Ctrl-C. SIGINT cannot: a non-interactive bash starts background jobs with SIGINT
    # ignored and Python leaves an inherited SIG_IGN alone, which is how the 2026-11-04 H723 session hung in wait.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    port = args.port or find_port()
    fd = open_port(port)
    os.write(fd, b"\n?\nS\n")
    if args.config:
        os.write(fd, f"C {args.config}\n".encode())

    csv = open(args.out, "w")
    log = open(os.path.splitext(args.out)[0] + ".log", "w")
    csv.write("host_s,seq,t_us,vshunt_raw,vbus_raw,i_mA,v_board_V,p_mW\n")
    log.write(f"# host port={port} shunt_ohm={args.shunt_ohm} gain={args.gain} started={time.time():.3f}\n")

    t_start = time.time()
    buf = b""
    last_seq = None
    wraps, last_t = 0, None
    drops = rows = 0
    win = []
    next_report = t_start + 1.0
    try:
        while not args.seconds or time.time() - t_start < args.seconds:
            r, _, _ = select.select([fd], [], [], 0.2)
            if r:
                buf += os.read(fd, 4096)
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.strip().decode("ascii", "replace")
                now = time.time()
                if not line:
                    continue
                if not line.startswith("D,"):
                    print(line, file=sys.stderr)
                    log.write(f"{now:.3f} {line}\n")
                    log.flush()
                    continue
                try:
                    _, seq, t_us, vsh, vb = line.split(",")
                    seq, t_us, vsh, vb = int(seq), int(t_us), int(vsh), int(vb)
                except ValueError:
                    log.write(f"{now:.3f} unparsed {line!r}\n")
                    continue
                if last_seq is not None and seq != last_seq + 1:
                    drops += max(seq - last_seq - 1, 0)
                    log.write(f"{now:.3f} gap seq {last_seq} -> {seq}\n")
                last_seq = seq
                if last_t is not None and t_us < last_t:
                    wraps += 1
                last_t = t_us
                t_unwrapped = t_us + wraps * (1 << 32)
                v_shunt = vsh * SHUNT_LSB_V
                i_a = v_shunt / args.shunt_ohm * args.gain
                v_board = vb * BUS_LSB_V - v_shunt
                p_w = v_board * i_a
                csv.write(f"{now:.4f},{seq},{t_unwrapped},{vsh},{vb},{i_a*1e3:.4f},{v_board:.5f},{p_w*1e3:.4f}\n")
                csv.flush()                                  # a kill loses no rows
                rows += 1
                win.append((i_a, v_board, p_w))
            if not args.quiet and time.time() >= next_report:
                next_report += 1.0
                if win:
                    n = len(win)
                    print(f"{time.time()-t_start:7.1f} s  n={n:3d}  I={sum(w[0] for w in win)/n*1e3:8.3f} mA  "
                          f"V={sum(w[1] for w in win)/n:6.4f} V  P={sum(w[2] for w in win)/n*1e3:8.2f} mW"
                          f"  rows={rows} drops={drops}", file=sys.stderr)
                    win = []
                else:
                    print(f"{time.time()-t_start:7.1f} s  no readings", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        csv.close()
        log.write(f"# stopped={time.time():.3f} rows={rows} drops={drops}\n")
        log.close()
        os.close(fd)
    print(f"{rows} readings, {drops} dropped, written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
