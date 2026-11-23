#!/usr/bin/env python3
"""Log the S3's UART console (the CH343 bridge, 115200 8N1) to a file, one
timestamped line per console line, until killed.

Written 2026-11-02 for the stamped RNNoise runs: the stage sums leave this
tier only on the console (main.c, BENCH_RNNOISE_STAMPS), and only once a run
has been quiet for 1 s, so the logger must be running BEFORE the batch starts
and still running after it ends. Opened with DTR and RTS both deasserted; the
board's auto-reset circuit may still reset it on open, which is harmless here
because the logger is started before the run (the sums start at zero after
any reset). Never open the console a second time while a run is in progress.

Needs pyserial, which the project venv does not carry: run it with ESP-IDF's
interpreter (~/.espressif/python_env/idf5.4_py3.12_env/bin/python).
  usage: console_capture.py <port> <outfile>
"""
import sys, time
import serial

port, outp = sys.argv[1], sys.argv[2]
s = serial.Serial()
s.port, s.baudrate, s.timeout = port, 115200, 0.5
s.dtr = False
s.rts = False
s.open()
with open(outp, "a", buffering=1) as f:
    f.write(f"# console_capture {port} opened {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    buf = b""
    while True:
        buf += s.read(4096)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            f.write(f"{time.strftime('%H:%M:%S')} {line.decode('ascii', 'replace').rstrip()}\n")
