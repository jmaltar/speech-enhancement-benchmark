#!/usr/bin/env python3
"""Parse a dump of g_bench (bench_report_t, bench.h) taken over SWD:

    openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -c init \
        -c "dump_image g_bench.bin 0x<addr of g_bench> 2336" -c shutdown
    python3 read_gbench.py g_bench.bin [--hz 96000000]

Layout (little-endian, natural alignment): header 7 x u32; then up to 32
results of {name[24], n, ops, elems, reps, cyc_min, cyc_max (u32 x6),
cyc_sum (u64 at +48), checksum, guard (u32), valid (u8), pad} = 72 B.
MOps/s is from the MINIMUM cycle count, as every recorded probe figure is.
"""
import struct, sys, argparse

HDR = "<7I"
RES = "<24s6IQ2IB7x"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--hz", type=int, default=None)
    a = ap.parse_args()
    b = open(a.dump, "rb").read()
    magic, done, dwt_ok, cpu_hz, overhead, irq, n = struct.unpack_from(HDR, b, 0)
    hz = a.hz or cpu_hz
    print(f"magic={magic:08x} done={done} dwt_ok={dwt_ok} cpu_hz={cpu_hz} overhead={overhead} irq_disabled={irq} n_results={n}")
    if magic != 0x42454E31 or not done or not dwt_ok: print("!! report not valid")
    off = 32; sz = struct.calcsize(RES)   # r[] holds a u64, so the array starts 8-aligned, at 32 not 28
    print(f"{'kernel':22s} {'n':>7s} {'reps':>4s} {'min':>9s} {'max':>9s} {'mean':>10s} {'cyc/elem':>9s} {'MOps/s':>8s} {'chk':>8s}")
    for i in range(n):
        name, nn, ops, elems, reps, cmin, cmax, csum, chk, guard, valid = struct.unpack_from(RES, b, off + i * sz)
        name = name.split(b"\0")[0].decode()
        mean = csum / reps if reps else 0
        mops = (ops * hz / cmin / 1e6) if (ops and cmin) else 0
        print(f"{name:22s} {nn:7d} {reps:4d} {cmin:9d} {cmax:9d} {mean:10.1f} {cmin/elems if elems else 0:9.2f} {mops:8.1f} {chk:08x}{'' if valid else '  INVALID'}")

if __name__ == "__main__": main()
