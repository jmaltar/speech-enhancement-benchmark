#!/usr/bin/env python3
"""Confirm the C codec's CRC-32 is byte-for-byte zlib's.

The spec says ABF1's CRC is IEEE 802.3, identical to zlib.crc32, so that
host-side integrity checking has nothing to reimplement incorrectly. That is a
claim about two independent implementations agreeing, so it is tested rather
than asserted: test_abf prints a VECTOR line, and this reproduces it here.
"""
import re
import sys
import zlib


def main(path: str) -> int:
    text = open(path).read()
    m_crc = re.search(r"VECTOR bytes=(\d+) crc=([0-9A-Fa-f]{8})", text)
    m_hdr = re.search(r"VECTOR hdr=([0-9a-fA-F]+)", text)
    if not (m_crc and m_hdr):
        print("no VECTOR lines in test output", file=sys.stderr)
        return 2

    total, want = int(m_crc.group(1)), int(m_crc.group(2), 16)
    hdr = bytes.fromhex(m_hdr.group(1))

    magic, typ, ver, plen, seq = (
        int.from_bytes(hdr[0:4], "little"), hdr[4], hdr[5],
        int.from_bytes(hdr[6:8], "little"),
        int.from_bytes(hdr[8:12], "little"),
    )
    if hdr[:4] != b"ABF1":
        print(f"magic is {hdr[:4]!r}, want b'ABF1'", file=sys.stderr)
        return 1
    if total != 12 + plen + 4:
        print(f"packet {total} != 12 + {plen} + 4", file=sys.stderr)
        return 1

    # test_abf fills the payload with (i & 0xFF).
    payload = bytes(i & 0xFF for i in range(plen))
    got = zlib.crc32(hdr + payload) & 0xFFFFFFFF

    print(f"magic=0x{magic:08X} type=0x{typ:02X} ver={ver} "
          f"payload_len={plen} seq=0x{seq:08X}")
    print(f"C codec: {want:08X}   python zlib: {got:08X}   "
          f"{'MATCH' if got == want else 'MISMATCH'}")
    return 0 if got == want else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdin"))
