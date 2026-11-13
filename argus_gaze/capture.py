"""Bit-exact recording of the live ETVision data stream, and reading it back.

A capture file keeps every data message exactly as ETVision sent it, together with the
local receive time, so that it can later be replayed byte for byte. This is useful when a
recording must preserve fields or timing that a csv export would round.

File format (.agscap):
    8-byte magic  b"AGSCAP01"
    then repeated records: float64 receive_time (Unix seconds), uint32 length, message bytes

Usage:
    python -m argus_gaze.capture --host 192.168.1.20 --port 5000 --out session01.agscap
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from typing import BinaryIO, Iterator, Tuple

from . import protocol as p

MAGIC = b"AGSCAP01"


class CaptureWriter:
    def __init__(self, path: str):
        self.f: BinaryIO = open(path, "wb")
        self.f.write(MAGIC)
        self.count = 0

    def write(self, raw: bytes, receive_time: float = None) -> None:
        t = time.time() if receive_time is None else receive_time
        self.f.write(struct.pack("<dI", t, len(raw)))
        self.f.write(raw)
        self.count += 1

    def close(self) -> None:
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_capture(path: str) -> Iterator[Tuple[float, bytes]]:
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError("%s is not an argus_gaze capture file" % path)
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            t, n = struct.unpack("<dI", head)
            raw = f.read(n)
            if len(raw) < n:
                return
            yield t, raw


def read_capture_records(path: str) -> Iterator[p.DataRecord]:
    for t, raw in read_capture(path):
        try:
            rec = p.parse_data_message(raw)
        except p.ProtocolError:
            continue
        rec.receive_time = t
        yield rec


def main(argv=None) -> int:
    from .client import ETVisionClient

    ap = argparse.ArgumentParser(description="Record the live ETVision data stream to a capture file")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True, help="output .agscap file")
    ap.add_argument("--tcp", action="store_true", help="use a TCP data socket instead of UDP")
    ap.add_argument("--idle", type=float, default=30.0, help="stop after this many seconds without data")
    args = ap.parse_args(argv)

    client = ETVisionClient(args.host, args.port)
    client.connect()
    if args.tcp:
        client.start_tcp()
    else:
        client.start_udp(0)
    print("recording to %s; press Ctrl-C to stop" % args.out, file=sys.stderr)
    with CaptureWriter(args.out) as w:
        try:
            for raw in client.messages(idle_timeout=args.idle):
                w.write(raw)
                if w.count % 900 == 0:
                    print("%d messages" % w.count, file=sys.stderr)
        except KeyboardInterrupt:
            pass
        finally:
            client.close()
        print("saved %d messages" % w.count, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
