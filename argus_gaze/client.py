"""Live client for ETVision: command socket plus a UDP or TCP data channel.

Usage (on a PC on the same network as the ETVision PC; click "Listen" in ETVision's
Network Configuration dialog and tick "Send Data Result" first):

    python -m argus_gaze.client --host 192.168.1.20 --port 5000 --udp 6000
    python -m argus_gaze.client --host 192.168.1.20 --port 5000 --tcp --units cm --gaze

The host and port are the ones shown in ETVision's Network Configuration dialog.
"""
from __future__ import annotations

import argparse
import json
import select
import socket
import sys
import time
from typing import Callable, Iterator, List, Optional

from . import protocol as p
from .gaze import to_gaze_sample


class ETVisionClient:
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self.cmd_sock: Optional[socket.socket] = None
        self.data_sock: Optional[socket.socket] = None
        self.mode: Optional[str] = None
        self._cmd_buf = bytearray()
        self._data_buf = bytearray()
        self.pending_responses: List[p.Message] = []

    # -- connection -------------------------------------------------------------------
    def connect(self) -> None:
        self.cmd_sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.cmd_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self) -> None:
        try:
            if self.mode == "udp" and self.cmd_sock:
                self.cmd_sock.sendall(p.stop_udp_data())
        except OSError:
            pass
        for s in (self.data_sock, self.cmd_sock):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        self.data_sock = self.cmd_sock = None
        self.mode = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # -- commands ---------------------------------------------------------------------
    def send(self, message: bytes) -> None:
        if self.cmd_sock is None:
            raise RuntimeError("not connected")
        self.cmd_sock.sendall(message)

    def set_xdat(self, value: int) -> None:
        self.send(p.set_xdat(value))

    def set_datafile_name(self, name: str) -> None:
        self.send(p.build_command(p.CMD_SET_DATAFILE_NAME, name))

    def open_datafile(self) -> None:
        self.send(p.build_command(p.CMD_OPEN_DATAFILE))

    def close_datafile(self) -> None:
        self.send(p.build_command(p.CMD_CLOSE_DATAFILE))

    def start_recording(self) -> None:
        self.send(p.build_command(p.CMD_START_DATAFILE_RECORDING))

    def stop_recording(self) -> None:
        self.send(p.build_command(p.CMD_STOP_DATAFILE_RECORDING))

    def read_response(self, cmd: int, timeout: Optional[float] = None) -> p.Message:
        """Wait for a response to `cmd` on the command socket."""
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            for i, m in enumerate(self.pending_responses):
                if m.base_cmd == cmd & 0xFF:
                    return self.pending_responses.pop(i)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("no response to command %d" % cmd)
            self.cmd_sock.settimeout(remaining)
            chunk = self.cmd_sock.recv(65536)
            if not chunk:
                raise ConnectionError("ETVision closed the command connection")
            self._cmd_buf.extend(chunk)
            for raw in p.split_stream(self._cmd_buf):
                self.pending_responses.append(p.parse_message(raw))

    def get_data_item(self, data_id: int) -> dict:
        self.send(p.get_data_item(data_id))
        msg = self.read_response(p.CMD_GET_DATAITEM)
        if msg.is_error:
            raise RuntimeError("ETVision returned an error for data item %d" % data_id)
        return p.decode_get_item_response(msg)

    def auto_calibrate(self, timeout: float = 10.0) -> bool:
        """Start auto calibration and wait for completion (manual command 28)."""
        self.send(p.build_command(p.CMD_START_AUTO_CALIBRATION))
        self.read_response(p.CMD_START_AUTO_CALIBRATION, timeout)
        try:
            self.read_response(p.CMD_STOP_AUTO_CALIBRATION, timeout)
            return True
        except TimeoutError:
            self.send(p.build_command(p.CMD_STOP_AUTO_CALIBRATION))
            return False

    # -- streaming --------------------------------------------------------------------
    def start_udp(self, local_port: int = 0, bind_host: str = "0.0.0.0") -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        sock.bind((bind_host, local_port))
        self.data_sock = sock
        port = sock.getsockname()[1]
        self.send(p.start_udp_data(port))
        self.mode = "udp"
        return port

    def start_tcp(self) -> None:
        self.send(p.set_connect_type_tcp_data())
        time.sleep(0.05)
        self.data_sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.mode = "tcp"

    def messages(self, idle_timeout: Optional[float] = None) -> Iterator[bytes]:
        """Yield raw data messages as they arrive. Stops after `idle_timeout` seconds of
        silence (None = wait forever). Command-channel responses are collected on the side."""
        if self.data_sock is None:
            raise RuntimeError("call start_udp() or start_tcp() first")
        socks = [self.data_sock] + ([self.cmd_sock] if self.cmd_sock else [])
        while True:
            ready, _, _ = select.select(socks, [], [], idle_timeout)
            if not ready:
                return
            for s in ready:
                if s is self.cmd_sock:
                    chunk = s.recv(65536)
                    if not chunk:
                        socks.remove(s)
                        continue
                    self._cmd_buf.extend(chunk)
                    for raw in p.split_stream(self._cmd_buf):
                        self.pending_responses.append(p.parse_message(raw))
                elif self.mode == "udp":
                    data, _addr = s.recvfrom(65536)
                    yield data
                else:
                    chunk = s.recv(65536)
                    if not chunk:
                        return
                    self._data_buf.extend(chunk)
                    for raw in p.split_stream(self._data_buf):
                        yield raw

    def records(self, idle_timeout: Optional[float] = None,
                on_error: Optional[Callable[[Exception, bytes], None]] = None) -> Iterator[p.DataRecord]:
        for raw in self.messages(idle_timeout):
            try:
                rec = p.parse_data_message(raw)
            except p.ProtocolError as exc:
                if on_error:
                    on_error(exc, raw)
                continue
            rec.receive_time = time.time()
            yield rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print live ETVision data")
    ap.add_argument("--host", required=True, help="ETVision PC address")
    ap.add_argument("--port", type=int, required=True, help="port shown in ETVision Network Configuration")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--udp", type=int, metavar="LOCAL_PORT", help="receive data over UDP on this port (0 = any)")
    mode.add_argument("--tcp", action="store_true", help="receive data over a TCP data socket")
    ap.add_argument("--units", default="cm", help="ETVision position units: cm or in (default cm)")
    ap.add_argument("--gaze", action="store_true", help="print GazeSample (SI units) instead of raw values")
    ap.add_argument("--count", type=int, default=0, help="stop after this many records (0 = run until Ctrl-C)")
    ap.add_argument("--idle", type=float, default=10.0, help="stop after this many seconds without data")
    args = ap.parse_args(argv)

    client = ETVisionClient(args.host, args.port)
    client.connect()
    if args.tcp:
        client.start_tcp()
    else:
        port = client.start_udp(args.udp or 0)
        print("receiving UDP data on port %d" % port, file=sys.stderr)
    n = 0
    try:
        for rec in client.records(idle_timeout=args.idle,
                                  on_error=lambda e, raw: print("skipped record: %s" % e, file=sys.stderr)):
            if args.gaze:
                out = to_gaze_sample(rec, args.units).as_dict()
            else:
                out = {"frame_no": rec.frame_no, "timestamp_s": rec.timestamp_s,
                       "update_rate": rec.update_rate, **rec.values}
            print(json.dumps(out))
            n += 1
            if args.count and n >= args.count:
                break
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
    print("%d records" % n, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
