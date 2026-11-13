"""ETVision emulator: replays a recorded session over the documented network protocol.

Software that talks to ETVision (this package's client, a ROS 2 node, LSL relays, the
Argus SDK samples) can be developed and tested against the emulator with no eye tracker,
motion-capture system or engineer present.

Supported:
  * TCP command socket on the given port (like ETVision after "Listen")
  * CMD_START_SDATA_UDP / CMD_STOP_SDATA_UDP: stream to the requesting host's UDP port
  * CMD_SET_CONNECT_TYPE (send data via TCP): the next TCP connection from the same host
    becomes a data socket and receives the stream
  * CMD_SET_XDAT: the value is written into subsequent records, as ETVision does
  * CMD_GET_DATAITEM: answered from the latest record
  * CMD_START_AUTO_CALIBRATION / CMD_STOP_AUTO_CALIBRATION: acknowledged
  * CMD_GET_AIOBJ_TOTALNUM / CMD_GET_AIOBJ_INFO: answered from an AI model names file
    (--object-names, one name per line, first line = object type 1); error responses
    when no file is given
  * data-file commands: acknowledged silently (nothing is recorded)

Sources:  --csv  ETVision text data file
          --capture  .agscap file from argus_gaze.capture (replayed byte for byte)

    python -m argus_gaze.replay --csv session01.csv --port 5000
"""
from __future__ import annotations

import argparse
import logging
import socket
import struct
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

from . import protocol as p

log = logging.getLogger("argus_gaze.replay")


class Source:
    """A list of (relative_time_s, record_or_raw_bytes)."""

    def __init__(self, items: List[Tuple[float, object]], raw: bool):
        self.items = items
        self.raw = raw

    @classmethod
    def from_csv(cls, path: str) -> "Source":
        from .csvfile import read_csv
        desc, recs = read_csv(path)
        items = []
        t0 = None
        for r in recs:
            t = r.time_s if r.time_s is not None else r.timestamp_100ns * 1e-7
            t0 = t if t0 is None else t0
            items.append((t - t0, r))
        if len(items) > 1 and items[-1][0] > 0:
            rate = round((len(items) - 1) / items[-1][0])  # csv times are rounded to 1 ms
            for _t, r in items:
                r.update_rate = rate
        if desc.unknown_columns:
            log.warning("columns not in the protocol table (not streamed): %s", ", ".join(desc.unknown_columns))
        return cls(items, raw=False)

    @classmethod
    def from_capture(cls, path: str) -> "Source":
        from .capture import read_capture
        items = []
        t0 = None
        for t, raw in read_capture(path):
            t0 = t if t0 is None else t0
            items.append((t - t0, raw))
        return cls(items, raw=True)

    @classmethod
    def from_records(cls, records: List[p.DataRecord]) -> "Source":
        items = []
        t0 = None
        for r in records:
            t = r.timestamp_100ns * 1e-7
            t0 = t if t0 is None else t0
            items.append((t - t0, r))
        return cls(items, raw=False)


class Emulator:
    def __init__(self, source: Source, host: str = "0.0.0.0", port: int = 5000,
                 speed: float = 1.0, loop: bool = True, live_timestamps: bool = True,
                 object_names: Optional[Dict[int, str]] = None):
        self.source = source
        self.object_names = object_names
        self.host, self.port = host, port
        self.speed = max(1e-6, speed)
        self.loop = loop
        self.live_timestamps = live_timestamps
        self._lock = threading.Lock()
        self._udp_targets: Dict[Tuple[str, int], bool] = {}
        self._tcp_targets: List[socket.socket] = []
        self._pending_tcp_data: Dict[str, int] = {}
        self._xdat: Optional[int] = None
        self.latest: Optional[p.DataRecord] = None
        self.sent = 0
        self._stop = threading.Event()
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._threads: List[threading.Thread] = []

    # -- lifecycle ----------------------------------------------------------------------
    def start(self) -> int:
        self._srv.bind((self.host, self.port))
        self._srv.listen(8)
        self.port = self._srv.getsockname()[1]
        for target in (self._accept_loop, self._stream_loop):
            th = threading.Thread(target=target, daemon=True)
            th.start()
            self._threads.append(th)
        log.info("emulator listening on %s:%d", self.host, self.port)
        return self.port

    def stop(self) -> None:
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass
        with self._lock:
            for s in self._tcp_targets:
                try:
                    s.close()
                except OSError:
                    pass
            self._tcp_targets.clear()
        for th in self._threads:
            th.join(timeout=2)

    # -- command handling ---------------------------------------------------------------
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except OSError:
                return
            ip = addr[0]
            with self._lock:
                pending = self._pending_tcp_data.pop(ip, None)
            if pending == p.SOCKET_TYPE_SDATA_TCP:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self._tcp_targets.append(conn)
                log.info("TCP data socket from %s", addr)
            else:
                th = threading.Thread(target=self._command_loop, args=(conn, addr), daemon=True)
                th.start()
                self._threads.append(th)

    def _command_loop(self, conn: socket.socket, addr) -> None:
        log.info("command socket from %s", addr)
        buf = bytearray()
        conn.settimeout(0.5)
        while not self._stop.is_set():
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf.extend(chunk)
            for raw in p.split_stream(buf):
                try:
                    msg = p.parse_message(raw)
                except p.ProtocolError as exc:
                    log.warning("bad command from %s: %s", addr, exc)
                    continue
                if not msg.checksum_ok:
                    log.warning("checksum mismatch on command %d from %s (accepted)", msg.base_cmd, addr)
                elif msg.checksum_style == "legacy":
                    log.info("command %d used the earlier example checksum (rule - 5)", msg.base_cmd)
                self._handle(conn, addr, msg)
        with self._lock:
            for key in [k for k in self._udp_targets if k[0] == addr[0]]:
                self._udp_targets.pop(key, None)
        try:
            conn.close()
        except OSError:
            pass

    def _handle(self, conn: socket.socket, addr, msg: p.Message) -> None:
        cmd = msg.base_cmd
        name = p.COMMAND_NAMES.get(cmd, str(cmd))
        log.debug("command %s from %s", name, addr)
        if cmd == p.CMD_START_SDATA_UDP:
            port = msg.arg_uint32()
            with self._lock:
                self._udp_targets[(addr[0], port)] = True
            log.info("streaming UDP to %s:%d", addr[0], port)
        elif cmd == p.CMD_STOP_SDATA_UDP:
            with self._lock:
                for key in [k for k in self._udp_targets if k[0] == addr[0]]:
                    self._udp_targets.pop(key, None)
        elif cmd == p.CMD_SET_CONNECT_TYPE:
            with self._lock:
                self._pending_tcp_data[addr[0]] = msg.arg_uint32()
        elif cmd == p.CMD_SET_XDAT:
            self._xdat = msg.arg_uint32() & 0xFFFF
        elif cmd == p.CMD_GET_DATAITEM:
            data_id = msg.arg_uint32()
            rec = self.latest or p.DataRecord()
            conn.sendall(p.encode_get_item_response(rec, data_id))
        elif cmd == p.CMD_START_AUTO_CALIBRATION:
            conn.sendall(p.build_message(p.CMD_RSP_BIT | p.CMD_START_AUTO_CALIBRATION, b"\x00" * 8))
            time.sleep(0.2)
            conn.sendall(p.build_message(p.CMD_RSP_BIT | p.CMD_STOP_AUTO_CALIBRATION, b"\x00" * 8))
        elif cmd == p.CMD_GET_AIOBJ_TOTALNUM:
            names = self.object_names
            conn.sendall(p.encode_ai_object_count_response(len(names or {}), error=names is None))
        elif cmd == p.CMD_GET_AIOBJ_INFO:
            object_id = msg.arg_uint32()
            name = (self.object_names or {}).get(object_id)
            info = None if name is None else p.AIObjectInfo(object_id, True, 0x00FFFFFF, name)
            conn.sendall(p.encode_ai_object_info_response(object_id, info))
        elif cmd == p.CMD_STOP_AUTO_CALIBRATION:
            conn.sendall(p.build_message(p.CMD_RSP_BIT | p.CMD_STOP_AUTO_CALIBRATION, b"\x00" * 8))
        else:
            pass  # file and video commands: acknowledged by doing nothing

    # -- streaming ------------------------------------------------------------------------
    def _prepare(self, item: object, frame_no: int, t_rel: float, t_wall0: float) -> bytes:
        if self.source.raw and self._xdat is None and not self.live_timestamps:
            self.latest = p.parse_data_message(item)
            return item  # byte for byte
        if self.source.raw:
            rec = p.parse_data_message(item)  # a fresh object we may modify
        else:
            src = item
            rec = p.DataRecord(frame_no=src.frame_no, timestamp_100ns=src.timestamp_100ns,
                               update_rate=src.update_rate, check_state=src.check_state,
                               values=dict(src.values), ai_objects=list(src.ai_objects), time_s=src.time_s)
        rec.frame_no = frame_no
        if self.live_timestamps:
            rec.timestamp_100ns = int((t_wall0 + t_rel / self.speed) * 1e7)
        if self._xdat is not None and "xdat" in rec.values:
            rec.values["xdat"] = self._xdat
        self.latest = rec
        return p.encode_data_message(rec)

    def _stream_loop(self) -> None:
        frame_no = 0
        items = self.source.items
        if not items:
            log.error("source has no records")
            return
        while not self._stop.is_set():
            t_mono0 = time.monotonic()
            t_wall0 = time.time()
            for t_rel, item in items:
                if self._stop.is_set():
                    return
                delay = t_mono0 + t_rel / self.speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                try:
                    payload = self._prepare(item, frame_no, t_rel, t_wall0)
                except p.ProtocolError as exc:
                    log.warning("skipping unreadable record: %s", exc)
                    continue
                frame_no += 1
                with self._lock:
                    udp = list(self._udp_targets)
                    tcp = list(self._tcp_targets)
                for target in udp:
                    try:
                        self._udp.sendto(payload, target)
                    except OSError as exc:
                        log.debug("UDP send to %s failed: %s", target, exc)
                for s in tcp:
                    try:
                        s.sendall(payload)
                    except OSError:
                        with self._lock:
                            if s in self._tcp_targets:
                                self._tcp_targets.remove(s)
                if udp or tcp:
                    self.sent += 1
            if not self.loop:
                return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay a recorded ETVision session over the network protocol")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="ETVision text (.csv) data file")
    src.add_argument("--capture", help=".agscap file from argus_gaze.capture")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    ap.add_argument("--once", action="store_true", help="play once instead of looping")
    ap.add_argument("--original-timestamps", action="store_true",
                    help="keep recorded time stamps instead of the current clock")
    ap.add_argument("--object-names", metavar="FILE",
                    help="AI model names file (one name per line, first line = object type 1), "
                         "used to answer AI object queries")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    source = Source.from_csv(args.csv) if args.csv else Source.from_capture(args.capture)
    names = None
    if args.object_names:
        from .align import load_object_names
        names = load_object_names(args.object_names)
    em = Emulator(source, args.host, args.port, args.speed, loop=not args.once,
                  live_timestamps=not args.original_timestamps, object_names=names)
    port = em.start()
    print("ETVision emulator on port %d with %d records. Ctrl-C to stop." % (port, len(source.items)),
          file=sys.stderr)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    em.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
