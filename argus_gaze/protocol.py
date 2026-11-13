"""ETVision network protocol: constants, command messages and streaming data records.

Source: ETVision Network Communication Manual v3.0 (October 2025), sections 1-4.
All multi-byte values are little-endian.

Message header (all messages):
    bytes 0-3   Argus signature 0x20414753 (" AGS")
    bytes 4-7   total message size
    bytes 8-11  command (one byte used, others 0)
    bytes 12-15 checksum (one byte used, others 0; 0 for data and JPEG messages)
    bytes 16-   argument, depending on the command

Streaming data message CMD_DATA_MSG (0x81):
    16 DataSize (uint32)       20 FrameSize (uint32, 0)   24 FrameNo (uint32)
    28 reserved                32 TimeStamp (int64, units of 100 ns)
    40 UpdateRate (4 bytes)    44 reserved                48 CheckState (uint64)
    56 data buffer: the items whose CheckState bit is set, in table order.

Implementation notes:
  * Checksums are computed by the rule in section 1. Some printed examples (commands
    1-22) carry a value 5 lower; incoming messages with either value are accepted.
  * CheckState is read from bytes 48-55; bit 59 (AI_Objects) is supported.
  * UpdateRate is decoded as uint32, with a float32 fallback if the integer reading is
    implausible.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
ARGUS_SIGNATURE = 0x20414753
HEADER_SIZE = 16
DATA_HEADER_SIZE = 56
START_OF_RECORD = 0xFA

CMD_START_DATAFILE_RECORDING = 1
CMD_STOP_DATAFILE_RECORDING = 2
CMD_OPEN_DATAFILE = 3
CMD_CLOSE_DATAFILE = 4
CMD_SET_XDAT = 5
CMD_SET_DATAFILE_NAME = 6
CMD_SET_CONNECT_TYPE = 7
CMD_START_SDATA_UDP = 8
CMD_STOP_SDATA_UDP = 9
CMD_START_SVIDEO_UDP = 10
CMD_STOP_SVIDEO_UDP = 11
CMD_START_RVIDEO_UDP = 12
CMD_STOP_RVIDEO_UDP = 13
CMD_START_SVFILE_RECORDING = 14
CMD_STOP_SVFILE_RECORDING = 15
CMD_OPEN_SVFILE = 16
CMD_CLOSE_SVFILE = 17
CMD_GET_DATAITEM = 25
CMD_GET_AIOBJ_TOTALNUM = 26
CMD_GET_AIOBJ_INFO = 27
CMD_START_AUTO_CALIBRATION = 28
CMD_STOP_AUTO_CALIBRATION = 29

CMD_RSP_BIT = 0x80000000
CMD_RSP_ERR_BIT = 0x40000000
CMD_DATA_MSG = 0x81
CMD_JPEG_MSG = 0x82

SOCKET_TYPE_SDATA_TCP = 3
SOCKET_TYPE_SVIDEO_TCP = 7
SOCKET_TYPE_RVIDEO_TCP = 9

COMMAND_NAMES = {v: k for k, v in globals().items() if k.startswith("CMD_") and isinstance(v, int)}

# Some printed checksum examples (commands 1-22) are 5 lower than the documented rule
# produces. A receiver therefore accepts either value.
LEGACY_CHECKSUM_OFFSET = 5


class ProtocolError(ValueError):
    """Raised when bytes do not form a valid ETVision message."""


# --------------------------------------------------------------------------------------
# Data item table (Network Comm Manual v3.0, section 3, "Streaming Data Buffer contents")
# Each entry: (CheckState bit, [field names in buffer order], struct code, scale factor)
# Struct codes: B uint8, b int8, H uint16, h int16, I uint32, f float32.
# Paired left/right items share one bit and appear left first, then right.
# Names are canonical snake_case; spelling variants (e.g. "hdrtk_el") are normalised.
# --------------------------------------------------------------------------------------
ItemSpec = Tuple[int, Tuple[str, ...], str, float]

DATA_ITEMS: Tuple[ItemSpec, ...] = (
    (0, ("start_of_record",), "B", 1.0),
    (1, ("status",), "B", 1.0),
    (2, ("overtime_count",), "H", 1.0),
    (3, ("mark_value",), "B", 1.0),
    (4, ("xdat",), "H", 1.0),
    (5, ("cu_video_field_num",), "H", 1.0),
    (6, ("left_pupil_pos_horz", "right_pupil_pos_horz"), "H", 1.0),
    (7, ("left_pupil_pos_vert", "right_pupil_pos_vert"), "H", 1.0),
    (8, ("left_pupil_diam", "right_pupil_diam"), "H", 0.01),
    (9, ("left_pupil_height", "right_pupil_height"), "H", 0.01),
    (10, ("left_cr_pos_horz", "right_cr_pos_horz"), "H", 1.0),
    (11, ("left_cr_pos_vert", "right_cr_pos_vert"), "H", 1.0),
    (12, ("left_cr_diam", "right_cr_diam"), "H", 1.0),
    (13, ("left_cr2_pos_horz", "right_cr2_pos_horz"), "H", 1.0),
    (14, ("left_cr2_pos_vert", "right_cr2_pos_vert"), "H", 1.0),
    (15, ("left_cr2_diam", "right_cr2_diam"), "H", 1.0),
    (16, ("horz_gaze_coord",), "h", 0.1),
    (17, ("vert_gaze_coord",), "h", 0.1),
    (18, ("horz_gaze_offset",), "h", 1.0),
    (19, ("vert_gaze_offset",), "h", 1.0),
    (20, ("vergence_angle",), "f", 1.0),
    (21, ("verg_gaze_coord_x",), "f", 1.0),
    (22, ("verg_gaze_coord_y",), "f", 1.0),
    (23, ("verg_gaze_coord_z",), "f", 1.0),
    (24, ("hdtrk_x",), "h", 0.01),
    (25, ("hdtrk_y",), "h", 0.01),
    (26, ("hdtrk_z",), "h", 0.01),
    (27, ("hdtrk_az",), "h", 0.01),
    (28, ("hdtrk_el",), "h", 0.01),
    (29, ("hdtrk_rl",), "h", 0.01),
    (30, ("et3s_scene_number",), "b", 1.0),  # -1 when gaze is on no defined plane
    (31, ("et3s_gaze_length",), "f", 1.0),
    (32, ("et3s_horz_gaze_coord",), "f", 1.0),
    (33, ("et3s_vert_gaze_coord",), "f", 1.0),
    (34, ("ssc_horz_gaze_coord",), "f", 1.0),
    (35, ("ssc_vert_gaze_coord",), "f", 1.0),
    (36, ("left_eye_location_x", "right_eye_location_x"), "h", 0.01),
    (37, ("left_eye_location_y", "right_eye_location_y"), "h", 0.01),
    (38, ("left_eye_location_z", "right_eye_location_z"), "h", 0.01),
    (39, ("left_gaze_dir_x", "right_gaze_dir_x"), "h", 0.001),
    (40, ("left_gaze_dir_y", "right_gaze_dir_y"), "h", 0.001),
    (41, ("left_gaze_dir_z", "right_gaze_dir_z"), "h", 0.001),
    (42, ("aux_sensor_x",), "h", 0.01),
    (43, ("aux_sensor_y",), "h", 0.01),
    (44, ("aux_sensor_z",), "h", 0.01),
    (45, ("aux_sensor_az",), "h", 0.01),
    (46, ("aux_sensor_el",), "h", 0.01),
    (47, ("aux_sensor_rl",), "h", 0.01),
    (48, ("left_eyelid_upper_vert", "right_eyelid_upper_vert"), "H", 1.0),
    (49, ("left_eyelid_lower_vert", "right_eyelid_lower_vert"), "H", 1.0),
    (50, ("left_blink_confidence", "right_blink_confidence"), "H", 1.0),
    (51, ("left_ellipse_angle", "right_ellipse_angle"), "f", 1.0),
    (52, ("gaze_laoi",), "I", 1.0),
    (53, ("laoi_horz_gaze_coord",), "f", 1.0),
    (54, ("laoi_vert_gaze_coord",), "f", 1.0),
    (55, ("fix_duration",), "f", 1.0),
    (56, ("horz_fix_coord",), "f", 1.0),
    (57, ("vert_fix_coord",), "f", 1.0),
    (58, ("gaze_ai_obj_id",), "I", 1.0),
)
AI_OBJECTS_BIT = 59  # variable-length set, handled separately
AI_OBJECT_FIELDS = ("obj_id", "obj_horz_cntr", "obj_vert_cntr", "obj_width",
                    "obj_height", "obj_gaze_horz", "obj_gaze_vert")

ITEM_BY_NAME: Dict[str, ItemSpec] = {}
for _spec in DATA_ITEMS:
    for _name in _spec[1]:
        ITEM_BY_NAME[_name] = _spec
BIT_BY_NAME = {name: spec[0] for name, spec in ITEM_BY_NAME.items()}
INTEGER_CODES = {"B", "b", "H", "h", "I"}

# Status byte bits for pupil and corneal reflection found, each eye (Network Comm
# Manual section 4; ET3Space Manual appendix E). The other bits are kept only as the
# raw byte.
STATUS_RIGHT_CR_FOUND = 1 << 2
STATUS_RIGHT_PUPIL_FOUND = 1 << 3
STATUS_LEFT_CR_FOUND = 1 << 4
STATUS_LEFT_PUPIL_FOUND = 1 << 5


# --------------------------------------------------------------------------------------
# Checksums and command messages
# --------------------------------------------------------------------------------------
def checksum(message: bytes) -> int:
    """Checksum byte per manual section 1: two's-complement negative of the LSB of the
    sum of all message bytes, with the checksum field (bytes 12-15) treated as zero."""
    total = sum(message[:12]) + sum(message[16:])
    return (-total) & 0xFF


def build_message(cmd: int, argument: bytes = b"") -> bytes:
    size = HEADER_SIZE + len(argument)
    body = struct.pack("<IIII", ARGUS_SIGNATURE, size, cmd, 0) + argument
    cs = checksum(body)
    return body[:12] + struct.pack("<I", cs) + body[16:]


def build_command(cmd: int, value: Optional[Union[int, str, bytes]] = None) -> bytes:
    """Build a command message. Integer arguments are sent as uint32; strings (file
    names) as bytes followed by a terminating zero byte."""
    if value is None:
        arg = b""
    elif isinstance(value, int):
        arg = struct.pack("<I", value & 0xFFFFFFFF)
    elif isinstance(value, str):
        arg = value.encode("ascii", errors="replace") + b"\x00"
    else:
        arg = bytes(value)
    return build_message(cmd, arg)


def set_xdat(value: int) -> bytes:
    return build_command(CMD_SET_XDAT, int(value) & 0xFFFF)


def start_udp_data(port: int) -> bytes:
    return build_command(CMD_START_SDATA_UDP, int(port))


def stop_udp_data() -> bytes:
    return build_command(CMD_STOP_SDATA_UDP)


def set_connect_type_tcp_data() -> bytes:
    return build_command(CMD_SET_CONNECT_TYPE, SOCKET_TYPE_SDATA_TCP)


def get_data_item(data_id: int) -> bytes:
    return build_command(CMD_GET_DATAITEM, int(data_id))


@dataclass
class Message:
    cmd: int
    size: int
    checksum: int
    argument: bytes
    checksum_ok: bool
    checksum_style: str  # "rule", "legacy" or "none"

    @property
    def is_response(self) -> bool:
        return bool(self.cmd & CMD_RSP_BIT)

    @property
    def is_error(self) -> bool:
        return bool(self.cmd & CMD_RSP_ERR_BIT)

    @property
    def base_cmd(self) -> int:
        return self.cmd & 0xFF

    def arg_uint32(self, offset: int = 0) -> int:
        return struct.unpack_from("<I", self.argument, offset)[0]


def parse_message(buf: bytes) -> Message:
    """Parse one complete message (command, response, data or JPEG)."""
    if len(buf) < HEADER_SIZE:
        raise ProtocolError("message shorter than 16-byte header")
    sig, size, cmd, cs = struct.unpack_from("<IIII", buf, 0)
    if sig != ARGUS_SIGNATURE:
        raise ProtocolError("bad Argus signature 0x%08X" % sig)
    if size != len(buf):
        raise ProtocolError("size field %d does not match %d bytes received" % (size, len(buf)))
    base = cmd & 0xFF
    if base in (CMD_DATA_MSG, CMD_JPEG_MSG) and not (cmd & CMD_RSP_BIT):
        return Message(cmd, size, cs, bytes(buf[16:]), True, "none")
    expected = checksum(buf)
    if cs & 0xFF == expected:
        ok, style = True, "rule"
    elif cs & 0xFF == (expected - LEGACY_CHECKSUM_OFFSET) & 0xFF:
        ok, style = True, "legacy"
    else:
        ok, style = False, "rule"
    return Message(cmd, size, cs, bytes(buf[16:]), ok, style)


def split_stream(buffer: bytearray) -> List[bytes]:
    """Remove and return every complete message at the front of a TCP byte buffer.
    Leaves any partial message in place. Resynchronises on the signature if needed."""
    out: List[bytes] = []
    sig = struct.pack("<I", ARGUS_SIGNATURE)
    while True:
        if len(buffer) < 8:
            return out
        if bytes(buffer[:4]) != sig:
            idx = bytes(buffer).find(sig, 1)
            if idx < 0:
                del buffer[:-3]
                return out
            del buffer[:idx]
            continue
        size = struct.unpack_from("<I", buffer, 4)[0]
        if size < HEADER_SIZE or size > 16 * 1024 * 1024:
            del buffer[:4]
            continue
        if len(buffer) < size:
            return out
        out.append(bytes(buffer[:size]))
        del buffer[:size]


# --------------------------------------------------------------------------------------
# Streaming data records
# --------------------------------------------------------------------------------------
@dataclass
class DataRecord:
    """One ETVision data record, scaled to the units the manual describes
    (position units = inches or centimetres as configured on ETVision; angles in
    degrees; gaze directions dimensionless)."""

    frame_no: int = 0
    timestamp_100ns: int = 0
    update_rate: float = 0.0
    check_state: int = 0
    values: Dict[str, Union[int, float]] = field(default_factory=dict)
    ai_objects: List[Dict[str, Union[int, float]]] = field(default_factory=list)
    time_s: Optional[float] = None  # seconds from start of file (csv files only)
    receive_time: Optional[float] = None  # local clock when received, if live

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_100ns * 1e-7

    def get(self, name: str, default=None):
        return self.values.get(name, default)

    def has(self, name: str) -> bool:
        return name in self.values


def _decode_update_rate(raw: bytes) -> float:
    as_int = struct.unpack("<I", raw)[0]
    if 1 <= as_int <= 100000:
        return float(as_int)
    as_float = struct.unpack("<f", raw)[0]
    if 0.0 < as_float < 100000.0:
        return float(as_float)
    return float(as_int)


def decode_buffer(buf: bytes, check_state: int) -> Tuple[Dict[str, Union[int, float]], List[dict]]:
    values: Dict[str, Union[int, float]] = {}
    ai_objects: List[dict] = []
    off = 0
    for bit, names, code, scale in DATA_ITEMS:
        if not (check_state >> bit) & 1:
            continue
        size = struct.calcsize("<" + code)
        for name in names:
            if off + size > len(buf):
                raise ProtocolError("data buffer ended inside item '%s'" % name)
            raw = struct.unpack_from("<" + code, buf, off)[0]
            off += size
            if code in INTEGER_CODES and scale == 1.0:
                values[name] = int(raw)
            else:
                values[name] = float(raw) * scale
    if (check_state >> AI_OBJECTS_BIT) & 1:
        if off + 4 > len(buf):
            raise ProtocolError("data buffer ended before AI object count")
        n = struct.unpack_from("<I", buf, off)[0]
        off += 4
        for _ in range(n):
            if off + 28 > len(buf):
                raise ProtocolError("data buffer ended inside AI object list")
            vals = struct.unpack_from("<Iffffff", buf, off)
            off += 28
            ai_objects.append(dict(zip(AI_OBJECT_FIELDS, vals)))
        values["no_of_ai_objects"] = n
    if off != len(buf):
        raise ProtocolError("data buffer has %d unexplained trailing bytes" % (len(buf) - off))
    return values, ai_objects


def parse_data_message(buf: bytes) -> DataRecord:
    """Parse a complete CMD_DATA_MSG (0x81) message into a DataRecord."""
    if len(buf) < DATA_HEADER_SIZE:
        raise ProtocolError("data message shorter than 56-byte header")
    sig, size, cmd, _cs, data_size, frame_size, frame_no = struct.unpack_from("<IIIIIII", buf, 0)
    if sig != ARGUS_SIGNATURE:
        raise ProtocolError("bad Argus signature")
    if cmd & 0xFF != CMD_DATA_MSG:
        raise ProtocolError("not a data message (cmd 0x%X)" % cmd)
    if size != len(buf) or size != DATA_HEADER_SIZE + data_size + frame_size:
        raise ProtocolError("inconsistent sizes: msg %d, data %d, frame %d, received %d"
                            % (size, data_size, frame_size, len(buf)))
    timestamp = struct.unpack_from("<q", buf, 32)[0]
    update_rate = _decode_update_rate(bytes(buf[40:44]))
    check_state = struct.unpack_from("<Q", buf, 48)[0]
    values, ai = decode_buffer(bytes(buf[56:56 + data_size]), check_state)
    if "start_of_record" in values and values["start_of_record"] != START_OF_RECORD:
        raise ProtocolError("start_of_record is 0x%02X, expected 0xFA" % values["start_of_record"])
    return DataRecord(frame_no=frame_no, timestamp_100ns=timestamp, update_rate=update_rate,
                      check_state=check_state, values=values, ai_objects=ai)


def check_state_for(names: Iterable[str], include_ai_objects: bool = False) -> int:
    """CheckState bits for a set of field names (always includes the 4 header items)."""
    cs = 0b1111
    for name in names:
        if name in BIT_BY_NAME:
            cs |= 1 << BIT_BY_NAME[name]
    if include_ai_objects:
        cs |= 1 << AI_OBJECTS_BIT
    return cs


def _quantise(value: Union[int, float], code: str, scale: float) -> Union[int, float]:
    if code == "f":
        return float(value)
    raw = int(round(float(value) / scale))
    lo, hi = {"B": (0, 255), "b": (-128, 127), "H": (0, 65535),
              "h": (-32768, 32767), "I": (0, 2**32 - 1)}[code]
    return max(lo, min(hi, raw))


def encode_buffer(values: Dict[str, Union[int, float]], check_state: int,
                  ai_objects: Sequence[dict] = ()) -> bytes:
    parts: List[bytes] = []
    for bit, names, code, scale in DATA_ITEMS:
        if not (check_state >> bit) & 1:
            continue
        for name in names:
            default = START_OF_RECORD if name == "start_of_record" else 0
            v = values.get(name, default)
            parts.append(struct.pack("<" + code, _quantise(v, code, scale)))
    if (check_state >> AI_OBJECTS_BIT) & 1:
        parts.append(struct.pack("<I", len(ai_objects)))
        for obj in ai_objects:
            parts.append(struct.pack("<Iffffff", int(obj.get("obj_id", 0)),
                                     *[float(obj.get(k, 0.0)) for k in AI_OBJECT_FIELDS[1:]]))
    return b"".join(parts)


def encode_data_message(record: DataRecord, update_rate_as_float: bool = False) -> bytes:
    """Encode a DataRecord as a CMD_DATA_MSG exactly as described in the manual."""
    buf = encode_buffer(record.values, record.check_state, record.ai_objects)
    rate = (struct.pack("<f", float(record.update_rate)) if update_rate_as_float
            else struct.pack("<I", int(round(record.update_rate))))
    header = struct.pack("<IIIIIIII", ARGUS_SIGNATURE, DATA_HEADER_SIZE + len(buf), CMD_DATA_MSG, 0,
                         len(buf), 0, record.frame_no & 0xFFFFFFFF, 0)
    header += struct.pack("<q", int(record.timestamp_100ns)) + rate + b"\x00" * 4
    header += struct.pack("<Q", record.check_state)
    return header + buf


# --------------------------------------------------------------------------------------
# Single data item request / response (CMD_GET_DATAITEM, manual command 25)
# Value types per the DATAID list in manual section 2: unsigned integer, integer,
# floating point, or two floating point values (left and right eye).
# --------------------------------------------------------------------------------------
GET_ITEM_RESPONSE_SIZE = 48
_GET_ITEM_UINT_IDS = {0, 1, 2, 3, 4, 5, 52, 58}
_GET_ITEM_INT_IDS = {30}


def _get_item_format(data_id: int) -> str:
    if data_id in _GET_ITEM_UINT_IDS:
        return "<I"
    if data_id in _GET_ITEM_INT_IDS:
        return "<i"
    return "<f"


def encode_get_item_response(record: DataRecord, data_id: int, error: bool = False) -> bytes:
    """Response to CMD_GET_DATAITEM: FrameNo, reserved, TimeStamp, UpdateRate, DataID,
    DataVal, DataVal2 (left/right items carry two values)."""
    spec = next((s for s in DATA_ITEMS if s[0] == data_id), None)
    cmd = CMD_RSP_BIT | CMD_GET_DATAITEM
    v1 = v2 = b"\x00" * 4
    if spec is None or spec[1][0] not in record.values:
        cmd |= CMD_RSP_ERR_BIT
    else:
        _bit, names, _code, _scale = spec
        vals = [record.values.get(n, 0) for n in names]
        fmt = _get_item_format(data_id)
        conv = float if fmt == "<f" else int
        v1 = struct.pack(fmt, conv(vals[0]))
        if len(vals) > 1:
            v2 = struct.pack(fmt, conv(vals[1]))
    if error:
        cmd |= CMD_RSP_ERR_BIT
    arg = struct.pack("<II", record.frame_no & 0xFFFFFFFF, 0) + struct.pack("<q", record.timestamp_100ns)
    arg += struct.pack("<I", int(round(record.update_rate))) + struct.pack("<I", data_id) + v1 + v2
    return build_message(cmd, arg)


def decode_get_item_response(msg: Message) -> Dict[str, Union[int, float, bytes]]:
    a = msg.argument
    frame_no, _res = struct.unpack_from("<II", a, 0)
    ts = struct.unpack_from("<q", a, 8)[0]
    rate = _decode_update_rate(a[16:20])
    data_id = struct.unpack_from("<I", a, 20)[0]
    raw1, raw2 = a[24:28], a[28:32]
    spec = next((s for s in DATA_ITEMS if s[0] == data_id), None)
    out: Dict[str, Union[int, float, bytes]] = {"frame_no": frame_no, "timestamp_100ns": ts,
                                                 "update_rate": rate, "data_id": data_id,
                                                 "raw1": raw1, "raw2": raw2}
    if spec is not None:
        _bit, names, _code, _scale = spec
        fmt = _get_item_format(data_id)
        out[names[0]] = struct.unpack(fmt, raw1)[0]
        if len(names) == 2:
            out[names[1]] = struct.unpack(fmt, raw2)[0]
    return out


# --------------------------------------------------------------------------------------
# AI model object types (manual commands 26 and 27)
# These describe the object types defined by the AI model loaded in ETVision (the names
# in the model's AINames.txt file), not the objects currently detected.
# The manual prints the response code of both as CMD_RSP_BIT | CMD_GET_DATAITEM; a
# response carrying either that code or the command's own code is accepted.
# --------------------------------------------------------------------------------------
@dataclass
class AIObjectInfo:
    object_id: int
    enabled: bool
    color: int  # XRGB
    name: str


def get_ai_object_count() -> bytes:
    return build_command(CMD_GET_AIOBJ_TOTALNUM)


def get_ai_object_info(object_id: int) -> bytes:
    return build_command(CMD_GET_AIOBJ_INFO, int(object_id))


def encode_ai_object_count_response(count: int, error: bool = False) -> bytes:
    cmd = CMD_RSP_BIT | CMD_GET_AIOBJ_TOTALNUM | (CMD_RSP_ERR_BIT if error else 0)
    return build_message(cmd, struct.pack("<I", 0 if error else int(count)))


def decode_ai_object_count_response(msg: Message) -> int:
    if msg.is_error:
        raise ProtocolError("ETVision returned an error for the AI object count")
    if len(msg.argument) < 4:
        raise ProtocolError("AI object count response too short")
    return msg.arg_uint32()


def encode_ai_object_info_response(object_id: int, info: Optional[AIObjectInfo] = None) -> bytes:
    """Response to CMD_GET_AIOBJ_INFO; an error response when `info` is None."""
    if info is None:
        cmd = CMD_RSP_BIT | CMD_RSP_ERR_BIT | CMD_GET_AIOBJ_INFO
        return build_message(cmd, struct.pack("<IIII", int(object_id), 0, 0, 0))
    arg = struct.pack("<III", int(info.object_id), 1 if info.enabled else 0, int(info.color) & 0xFFFFFFFF)
    arg += info.name.encode("ascii", errors="replace") + b"\x00"
    return build_message(CMD_RSP_BIT | CMD_GET_AIOBJ_INFO, arg)


def decode_ai_object_info_response(msg: Message) -> AIObjectInfo:
    if msg.is_error:
        raise ProtocolError("ETVision returned an error for the AI object info")
    a = msg.argument
    if len(a) < 12:
        raise ProtocolError("AI object info response too short")
    object_id, status, color = struct.unpack_from("<III", a, 0)
    name = a[12:].split(b"\x00", 1)[0].decode("ascii", errors="replace")
    return AIObjectInfo(object_id=object_id, enabled=bool(status), color=color, name=name)
