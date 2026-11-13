"""Reader for ETVision text (.csv) data files.

Format, per ETVision Manual appendix I (section 9.9.1):
  line 1  comma-separated: ETVision version, position units (inches or centimeters),
          whether ET3Space is enabled (Yes/No), date and time recording started
  line 2  column labels
  then    one record per line. The first field is time in seconds from the start of
          the file; the second is the status byte written as a binary number. Values are
          already scaled. AI_objects is one field holding values separated by '>'.

Checked against real recordings from ETVision 1.0.12 (November 2026): the header line reads
"ETVision: 1.0.12.0,Units: centimeters,3Space: Yes,Start_Recording: <date time>", the time
column is labelled "start_of_record", and each line ends with a comma. Label matching stays
tolerant (case, spaces and the left/right word order are normalised). Unknown columns are
kept under their normalised label.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from . import protocol as p

_SYNONYMS = {
    "time": "time_s", "time(s)": "time_s", "time_sec": "time_s",
    "cu_video_field_number": "cu_video_field_num",
    "et3s_scene_num": "et3s_scene_number", "et3sscene_num": "et3s_scene_number",
    "hdrtk_el": "hdtrk_el",
    "markvalue": "mark_value", "mark": "mark_value",
    "gaze_ai_id": "gaze_ai_obj_id",
    "ai_objects": "ai_objects",
}


def normalise_label(label: str) -> str:
    s = label.strip().strip('"').lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    s = s.replace("eyelocation", "eye_location")
    s = _SYNONYMS.get(s, s)
    if s in p.ITEM_BY_NAME or s in ("time_s", "ai_objects"):
        return s
    m = re.match(r"^(.*)_(left|right)$", s)
    if m and "%s_%s" % (m.group(2), m.group(1)) in p.ITEM_BY_NAME:
        return "%s_%s" % (m.group(2), m.group(1))
    return s


@dataclass
class CsvHeader:
    raw: List[str]
    version: Optional[str] = None
    units: Optional[str] = None          # "cm" or "in"
    et3space: Optional[bool] = None
    started: Optional[str] = None


def parse_header(fields: List[str]) -> CsvHeader:
    h = CsvHeader(raw=[f.strip() for f in fields])
    for f in h.raw:
        low = f.lower()
        if h.units is None and ("centimet" in low or low in ("cm",)):
            h.units = "cm"
        elif h.units is None and ("inch" in low or low in ("in",)):
            h.units = "in"
        elif h.et3space is None and re.search(r"\b(yes|no)\b", low) and ("3s" in low or "3space" in low or low in ("yes", "no")):
            h.et3space = "yes" in low
        elif h.version is None and re.search(r"\d+\.\d+", f) and ("version" in low or "etvision" in low or re.fullmatch(r"[\d.]+", f)):
            h.version = f
        elif h.started is None and re.search(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", f):
            h.started = f
    return h


@dataclass
class CsvFile:
    header: CsvHeader
    columns: List[str]
    unknown_columns: List[str] = field(default_factory=list)


def _to_number(name: str, text: str):
    t = text.strip()
    if t == "":
        return None
    if name == "status":
        if re.fullmatch(r"[01]{1,8}", t):
            return int(t, 2)
        return int(float(t))
    spec = p.ITEM_BY_NAME.get(name)
    if spec is not None and spec[2] in p.INTEGER_CODES and spec[3] == 1.0:
        return int(float(t))
    return float(t)


def _parse_ai_objects(text: str) -> List[dict]:
    parts = [x for x in re.split(r"\s*>\s*", text.strip()) if x != ""]
    if not parts:
        return []
    n = int(float(parts[0]))
    objs = []
    vals = parts[1:]
    for i in range(n):
        chunk = vals[i * 7:(i + 1) * 7]
        if len(chunk) < 7:
            break
        obj = {"obj_id": int(float(chunk[0]))}
        for k, v in zip(p.AI_OBJECT_FIELDS[1:], chunk[1:]):
            obj[k] = float(v)
        objs.append(obj)
    return objs


def read_csv(path_or_text: str, is_text: bool = False) -> Tuple[CsvFile, Iterator[p.DataRecord]]:
    """Return the file description and an iterator of DataRecords.
    Each record's timestamp is set from the file time so it can be replayed."""
    if is_text:
        lines = io.StringIO(path_or_text)
    else:
        lines = open(path_or_text, "r", newline="", encoding="utf-8-sig", errors="replace")
    reader = csv.reader(lines)
    header = parse_header(next(reader))
    labels = next(reader)
    columns = [normalise_label(x) for x in labels]
    # ETVision 1.0.12 labels the first column "start_of_record" but writes the time in
    # seconds from the start of the file there (checked on a real recording, Nov 2026).
    if columns and columns[0] in ("start_of_record", "time_s"):
        columns[0] = "time_s"
    # a trailing comma on each line gives an empty last label
    while columns and columns[-1] == "":
        columns.pop()
    unknown = [c for c in columns if c not in p.ITEM_BY_NAME and c not in ("time_s", "ai_objects")]
    desc = CsvFile(header=header, columns=columns, unknown_columns=unknown)
    known_items = [c for c in columns if c in p.ITEM_BY_NAME]
    cs = p.check_state_for(known_items, include_ai_objects="ai_objects" in columns)

    def gen() -> Iterator[p.DataRecord]:
        prev_t = None
        rate = 0.0
        for i, row in enumerate(reader):
            if not row or all(not x.strip() for x in row):
                continue
            values: Dict[str, object] = {"start_of_record": p.START_OF_RECORD}
            ai: List[dict] = []
            t = None
            for name, text in zip(columns, row):
                if name == "time_s":
                    t = float(text) if text.strip() else None
                elif name == "ai_objects":
                    ai = _parse_ai_objects(text)
                    values["no_of_ai_objects"] = len(ai)
                else:
                    try:
                        v = _to_number(name, text)
                    except ValueError:
                        v = text.strip()
                    if v is not None:
                        values[name] = v
            if t is not None and prev_t is not None and t > prev_t:
                inst = 1.0 / (t - prev_t)
                rate = inst if rate == 0 else 0.98 * rate + 0.02 * inst
            prev_t = t if t is not None else prev_t
            yield p.DataRecord(frame_no=i, timestamp_100ns=int(round((t or 0.0) * 1e7)),
                               update_rate=round(rate) if rate else 0.0, check_state=cs,
                               values=values, ai_objects=ai, time_s=t)
        if not is_text:
            lines.close()

    return desc, gen()


def write_csv(path: str, records, units: str = "centimeters", et3space: bool = True,
              version: str = "argus_gaze synthetic") -> None:
    """Write records in the documented csv layout. Used for tests and synthetic examples."""
    records = list(records)
    names = [n for spec in p.DATA_ITEMS for n in spec[1]
             if any(n in r.values for r in records) and n != "start_of_record"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([version, units, "ET3Space Yes" if et3space else "ET3Space No", "2026-10-28 10:00:00"])
        w.writerow(["time"] + names)
        t0 = records[0].timestamp_100ns if records else 0
        for r in records:
            row = ["%.3f" % ((r.timestamp_100ns - t0) * 1e-7)]
            for n in names:
                v = r.values.get(n, "")
                if n == "status" and v != "":
                    row.append(format(int(v), "08b"))
                elif isinstance(v, float):
                    row.append("%.3f" % v)
                else:
                    row.append(str(v))
            w.writerow(row)
