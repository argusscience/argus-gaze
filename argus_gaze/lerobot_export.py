"""Convert an ETVision / ET3Space recording to a LeRobot dataset (format v3.0).

Field names, units and frames follow the Argus gaze field specification (draft v0.2):
each gaze field is stored as ``observation.gaze.<name>``, in SI units, in the ET3Space
world frame. Camera videos become ``observation.images.<camera>``.

Gaze runs at 180 Hz; a dataset frame takes the ETVision sample nearest to its time.
The full-rate stream is written next to the dataset in ``extras/``.

Two steps, so the first can be tested without LeRobot installed:

* :func:`gaze_table` reads the ETVision .csv file into full-rate arrays (numpy only).
* :func:`write_lerobot` samples them at the frame rate, reads the videos with ffmpeg,
  and writes the dataset with ``lerobot`` (``pip install lerobot``).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .csvfile import read_csv
from .gaze import GAZE_FIELDS, spec_values, to_gaze_sample  # noqa: F401 (GAZE_FIELDS re-exported)

NAN = float("nan")



@dataclass
class GazeTable:
    """Full-rate gaze, one row per ETVision sample."""
    time_s: np.ndarray                 # seconds from the start of the file
    mark: np.ndarray                   # ETVision Mark value (int)
    fields: Dict[str, np.ndarray]      # name -> (n, size) arrays, as in GAZE_FIELDS
    header: dict                       # what the file says about itself


def gaze_table(csv_path: str) -> GazeTable:
    desc, records = read_csv(csv_path)
    units = desc.header.units or "cm"
    et3s = bool(desc.header.et3space)
    rows = {k: [] for k in GAZE_FIELDS}
    t, mark = [], []
    for rec in records:
        g = to_gaze_sample(rec, units, et3space=et3s)
        t.append(rec.time_s if rec.time_s is not None else NAN)
        mark.append(int(g.mark) if g.mark is not None else 0)
        vals = spec_values(g)
        vals["device_time"] = [t[-1]]  # in a dataset: seconds from the start of the file
        for k in GAZE_FIELDS:
            rows[k].append(vals[k])
    fields = {k: np.asarray(v, dtype=GAZE_FIELDS[k][0]) for k, v in rows.items()}
    header = dict(etvision=desc.header.version, position_units=units, et3space=et3s,
                  started=desc.header.started, columns=desc.columns)
    return GazeTable(np.asarray(t, dtype=float), np.asarray(mark, dtype=int), fields, header)


def segments_by_mark(table: GazeTable, keep: Sequence[int], min_s: float = 1.0) -> List[Tuple[int, float, float]]:
    """(mark, start, end) for each run of a Mark value listed in `keep`.
    A run that starts with the file is skipped: its Mark value may be left over from
    the previous recording."""
    out = []
    t, m = table.time_s, table.mark
    start = 0
    for i in range(1, len(m) + 1):
        if i == len(m) or m[i] != m[start]:
            if start > 0 and m[start] in keep and t[i - 1] - t[start] >= min_s:
                out.append((int(m[start]), float(t[start]), float(t[i - 1])))
            start = i
    return out


def sample_at(table: GazeTable, times: np.ndarray) -> np.ndarray:
    """Index of the ETVision sample nearest to each time."""
    idx = np.searchsorted(table.time_s, times)
    idx = np.clip(idx, 1, len(table.time_s) - 1)
    left = table.time_s[idx - 1]
    right = table.time_s[idx]
    return np.where(times - left <= right - times, idx - 1, idx)


def lerobot_features(cameras: Dict[str, Tuple[int, int]]) -> dict:
    feats = {}
    for name, (dtype, size, names) in GAZE_FIELDS.items():
        feats["observation.gaze." + name] = {"dtype": dtype, "shape": (size,), "names": names}
    for cam, (w, h) in cameras.items():
        feats["observation.images." + cam] = {"dtype": "video", "shape": (h, w, 3),
                                             "names": ["height", "width", "channels"]}
    return feats


def _video_frames(path: str, fps: int, size: Tuple[int, int]):
    """Yield RGB frames resampled to a constant frame rate, frame n at time n / fps."""
    w, h = size
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", "fps=%d,scale=%d:%d" % (fps, w, h),
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.wait()


def write_lerobot(table: GazeTable, segments: Sequence[Tuple[int, float, float]], tasks: Dict[int, str],
                  videos: Dict[str, str], root: str, repo_id: str, fps: int = 30,
                  size: Tuple[int, int] = (640, 360), metadata: Optional[dict] = None):
    """Write the dataset. One episode per segment; `tasks` maps Mark value -> task text."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # optional dependency

    root = Path(root)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, root=root, robot_type=None,
                               features=lerobot_features({c: size for c in videos}), use_videos=True)
    streams = {c: _video_frames(p, fps, size) for c, p in videos.items()}
    episodes = []
    current = None
    n = 0
    while True:
        try:
            frames = {c: next(s) for c, s in streams.items()}
        except StopIteration:
            break
        t = n / fps
        n += 1
        seg = next((k for k, (m, a, b) in enumerate(segments) if a <= t <= b), None)
        if seg != current:
            if current is not None:
                ds.save_episode()
            current = seg
        if seg is None:
            continue
        i = int(sample_at(table, np.array([t]))[0])
        frame = {"task": tasks[segments[seg][0]]}
        for name in GAZE_FIELDS:
            frame["observation.gaze." + name] = table.fields[name][i]
        for c, img in frames.items():
            frame["observation.images." + c] = img
        ds.add_frame(frame)
        if not episodes or episodes[-1]["segment"] != seg:
            episodes.append(dict(segment=seg, mark=segments[seg][0], first_frame_time_s=t, frames=0))
        episodes[-1]["frames"] += 1
        episodes[-1]["last_frame_time_s"] = t
    if current is not None:
        ds.save_episode()
    ds.finalize()

    # full-rate gaze and Argus metadata, next to the LeRobot files
    write_full_rate(table, segments, root / "extras" / "gaze_full_rate.parquet")
    meta = dict(metadata or {})
    meta["episodes"] = [dict(episode_index=k, mark=e["mark"], task=tasks[e["mark"]], frames=e["frames"],
                             recording_time_s=[round(e["first_frame_time_s"], 3), round(e["last_frame_time_s"], 3)])
                        for k, e in enumerate(episodes)]
    (root / "meta" / "argus_gaze.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return ds


def flat_columns() -> List[str]:
    cols = []
    for name, (_, size, names) in GAZE_FIELDS.items():
        cols += [name] if size == 1 else ["%s_%s" % (name, x) for x in names]
    return cols


def flat_table(table: GazeTable, rows: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    rows = np.arange(len(table.time_s)) if rows is None else rows
    out = {}
    for name, (_, size, names) in GAZE_FIELDS.items():
        a = table.fields[name][rows]
        if size == 1:
            out[name] = a[:, 0]
        else:
            for j, x in enumerate(names):
                out["%s_%s" % (name, x)] = a[:, j]
    return out


def write_full_rate(table: GazeTable, segments, path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    ep = np.full(len(table.time_s), -1, dtype=np.int32)
    for k, (_, a, b) in enumerate(segments):
        ep[(table.time_s >= a) & (table.time_s <= b)] = k
    cols = {"episode_index": ep, "mark": table.mark.astype(np.int32)}
    cols.update(flat_table(table))
    pq.write_table(pa.table(cols), path)
