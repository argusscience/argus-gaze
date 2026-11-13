# argus_gaze

Read, record and replay real-time data from Argus Science ETVision and ET3Space, using the
documented ETVision network protocol.

**Version 0.2.3, November 2026.** Written against the public *ETVision Network Communication
Manual* v3.0 (October 2025), the *ETVision Manual* (September 2026) and the *ET3Space Manual*
v1.9 (April 2026). The file reader has been run on real ETVision 1.0.12 recordings; the live
network client has not yet been run against a real system.

A sample dataset built with this code is on Hugging Face:
[argus-science/et3space-workcell](https://huggingface.co/datasets/argus-science/et3space-workcell).

Changes in 0.2.3: released under the Apache License 2.0; documentation for publication.

Changes in 0.2.2: `examples/build_demo_dataset.py` takes the camera videos as ETVision recorded
them, with the gaze cursor and surface outlines, writes the full-rate gaze file as well, and uses
shorter task descriptions.

Changes in 0.2.1: a full demo recording in `examples/data`, so the emulator and the dataset
converter can be tried without an eye tracker; `examples/build_demo_dataset.py` builds the
published dataset from it.

Changes in 0.2.0: the `.csv` reader handles the header and time column that ETVision 1.0.12
actually writes; new `lerobot_export` module converts a recording and its videos to a
LeRobot dataset; tests on rows from a real recording.

## What is here

| Module | What it does |
|---|---|
| `protocol` | Message layout, command builders and checksums, streaming data-record decoder and encoder (all 60 items, including the variable-length AI object list) |
| `client` | Live client: TCP command socket plus a UDP or TCP data channel; XDAT, data-file control, single data items, auto calibration |
| `capture` | Records the live stream byte for byte to a `.agscap` file, and reads it back |
| `csvfile` | Reads ETVision text (`.csv`) data files |
| `replay` | ETVision emulator: replays a `.csv` or `.agscap` file over the same protocol, so software can be developed with no eye tracker present |
| `geometry` | Units, head-pose rotation (azimuth, elevation, roll), convergence point of the two lines of gaze |
| `gaze` | `GazeSample`: one record in SI units and named fields, the input for a ROS 2 node or a dataset converter. Includes the AI LMOT object type at the point of gaze and the list of recognized objects |
| `align` | Matches gaze to objects a robot system already tracks: by geometry (line of gaze and convergence point against 3D object positions) or by AI LMOT object name |
| `checks` | Consistency checks and a summary for a recorded file |
| `lerobot_export` | Converts a recording and its videos to a LeRobot dataset (format v3.0), with the gaze fields of the field specification. Needs `numpy`, `lerobot[dataset]` and ffmpeg |
| `synthetic` | **Synthetic test data for software testing only.** Never publish it or present it as a recording. |

Pure Python 3.8+, no dependencies, except `lerobot_export`: `pip install ".[lerobot]"`.

## Try it without an eye tracker

`examples/data/workcell_demo_2026-11-12.csv` is a real 85-second ETVision and ET3Space
recording: a person looks at nine targets, moves the head while looking at one point, then
moves two parts from a tray to a fixture plate and back. Replay it as if it were a live
ETVision, and read it with the client:

```bash
python -m argus_gaze.replay --csv examples/data/workcell_demo_2026-11-12.csv --port 5000
python -m argus_gaze.client --host 127.0.0.1 --port 5000 --tcp --gaze --count 10
```

## Quick start

```bash
pip install .

# Replay a recorded session as if it were a live ETVision on port 5000
python -m argus_gaze.replay --csv session01.csv --port 5000

# Read live data (from ETVision, or from the emulator above)
python -m argus_gaze.client --host 192.168.1.20 --port 5000 --udp 0 --gaze --count 10
python -m argus_gaze.client --host 192.168.1.20 --port 5000 --tcp

# Record the live stream exactly as sent
python -m argus_gaze.capture --host 192.168.1.20 --port 5000 --out session01.agscap

# Check a recording
python -m argus_gaze.checks session01.csv

# Run the tests
python -m unittest discover -s tests
```

On the ETVision PC: click **Listen** in the Network Configuration dialog and tick
**Send Data Result**. Use the address and port shown there.

## Convert a recording to a LeRobot dataset

```python
from argus_gaze.lerobot_export import gaze_table, segments_by_mark, write_lerobot

table = gaze_table("session01.csv")                  # 180 Hz, SI units
segments = segments_by_mark(table, keep=[1, 2, 3])   # one episode per Mark value
tasks = {1: "Look at targets 1 to 9 in turn.", 2: "...", 3: "..."}
videos = {"workcell_camera": "session01_SSC.wmv", "head_camera": "session01_Scene.wmv"}
write_lerobot(table, segments, tasks, videos, root="out/dataset",
              repo_id="argus-science/et3space-workcell", fps=30, size=(640, 360))
```

Each frame stores the two camera images and the gaze fields as `observation.gaze.<name>`,
taking the gaze sample nearest to the frame time. The full-rate stream goes to
`extras/gaze_full_rate.parquet` and the frames, units, surfaces and targets to
`meta/argus_gaze.json`. `examples/build_demo_dataset.py` builds the sample dataset from the
demo recording in `examples/data`.

## Units and frames

* Streaming data does not say whether ETVision is set to inches or centimetres. Pass
  `--units` (default `cm`); `.csv` files state it in their first line.
* `GazeSample` converts positions to metres. The `world` frame is the ET3Space global
  frame, which is the motion-capture frame. A robot application relates it to the robot
  base with one static transform.
* With ET3Space on, `verg_gaze_coord` is a unit direction (the cyclopean line of gaze), not a
  point. The convergence point is computed here from the two eyes' positions and gaze
  directions (closest approach of the two lines).

## Protocol notes

What the code does at points where the protocol leaves a choice:

1. **Checksum.** Outgoing commands use the rule in section 1 (two's complement of the byte
   sum). Some printed examples carry a value 5 lower; the emulator accepts both.
2. **CheckState** is read from bytes 48 to 55, and bit 59 (AI_Objects) is supported.
3. **UpdateRate** (bytes 40 to 43) is read as an integer, with a float fallback if the
   integer is implausible.
4. **Time stamps** are in units of 100 ns from the device's own zero point; use
   differences between them.
5. **Item order** follows the Network Communication Manual v3.0, with the eyelid and blink
   items after the aux sensor items (bits 48 to 50).
6. **Status bits.** Only bits 2 to 5 (pupil and corneal reflection found, each eye) are
   interpreted.
7. **`Gaze_AI_Obj_ID`** is read as 4 bytes.
8. **Convergence point.** Computed from the two eyes' lines of gaze (closest approach).

## License

Copyright 2026 Argus Science LLC. Licensed under the [Apache License 2.0](LICENSE).

Questions and requests: info@argusscience.com
