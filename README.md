# argus_gaze

Read, record and replay real-time data from Argus Science ETVision and ET3Space, using the
documented ETVision network protocol.

**Status: internal draft, version 0.2.2, November 12, 2026. Not released.**
Written against the public *ETVision Network Communication Manual* v3.0 (October 2025),
the *ETVision Manual* (September 2026) and the *ET3Space Manual* v1.9 (April 2026).
The file reader has now been run on real ETVision 1.0.12 recordings (trial recordings,
November 2026); the live network client has not yet been run against a real system.

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

table = gaze_table("S1-8.csv")                       # 180 Hz, SI units
segments = segments_by_mark(table, keep=[1, 2, 3])   # one episode per Mark value
tasks = {1: "Look at targets 1 to 9 in turn.", 2: "...", 3: "..."}
videos = {"workcell_camera": "S1-8_SSC.wmv", "head_camera": "S1-8_Scene.wmv"}
write_lerobot(table, segments, tasks, videos, root="out/dataset",
              repo_id="argus-science/et3space-workcell", fps=30, size=(640, 360))
```

Each frame stores the two camera images and the gaze fields as `observation.gaze.<name>`,
taking the gaze sample nearest to the frame time. The full-rate stream goes to
`extras/gaze_full_rate.parquet` and the frames, units, surfaces and targets to
`meta/argus_gaze.json`. `examples/build_trial_dataset.py` builds the preview dataset from
trial recording S1-8.

## Units and frames

* Streaming data does not say whether ETVision is set to inches or centimetres. Pass
  `--units` (default `cm`); `.csv` files state it in their first line.
* `GazeSample` converts positions to metres. The `world` frame is the ET3Space global
  frame, which is the motion-capture frame. A robot application relates it to the robot
  base with one static transform.
* With ET3Space on, `verg_gaze_coord` is a unit direction (the cyclopean line of gaze), not a
  point. The convergence point is computed here from the two eyes' positions and gaze
  directions (closest approach of the two lines).

## Open questions from the documentation review

Most of these will be settled by the first real recording and one live test. None blocks
development.

1. **Checksum.** The stated rule (two's complement of the byte sum) reproduces the manual's
   examples for commands 28 and 29, but the examples for commands 1 to 22, including the
   worked XDAT example, are exactly 5 lower. The client sends the rule; the emulator
   accepts both.
2. **CheckState position.** Section 1 places its least significant bit in byte 47; the
   message table puts CheckState at bytes 48 to 55. We use bytes 48 to 55.
3. **CheckState bit count.** Section 1 says bits 0 to 58 are used; the item table lists
   AI_Objects as bit 59. We support bit 59.
4. **UpdateRate type** (bytes 40 to 43) is not stated. We read it as an integer and fall
   back to a float if the integer is implausible.
5. **Time stamp origin.** Units are 100 ns; the zero point is not stated.
6. **Item order.** The ET3Space Manual appendix E lists the eyelid and blink items right
   after pupil height; the Network Comm Manual v3.0 and the ETVision Manual list them after
   the aux sensor items (bits 48 to 50). We follow the latter two.
7. **Status bits** 0, 1, 6 and 7 are described differently in the two manuals. We use only
   bits 2 to 5 (pupil and corneal reflection found, each eye), where they agree.
8. **Item sizes.** `Gaze_AI_Obj_ID` is 4 bytes in the Network Comm Manual and 2 in the
   appendix tables. We use 4.
9. **Convergence point.** Is a room-coordinate convergence point available directly when
   ET3Space is on? If not, is computing it from the two lines of gaze consistent with the
   internal computation?

## Next steps

1. Field specification v0.3: add `workcell_camera_point` (gaze on the fixed camera image) and `device_time`, as used in the LeRobot export.
2. ROS 2 node publishing `GazeSample` in the robot's transform tree.
3. Release dataset from the recording session; then a live test.

## License

Not yet decided. Internal use only until Argus Science decides how to publish it.
