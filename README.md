# argus_gaze

Read, record and replay real-time data from Argus Science ETVision and ET3Space, using the
documented ETVision network protocol.

**Version 0.3.1, November 2026.** Written against the public *ETVision Network Communication
Manual* v3.0 (October 2025), the *ETVision Manual* (September 2026) and the *ET3Space Manual*
v1.9 (April 2026). The live client has been checked line by line against Argus Science's
ETVision SDK samples, which have been run thousands of times with live ETVision data and video
streams: the same commands, the same TCP connection sequence, and the same order, type and
scale for every data item. The file reader has been run on real ETVision 1.0.12 recordings,
and the ROS 2 node has been tested on ROS 2 Jazzy with a real recording replayed over this
protocol.

A sample dataset built with this code is on Hugging Face:
[argus-science/et3space-workcell](https://huggingface.co/datasets/argus-science/et3space-workcell).

Changes in 0.3.1: commands are sent as 20-byte messages, and the client asks ETVision for its
network protocol version when it connects; the eye-camera image items are read as signed
values; AI object types are read from 1; strings are UTF-8.

Changes in 0.3.0: ROS 2 packages in `ros2/` publish the gaze fields live (see [ROS 2](#ros-2));
the client reads the names of the AI model's object types (commands 26 and 27), and the
emulator answers them from a names file; the dataset converter and the ROS 2 node share one
field conversion, `gaze.spec_values`.

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
| `client` | Live client: TCP command socket plus a UDP or TCP data channel; XDAT, data-file control, single data items, auto calibration, AI model object names |
| `capture` | Records the live stream byte for byte to a `.agscap` file, and reads it back |
| `csvfile` | Reads ETVision text (`.csv`) data files |
| `replay` | ETVision emulator: replays a `.csv` or `.agscap` file over the same protocol, so software can be developed with no eye tracker present |
| `geometry` | Units, head-pose rotation (azimuth, elevation, roll), convergence point of the two lines of gaze |
| `gaze` | `GazeSample`: one record in SI units and named fields, including the AI LMOT object type at the point of gaze and the list of recognized objects. `spec_values` gives the fields of the specification, as used by the ROS 2 node and the dataset converter |
| `align` | Matches gaze to objects a robot system already tracks: by geometry (line of gaze and convergence point against 3D object positions) or by AI LMOT object name |
| `checks` | Consistency checks and a summary for a recorded file |
| `lerobot_export` | Converts a recording and its videos to a LeRobot dataset (format v3.0), with the gaze fields of the field specification. Needs `numpy`, `lerobot[dataset]` and ffmpeg |
| `synthetic` | **Synthetic test data for software testing only.** Never publish it or present it as a recording. |
| `ros2/argus_gaze_msgs` | ROS 2 messages: `Gaze`, `Object`, `ObjectArray` |
| `ros2/argus_gaze_ros` | ROS 2 node that publishes live gaze; launch files for a live ETVision and for the demo recording |

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

## ROS 2

`ros2/argus_gaze_ros` publishes live gaze in ROS 2, and `ros2/argus_gaze_msgs` defines its
messages. Tested on ROS 2 Jazzy. Clone the repository into a colcon workspace; the node
installs the `argus_gaze` library with it.

```bash
mkdir -p ~/argus_ws/src && cd ~/argus_ws/src
git clone https://github.com/argusscience/argus-gaze.git
cd ~/argus_ws && colcon build && source install/setup.bash

# Without an eye tracker: replay the demo recording as a live ETVision and publish it
ros2 launch argus_gaze_ros demo.launch.py

# With ETVision (address and port from its Network Configuration dialog)
ros2 launch argus_gaze_ros argus_gaze.launch.py host:=192.168.1.20 port:=5000
```

| Topic | Type | Published |
|---|---|---|
| `/argus/gaze` | `argus_gaze_msgs/Gaze` | Every sample (180 Hz), in frame `et3space_world` |
| `/argus/gaze/convergence_point` | `geometry_msgs/PointStamped` | Where the two lines of gaze meet, in every sample in which both eyes are valid |
| `/argus/objects` | `argus_gaze_msgs/ObjectArray` | The objects AI LMOT recognizes in the scene camera image, with their names, when the list changes |
| `/argus/event` | `std_msgs/UInt16` | The event code (XDAT), when it changes |
| `/tf` | | `et3space_world` → `argus_head_tracker`: pose of the head-tracker sensor |
| `/argus/set_event` | `std_msgs/UInt16` | Subscribed: sends an event code to ETVision, so robot actions are marked in the gaze record |

`Gaze` carries the fields of the dataset, with the same names, units and values:
`left_eye_position`, `right_eye_position`, `left_direction`, `right_direction`, `direction`,
`convergence_point`, `convergence_miss`, `vergence_angle`, `valid`, `surface_id`,
`surface_point`, `gaze_length`, `head_pose`, `pupil_diameter`, `fixation_duration`, `event`,
`gaze_object` and `workcell_camera_point`, plus the ETVision time stamp and frame number.
Messages are stamped with the host time at arrival. Units are in
[`Gaze.msg`](ros2/argus_gaze_msgs/msg/Gaze.msg).

Relate the gaze to the robot with one static transform from the robot base to
`et3space_world`, for example:

```bash
ros2 run tf2_ros static_transform_publisher --x 0.6 --y 0 --z 0 --yaw 3.1416 \
    --frame-id base_link --child-frame-id et3space_world
```

Parameters (transport, TCP by default; position units; frame names; AI model names file) are in
[`config/argus_gaze.yaml`](ros2/argus_gaze_ros/config/argus_gaze.yaml). The names of the AI
model's object types are read from ETVision when the node connects. Topic names are relative
to the node's namespace, so several eye trackers can run side by side
(`--ros-args -r __ns:=/operator2`). `colcon test --packages-select argus_gaze_ros` runs the
node against the emulator over TCP and UDP.

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

How the code reads the protocol of the Network Communication Manual v3.0:

1. **Commands** are 20-byte messages, with a zero argument where a command takes none, and the
   checksum rule of section 1 (two's complement of the byte sum). Some printed examples carry
   a value 5 lower; the emulator accepts both.
2. **Protocol version.** On connecting, the client asks ETVision for its network protocol
   version (command 0). Version 1, the layout of the manual v3.0, is read here; version 0,
   used by earlier systems, has a different layout and is refused.
3. **CheckState** is read from bytes 48 to 55, and bit 59 (AI_Objects) is supported.
4. **UpdateRate** (bytes 40 to 43) is read as an integer.
5. **Time stamps** are in units of 100 ns from the device's own zero point; use
   differences between them.
6. **Item order and types** follow section 3, with the eyelid and blink items after the aux
   sensor items (bits 48 to 50). The eye-camera image items (bits 6 to 15) are signed
   16-bit values.
7. **Status bits.** Only bits 2 to 5 (pupil and corneal reflection found, each eye) are
   interpreted.
8. **`Gaze_AI_Obj_ID`** is read as 4 bytes.
9. **Convergence point.** Computed from the two eyes' lines of gaze (closest approach).
10. **AI model object names** (commands 26 and 27). Object types are numbered from 1 to the
    count ETVision reports; 0 means no object.

## License

Copyright 2026 Argus Science LLC. Licensed under the [Apache License 2.0](LICENSE).

Questions and requests: info@argusscience.com
