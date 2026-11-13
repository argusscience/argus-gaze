"""Build the LeRobot dataset from the demo recording of November 12, 2026.

    python examples/build_demo_dataset.py <folder with the recording> <output folder> [--videos <folder>]

The camera videos are taken as ETVision recorded them (<stem>_SSC.wmv and <stem>_Scene.wmv in the
recording folder), with the gaze cursor and surface outlines that ETVision draws into them.
--videos takes <stem>_SSC_clean.mp4 and <stem>_Scene_clean.mp4 from another folder instead.

Needs ffmpeg and `pip install "lerobot[dataset]"`.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from argus_gaze.lerobot_export import gaze_table, segments_by_mark, write_full_rate, write_lerobot  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
other = Path(sys.argv[sys.argv.index("--videos") + 1]) if "--videos" in sys.argv else None
stem = "PO1_2026_11_12_12_58_19"

TASKS = {
    1: "Look at targets 1 to 9 in turn.",
    2: "Keep eyes on the table center; move the head.",
    3: "Move two parts from tray to plate and back.",
}

CM = 0.01
METADATA = {
    "field_specification": "Argus gaze field specification, draft v0.2 (October 30, 2026), with two proposed "
                           "additions: workcell_camera_point and device_time.",
    "recording": {"date": "2026-11-12", "participant": "P01", "etvision_version": "1.0.12.0", "et3space": True,
                  "position_units_on_etvision": "centimeters", "gaze_rate_hz": 180,
                  "head_tracker": "OptiTrack V120:Trio", "environment": "environment 2"},
    "frames": {
        "world": "et3space_world: the ET3Space global frame, which is the motion-capture frame, in metres. "
                 "A robot application relates it to its robot base with one static transform.",
        "world_to_robot_transform": None,
        "surface": "2D frame of each surface: horizontal = surface Y axis, vertical = surface Z axis, "
                   "origin at the surface centre, in metres.",
        "head_pose": "Pose of the head-tracker sensor in the world frame: x, y, z, then quaternion qx, qy, qz, qw. "
                     "Azimuth, elevation and roll applied about Z, Y and X (to be confirmed, specification question 3).",
    },
    "units": {"position": "m", "direction": "unit vector", "angle": "rad", "time": "s",
              "pupil_diameter": "as set on ETVision (millimetres by default)",
              "workcell_camera_point": "pixels of the 1280 x 720 workcell camera image"},
    "sampling": "Each dataset frame takes the ETVision sample nearest to the frame time. The full-rate stream "
                "is in extras/gaze_full_rate.parquet.",
    "validity": "valid = 1 when both the pupil and the corneal reflection of that eye are found. Fields from an "
                "invalid eye are NaN.",
    "surfaces": [
        {"id": 0, "name": "table", "size_m": [60.96 * CM, 91.44 * CM]},
        {"id": 1, "name": "back panel", "size_m": [48.26 * CM, 55.88 * CM]},
        {"id": 2, "name": "fixture plate", "size_m": [24.765 * CM, 24.765 * CM]},
        {"id": 3, "name": "parts tray", "size_m": [11.43 * CM, 11.43 * CM]},
    ],
    "targets": [
        {"target": 1, "surface_id": 0, "point_m": [-12.7 * CM, -25.4 * CM]},
        {"target": 2, "surface_id": 0, "point_m": [22.9 * CM, -12.7 * CM]},
        {"target": 3, "surface_id": 0, "point_m": [21.6 * CM, 22.9 * CM]},
        {"target": 4, "surface_id": 2, "point_m": [-4.3 * CM, -4.6 * CM]},
        {"target": 5, "surface_id": 2, "point_m": [6.4 * CM, 2.9 * CM]},
        {"target": 6, "surface_id": 1, "point_m": [2.54 * CM, -0.635 * CM]},
        {"target": 7, "surface_id": 1, "point_m": [-16.51 * CM, 20.955 * CM]},
        {"target": 8, "surface_id": 1, "point_m": [2.54 * CM, 20.955 * CM]},
        {"target": 9, "surface_id": 1, "point_m": [21.59 * CM, 20.955 * CM]},
    ],
    "cameras": {
        "workcell_camera": "Stationary Scene Camera, fixed above the work area, the view of a robot's external "
                           "camera. Recorded at 1280 x 720 with a variable frame rate; resampled to 30 fps and "
                           "640 x 360.",
        "head_camera": "ETVision head-mounted scene camera, the participant's view. Recorded at 1280 x 720, "
                       "30 fps; resized to 640 x 360.",
        "processing": "Both videos are as ETVision recorded them, with the gaze cursor and the outlines of the four "
                      "surfaces drawn in. Versions without the cursor and outlines are available on request from "
                      "info@argusscience.com.",
    },
    "ai_lmot": None,
    "notes": [
        "ETVision delivered about 171 samples per second rather than 180 (about 5% missed), with gaps of up to 0.21 s.",
        "The head tracker dropped out five times, 0.75 s in total, each under 0.25 s.",
        "AI LMOT was off, so gaze_object is 0 throughout.",
        "The first 6.8 s, before Part 1 starts, belong to no episode.",
    ],
}

if out.exists():
    shutil.rmtree(out)
T = gaze_table(str(src / (stem + ".csv")))
segs = segments_by_mark(T, keep=TASKS.keys())
if other is None:
    videos = {"workcell_camera": str(src / (stem + "_SSC.wmv")), "head_camera": str(src / (stem + "_Scene.wmv"))}
else:
    videos = {"workcell_camera": str(other / (stem + "_SSC_clean.mp4")), "head_camera": str(other / (stem + "_Scene_clean.mp4"))}
write_lerobot(T, segs, TASKS, videos, root=str(out), repo_id="argus-science/et3space-workcell",
              fps=30, size=(640, 360), metadata=METADATA)
write_full_rate(T, segs, out / "extras" / "gaze_full_rate.parquet")
print("done", out)
