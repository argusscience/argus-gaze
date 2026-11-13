"""GazeSample: one ETVision/ET3Space record expressed in SI units for downstream software.

This is the shape a ROS 2 node or a LeRobot dataset converter would consume. Field names
here are a draft for the public field specification and may change after review.

Frames
------
world : the ET3Space global frame, which is the motion-capture frame (right-handed).
        Values are converted from ETVision position units (cm or inches) to metres.
        A robot application relates it to the robot base with one static transform.
surface : the 2D frame of a defined ET3Space surface (horizontal = surface Y axis,
        vertical = surface Z axis, per the ET3Space Manual).
scene image : pixels of the head-mounted scene camera image (AI LMOT objects).

Object types (AI LMOT) are indices into the names of the AI model loaded in ETVision.
A type identifies a kind of object (two cups share one type); 0 means no object.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Dict, List, Optional, Tuple

from . import geometry as g
from .protocol import (DataRecord, STATUS_LEFT_PUPIL_FOUND, STATUS_RIGHT_PUPIL_FOUND,
                       STATUS_LEFT_CR_FOUND, STATUS_RIGHT_CR_FOUND)

Vec3 = Tuple[float, float, float]
NAN = float("nan")

# name: (dtype, size, names) -- the order and names follow the field specification
GAZE_FIELDS: Dict[str, Tuple[str, int, List[str]]] = {
    "left_eye_position": ("float32", 3, ["x", "y", "z"]),
    "right_eye_position": ("float32", 3, ["x", "y", "z"]),
    "left_direction": ("float32", 3, ["x", "y", "z"]),
    "right_direction": ("float32", 3, ["x", "y", "z"]),
    "direction": ("float32", 3, ["x", "y", "z"]),
    "convergence_point": ("float32", 3, ["x", "y", "z"]),
    "convergence_miss": ("float32", 1, ["m"]),
    "vergence_angle": ("float32", 1, ["rad"]),
    "valid": ("float32", 2, ["left", "right"]),
    "surface_id": ("int32", 1, ["id"]),
    "surface_point": ("float32", 2, ["horizontal", "vertical"]),
    "gaze_length": ("float32", 1, ["m"]),
    "head_pose": ("float32", 7, ["x", "y", "z", "qx", "qy", "qz", "qw"]),
    "pupil_diameter": ("float32", 2, ["left", "right"]),
    "fixation_duration": ("float32", 1, ["s"]),
    "event": ("int32", 1, ["code"]),
    "gaze_object": ("int32", 1, ["type"]),
    # proposed for v0.3 of the specification
    "workcell_camera_point": ("float32", 2, ["u", "v"]),
    "device_time": ("float32", 1, ["s"]),
}


@dataclass
class GazeSample:
    # timing and markers
    frame_no: int
    device_time_s: float                 # ETVision time stamp (100 ns units) in seconds
    file_time_s: Optional[float]         # seconds from start of a csv file, if any
    xdat: Optional[int]
    mark: Optional[int]
    status: Optional[int]
    left_valid: bool                     # pupil and corneal reflection found, left eye
    right_valid: bool
    # eyes in the world frame (metres) and unit gaze directions
    left_eye_position: Optional[Vec3]
    right_eye_position: Optional[Vec3]
    left_gaze_direction: Optional[Vec3]
    right_gaze_direction: Optional[Vec3]
    gaze_direction: Optional[Vec3]       # cyclopean line of gaze (verg_gaze_coord with ET3Space on)
    vergence_angle_deg: Optional[float]
    # convergence point of the two lines of gaze, computed here (metres)
    convergence_point: Optional[Vec3]
    convergence_miss: Optional[float]    # shortest distance between the two lines (metres)
    # ET3Space surface hit
    surface_id: Optional[int]            # -1 when gaze is on no defined surface
    surface_point: Optional[Tuple[float, float]]  # metres in the surface frame
    gaze_length: Optional[float]         # metres, eye to point of gaze on the surface
    # head tracker sensor pose in the world frame
    head_position: Optional[Vec3]        # metres
    head_orientation_deg: Optional[Vec3]  # azimuth, elevation, roll
    head_quaternion: Optional[Tuple[float, float, float, float]]  # x, y, z, w
    # other useful items
    left_pupil_diameter: Optional[float]
    right_pupil_diameter: Optional[float]
    fixation_duration_s: Optional[float]
    stationary_camera_point: Optional[Tuple[float, float]]  # pixels on the room camera image
    # AI LMOT
    gaze_object: Optional[int] = None    # object type at the point of gaze; 0 when none
    objects: Optional[List[dict]] = None  # every recognized object: type, centre_px, size_px, gaze_in_box

    def as_dict(self) -> dict:
        return asdict(self)


def _vec(rec: DataRecord, prefix: str, scale_m: float = 1.0) -> Optional[Vec3]:
    keys = (prefix + "_x", prefix + "_y", prefix + "_z")
    if not all(k in rec.values for k in keys):
        return None
    return tuple(float(rec.values[k]) * scale_m for k in keys)  # type: ignore[return-value]


def to_gaze_sample(rec: DataRecord, position_units: str = "cm", et3space: bool = True) -> GazeSample:
    """Convert a DataRecord. `position_units` must match the ETVision setting
    (streaming data does not carry it; csv files state it in their first line)."""
    s = g.unit_scale(position_units)
    status = rec.values.get("status")
    left_valid = right_valid = False
    if status is not None:
        st = int(status)
        left_valid = bool(st & STATUS_LEFT_PUPIL_FOUND) and bool(st & STATUS_LEFT_CR_FOUND)
        right_valid = bool(st & STATUS_RIGHT_PUPIL_FOUND) and bool(st & STATUS_RIGHT_CR_FOUND)

    le = _vec(rec, "left_eye_location", s)
    re_ = _vec(rec, "right_eye_location", s)
    ld = _vec(rec, "left_gaze_dir")
    rd = _vec(rec, "right_gaze_dir")
    cyc = _vec(rec, "verg_gaze_coord") if et3space else None

    conv = None
    if le and re_ and ld and rd and left_valid and right_valid:
        try:
            conv = g.convergence_point(le, ld, re_, rd)
        except ValueError:
            conv = None

    surface_id = rec.values.get("et3s_scene_number")
    surface_point = None
    if "et3s_horz_gaze_coord" in rec.values and "et3s_vert_gaze_coord" in rec.values:
        surface_point = (rec.values["et3s_horz_gaze_coord"] * s, rec.values["et3s_vert_gaze_coord"] * s)
    gaze_length = rec.values.get("et3s_gaze_length")

    head_pos = _vec(rec, "hdtrk", s)
    head_ori = None
    head_q = None
    if all(k in rec.values for k in ("hdtrk_az", "hdtrk_el", "hdtrk_rl")):
        head_ori = (rec.values["hdtrk_az"], rec.values["hdtrk_el"], rec.values["hdtrk_rl"])
        head_q = g.quaternion_from_matrix(g.rotation_from_az_el_rl(*head_ori))

    ssc = None
    if "ssc_horz_gaze_coord" in rec.values and "ssc_vert_gaze_coord" in rec.values:
        ssc = (rec.values["ssc_horz_gaze_coord"], rec.values["ssc_vert_gaze_coord"])

    objects = None
    if rec.ai_objects or "no_of_ai_objects" in rec.values:
        objects = []
        for o in rec.ai_objects:
            gh, gv = float(o.get("obj_gaze_horz", -1.0)), float(o.get("obj_gaze_vert", -1.0))
            objects.append(dict(
                type=int(o["obj_id"]),
                centre_px=(float(o["obj_horz_cntr"]), float(o["obj_vert_cntr"])),
                size_px=(float(o["obj_width"]), float(o["obj_height"])),
                gaze_in_box=None if (gh < 0 or gv < 0) else (gh, gv),
            ))
    gaze_object = rec.values.get("gaze_ai_obj_id")

    return GazeSample(
        frame_no=rec.frame_no,
        device_time_s=rec.timestamp_s,
        file_time_s=rec.time_s,
        xdat=rec.values.get("xdat"),
        mark=rec.values.get("mark_value"),
        status=status,
        left_valid=left_valid,
        right_valid=right_valid,
        left_eye_position=le,
        right_eye_position=re_,
        left_gaze_direction=ld,
        right_gaze_direction=rd,
        gaze_direction=cyc,
        vergence_angle_deg=rec.values.get("vergence_angle"),
        convergence_point=conv.point if conv else None,
        convergence_miss=conv.miss_distance if conv else None,
        surface_id=int(surface_id) if surface_id is not None else None,
        surface_point=surface_point,
        gaze_length=gaze_length * s if gaze_length is not None else None,
        head_position=head_pos,
        head_orientation_deg=head_ori,
        head_quaternion=head_q,
        left_pupil_diameter=rec.values.get("left_pupil_diam"),
        right_pupil_diameter=rec.values.get("right_pupil_diam"),
        fixation_duration_s=rec.values.get("fix_duration"),
        stationary_camera_point=ssc,
        gaze_object=int(gaze_object) if gaze_object is not None else None,
        objects=objects,
    )


def _v(x, size: int) -> List[float]:
    if x is None:
        return [NAN] * size
    return [NAN if v is None else float(v) for v in x]


def spec_values(s: GazeSample) -> Dict[str, List[float]]:
    """The fields of the gaze field specification (GAZE_FIELDS) for one sample: SI units,
    world frame, NaN where a value is not available. Values from an invalid eye are NaN.
    The LeRobot converter and the ROS 2 node both use this, so they carry the same values."""
    lv, rv = s.left_valid, s.right_valid
    d = s.gaze_direction
    if d is not None:
        n = math.sqrt(sum(c * c for c in d))
        d = tuple(c / n for c in d) if n > 0 else None
    va = s.vergence_angle_deg
    if s.head_position is not None and s.head_quaternion is not None:
        head = list(s.head_position) + list(s.head_quaternion)
    else:
        head = [NAN] * 7
    return {
        "left_eye_position": _v(s.left_eye_position if lv else None, 3),
        "right_eye_position": _v(s.right_eye_position if rv else None, 3),
        "left_direction": _v(s.left_gaze_direction if lv else None, 3),
        "right_direction": _v(s.right_gaze_direction if rv else None, 3),
        "direction": _v(d, 3),
        "convergence_point": _v(s.convergence_point, 3),
        "convergence_miss": [NAN if s.convergence_miss is None else s.convergence_miss],
        "vergence_angle": [NAN if va is None else math.radians(va)],
        "valid": [float(lv), float(rv)],
        "surface_id": [-1 if s.surface_id is None else s.surface_id],
        "surface_point": _v(s.surface_point, 2),
        "gaze_length": [NAN if s.gaze_length is None else s.gaze_length],
        "head_pose": head,
        "pupil_diameter": [NAN if s.left_pupil_diameter is None else s.left_pupil_diameter,
                           NAN if s.right_pupil_diameter is None else s.right_pupil_diameter],
        "fixation_duration": [NAN if s.fixation_duration_s is None else s.fixation_duration_s],
        "event": [0 if s.xdat is None else int(s.xdat)],
        "gaze_object": [0 if s.gaze_object is None else int(s.gaze_object)],
        "workcell_camera_point": _v(s.stationary_camera_point, 2),
        "device_time": [s.device_time_s],
    }
