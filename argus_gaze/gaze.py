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
from typing import List, Optional, Tuple

from . import geometry as g
from .protocol import (DataRecord, STATUS_LEFT_PUPIL_FOUND, STATUS_RIGHT_PUPIL_FOUND,
                       STATUS_LEFT_CR_FOUND, STATUS_RIGHT_CR_FOUND)

Vec3 = Tuple[float, float, float]


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
