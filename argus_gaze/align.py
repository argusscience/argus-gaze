"""Match gaze to objects that a robot system already tracks.

Robot systems usually track objects already (motion-capture markers, pose estimation,
detectors with labels). These helpers match a GazeSample against what such a system
provides, after the gaze has been put in the same frame as the objects.

Geometric matching (objects with 3D positions)
    For each object: the angle between the line of gaze and the direction from the eyes
    to the object, less the object's own angular radius; and the difference in depth
    between the object and the convergence point, along the line of gaze. The object
    looked at has the smallest angle; the depth difference, judged against the expected
    depth error at that distance, separates objects that lie along nearly the same line.
    This identifies a particular object, also among identical ones, and works sample by
    sample, so it applies to objects being moved as long as their positions are
    time-stamped.

Semantic matching (objects with labels)
    AI LMOT reports the type of object at the point of gaze (an index into the names of
    the loaded AI model). `object_name` turns it into a name to compare with the robot
    system's labels; the convergence point gives where that object was when looked at.
"""
from __future__ import annotations

import math
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from . import geometry as g
from .gaze import GazeSample

Vec3 = Tuple[float, float, float]


@dataclass
class Candidate:
    key: Hashable
    distance: float                 # eyes to object (m)
    angle_deg: float                # line of gaze to object, less the object's angular radius
    depth_offset: Optional[float]   # object minus convergence point, along the line of gaze (m)
    depth_tolerance: float          # expected depth error at this distance (m)
    score: float                    # lower is better


def gaze_ray(sample: GazeSample) -> Optional[Tuple[Vec3, Vec3]]:
    """Origin (midpoint of the eyes) and unit direction of the line of gaze."""
    if not (sample.left_eye_position and sample.right_eye_position):
        return None
    origin = g.scale(g.add(sample.left_eye_position, sample.right_eye_position), 0.5)
    d = sample.gaze_direction
    if d is None and sample.left_gaze_direction and sample.right_gaze_direction:
        d = g.add(sample.left_gaze_direction, sample.right_gaze_direction)
    if d is None or g.norm(d) == 0:
        return None
    return origin, g.normalise(d)


def rank_objects(sample: GazeSample, objects: Mapping[Hashable, Vec3],
                 radius: float = 0.03, angle_scale_deg: float = 1.0,
                 angle_error_deg: float = 0.5) -> List[Candidate]:
    """Rank tracked objects (positions in metres, same frame as the gaze) by how well
    they match the line of gaze and the convergence point. `radius` is the approximate
    object radius in metres; `angle_scale_deg` weights the angle against depth."""
    ray = gaze_ray(sample)
    if ray is None:
        return []
    origin, d = ray
    conv_depth = None
    if sample.convergence_point is not None:
        conv_depth = g.dot(g.sub(sample.convergence_point, origin), d)
    out: List[Candidate] = []
    for key, pos in objects.items():
        v = g.sub(pos, origin)
        dist = g.norm(v)
        if dist <= 0:
            continue
        ang = max(0.0, g.angle_deg(v, d) - math.degrees(math.atan2(radius, dist)))
        _, depth_tol = g.lateral_and_depth_error(dist, angle_error_deg)
        depth_tol = max(depth_tol, radius)
        offset = None
        score = (ang / angle_scale_deg) ** 2
        if conv_depth is not None:
            offset = g.dot(v, d) - conv_depth
            score += (offset / depth_tol) ** 2
        out.append(Candidate(key, dist, ang, offset, depth_tol, score))
    out.sort(key=lambda c: c.score)
    return out


def looked_at(sample: GazeSample, objects: Mapping[Hashable, Vec3], radius: float = 0.03,
              max_angle_deg: float = 2.0, max_depth_sigmas: float = 2.0) -> Optional[Hashable]:
    """The tracked object being looked at, or None. An object qualifies when the line of
    gaze passes within `max_angle_deg` of its edge and, if a convergence point exists,
    its depth is within `max_depth_sigmas` expected depth errors of that point."""
    for c in rank_objects(sample, objects, radius=radius):
        if c.angle_deg > max_angle_deg:
            continue
        if c.depth_offset is not None and abs(c.depth_offset) > max_depth_sigmas * c.depth_tolerance:
            continue
        return c.key
    return None


def object_name(sample: GazeSample, names: Sequence[str] | Mapping[int, str]) -> Optional[str]:
    """Name of the AI LMOT object type at the point of gaze, or None when there is none.
    `names` maps type index to name, as read from the loaded model (ETVision commands
    26 and 27, or the model's names file). A mapping is safest: whether the model's
    first name is index 0 or 1 is still to be confirmed (0 is reported as no object)."""
    t = sample.gaze_object
    if not t:
        return None
    if isinstance(names, _MappingABC):
        return names.get(t)
    return names[t] if 0 <= t < len(names) else None


def load_object_names(path: str, first_id: int = 1) -> Dict[int, str]:
    """Read an AI model names file (one object name per line, as in the model's
    AINames.txt) into object type -> name. Type 0 means no object, so the first line is
    numbered `first_id` (1 by default). Prefer the names ETVision reports
    (ETVisionClient.get_ai_object_names) when it is reachable."""
    names: Dict[int, str] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh]
    while lines and not lines[-1]:
        lines.pop()
    for i, name in enumerate(lines):
        if name:
            names[first_id + i] = name
    return names
