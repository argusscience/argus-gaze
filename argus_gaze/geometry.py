"""Geometry helpers: units, head-pose rotation, and the convergence point of two lines of gaze.

Pure Python (no numpy) so the package installs anywhere, including a ROS 2 machine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]

UNIT_TO_METRES = {"cm": 0.01, "centimeters": 0.01, "centimetres": 0.01,
                  "in": 0.0254, "inch": 0.0254, "inches": 0.0254, "m": 1.0}


def unit_scale(units: str) -> float:
    try:
        return UNIT_TO_METRES[units.strip().lower()]
    except KeyError:
        raise ValueError("unknown position units %r (use 'cm' or 'in')" % units)


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalise(a: Vec3) -> Vec3:
    n = norm(a)
    if n == 0:
        raise ValueError("zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def angle_deg(a: Vec3, b: Vec3) -> float:
    c = dot(normalise(a), normalise(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass
class Convergence:
    point: Vec3            # midpoint of the shortest segment between the two lines
    miss_distance: float   # length of that segment (0 if the lines truly intersect)
    distance_left: float   # along the left line, from the left eye
    distance_right: float  # along the right line, from the right eye
    vergence_deg: float    # angle between the two gaze directions


def convergence_point(left_eye: Vec3, left_dir: Vec3, right_eye: Vec3, right_dir: Vec3,
                      min_vergence_deg: float = 0.05) -> Optional[Convergence]:
    """Closest approach of the two lines of gaze.

    Returns None when the lines are (nearly) parallel or the closest approach lies behind
    either eye, in which case no meaningful convergence point exists."""
    d1, d2 = normalise(left_dir), normalise(right_dir)
    w0 = sub(left_eye, right_eye)
    b = dot(d1, d2)
    d = dot(d1, w0)
    e = dot(d2, w0)
    denom = 1.0 - b * b
    verg = angle_deg(d1, d2)
    if denom < 1e-12 or verg < min_vergence_deg:
        return None
    t1 = (b * e - d) / denom
    t2 = (e - b * d) / denom
    if t1 <= 0 or t2 <= 0:
        return None
    p1 = add(left_eye, scale(d1, t1))
    p2 = add(right_eye, scale(d2, t2))
    mid = scale(add(p1, p2), 0.5)
    return Convergence(mid, norm(sub(p1, p2)), t1, t2, verg)


def rotation_from_az_el_rl(az_deg: float, el_deg: float, rl_deg: float):
    """3x3 rotation for azimuth (about Z), then elevation (about the new Y), then roll
    (about the new X), the order stated in the ET3Space Manual section 3.1.
    Columns are the object's axes expressed in the global frame."""
    a, e, r = (math.radians(x) for x in (az_deg, el_deg, rl_deg))
    ca, sa, ce, se, cr, sr = math.cos(a), math.sin(a), math.cos(e), math.sin(e), math.cos(r), math.sin(r)
    return (
        (ca * ce, ca * se * sr - sa * cr, ca * se * cr + sa * sr),
        (sa * ce, sa * se * sr + ca * cr, sa * se * cr - ca * sr),
        (-se, ce * sr, ce * cr),
    )


def quaternion_from_matrix(m) -> Tuple[float, float, float, float]:
    """(x, y, z, w) quaternion from a 3x3 rotation matrix."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def matvec(m, v: Sequence[float]) -> Vec3:
    return (m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2])


def lateral_and_depth_error(distance: float, angle_error_deg: float = 0.5,
                            ipd: float = 0.063) -> Tuple[float, float]:
    """Order-of-magnitude error at a viewing distance (same units as distance and ipd):
    lateral error of one line of gaze, and depth error from a vergence-angle error."""
    t = math.radians(angle_error_deg)
    lateral = distance * math.tan(t)
    half = ipd / 2.0
    verg = 2 * math.atan(half / distance)
    far = half / math.tan(max(1e-9, (verg - t) / 2))
    near = half / math.tan((verg + t) / 2)
    return lateral, (far - near) / 2.0
