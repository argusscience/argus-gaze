"""Consistency checks and a summary for a recorded session (csv or capture file).

The first real ETVision/ET3Space file will be checked with this before anything else is
built on it. Each check tests an assumption taken from the manuals, so a failure points
directly at the assumption to correct (field order, units, frames, rates).

    python -m argus_gaze.checks session01.csv
    python -m argus_gaze.checks session01.agscap --units cm
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
from typing import Dict, Iterable, List, Optional

from . import geometry as g
from .gaze import to_gaze_sample
from .protocol import DataRecord


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def summarise(records: Iterable[DataRecord], units: str = "cm", et3space: bool = True) -> Dict[str, object]:
    recs = list(records)
    out: Dict[str, object] = {"records": len(recs)}
    if not recs:
        return out
    present = sorted({k for r in recs for k in r.values})
    out["fields_present"] = present

    # timing
    times = [(r.time_s if r.time_s is not None else r.timestamp_s) for r in recs]
    dts = [b - a for a, b in zip(times, times[1:]) if b > a]
    if dts:
        span = times[-1] - times[0]
        mean_dt = span / (len(times) - 1) if span > 0 else 0.0
        out["duration_s"] = round(span, 3)
        out["mean_rate_hz"] = round(1.0 / mean_dt, 1) if mean_dt else None
        out["gaps_over_3_samples"] = sum(1 for d in dts if mean_dt and d > 3 * mean_dt)
    out["overtime_total"] = int(sum(r.values.get("overtime_count", 0) for r in recs))

    samples = [to_gaze_sample(r, units, et3space) for r in recs]
    valid_both = [s for s in samples if s.left_valid and s.right_valid]
    out["both_eyes_valid_pct"] = round(_pct(len(valid_both), len(samples)), 1)

    # check 1: gaze directions should be unit vectors
    norms = [g.norm(d) for s in valid_both for d in (s.left_gaze_direction, s.right_gaze_direction) if d]
    if norms:
        bad = sum(1 for n in norms if abs(n - 1.0) > 0.02)
        out["check_gaze_dir_unit_length"] = {"pass": bad == 0, "outside_2pct": bad,
                                             "median_norm": round(stats.median(norms), 4)}

    # check 2: distance between the eyes should be a plausible interpupillary distance
    ipds = [g.norm(g.sub(s.left_eye_position, s.right_eye_position)) * 1000
            for s in valid_both if s.left_eye_position and s.right_eye_position]
    if ipds:
        med_ipd = stats.median(ipds)
        out["check_eye_separation_mm"] = {"pass": 50 <= med_ipd <= 75, "median": round(med_ipd, 1),
                                          "note": "expected 50-75 mm; ~10x off suggests wrong units"}

    # check 3: lines of gaze should nearly meet, in front of the eyes
    conv = [s for s in valid_both if s.convergence_point is not None]
    if valid_both:
        out["convergence_found_pct"] = round(_pct(len(conv), len(valid_both)), 1)
    if conv:
        misses = [s.convergence_miss * 1000 for s in conv]
        dists = []
        for s in conv:
            mid_eye = g.scale(g.add(s.left_eye_position, s.right_eye_position), 0.5)
            dists.append(g.norm(g.sub(s.convergence_point, mid_eye)))
        out["check_convergence"] = {
            "median_miss_mm": round(stats.median(misses), 1),
            "median_distance_m": round(stats.median(dists), 3),
            "pass": stats.median(misses) < 30,
        }

    # check 4: reported vergence angle vs angle between the two gaze directions
    diffs = []
    for s in valid_both:
        if s.vergence_angle_deg is not None and s.left_gaze_direction and s.right_gaze_direction:
            diffs.append(abs(g.angle_deg(s.left_gaze_direction, s.right_gaze_direction) - s.vergence_angle_deg))
    if diffs:
        out["check_vergence_consistency_deg"] = {"pass": stats.median(diffs) < 0.5,
                                                 "median_abs_diff": round(stats.median(diffs), 3)}

    # check 5: cyclopean direction should lie between the two eye directions
    if et3space:
        cyc_diffs = []
        for s in valid_both:
            if s.gaze_direction and s.left_gaze_direction and s.right_gaze_direction:
                mean_dir = g.normalise(g.add(s.left_gaze_direction, s.right_gaze_direction))
                cyc_diffs.append(g.angle_deg(mean_dir, s.gaze_direction))
        if cyc_diffs:
            out["check_cyclopean_direction_deg"] = {"pass": stats.median(cyc_diffs) < 2.0,
                                                    "median_angle_to_mean_eye_dir": round(stats.median(cyc_diffs), 3)}

    # surfaces
    surf: Dict[str, int] = {}
    for s in samples:
        if s.surface_id is not None:
            surf[str(s.surface_id)] = surf.get(str(s.surface_id), 0) + 1
    if surf:
        out["surface_hits_pct"] = {k: round(_pct(v, len(samples)), 1) for k, v in sorted(surf.items())}

    # event marks
    xd = [s.xdat for s in samples if s.xdat]
    if xd:
        out["xdat_values_seen"] = sorted(set(xd))
    return out


def load(path: str, units: Optional[str]):
    if path.lower().endswith(".csv"):
        from .csvfile import read_csv
        desc, recs = read_csv(path)
        hdr_units = desc.header.units
        return list(recs), (units or hdr_units or "cm"), (desc.header.et3space is not False), desc
    from .capture import read_capture_records
    return list(read_capture_records(path)), (units or "cm"), True, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check and summarise a recorded ETVision session")
    ap.add_argument("path", help=".csv or .agscap file")
    ap.add_argument("--units", help="position units if not in the file: cm or in")
    args = ap.parse_args(argv)
    recs, units, et3s, desc = load(args.path, args.units)
    report = summarise(recs, units, et3s)
    report["units"] = units
    if desc is not None:
        report["csv_header"] = desc.header.raw
        report["unrecognised_columns"] = desc.unknown_columns
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
