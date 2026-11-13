"""Rows copied from a real ETVision 1.0.12 recording (trial S1-8, November 11, 2026)."""
import math
import os
import unittest

from argus_gaze.csvfile import read_csv
from argus_gaze.gaze import to_gaze_sample

HERE = os.path.dirname(__file__)
EXCERPT = os.path.join(HERE, "data", "etvision_1.0.12_excerpt.csv")

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class RealCsv(unittest.TestCase):
    def test_header_and_columns(self):
        desc, recs = read_csv(EXCERPT)
        self.assertEqual(desc.header.units, "cm")
        self.assertTrue(desc.header.et3space)
        self.assertIn("1.0.12", desc.header.version)
        # first column holds time in seconds although it is labelled start_of_record
        self.assertEqual(desc.columns[0], "time_s")
        self.assertNotIn("", desc.columns)  # trailing comma ignored
        recs = list(recs)
        self.assertEqual(len(recs), 10)
        self.assertAlmostEqual(recs[0].time_s, 2.556)
        self.assertEqual(recs[0].values["status"], 0b11111110)

    def test_gaze_sample_units(self):
        _, recs = read_csv(EXCERPT)
        g = to_gaze_sample(next(recs), "cm")
        self.assertTrue(g.left_valid and g.right_valid)
        self.assertEqual(g.surface_id, 0)
        self.assertLess(abs(g.surface_point[0]), 1.0)        # metres, not centimetres
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in g.gaze_direction)), 1.0, places=2)
        self.assertIsNone(g.left_eye_position)               # not recorded in this file

    @unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
    def test_table_and_segments(self):
        from argus_gaze.lerobot_export import gaze_table, segments_by_mark, lerobot_features, GAZE_FIELDS
        t = gaze_table(EXCERPT)
        self.assertEqual(len(t.time_s), 10)
        self.assertTrue(math.isnan(t.fields["convergence_point"][0, 0]))
        self.assertAlmostEqual(float(t.fields["vergence_angle"][0, 0]), math.radians(2.0), delta=0.02)
        # the leading Mark 3 is left over from the previous recording and is skipped
        segs = segments_by_mark(t, keep=[1, 3], min_s=0.0)
        self.assertEqual([s[0] for s in segs], [1])
        f = lerobot_features({"workcell_camera": (640, 360)})
        self.assertEqual(f["observation.images.workcell_camera"]["shape"], (360, 640, 3))
        self.assertEqual(len([k for k in f if k.startswith("observation.gaze.")]), len(GAZE_FIELDS))


if __name__ == "__main__":
    unittest.main()
