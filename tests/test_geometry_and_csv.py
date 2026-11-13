import math
import os
import tempfile
import unittest

from argus_gaze import geometry as g
from argus_gaze import synthetic
from argus_gaze.checks import summarise
from argus_gaze.csvfile import normalise_label, read_csv, write_csv
from argus_gaze.gaze import to_gaze_sample


class GeometryTests(unittest.TestCase):
    def test_convergence_of_exact_rays(self):
        target = (50.0, 0.0, 0.0)
        le, re_ = (0.0, -3.15, -55.0), (0.0, 3.15, -55.0)
        c = g.convergence_point(le, g.sub(target, le), re_, g.sub(target, re_))
        for a, b in zip(c.point, target):
            self.assertAlmostEqual(a, b, places=6)
        self.assertAlmostEqual(c.miss_distance, 0.0, places=6)

    def test_parallel_rays_have_no_convergence(self):
        self.assertIsNone(g.convergence_point((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)))

    def test_error_table_matches_earlier_estimate(self):
        lat, depth = g.lateral_and_depth_error(0.5)
        self.assertAlmostEqual(lat * 100, 0.44, places=2)      # cm
        self.assertAlmostEqual(depth * 100, 3.5, delta=0.2)   # cm

    def test_rotation_is_orthonormal(self):
        m = g.rotation_from_az_el_rl(30, -20, 10)
        for i in range(3):
            col = (m[0][i], m[1][i], m[2][i])
            self.assertAlmostEqual(g.norm(col), 1.0, places=9)
        q = g.quaternion_from_matrix(m)
        self.assertAlmostEqual(sum(x * x for x in q), 1.0, places=9)


class CsvTests(unittest.TestCase):
    def test_label_normalisation(self):
        self.assertEqual(normalise_label("ET3S _gaze_length"), "et3s_gaze_length")
        self.assertEqual(normalise_label("ET3S scene_num"), "et3s_scene_number")
        self.assertEqual(normalise_label("hdrtk_el"), "hdtrk_el")
        self.assertEqual(normalise_label("left_eyelocation_x"), "left_eye_location_x")
        self.assertEqual(normalise_label("eyelid_upper_vert_left"), "left_eyelid_upper_vert")
        self.assertEqual(normalise_label("blink_confidence_right"), "right_blink_confidence")

    def test_write_then_read(self):
        recs = synthetic.generate(duration_per_target_s=0.2)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "synthetic.csv")
            write_csv(path, recs)
            desc, it = read_csv(path)
            back = list(it)
        self.assertEqual(desc.header.units, "cm")
        self.assertTrue(desc.header.et3space)
        self.assertEqual(desc.unknown_columns, [])
        self.assertEqual(len(back), len(recs))
        self.assertAlmostEqual(back[10].values["left_gaze_dir_x"], recs[10].values["left_gaze_dir_x"], places=3)
        self.assertEqual(back[10].values["status"], recs[10].values["status"])


class GazeSampleTests(unittest.TestCase):
    def test_convergence_lands_near_targets_in_metres(self):
        recs = synthetic.generate(duration_per_target_s=1.0, noise_deg=0.0)
        s = to_gaze_sample(recs[90], "cm")  # first target T1, mid-fixation
        t1 = tuple(x / 100 for x in synthetic.TARGETS["T1"])
        self.assertIsNotNone(s.convergence_point)
        self.assertLess(g.norm(g.sub(s.convergence_point, t1)), 0.003)  # quantisation only
        self.assertEqual(s.surface_id, 0)

    def test_checks_pass_on_consistent_data(self):
        report = summarise(synthetic.generate(duration_per_target_s=1.0))
        self.assertTrue(report["check_gaze_dir_unit_length"]["pass"])
        self.assertTrue(report["check_eye_separation_mm"]["pass"])
        self.assertTrue(report["check_convergence"]["pass"])
        self.assertTrue(report["check_vergence_consistency_deg"]["pass"])

    def test_checks_flag_wrong_units(self):
        report = summarise(synthetic.generate(duration_per_target_s=0.5), units="in")
        self.assertFalse(report["check_eye_separation_mm"]["pass"])


if __name__ == "__main__":
    unittest.main()
