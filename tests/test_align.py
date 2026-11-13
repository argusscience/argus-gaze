import math
import unittest
from dataclasses import fields

from argus_gaze import geometry as g
from argus_gaze.align import gaze_ray, looked_at, object_name, rank_objects
from argus_gaze.gaze import GazeSample, to_gaze_sample
from argus_gaze.protocol import DataRecord


def sample_looking_at(point, eyes_mid=(0.0, 0.0, 0.0), ipd=0.063):
    """A GazeSample whose two lines of gaze meet exactly at `point` (metres)."""
    le = (eyes_mid[0], eyes_mid[1] - ipd / 2, eyes_mid[2])
    re_ = (eyes_mid[0], eyes_mid[1] + ipd / 2, eyes_mid[2])
    ld, rd = g.normalise(g.sub(point, le)), g.normalise(g.sub(point, re_))
    kw = {f.name: None for f in fields(GazeSample)}
    kw.update(frame_no=0, device_time_s=0.0, left_valid=True, right_valid=True,
              left_eye_position=le, right_eye_position=re_,
              left_gaze_direction=ld, right_gaze_direction=rd,
              gaze_direction=g.normalise(g.sub(point, eyes_mid)))
    conv = g.convergence_point(le, ld, re_, rd)
    kw.update(convergence_point=conv.point, convergence_miss=conv.miss_distance)
    return GazeSample(**kw)


def along(direction_deg_down, distance, sideways=0.0):
    """Point ahead (+x), below (+z) the eyes, at `distance` along a line `direction_deg_down`
    below horizontal, shifted `sideways` metres to the right (+y)."""
    a = math.radians(direction_deg_down)
    return (distance * math.cos(a), sideways, distance * math.sin(a))


class AlignTests(unittest.TestCase):
    def setUp(self):
        # three objects on nearly one line of sight, as in the recording layout
        self.objects = {'near': along(35, 0.40, -0.03), 'middle': along(35, 0.55), 'far': along(35, 0.70, 0.03),
                        'side': along(35, 0.55, 0.25)}

    def test_depth_separates_objects_on_one_line(self):
        for key in ('near', 'middle', 'far'):
            s = sample_looking_at(self.objects[key])
            self.assertEqual(looked_at(s, self.objects), key)
            self.assertEqual(rank_objects(s, self.objects)[0].key, key)

    def test_nothing_when_gaze_is_elsewhere(self):
        s = sample_looking_at(along(10, 0.8, -0.3))
        self.assertIsNone(looked_at(s, self.objects))

    def test_gaze_ray_origin_is_between_the_eyes(self):
        s = sample_looking_at((0.5, 0.0, 0.2))
        origin, d = gaze_ray(s)
        self.assertAlmostEqual(origin[1], 0.0)
        self.assertAlmostEqual(g.norm(d), 1.0)

    def test_ai_lmot_fields_and_names(self):
        rec = DataRecord(frame_no=1, timestamp_100ns=0, update_rate=180, check_state=0,
                         values={'gaze_ai_obj_id': 42, 'no_of_ai_objects': 2},
                         ai_objects=[dict(obj_id=42, obj_horz_cntr=600.0, obj_vert_cntr=400.0, obj_width=80.0,
                                          obj_height=120.0, obj_gaze_horz=0.4, obj_gaze_vert=0.6),
                                     dict(obj_id=47, obj_horz_cntr=900.0, obj_vert_cntr=420.0, obj_width=60.0,
                                          obj_height=60.0, obj_gaze_horz=-1.0, obj_gaze_vert=-1.0)])
        s = to_gaze_sample(rec)
        self.assertEqual(s.gaze_object, 42)
        self.assertEqual(len(s.objects), 2)
        self.assertEqual(s.objects[0]['gaze_in_box'], (0.4, 0.6))
        self.assertIsNone(s.objects[1]['gaze_in_box'])
        self.assertEqual(object_name(s, {42: 'cup', 47: 'apple'}), 'cup')
        s.gaze_object = 0
        self.assertIsNone(object_name(s, {42: 'cup'}))


if __name__ == '__main__':
    unittest.main()
