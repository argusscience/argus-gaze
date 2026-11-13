import os
import tempfile
import threading
import time
import unittest

from argus_gaze import protocol as p
from argus_gaze import synthetic
from argus_gaze.capture import CaptureWriter
from argus_gaze.client import ETVisionClient
from argus_gaze.csvfile import write_csv
from argus_gaze.replay import Emulator, Source


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.recs = synthetic.generate(duration_per_target_s=0.5)
        self.em = Emulator(Source.from_records(self.recs), host="127.0.0.1", port=0, speed=4.0, loop=True)
        self.port = self.em.start()

    def tearDown(self):
        self.em.stop()

    def _collect(self, client, n):
        out = []
        for rec in client.records(idle_timeout=3.0):
            out.append(rec)
            if len(out) >= n:
                break
        return out

    def test_udp_stream_and_xdat(self):
        with ETVisionClient("127.0.0.1", self.port) as c:
            c.start_udp(0)
            got = self._collect(c, 50)
            self.assertEqual(len(got), 50)
            self.assertTrue(all("left_gaze_dir_x" in r.values for r in got))
            frames = [r.frame_no for r in got]
            self.assertEqual(frames, sorted(frames))
            c.set_xdat(321)
            time.sleep(0.1)
            later = self._collect(c, 20)
            self.assertTrue(any(r.values.get("xdat") == 321 for r in later))

    def test_tcp_stream(self):
        with ETVisionClient("127.0.0.1", self.port) as c:
            c.start_tcp()
            got = self._collect(c, 40)
            self.assertEqual(len(got), 40)

    def test_get_data_item_and_auto_calibration(self):
        with ETVisionClient("127.0.0.1", self.port) as c:
            c.start_udp(0)
            self._collect(c, 5)
            item = c.get_data_item(20)
            self.assertIn("vergence_angle", item)
            self.assertTrue(c.auto_calibrate(timeout=3.0))

    def test_replay_from_csv_and_capture(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "s.csv")
            write_csv(csv_path, self.recs)
            cap_path = os.path.join(d, "s.agscap")
            with CaptureWriter(cap_path) as w:
                for i, r in enumerate(self.recs):
                    w.write(p.encode_data_message(r), receive_time=1000.0 + i / 180.0)
            for source in (Source.from_csv(csv_path), Source.from_capture(cap_path)):
                em = Emulator(source, host="127.0.0.1", port=0, speed=4.0)
                port = em.start()
                try:
                    with ETVisionClient("127.0.0.1", port) as c:
                        c.start_udp(0)
                        got = self._collect(c, 30)
                        self.assertEqual(len(got), 30)
                finally:
                    em.stop()


if __name__ == "__main__":
    unittest.main()
