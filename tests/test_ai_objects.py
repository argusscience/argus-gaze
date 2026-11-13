import os
import tempfile
import unittest

from argus_gaze import protocol as p
from argus_gaze import synthetic
from argus_gaze.align import load_object_names
from argus_gaze.client import ETVisionClient
from argus_gaze.replay import Emulator, Source


class AIObjectMessageTests(unittest.TestCase):
    def test_count_round_trip(self):
        msg = p.parse_message(p.encode_ai_object_count_response(80))
        self.assertTrue(msg.checksum_ok)
        self.assertEqual(msg.size, 20)  # manual: total message size 20
        self.assertEqual(p.decode_ai_object_count_response(msg), 80)

    def test_info_round_trip(self):
        info = p.AIObjectInfo(object_id=42, enabled=True, color=0x00FF8800, name="cup")
        raw = p.encode_ai_object_info_response(42, info)
        msg = p.parse_message(raw)
        self.assertTrue(msg.checksum_ok)
        self.assertEqual(msg.size, 28 + len("cup") + 1)  # manual: 28 + name length + 1
        self.assertEqual(p.decode_ai_object_info_response(msg), info)

    def test_info_error(self):
        msg = p.parse_message(p.encode_ai_object_info_response(7, None))
        self.assertTrue(msg.is_error)
        with self.assertRaises(p.ProtocolError):
            p.decode_ai_object_info_response(msg)

    def test_command_messages(self):
        m = p.parse_message(p.get_ai_object_count())
        self.assertEqual((m.base_cmd, m.size), (26, 16))
        m = p.parse_message(p.get_ai_object_info(5))
        self.assertEqual((m.base_cmd, m.size, m.arg_uint32()), (27, 20, 5))

    def test_names_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "AINames.txt")
            with open(path, "w") as fh:
                fh.write("person\nbicycle\ncup\n\n")
            self.assertEqual(load_object_names(path), {1: "person", 2: "bicycle", 3: "cup"})
            self.assertEqual(load_object_names(path, first_id=0)[0], "person")


class AIObjectClientTests(unittest.TestCase):
    def test_names_from_emulator(self):
        names = {1: "person", 2: "bicycle", 3: "cup"}
        em = Emulator(Source.from_records(synthetic.generate(duration_per_target_s=0.2)),
                      host="127.0.0.1", port=0, object_names=names)
        port = em.start()
        try:
            with ETVisionClient("127.0.0.1", port) as c:
                self.assertEqual(c.get_ai_object_count(), 3)
                self.assertIsNone(c.get_ai_object_info(0))
                self.assertEqual(c.get_ai_object_info(3).name, "cup")
                self.assertEqual(c.get_ai_object_names(), names)
        finally:
            em.stop()

    def test_no_model(self):
        em = Emulator(Source.from_records(synthetic.generate(duration_per_target_s=0.2)),
                      host="127.0.0.1", port=0)
        port = em.start()
        try:
            with ETVisionClient("127.0.0.1", port) as c:
                with self.assertRaises(p.ProtocolError):
                    c.get_ai_object_count()
        finally:
            em.stop()


if __name__ == "__main__":
    unittest.main()
