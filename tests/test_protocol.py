import struct
import unittest

from argus_gaze import protocol as p
from argus_gaze import synthetic


class ChecksumTests(unittest.TestCase):
    def test_rule_matches_manual_examples_for_auto_calibration(self):
        # Manual: 16-byte CMD_START_AUTO_CALIBRATION checksum 0xD9, CMD_STOP_AUTO_CALIBRATION 0xD8
        self.assertEqual(p.build_message(p.CMD_START_AUTO_CALIBRATION)[12], 0xD9)
        self.assertEqual(p.build_message(p.CMD_STOP_AUTO_CALIBRATION)[12], 0xD8)

    def test_commands_without_argument_are_20_bytes(self):
        # A zero argument, and the whole message sums to 0 (checksum rule of section 1)
        for cmd in (p.CMD_GET_VERSION, p.CMD_START_AUTO_CALIBRATION, p.CMD_STOP_SDATA_UDP,
                    p.CMD_GET_AIOBJ_TOTALNUM):
            m = p.build_command(cmd)
            self.assertEqual(len(m), 20)
            self.assertEqual(struct.unpack_from("<II", m, 4), (20, cmd))
            self.assertEqual(m[16:], b"\x00" * 4)
            self.assertEqual(sum(m) & 0xFF, 0)

    def test_manual_example_bytes_for_set_xdat_are_accepted_as_legacy(self):
        # The 20 bytes printed in manual section 1 for XDAT = 100
        manual = bytes([0x53, 0x47, 0x41, 0x20, 0x14, 0, 0, 0, 0x05, 0, 0, 0,
                        0x83, 0, 0, 0, 0x64, 0, 0, 0])
        msg = p.parse_message(manual)
        self.assertTrue(msg.checksum_ok)
        self.assertEqual(msg.checksum_style, "legacy")
        self.assertEqual(msg.arg_uint32(), 100)
        # Our message differs only in the checksum byte, and satisfies the stated rule
        ours = p.set_xdat(100)
        self.assertEqual(ours[:12], manual[:12])
        self.assertEqual(ours[16:], manual[16:])
        self.assertEqual(sum(ours) & 0xFF, 0)

    def test_fixed_examples_for_commands_1_to_17_differ_by_five(self):
        manual = {p.CMD_START_DATAFILE_RECORDING: 0xEF, p.CMD_STOP_DATAFILE_RECORDING: 0xEE,
                  p.CMD_OPEN_DATAFILE: 0xED, p.CMD_CLOSE_DATAFILE: 0xEC, p.CMD_STOP_SDATA_UDP: 0xE7,
                  p.CMD_STOP_SVIDEO_UDP: 0xE5, p.CMD_STOP_RVIDEO_UDP: 0xE3,
                  p.CMD_START_SVFILE_RECORDING: 0xE2, p.CMD_CLOSE_SVFILE: 0xDF}
        for cmd, cs in manual.items():
            self.assertEqual((p.build_message(cmd)[12] - cs) & 0xFF, 5, "command %d" % cmd)


class DataRecordTests(unittest.TestCase):
    def test_round_trip_every_item(self):
        rec = synthetic.generate(duration_per_target_s=0.05)[3]
        cs = 0
        for bit in range(59):
            cs |= 1 << bit
        values = dict(rec.values)
        for spec in p.DATA_ITEMS:
            for name in spec[1]:
                values.setdefault(name, 1.0 if spec[2] == "f" else 1)
        rec2 = p.DataRecord(frame_no=42, timestamp_100ns=123456789, update_rate=180,
                            check_state=cs | (1 << p.AI_OBJECTS_BIT), values=values,
                            ai_objects=[{"obj_id": 3, "obj_horz_cntr": 640.0, "obj_vert_cntr": 360.0,
                                         "obj_width": 50.0, "obj_height": 40.0,
                                         "obj_gaze_horz": 0.5, "obj_gaze_vert": -1.0}])
        raw = p.encode_data_message(rec2)
        back = p.parse_data_message(raw)
        self.assertEqual(back.frame_no, 42)
        self.assertEqual(back.timestamp_100ns, 123456789)
        self.assertEqual(back.update_rate, 180)
        self.assertEqual(back.ai_objects[0]["obj_id"], 3)
        for spec in p.DATA_ITEMS:
            bit, names, code, scale = spec
            for name in names:
                tol = scale / 2 + 1e-6 if code != "f" else 1e-4
                self.assertAlmostEqual(float(back.values[name]), float(values[name]), delta=tol, msg=name)

    def test_header_layout(self):
        rec = synthetic.generate(duration_per_target_s=0.02)[0]
        raw = p.encode_data_message(rec)
        sig, size, cmd, cs, data_size, frame_size, frame_no = struct.unpack_from("<IIIIIII", raw)
        self.assertEqual(sig, 0x20414753)
        self.assertEqual(raw[:4], b"SGA ")  # least significant byte first
        self.assertEqual(cmd, 0x81)
        self.assertEqual(cs, 0)
        self.assertEqual(size, len(raw))
        self.assertEqual(size, 56 + data_size)
        self.assertEqual(raw[56], 0xFA)  # start of record
        self.assertEqual(struct.unpack_from("<Q", raw, 48)[0], rec.check_state)

    def test_rejects_wrong_start_of_record(self):
        rec = synthetic.generate(duration_per_target_s=0.02)[0]
        raw = bytearray(p.encode_data_message(rec))
        raw[56] = 0x00
        with self.assertRaises(p.ProtocolError):
            p.parse_data_message(bytes(raw))

    def test_scene_number_minus_one(self):
        rec = p.DataRecord(check_state=p.check_state_for(["et3s_scene_number"]),
                           values={"status": 0, "et3s_scene_number": -1})
        back = p.parse_data_message(p.encode_data_message(rec))
        self.assertEqual(back.values["et3s_scene_number"], -1)

    def test_split_stream_handles_fragments(self):
        recs = synthetic.generate(duration_per_target_s=0.02)[:3]
        blob = b"".join(p.encode_data_message(r) for r in recs)
        buf = bytearray()
        got = []
        for i in range(0, len(blob), 37):
            buf.extend(blob[i:i + 37])
            got.extend(p.split_stream(buf))
        self.assertEqual(len(got), 3)
        self.assertEqual(len(buf), 0)

    def test_get_item_response_round_trip(self):
        rec = synthetic.generate(duration_per_target_s=0.02)[1]
        msg = p.parse_message(p.encode_get_item_response(rec, 20))  # vergence angle
        self.assertTrue(msg.is_response and not msg.is_error)
        out = p.decode_get_item_response(msg)
        self.assertAlmostEqual(out["vergence_angle"], rec.values["vergence_angle"], places=4)
        msg = p.parse_message(p.encode_get_item_response(rec, 36))  # eye location x, two values
        out = p.decode_get_item_response(msg)
        self.assertAlmostEqual(out["left_eye_location_x"], rec.values["left_eye_location_x"], places=3)
        self.assertAlmostEqual(out["right_eye_location_x"], rec.values["right_eye_location_x"], places=3)


if __name__ == "__main__":
    unittest.main()
