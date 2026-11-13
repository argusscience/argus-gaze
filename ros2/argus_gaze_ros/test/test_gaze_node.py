"""Integration tests: the node against the ETVision emulator (argus_gaze.replay).

Run with `colcon test --packages-select argus_gaze_ros`, or `python -m pytest` in this
directory after sourcing the workspace.
"""
import math
import os
import time

import pytest
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt16

from argus_gaze import protocol as p
from argus_gaze import synthetic
from argus_gaze.replay import Emulator, Source
from argus_gaze_msgs.msg import Gaze, ObjectArray
from argus_gaze_ros.gaze_node import ArgusGazeNode

NAMES = {1: "person", 2: "bicycle", 3: "cup"}  # a model numbers its types consecutively
LATCHED = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _records_with_objects():
    """Synthetic records with an AI LMOT object list that changes every 60 samples."""
    recs = synthetic.generate(duration_per_target_s=0.5)
    for i, r in enumerate(recs):
        r.check_state |= (1 << p.AI_OBJECTS_BIT) | (1 << p.BIT_BY_NAME["gaze_ai_obj_id"])
        step = (i // 60) % 3
        objs = [dict(obj_id=3, obj_horz_cntr=320.0 + step, obj_vert_cntr=240.0, obj_width=80.0,
                     obj_height=60.0, obj_gaze_horz=0.5, obj_gaze_vert=0.25)]
        if step:
            objs.append(dict(obj_id=1, obj_horz_cntr=100.0, obj_vert_cntr=200.0, obj_width=50.0,
                             obj_height=150.0, obj_gaze_horz=-1.0, obj_gaze_vert=-1.0))
        r.ai_objects = objs
        r.values["no_of_ai_objects"] = len(objs)
        r.values["gaze_ai_obj_id"] = 3
    return recs


@pytest.fixture
def emulator():
    em = Emulator(Source.from_records(_records_with_objects()), host="127.0.0.1", port=0,
                  speed=1.0, loop=True, object_names=NAMES)
    port = em.start()
    yield port
    em.stop()


def _run(port, transport, seconds=3.0):
    rclpy.init(args=["--ros-args", "-p", "host:=127.0.0.1", "-p", "port:=%d" % port,
                     "-p", "transport:=%s" % transport])
    node = ArgusGazeNode()
    probe = rclpy.create_node("argus_gaze_probe")
    got = {"gaze": [], "point": [], "objects": [], "event": []}
    probe.create_subscription(Gaze, "/argus/gaze", got["gaze"].append, 500)
    probe.create_subscription(PointStamped, "/argus/gaze/convergence_point", got["point"].append, 500)
    probe.create_subscription(ObjectArray, "/argus/objects", got["objects"].append, LATCHED)
    probe.create_subscription(UInt16, "/argus/event", lambda m: got["event"].append(m.data), LATCHED)
    set_event = probe.create_publisher(UInt16, "/argus/set_event", 10)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(probe)

    def spin(s):
        end = time.monotonic() + s
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=0.05)

    try:
        spin(seconds)
        set_event.publish(UInt16(data=9))
        spin(1.0)
        names = dict(node.names)
    finally:
        node.shutdown()
        executor.shutdown()
        node.destroy_node()
        probe.destroy_node()
        rclpy.shutdown()
    return got, names


@pytest.mark.parametrize("transport", ["tcp", "udp"])
def test_topics(emulator, transport):
    got, names = _run(emulator, transport)
    assert names == NAMES                                  # read from ETVision at start
    assert len(got["gaze"]) > 300                          # about 180 per second
    g = got["gaze"][-1]
    assert g.header.frame_id == "et3space_world"
    assert all(g.valid)
    for v in (g.direction, g.left_direction, g.right_direction):
        assert abs(math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2) - 1.0) < 1e-3
    assert got["point"], "convergence point published when both eyes are valid"
    assert g.gaze_object == 3

    objs = got["objects"]
    assert objs, "object list published"
    assert len(objs) < len(got["gaze"]) / 10               # only when the list changes
    o = objs[-1].objects[0]
    assert (o.type, o.name) == (3, "cup")
    assert objs[-1].header.frame_id == "argus_scene_camera"
    assert list(o.gaze_in_box) == [0.5, 0.25]
    two = [a for a in objs if len(a.objects) == 2]
    assert two and two[-1].objects[1].name == "person"
    assert list(two[-1].objects[1].gaze_in_box) == [-1.0, -1.0]

    assert got["event"][0] == got["gaze"][0].event         # current event code on connect
    assert 9 in got["event"]                               # set_event reaches ETVision and comes back
    assert any(m.event == 9 for m in got["gaze"])
