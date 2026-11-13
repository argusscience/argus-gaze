"""ROS 2 node: live gaze from ETVision with ET3Space (Argus gaze field specification v0.2).

Published (names relative to the node's namespace; /argus/... with none):
  argus/gaze                    argus_gaze_msgs/Gaze         every sample (180 Hz), in world_frame
  argus/gaze/convergence_point  geometry_msgs/PointStamped   every sample in which both eyes are valid
  argus/objects                 argus_gaze_msgs/ObjectArray  when the AI LMOT object list changes
  argus/event                   std_msgs/UInt16              when the event code (XDAT) changes
  TF                            world_frame -> head_frame    head-tracker sensor pose
Subscribed:
  argus/set_event               std_msgs/UInt16              sends an event code to ETVision (XDAT),
                                                             so robot actions are marked in the gaze record

Run (ETVision's Network Configuration dialog shows host and port; click "Listen" first):
  ros2 run argus_gaze_ros gaze_node --ros-args -p host:=192.168.1.20 -p port:=5000
"""
from __future__ import annotations

import math
import threading
from typing import Dict, Optional, Tuple

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, PointStamped, Pose, Quaternion, TransformStamped, Vector3
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt16
from tf2_ros import TransformBroadcaster

from argus_gaze import protocol as p
from argus_gaze.align import load_object_names
from argus_gaze.client import ETVisionClient
from argus_gaze.gaze import spec_values, to_gaze_sample
from argus_gaze_msgs.msg import Gaze, Object, ObjectArray


def _stamp(t: float) -> Time:
    sec = int(math.floor(t))
    return Time(sec=sec, nanosec=min(int(round((t - sec) * 1e9)), 999_999_999))


def _point(v) -> Point:
    return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def _vector(v) -> Vector3:
    return Vector3(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def _finite(v) -> bool:
    return all(math.isfinite(float(x)) for x in v)


class ArgusGazeNode(Node):
    def __init__(self) -> None:
        super().__init__("argus_gaze")
        self.host = self.declare_parameter("host", "").value
        self.port = int(self.declare_parameter("port", 5000).value)
        self.transport = str(self.declare_parameter("transport", "tcp").value).lower()
        self.udp_port = int(self.declare_parameter("udp_port", 0).value)
        self.units = str(self.declare_parameter("position_units", "cm").value)
        self.et3space = bool(self.declare_parameter("et3space", True).value)
        self.world_frame = self.declare_parameter("world_frame", "et3space_world").value
        self.head_frame = self.declare_parameter("head_frame", "argus_head_tracker").value
        self.scene_frame = self.declare_parameter("scene_camera_frame", "argus_scene_camera").value
        self.publish_tf = bool(self.declare_parameter("publish_tf", True).value)
        self.names_file = self.declare_parameter("object_names_file", "").value
        self.reconnect_period = float(self.declare_parameter("reconnect_period", 2.0).value)
        if not self.host:
            raise ValueError("parameter 'host' (the ETVision PC address) is required")
        if self.transport not in ("tcp", "udp"):
            raise ValueError("parameter 'transport' must be 'tcp' or 'udp'")

        self.pub_gaze = self.create_publisher(Gaze, "argus/gaze", 200)
        self.pub_point = self.create_publisher(PointStamped, "argus/gaze/convergence_point", 200)
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_objects = self.create_publisher(ObjectArray, "argus/objects", latched)
        self.pub_event = self.create_publisher(UInt16, "argus/event", latched)
        self.tf = TransformBroadcaster(self) if self.publish_tf else None
        self.create_subscription(UInt16, "argus/set_event", self._on_set_event, 10)

        self.names: Dict[int, str] = {}
        self._last_event: Optional[int] = None
        self._last_objects: Optional[Tuple] = None
        self._client: Optional[ETVisionClient] = None
        self._client_lock = threading.Lock()
        self._stop = threading.Event()
        self.samples = 0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # -- connection ---------------------------------------------------------------------
    def _load_names(self, client: ETVisionClient) -> Dict[int, str]:
        try:
            names = client.get_ai_object_names(timeout=1.0)
            if names:
                self.get_logger().info("read %d AI object names from ETVision" % len(names))
                return names
        except (p.ProtocolError, TimeoutError, OSError) as exc:
            self.get_logger().info("ETVision did not report AI object names (%s)" % exc)
        if self.names_file:
            names = load_object_names(self.names_file)
            self.get_logger().info("read %d AI object names from %s" % (len(names), self.names_file))
            return names
        return {}

    def _run(self) -> None:
        while not self._stop.is_set() and rclpy.ok():
            client = ETVisionClient(self.host, self.port)
            try:
                client.connect()
                self.names = self._load_names(client)
                if self.transport == "tcp":
                    client.start_tcp()
                    how = "TCP"
                else:
                    how = "UDP port %d" % client.start_udp(self.udp_port)
                with self._client_lock:
                    self._client = client
                self.get_logger().info("receiving gaze from %s:%d over %s" % (self.host, self.port, how))
                for rec in client.records(idle_timeout=5.0):
                    if self._stop.is_set():
                        break
                    self.publish(rec)
                if not self._stop.is_set():
                    self.get_logger().warning("no data from ETVision for 5 s; reconnecting")
            except (OSError, TimeoutError, ConnectionError, p.ProtocolError) as exc:
                if not self._stop.is_set():
                    self.get_logger().warning("ETVision at %s:%d: %s; retrying in %.0f s"
                                              % (self.host, self.port, exc, self.reconnect_period))
            finally:
                with self._client_lock:
                    self._client = None
                client.close()
            self._stop.wait(self.reconnect_period)

    def _on_set_event(self, msg: UInt16) -> None:
        with self._client_lock:
            client = self._client
            if client is None:
                self.get_logger().warning("not connected to ETVision; event %d not sent" % msg.data)
                return
            try:
                client.set_xdat(int(msg.data))
            except OSError as exc:
                self.get_logger().warning("could not send event %d: %s" % (msg.data, exc))

    # -- publishing -----------------------------------------------------------------------
    def publish(self, rec: p.DataRecord) -> None:
        s = to_gaze_sample(rec, self.units, et3space=self.et3space)
        v = spec_values(s)
        stamp = _stamp(rec.receive_time) if rec.receive_time else self.get_clock().now().to_msg()

        m = Gaze()
        m.header.stamp = stamp
        m.header.frame_id = self.world_frame
        m.device_time = float(s.device_time_s)
        m.frame_number = int(rec.frame_no) & 0xFFFFFFFF
        m.left_eye_position = _point(v["left_eye_position"])
        m.right_eye_position = _point(v["right_eye_position"])
        m.left_direction = _vector(v["left_direction"])
        m.right_direction = _vector(v["right_direction"])
        m.direction = _vector(v["direction"])
        m.convergence_point = _point(v["convergence_point"])
        m.convergence_miss = float(v["convergence_miss"][0])
        m.vergence_angle = float(v["vergence_angle"][0])
        m.valid = [bool(v["valid"][0]), bool(v["valid"][1])]
        m.surface_id = int(v["surface_id"][0])
        m.surface_point = [float(x) for x in v["surface_point"]]
        m.gaze_length = float(v["gaze_length"][0])
        hp = v["head_pose"]
        m.head_pose = Pose(position=_point(hp[:3]),
                           orientation=Quaternion(x=float(hp[3]), y=float(hp[4]), z=float(hp[5]), w=float(hp[6])))
        m.pupil_diameter = [float(x) for x in v["pupil_diameter"]]
        m.fixation_duration = float(v["fixation_duration"][0])
        event = int(v["event"][0]) & 0xFFFF
        m.event = event
        m.gaze_object = max(0, int(v["gaze_object"][0]))
        m.workcell_camera_point = [float(x) for x in v["workcell_camera_point"]]
        self.pub_gaze.publish(m)
        self.samples += 1

        cp = v["convergence_point"]
        if _finite(cp):
            self.pub_point.publish(PointStamped(header=m.header, point=_point(cp)))

        if self.tf is not None and _finite(hp):
            t = TransformStamped()
            t.header = m.header
            t.child_frame_id = self.head_frame
            t.transform.translation = _vector(hp[:3])
            t.transform.rotation = m.head_pose.orientation
            self.tf.sendTransform(t)

        if event != self._last_event:
            self._last_event = event
            self.pub_event.publish(UInt16(data=event))

        if s.objects is not None:
            key = (m.gaze_object, tuple((o["type"], o["centre_px"], o["size_px"], o["gaze_in_box"])
                                        for o in s.objects))
            if key != self._last_objects:
                self._last_objects = key
                arr = ObjectArray()
                arr.header.stamp = stamp
                arr.header.frame_id = self.scene_frame
                arr.gaze_object = m.gaze_object
                for o in s.objects:
                    ob = Object()
                    ob.type = max(0, int(o["type"]))
                    ob.name = self.names.get(ob.type, "")
                    ob.center = [float(x) for x in o["centre_px"]]
                    ob.size = [float(x) for x in o["size_px"]]
                    ob.gaze_in_box = [float(x) for x in (o["gaze_in_box"] or (-1.0, -1.0))]
                    arr.objects.append(ob)
                self.pub_objects.publish(arr)

    def shutdown(self) -> None:
        self._stop.set()
        with self._client_lock:
            client = self._client
        if client is not None:
            client.close()
        self._worker.join(timeout=3.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ArgusGazeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ValueError as exc:
        print("argus_gaze: %s" % exc)
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
