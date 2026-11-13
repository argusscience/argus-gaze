"""Try the node without an eye tracker: replay the recorded work-cell session
(examples/data/workcell_demo_2026-11-12.csv) as a live ETVision on this computer and
connect the node to it.

    ros2 launch argus_gaze_ros demo.launch.py
"""
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("argus_gaze_ros")
    csv = os.path.join(share, "data", "workcell_demo_2026-11-12.csv")
    replay = os.path.join(get_package_prefix("argus_gaze_ros"), "lib", "argus_gaze_ros", "replay")
    port = LaunchConfiguration("port")
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="5000", description="local port for the replayed ETVision"),
        DeclareLaunchArgument("csv", default_value=csv, description="ETVision .csv recording to replay"),
        ExecuteProcess(
            cmd=[replay, "--csv", LaunchConfiguration("csv"), "--host", "127.0.0.1", "--port", port],
            name="etvision_replay",
            output="screen",
        ),
        TimerAction(period=2.0, actions=[
            Node(
                package="argus_gaze_ros",
                executable="gaze_node",
                name="argus_gaze",
                output="screen",
                parameters=[{
                    "host": "127.0.0.1",
                    "port": ParameterValue(port, value_type=int),
                    "transport": "tcp",
                    "position_units": "cm",
                }],
            ),
        ]),
    ])
