"""Start the argus_gaze node for a live ETVision.

    ros2 launch argus_gaze_ros argus_gaze.launch.py host:=192.168.1.20 port:=5000

Other parameters (frames, UDP, names file) are in config/argus_gaze.yaml; pass a copy
with params_file:=/path/to/argus_gaze.yaml.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory("argus_gaze_ros"), "config", "argus_gaze.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("host", description="ETVision PC address"),
        DeclareLaunchArgument("port", default_value="5000", description="ETVision port"),
        DeclareLaunchArgument("transport", default_value="tcp", description="tcp or udp"),
        DeclareLaunchArgument("position_units", default_value="cm", description="as set on ETVision: cm or in"),
        DeclareLaunchArgument("params_file", default_value=default_params),
        Node(
            package="argus_gaze_ros",
            executable="gaze_node",
            name="argus_gaze",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {
                    "host": LaunchConfiguration("host"),
                    "port": ParameterValue(LaunchConfiguration("port"), value_type=int),
                    "transport": LaunchConfiguration("transport"),
                    "position_units": LaunchConfiguration("position_units"),
                },
            ],
        ),
    ])
