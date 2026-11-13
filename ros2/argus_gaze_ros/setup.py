"""ament_python package for the Argus gaze ROS 2 node.

The node uses the argus_gaze library at the root of this repository; it is installed
with this package, so cloning the repository into a colcon workspace is enough.
"""
import os
from glob import glob

from setuptools import setup

PACKAGE = "argus_gaze_ros"
HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY = os.path.relpath(os.path.join(HERE, "..", "..", "argus_gaze"), HERE)
DEMO_CSV = os.path.relpath(os.path.join(HERE, "..", "..", "examples", "data",
                                        "workcell_demo_2026-11-12.csv"), HERE)

setup(
    name=PACKAGE,
    version="0.3.1",
    packages=[PACKAGE, "argus_gaze"],
    package_dir={"argus_gaze": LIBRARY},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE]),
        ("share/" + PACKAGE, ["package.xml"]),
        ("share/" + PACKAGE + "/launch", glob("launch/*.launch.py")),
        ("share/" + PACKAGE + "/config", glob("config/*.yaml")),
        ("share/" + PACKAGE + "/data", [DEMO_CSV]),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Argus Science",
    maintainer_email="info@argusscience.com",
    description="ROS 2 node that publishes live gaze from an ETVision eye tracker with ET3Space.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "gaze_node = argus_gaze_ros.gaze_node:main",
            "replay = argus_gaze.replay:main",
        ],
    },
)
