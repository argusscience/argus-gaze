"""argus_gaze: read, record and replay real-time data from Argus Science ETVision / ET3Space.

Everything here is written against the public "ETVision Network Communication Manual"
(version 3.0, October 2025) and the ETVision / ET3Space manuals published on
argusscience.com. It talks to ETVision only through that documented protocol.

Modules
-------
protocol   message layout, command builders, data-record encoder/decoder
client     live client: command socket plus UDP or TCP data channel
capture    bit-exact recording of the live stream to a file, and reading it back
csvfile    reader for ETVision text (.csv) data files
replay     an ETVision emulator that streams recorded data over the same protocol
geometry   unit conversion, head-pose rotation, convergence point of the two lines of gaze
gaze       GazeSample: a tidy, SI-unit view of one record for downstream software
checks     consistency checks and a summary report for a recorded file
align      match gaze to objects a robot system already tracks (geometry or AI LMOT name)
lerobot_export  convert a recording and its videos to a LeRobot dataset (needs numpy, lerobot)
"""

__version__ = "0.2.3"

from .protocol import DataRecord  # noqa: F401
from .gaze import GazeSample, to_gaze_sample  # noqa: F401
