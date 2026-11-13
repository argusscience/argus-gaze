# The objects AI LMOT recognizes in the scene camera image (Argus gaze field
# specification v0.2). Published on /argus/objects when the list changes.

std_msgs/Header header                # stamp: host time; frame_id: the scene camera
uint32 gaze_object                    # object type at the point of gaze; 0 when none
Object[] objects
