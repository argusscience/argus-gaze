# One ETVision / ET3Space gaze sample (Argus gaze field specification v0.2).
# Published on /argus/gaze at the ETVision rate (180 Hz).
#
# Units: metres, radians, seconds. Positions and directions are in header.frame_id,
# the ET3Space world frame (et3space_world, the motion-capture frame). A value that is
# not available is NaN: values from an invalid eye are NaN, and the convergence point
# is NaN unless both eyes are valid. Field names match the LeRobot dataset fields
# observation.gaze.<name>.

std_msgs/Header header                # stamp: host time when the sample arrived
float64 device_time                   # ETVision time stamp, s
uint32 frame_number                   # ETVision frame number

geometry_msgs/Point left_eye_position
geometry_msgs/Point right_eye_position
geometry_msgs/Vector3 left_direction  # unit vector along the left line of gaze
geometry_msgs/Vector3 right_direction # unit vector along the right line of gaze
geometry_msgs/Vector3 direction       # unit vector along the cyclopean line of gaze
geometry_msgs/Point convergence_point # where the two lines of gaze meet (closest approach)
float32 convergence_miss              # m, gap between the two lines at closest approach
float32 vergence_angle                # rad
bool[2] valid                         # left, right: pupil and corneal reflection found

int32 surface_id                      # ET3Space surface looked at; -1 when on no defined surface
float32[2] surface_point              # m, point of gaze on that surface: horizontal, vertical
float32 gaze_length                   # m, eye to the point of gaze on the surface

geometry_msgs/Pose head_pose          # head-tracker sensor pose; also published as TF
float32[2] pupil_diameter             # left, right; unit as set on ETVision
float32 fixation_duration             # s; 0 when no fixation is in progress
uint16 event                          # event code (XDAT) sent by a robot controller or operator
uint32 gaze_object                    # AI LMOT object type at the point of gaze; 0 when none
float32[2] workcell_camera_point      # px, point of gaze on the stationary scene camera image
