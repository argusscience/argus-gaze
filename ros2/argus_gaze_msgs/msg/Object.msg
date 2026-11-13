# One object recognized by AI LMOT in the head-mounted scene camera image.

uint32 type                           # object type: index into the names of the AI model
string name                           # name of the type, read from ETVision; empty when unknown
float32[2] center                     # px, box center in the scene camera image: horizontal, vertical
float32[2] size                       # px, box width and height
float32[2] gaze_in_box                # position of gaze within the box, 0-1; -1 when gaze is outside
