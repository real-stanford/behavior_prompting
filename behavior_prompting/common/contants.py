IPHUMI_TCP_OFFSET_FROM_MAIN_CAMERA = [0, -0.065, -0.212] # in ARKit main camera coordinate system (optical center of main camera); TCP is centered at the tip of the fingers in the middle vertically
IPHUMI_ULTRAWIDE_OFFSET_FROM_MAIN_CAMERA_X = -0.0192 # in ARKit main camera coordinate system
IPHUMI_ULTRAWIDE_OFFSET_FROM_CENTER_X = IPHUMI_ULTRAWIDE_OFFSET_FROM_MAIN_CAMERA_X - IPHUMI_TCP_OFFSET_FROM_MAIN_CAMERA[0] # in ARKit main camera coordinate system
IPHUMI_ULTRAWIDE_OFFSET_FROM_FINGER_AR_TAG_Z = 0.098079 # distance along z axis from ultrawide camera center to the AR tags on the finger (unlike ARkit, here we assume Z axis is pointing outwards from the camera (away from you))
