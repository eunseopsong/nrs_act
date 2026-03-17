import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/eunseop/nrs_act/behavior_ws/install/vive_tracker_ros2'
