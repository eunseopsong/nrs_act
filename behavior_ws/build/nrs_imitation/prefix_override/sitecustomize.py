import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/eunseop/nrs_act/behavior_ws/install/nrs_imitation'
