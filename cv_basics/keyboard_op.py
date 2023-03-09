import rclpy  # Python Client Library for ROS 2
from rclpy.node import Node  # Handles the creation of nodes
import os
import select
import sys
import rclpy
from rclpy.qos import QoSProfile
from collections import deque

import std_msgs.msg

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty
    

def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 1./30)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

log_msg = \
'''
T: toggle trajectory recording
E: erase all trajectory
Q: kill subscriber node
'''

def main():
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    rclpy.init()
    qos = QoSProfile(depth=10)
    node = rclpy.create_node('keyboard_op')
    try:
        node.get_logger().log(log_msg)
        pub = node.create_publisher(std_msgs.msg.String, 'key_pressed', qos)
        node.prev_keys = deque(maxlen=3)

        while True:
            key = get_key(settings)
            if (not key in node.prev_keys) and (key in ('t', 'e', 'q')):
                msg = std_msgs.msg.String()
                msg.data = key
                pub.publish(msg)
            node.prev_keys.append(key)
            
    finally:
        return




