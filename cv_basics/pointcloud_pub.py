# Basic ROS 2 program to publish real-time streaming
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com

import rclpy  # Python Client Library for ROS 2
from rclpy.node import Node  # Handles the creation of nodes
import geometry_msgs.msg
import sensor_msgs.msg as msg# Image is the message type
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library

import os
import select
import sys
import rclpy
from collections import deque
import scipy as sp
import numpy as np
import mediapipe as mp
import time
import open3d as o3d

from rclpy.qos import QoSProfile

from cv_basics.constants import *
from cv_basics.recon_3d_hand import recon_3d_hand

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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

def preprocess(image: np.ndarray, intrinsic_mat: np.ndarray, distortion_coeff: np.ndarray, hands):
    image = cv2.undistort(image, intrinsic_mat, distortion_coeff)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results

def draw_hand(results, image):
    for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
        print(f'HAND NUMBER: {hand_no+1}')
        
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

class PointCloudPublisher(Node):
    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('pointcloud_publisher')

        # Create the publisher. This publisher will publish an Image
        # to the video_frames topic. The queue size is 10 messages.
        self.publisher_ = self.create_publisher(msg.PointCloud, 'pointcloud', 5)

        self.recent_points = deque()
        self.recent_timings = deque()
        self.frame_no = 0

        # We will publish a message every 0.1 seconds
        timer_period = 1./30  # seconds

        self.cap_l = cv2.VideoCapture("./src/hyper-2023-winter-ros2/output_L_near.mp4")
        self.cap_r = cv2.VideoCapture("./src/hyper-2023-winter-ros2/output_R_near.mp4")

        self.hands_l = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.hands_r = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Create the timer
        self.timer = self.create_timer(1./30, self.timer_callback)

        # TODO implement PointCloud2 Bridge
        self.br = CvBridge()

    def timer_callback(self):
        """
        Callback function.
        This function gets called every 0.1 seconds.
        """
        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.
        ret_l, frame_l = self.cap_l.read()
        ret_r, frame_r = self.cap_r.read()
        
        if not ret_l or not ret_r:
            print("Ignoring empty camera frame.")
            return
        
        if frame_l.shape != frame_r.shape:
            cv2.resize(frame_l, frame_r.shape[1], frame_r.shape[0])

        result_l =preprocess(frame_l, L_INTRINSIC, L_DISTORTION, self.hands_l)
        result_r = preprocess(frame_r, R_INTRINSIC, R_DISTORTION, self.hands_r)
        hand_3d = recon_3d_hand(frame_l.shape[0], frame_r.shape[0], result_l, result_r)
        if type(hand_3d) != type(None):
            # 8번점 = 검지끝
            if len(self.recent_points) == 0:
                self.recent_points.append(hand_3d[0].points[8])
                self.recent_timings.append(self.frame_no / FRAME_RATE)

            else:
                speed = np.linalg.norm(hand_3d[0].points[8] - self.recent_points[-1]) / ((self.frame_no / FRAME_RATE) - self.recent_timings[-1])
                if np.linalg.norm(hand_3d[0].points[8]) <= DIST_THRESHOLD and speed <= SPEED_THRESHOLD and np.abs(hand_3d[0].points[8][2]) >= DEPTH_MIN_THRESHOLD:
                    cur_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([hand_3d[0].points[8], self.recent_points[-1]])))
                    # vis.add_geometry(cur_points)
                    line = o3d.geometry.LineSet()
                    line.points = cur_points.points
                    line.lines = o3d.utility.Vector2iVector([[0, 1]])
                    # vis.add_geometry(line)
                    self.recent_points.append(hand_3d[0].points[8])
                    self.recent_timings.append(self.frame_no / FRAME_RATE)

            while len(self.recent_points) > 5:
                self.recent_points.popleft()
                self.recent_timings.popleft()

            # spline interpolation을 위해서는 기본 4개의 점이 필요 + 1개는 안정성을 위해 추가
            if len(self.recent_points) == 5:
                t = np.array(self.recent_timings)
                dt = t[1:] - t[:-1]

                # 손가락이 화면에서 너무 오랬동안 없어져 있던 경우에는 보간 중지
                if max(dt) <= .5:
                    xy = np.array(self.recent_points)
                    tt = np.linspace(self.recent_timings[-2], self.recent_timings[-1], 30)
                    bspl = sp.interpolate.make_interp_spline(t, xy)
                    points = bspl(tt)

                    # TODO publish poinets here!!!!
                    ret = msg.PointCloud()
                    # TODO 이거 형변환 해주고 들어가야되나보네
                    ret.points = [geometry_msgs.msg.Point32()] * len(points)
                    for i, [x, y, z] in enumerate(points):
                        ret.points[i].x = x / 1000
                        ret.points[i].y = y / 1000
                        ret.points[i].z = z / 1000
                    ret.header.frame_id = 'pointcloud'
                    cur_time = time.time()
                    ret.header.stamp.sec = int(cur_time)
                    ret.header.stamp.nanosec = int((cur_time - int(cur_time)) * 1e9)
                    self.publisher_.publish(ret)
                
                    # Display the message on the console
                    self.get_logger().info('Publishing PointCloud')

        self.frame_no += 1




def main(args=None):
    print(os.getcwd())
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    pointcolut_publisher = PointCloudPublisher()

    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    spinning = True
    while True:
        
        # toggle interrupt mode when key is pressed
        key = get_key(settings)
        if key != '':
            if ord(key) == 3 or ord(key) == 27:
                break
            else:
                spinning = True

        # Spin the node so the callback function is called.
        if spinning:
            rclpy.spin_once(pointcolut_publisher)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
    pointcolut_publisher.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()


if __name__ == '__main__':
    main()
