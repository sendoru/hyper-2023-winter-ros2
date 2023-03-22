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
import argparse
from sklearn.linear_model import LinearRegression

from rclpy.qos import QoSProfile

from cv_basics.constants import *
from cv_basics.utils import *

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

class PointCloudPublisher(Node):
    def __init__(self, video_l, video_r, use_file, show_video):
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
        self.regression_l = LinearRegression()
        self.regression_l.coef_ = np.zeros((3, 3))
        self.regression_l.intercept_ = np.zeros(3)
        self.regression_r = LinearRegression()
        self.regression_r.coef_ = np.zeros((3, 3))
        self.regression_r.intercept_ = np.zeros(3)
        self.alpha_sum = 0

        # We will publish a message every 0.1 seconds
        timer_period = 1./30 # seconds

        self.cap_l = cv2.VideoCapture(video_l)
        self.cap_r = cv2.VideoCapture(video_r)

        self.use_file = use_file
        self.show_video = show_video

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

    def timer_callback(self):
        """
        Callback function.
        This function gets called every 0.1 seconds.
        """
        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.
        time_elapsed_for_fps = time_elapsed = time.time()
        if self.use_file:
            time_elapsed = self.frame_no / FRAME_RATE
        ret_l, frame_l = self.cap_l.read()
        ret_r, frame_r = self.cap_r.read()
        
        if not ret_l or not ret_r:
            self.get_logger().info("Ignoring empty camera frame.")
            cv2.destroyAllWindows()
            return
        
        if frame_l.shape != frame_r.shape:
            cv2.resize(frame_l, frame_r.shape[1], frame_r.shape[0])
        cur_point = None

        result_l = preprocess(frame_l, L_INTRINSIC, L_DISTORTION, self.hands_l)
        draw_hand(result_l, frame_l)
        result_r = preprocess(frame_r, R_INTRINSIC, R_DISTORTION, self.hands_r)
        draw_hand(result_r, frame_r)

        if self.show_video:
            output_image = cv2.hconcat([frame_l, frame_r])
            cv2.imshow("Debug Image",output_image)
            cv2.waitKey(1)

        hand_3d = recon_3d_hand(frame_l.shape[1], frame_l.shape[0], result_l, result_r)
        if type(hand_3d) != type(None):
            # 8번점 = 검지끝
            cur_point = hand_3d[0].points[8]
            if len(self.recent_points) == 0:
                pass

            else:
                speed = np.linalg.norm(cur_point - self.recent_points[-1]) / (time_elapsed - self.recent_timings[-1])
                if np.linalg.norm(cur_point) <= DIST_THRESHOLD and speed <= SPEED_THRESHOLD and np.abs(cur_point[2]) >= DEPTH_MIN_THRESHOLD:

                    # linear regression
                    self.alpha_sum = ALPHA + (1 - ALPHA) * self.alpha_sum

                    linear_regression_update_with_moving_average(
                        hand_landmarks_to_array(result_l.multi_hand_landmarks[0]),
                        hand_3d[0].points,
                        self.regression_l)
                    linear_regression_update_with_moving_average(
                        hand_landmarks_to_array(result_r.multi_hand_landmarks[0]),
                        hand_3d[0].points,
                        self.regression_r)
                    

        # 제대로 인식된 손이 1개 이하일 경우 
        else:
            if result_l.multi_hand_landmarks:
                hand_landmars_array = hand_landmarks_to_array(result_l.multi_hand_landmarks[0])
                cur_point = self.regression_l.predict(hand_landmars_array[8].reshape((1, -1))) / self.alpha_sum
                cur_point = cur_point.reshape(3)
                pass
            elif result_r.multi_hand_landmarks:
                hand_landmars_array = hand_landmarks_to_array(result_r.multi_hand_landmarks[0])
                cur_point = self.regression_r.predict(hand_landmars_array[8].reshape((1, -1))) / self.alpha_sum
                cur_point = cur_point.reshape(3)

        if type(cur_point) != type(None):
            if len(self.recent_points) == 0:
                self.recent_points.append(cur_point)
                self.recent_timings.append(time_elapsed)
            else:
                speed = np.linalg.norm(cur_point - self.recent_points[-1]) / (time_elapsed - self.recent_timings[-1])
                if np.linalg.norm(cur_point) <= DIST_THRESHOLD and speed <= SPEED_THRESHOLD and np.abs(cur_point[2]) >= DEPTH_MIN_THRESHOLD:
                    self.recent_points.append(cur_point)
                    self.recent_timings.append(time_elapsed)

            while len(self.recent_points) > 5:
                self.recent_points.popleft()
                self.recent_timings.popleft()

            # spline interpolation을 위해서는 기본 4개의 점이 필요 + 1개는 안정성을 위해 추가
            if len(self.recent_points) == 5:
                t = np.array(self.recent_timings)
                dt = t[1:] - t[:-1]

                # 손가락이 화면에서 너무 오랬동안 없어져 있던 경우에는 보간 중지
                if dt[-1] <= .5:
                    xy = np.array(self.recent_points)
                    tt = np.linspace(self.recent_timings[-2], self.recent_timings[-1], 30)
                    bspl = sp.interpolate.make_interp_spline(t, xy)
                    points = bspl(tt)

                    ret = msg.PointCloud()
                    ret.points = [geometry_msgs.msg.Point32() for _ in range(len(points))]
                    # cv2에선 단위가 mm인데 ROS에선 m임
                    for i, [x, y, z] in enumerate(points):
                        ret.points[i].x = float(x / 1000)
                        ret.points[i].y = float(y / 1000)
                        ret.points[i].z = float(z / 1000)
                    ret.header.frame_id = 'pointcloud'
                    cur_time = time.time()
                    ret.header.stamp.sec = int(cur_time)
                    ret.header.stamp.nanosec = int((cur_time - int(cur_time)) * 1e9)
                    self.publisher_.publish(ret)
                
                    # Display the message on the console
                    self.get_logger().info(f'Publishing PointCloud with {len(ret.points)} point(s)')

        self.frame_no += 1

def main(args=None):
    # print(os.getcwd())
    # Initialize the rclpy library
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_l", help="file/device name of left video", default='0', required=False)
    parser.add_argument("--video_r", help="file/device name of right video", default='1', required=False)
    parser.add_argument("--debug", help="use debug mode: ", action="store_true")
    parser.add_argument("--use_file", help="use actual file instead of device", action="store_true")
    parser.add_argument("--show_video", help="use actual file instead of device", action="store_true")

    args = parser.parse_args()

    if args.debug:
        args.use_file = True
        args.show_video = True
        args.video_l = "/home/sendol/colcon_ws/src/hyper-2023-winter-ros2/output_L_near.mp4"
        args.video_r = "/home/sendol/colcon_ws/src/hyper-2023-winter-ros2/output_R_near.mp4"

    # print(args.video_l, args.video_r)

    # Create the node
    if not args.use_file:
        args.video_l = int(args.video_l)
        args.video_r = int(args.video_r)
    pointcolut_publisher = PointCloudPublisher(args.video_l, args.video_r, args.use_file, args.show_video)

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
            time_prev = time.time()
            rclpy.spin_once(pointcolut_publisher)
            time_curr = time.time()
            pointcolut_publisher.get_logger().info(f"FPS: {1./(time_curr - time_prev):.3f}")

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
    pointcolut_publisher.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()


if __name__ == '__main__':
    main()