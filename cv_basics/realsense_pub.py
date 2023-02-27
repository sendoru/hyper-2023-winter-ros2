# Basic ROS 2 program to publish real-time streaming
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com

# reference from https://github.com/IntelRealSense/librealsense
# librealsense/wrappers/python/examples/opencv_viewer_example.py

import pyrealsense2 as rs
import rclpy  # Python Client Library for ROS 2
from rclpy.node import Node  # Handles the creation of nodes
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library
import numpy as np
import rclpy  # Python Client Library for ROS 2
from rclpy.node import Node  # Handles the creation of nodes
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library

import os
import select
import sys
import rclpy

from rclpy.qos import QoSProfile
if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class ImagePublisher(Node):
    """
    Create an ImagePublisher class, which is a subclass of the Node class.
    """

    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('image_publisher')

        # Create the publisher. This publisher will publish an Image
        # to the video_frames topic. The queue size is 10 messages.
        self.publisher_ = self.create_publisher(Image, 'video_frames', 3)

        # We will publish a message every 0.1 seconds
        timer_period = 1./30  # seconds

        # Create the timer
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Create a VideoCapture object
        # The argument '0' gets the default webcam.
        self.cap = cv2.VideoCapture(0)

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(
            self.device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(
                rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(
                rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

    def timer_callback(self):
        """
        Callback function.
        This function gets called every 0.1 seconds.
        """

        ret = True
        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            ret = False

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # determines which image would be published
        # change below line to "frame = color_image" to publish color image instead
        frame = depth_colormap

        if ret == True:
            # Publish the image.
            # The 'cv2_to_imgmsg' method converts an OpenCV
            # image to a ROS 2 image message
            self.publisher_.publish(self.br.cv2_to_imgmsg(frame))

        # Display the message on the console
        self.get_logger().info('Publishing video frame')


def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    image_publisher = ImagePublisher()

    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    spinning = True
    while True:
        # toggle interrupt mode when key is pressed
        key = get_key(settings)
        if key != '':
            spinning = not spinning

        # Spin the node so the callback function is called.
        if spinning:
            rclpy.spin_once(image_publisher)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
    image_publisher.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
