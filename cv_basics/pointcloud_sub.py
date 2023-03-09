# Basic ROS 2 program to subscribe to real-time streaming
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com

# Import the necessary libraries
import rclpy  # Python library for ROS 2
from rclpy.node import Node  # Handles the creation of nodes
import geometry_msgs.msg as geo_msg
import sensor_msgs.msg as sensor_msg# Image is the message type
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library
import open3d as o3d
import scipy as sp
import numpy as np
import mediapipe as mp
import std_msgs.msg
from cv_basics.constants import *
from cv_basics.utils import *

class PointcloudSubscriber(Node):
    """
    Create an ImageSubscriber class, which is a subclass of the Node class.
    """

    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('ponitcloud_subscriber')

        # Create the subscriber. This subscriber will receive an Image
        # from the video_frames topic. The queue size is 10 messages.
        self.pointcloud_subscription = self.create_subscription(
            sensor_msg.PointCloud,
            'pointcloud',
            self.pointcloud_listener_callback,
            5)
        
        self.keyboard_subscription = self.create_subscription(
            std_msgs.msg.Strnig,
            'keyboard_op',
            self.keyboard_listener_callback,
            5
        )
        self.pointcloud_subscription  # prevent unused variable warning

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()
        self.interrupt = False
        self.pointcloud = o3d.geometry.PointCloud()
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=1600, height=900)
        self.record = True

    def pointcloud_listener_callback(self, data: sensor_msg.PointCloud):
        """
        Callback function.
        """
        # Display the message on the console
        self.get_logger().info('Receiving pointcloud')

        # Convert ROS Image message to OpenCV image
        # print(data.points[0])
        cur_points = []
        for point in data.points:
            cur_points.append([point.x, point.y, point.z])
        cur_points = o3d.utility.Vector3dVector(np.array(cur_points))
        cur_pointcloud = o3d.geometry.PointCloud(cur_points)
        if self.record:
            self.vis.add_geometry(cur_pointcloud)
            self.vis.poll_events()
            self.vis.update_renderer()

    def keyboard_listener_callback(self, data: std_msgs.msg.String):
        key = data.data
        if key == 't':
            # toggle
            self.record  = not self.record
        elif key == 'e':
            # erase
            self.vis.clear_geometries()
        elif key == 'q':
            # quit
            self.destroy_node()
        self.get_logger().info("")

def main(args=None):

    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    pointcloud_subscriber = PointcloudSubscriber()

    # Spin the node so the callback function is called.
    rclpy.spin(pointcloud_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pointcloud_subscriber.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
