#!/usr/bin/env python3
"""ROS 2 Sensor Feed Publisher Node.

Publishes live or recorded sensor streams onto:
- /camera/image_raw (sensor_msgs/msg/Image: rgb8)
- /camera/camera_info (sensor_msgs/msg/CameraInfo)
- /camera/depth/image_raw (sensor_msgs/msg/Image: 32FC1)
"""

import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

# Add project root to sys.path to access src.sensors
candidate_roots = [
    os.environ.get("SIH_ROOT", ""),
    "/mnt/c/Users/harsh/SIH-26126-Vision-UGV",
    "C:/Users/harsh/SIH-26126-Vision-UGV",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")),
]
for root in candidate_roots:
    if root and os.path.isdir(os.path.join(root, "src", "sensors")):
        if root not in sys.path:
            sys.path.insert(0, root)
        break

from src.sensors.camera_pipeline import (
    BaseCameraSource,
    create_camera_source,
)


class SensorFeedPublisherNode(Node):
    """ROS 2 Node publishing synchronized camera frames and intrinsics."""

    def __init__(self):
        super().__init__('sensor_feed_publisher_node')

        # Declare parameters
        self.declare_parameter('source_uri', 'datasets/processed/scenario_1_open_path')
        self.declare_parameter('target_fps', 10.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('frame_id', 'camera_optical_frame')

        source_uri = self.get_parameter('source_uri').get_parameter_value().string_value
        target_fps = self.get_parameter('target_fps').get_parameter_value().double_value
        loop = self.get_parameter('loop').get_parameter_value().bool_value
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        # Initialize Camera Source
        self.source: BaseCameraSource = create_camera_source(
            uri=source_uri,
            target_fps=target_fps,
            loop=loop,
            frame_id=frame_id,
        )
        if not self.source.open():
            self.get_logger().error(f'Failed to open camera source: {source_uri}')
            raise RuntimeError(f'Cannot open camera source {source_uri}')

        # Setup Publishers
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)

        # Timer loop for non-blocking publishing
        timer_period = 1.0 / max(target_fps, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f'SensorFeedPublisherNode active. Ingesting from "{source_uri}" at {target_fps:.1f} FPS.'
        )

    def timer_callback(self):
        packet = self.source.read()
        if packet is None:
            return

        now = self.get_clock().now().to_msg()

        # 1. RGB Image
        header = Header()
        header.stamp = now
        header.frame_id = packet.frame_id

        rgb_msg = Image()
        rgb_msg.header = header
        rgb_msg.height = packet.height
        rgb_msg.width = packet.width
        rgb_msg.encoding = 'rgb8'
        rgb_msg.is_bigendian = 0
        rgb_msg.step = packet.width * 3
        rgb_msg.data = packet.rgb.tobytes()
        self.image_pub.publish(rgb_msg)

        # 2. CameraInfo
        info_msg = CameraInfo()
        info_msg.header = header
        info_msg.height = packet.height
        info_msg.width = packet.width
        info_msg.distortion_model = 'plumb_bob'
        info_msg.d = [
            packet.intrinsics.k1,
            packet.intrinsics.k2,
            packet.intrinsics.p1,
            packet.intrinsics.p2,
            0.0,
        ]
        info_msg.k = [
            packet.intrinsics.fx, 0.0, packet.intrinsics.cx,
            0.0, packet.intrinsics.fy, packet.intrinsics.cy,
            0.0, 0.0, 1.0,
        ]
        info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info_msg.p = [
            packet.intrinsics.fx, 0.0, packet.intrinsics.cx, 0.0,
            0.0, packet.intrinsics.fy, packet.intrinsics.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self.info_pub.publish(info_msg)

        # 3. Depth (if available)
        if packet.depth is not None:
            depth_msg = Image()
            depth_msg.header = header
            depth_msg.height = packet.depth.shape[0]
            depth_msg.width = packet.depth.shape[1]
            depth_msg.encoding = '32FC1'
            depth_msg.is_bigendian = 0
            depth_msg.step = packet.depth.shape[1] * 4
            depth_msg.data = packet.depth.astype(np.float32).tobytes()
            self.depth_pub.publish(depth_msg)

    def destroy_node(self):
        if hasattr(self, 'source') and self.source is not None:
            self.source.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SensorFeedPublisherNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
