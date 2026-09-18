#!/usr/bin/env python3
"""ROS 2 Visual Odometry Node for 6-DoF vehicle pose estimation and TF2 broadcasting.

Subscribes:
    /camera/image_raw (sensor_msgs/Image, rgb8 or bgr8)
    /camera/depth/image_raw (sensor_msgs/Image, 32FC1)

Publishes:
    /localization/odometry (nav_msgs/Odometry) — pose + twist in odom frame
    /localization/status (std_msgs/String) — tracking state as JSON
    /localization/confidence (std_msgs/Float32) — scalar VO confidence
    /localization/relocalization_state (std_msgs/String) — relocalization state

TF Broadcasts:
    odom -> base_link (dynamic, from VO)
    base_link -> camera_link (static, from params)
    camera_link -> camera_optical_frame (static, REP-103)

Does NOT claim physical UGV localization. All evaluation is on recorded sensor data.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Optional

import numpy as np

# Ensure project root is in sys.path for importing src modules
candidate_roots = [
    os.environ.get("SIH_ROOT", ""),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")),
]
for root in candidate_roots:
    if root and os.path.isdir(os.path.join(root, "src", "localization")):
        if root not in sys.path:
            sys.path.insert(0, root)
        break

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String, Float32
    from geometry_msgs.msg import (
        TransformStamped,
        Quaternion,
        Point,
        Pose,
        PoseWithCovariance,
        Twist,
        TwistWithCovariance,
        Vector3,
    )
    from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    # Mock base class for non-ROS environments
    class Node:
        def __init__(self, name: str) -> None:
            self.name = name

from src.interfaces.types import CameraIntrinsics, TrackingStatus
from src.localization.visual_odometry import VisualOdometry


def yaw_to_quaternion(yaw: float) -> tuple:
    """Convert yaw angle (radians) to quaternion (x, y, z, w)."""
    half_yaw = yaw * 0.5
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


class VisualOdometryNode(Node):
    """Tracks keypoints across camera frames and broadcasts odom -> base_link transforms."""

    def __init__(self):
        super().__init__('visual_odometry_node')

        # Declare parameters
        self.declare_parameter('max_features', 400)
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('camera_frame_id', 'camera_optical_frame')
        self.declare_parameter('fx', 385.0)
        self.declare_parameter('fy', 385.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('camera_height_m', 0.45)
        self.declare_parameter('camera_pitch_rad', 0.209)

        # Read parameters
        max_features = self.get_parameter('max_features').value
        self.odom_frame = self.get_parameter('odom_frame_id').value
        self.base_frame = self.get_parameter('base_frame_id').value
        self.camera_frame = self.get_parameter('camera_frame_id').value
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        cam_height = self.get_parameter('camera_height_m').value
        cam_pitch = self.get_parameter('camera_pitch_rad').value

        # Build intrinsics
        intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            camera_height_m=cam_height,
            camera_pitch_rad=cam_pitch,
        )

        # Initialize VO engine
        self.vo = VisualOdometry(intrinsics, max_features=max_features)

        # Sensor QoS profile
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, sensor_qos
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, sensor_qos
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/localization/odometry', 10)
        self.status_pub = self.create_publisher(String, '/localization/status', 10)
        self.confidence_pub = self.create_publisher(Float32, '/localization/confidence', 10)
        self.reloc_pub = self.create_publisher(String, '/localization/relocalization_state', 10)

        # TF broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Publish static transforms
        self._publish_static_transforms(cam_height, cam_pitch)

        # State
        self._latest_depth: Optional[np.ndarray] = None
        self._frame_count = 0
        self._total_latency_ms = 0.0

        self.get_logger().info(
            f'VisualOdometryNode initialized: max_features={max_features}, '
            f'fx={fx}, fy={fy}, camera_height={cam_height}m, pitch={cam_pitch:.3f}rad'
        )

    def _publish_static_transforms(self, cam_height: float, cam_pitch: float) -> None:
        """Publish static base_link -> camera_link -> camera_optical_frame transforms."""
        now = self.get_clock().now().to_msg()

        # base_link -> camera_link (height + pitch)
        t_cam = TransformStamped()
        t_cam.header.stamp = now
        t_cam.header.frame_id = self.base_frame
        t_cam.child_frame_id = 'camera_link'
        t_cam.transform.translation.x = 0.1  # 10cm forward of base
        t_cam.transform.translation.y = 0.0
        t_cam.transform.translation.z = cam_height
        # Pitch rotation around Y axis
        half_pitch = cam_pitch * 0.5
        t_cam.transform.rotation.x = 0.0
        t_cam.transform.rotation.y = math.sin(half_pitch)
        t_cam.transform.rotation.z = 0.0
        t_cam.transform.rotation.w = math.cos(half_pitch)

        # camera_link -> camera_optical_frame (REP-103: 90deg rotation)
        t_opt = TransformStamped()
        t_opt.header.stamp = now
        t_opt.header.frame_id = 'camera_link'
        t_opt.child_frame_id = self.camera_frame
        t_opt.transform.translation.x = 0.0
        t_opt.transform.translation.y = 0.0
        t_opt.transform.translation.z = 0.0
        # Rotation: camera_link (X-fwd, Y-left, Z-up) -> optical (X-right, Y-down, Z-fwd)
        # Quaternion for this convention rotation
        t_opt.transform.rotation.x = -0.5
        t_opt.transform.rotation.y = 0.5
        t_opt.transform.rotation.z = -0.5
        t_opt.transform.rotation.w = 0.5

        self.static_tf_broadcaster.sendTransform([t_cam, t_opt])
        self.get_logger().info('Published static transforms: base_link -> camera_link -> camera_optical_frame')

    def depth_callback(self, msg: Image) -> None:
        """Cache latest depth image for synchronization with RGB."""
        try:
            if msg.encoding == '32FC1':
                depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            elif msg.encoding == '16UC1':
                depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                depth = depth.astype(np.float32) / 1000.0  # mm to meters
            else:
                self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}', throttle_duration_sec=5.0)
                return
            self._latest_depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            self.get_logger().error(f'Depth decode error: {e}', throttle_duration_sec=5.0)

    def image_callback(self, msg: Image) -> None:
        """Process incoming RGB frame through visual odometry pipeline."""
        try:
            # Decode RGB image
            if msg.encoding in ('rgb8', 'RGB8'):
                rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            elif msg.encoding in ('bgr8', 'BGR8'):
                bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                rgb = bgr[:, :, ::-1].copy()
            else:
                self.get_logger().warn(f'Unsupported image encoding: {msg.encoding}', throttle_duration_sec=5.0)
                return

            # Use cached depth or zero depth
            depth = self._latest_depth
            if depth is None or depth.shape[:2] != (msg.height, msg.width):
                depth = np.zeros((msg.height, msg.width), dtype=np.float32)

            # Run visual odometry
            result = self.vo.process_frame(rgb, depth)
            self._frame_count += 1
            self._total_latency_ms += result.latency_ms

            # Build and publish odometry message
            stamp = msg.header.stamp
            self._publish_odometry(result, stamp)
            self._publish_tf(result, stamp)
            self._publish_status(result, stamp)
            self._publish_confidence(result)
            self._publish_relocalization_state(result)

        except Exception as e:
            self.get_logger().error(f'VO processing error: {e}', throttle_duration_sec=2.0)

    def _publish_odometry(self, result, stamp) -> None:
        """Publish nav_msgs/Odometry with pose and twist."""
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        # Pose
        qx, qy, qz, qw = yaw_to_quaternion(result.yaw)
        odom.pose.pose.position = Point(x=result.x, y=result.y, z=result.z)
        odom.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        # Twist (frame-to-frame velocities, approximate)
        odom.twist.twist.linear = Vector3(x=result.delta_x, y=result.delta_y, z=0.0)
        odom.twist.twist.angular = Vector3(x=0.0, y=0.0, z=result.delta_yaw)

        self.odom_pub.publish(odom)

    def _publish_tf(self, result, stamp) -> None:
        """Broadcast odom -> base_link dynamic transform."""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = result.x
        t.transform.translation.y = result.y
        t.transform.translation.z = result.z

        qx, qy, qz, qw = yaw_to_quaternion(result.yaw)
        t.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        self.tf_broadcaster.sendTransform(t)

    def _publish_status(self, result, stamp) -> None:
        """Publish JSON tracking status."""
        status_dict = {
            "tracking_status": result.tracking_status.value,
            "relocalization_state": result.relocalization_state,
            "confidence": round(result.confidence, 4),
            "inlier_count": result.inlier_count,
            "consecutive_lost": result.consecutive_lost_frames,
            "latency_ms": round(result.latency_ms, 2),
            "frame_count": self._frame_count,
        }
        msg = String()
        msg.data = json.dumps(status_dict)
        self.status_pub.publish(msg)

    def _publish_confidence(self, result) -> None:
        """Publish scalar confidence."""
        msg = Float32()
        msg.data = float(result.confidence)
        self.confidence_pub.publish(msg)

    def _publish_relocalization_state(self, result) -> None:
        """Publish relocalization state."""
        msg = String()
        msg.data = result.relocalization_state
        self.reloc_pub.publish(msg)

    def destroy_node(self) -> None:
        """Clean shutdown with summary statistics."""
        if self._frame_count > 0:
            avg_lat = self._total_latency_ms / self._frame_count
            self.get_logger().info(
                f'VO shutdown summary: {self._frame_count} frames processed, '
                f'avg latency {avg_lat:.2f}ms, '
                f'{self.vo.total_recoveries} recoveries'
            )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
