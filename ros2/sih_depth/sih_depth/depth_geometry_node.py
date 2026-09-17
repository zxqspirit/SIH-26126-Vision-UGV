#!/usr/bin/env python3
"""ROS 2 Depth Geometry Node for SIH 26126 Autonomous UGV.

Subscribes to:
  /camera/depth/image_raw (sensor_msgs/msg/Image: 32FC1 meters or 16UC1 millimeters)

Publishes:
  /depth/geometric_cost (sensor_msgs/msg/Image: mono8 0..100)
  /depth/positive_obstacles (sensor_msgs/msg/Image: mono8 0 or 255)
  /depth/negative_obstacles (sensor_msgs/msg/Image: mono8 0 or 255)
  /depth/discontinuities (sensor_msgs/msg/Image: mono8 0 or 255)
  /depth/quality_metric (std_msgs/msg/Float32: 0.0..1.0)

Safety Guarantee (Rule 12):
  Invalid depth returns (NaN, Inf, 0.0, out-of-range) are strictly mapped to
  an uncertainty penalty cost (60/100) and NEVER to free space (0/100).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Any

import cv2
import numpy as np

# Ensure project root is in sys.path
candidate_roots = [
    os.environ.get("SIH_ROOT", ""),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")),
]
for root in candidate_roots:
    if root and os.path.isdir(os.path.join(root, "src", "depth_geometry")):
        if root not in sys.path:
            sys.path.insert(0, root)
        break

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32, Header
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    # Mock base class for non-ROS testing
    class Node:
        def __init__(self, name: str) -> None:
            self.name = name

from src.interfaces.types import CameraIntrinsics, DepthGeometryResult
from src.depth_geometry.depth_geometry_engine import DepthGeometryEngine


def numpy_to_image_msg(img_np: np.ndarray, encoding: str, header: Any) -> Any:
    """Converts a numpy array directly into a sensor_msgs/Image message without cv_bridge dependency."""
    msg = Image()
    msg.header = header
    msg.height = img_np.shape[0]
    msg.width = img_np.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = False

    if encoding == "mono8":
        msg.step = msg.width
        msg.data = img_np.astype(np.uint8).tobytes()
    elif encoding == "32FC1":
        msg.step = msg.width * 4
        msg.data = img_np.astype(np.float32).tobytes()
    elif encoding == "rgb8":
        msg.step = msg.width * 3
        msg.data = img_np.astype(np.uint8).tobytes()
    else:
        msg.step = msg.width
        msg.data = img_np.tobytes()

    return msg


def image_msg_to_numpy(msg: Any) -> np.ndarray:
    """Decodes a sensor_msgs/Image message into a float32 depth map in meters."""
    encoding = getattr(msg, "encoding", "32FC1")
    h = getattr(msg, "height", 480)
    w = getattr(msg, "width", 640)
    data = getattr(msg, "data", b"")

    if encoding == "32FC1":
        arr = np.frombuffer(data, dtype=np.float32).reshape((h, w))
        return arr
    elif encoding in ("16UC1", "mono16"):
        arr_mm = np.frombuffer(data, dtype=np.uint16).reshape((h, w))
        return arr_mm.astype(np.float32) / 1000.0
    else:
        # Fallback assume float32
        arr = np.frombuffer(data, dtype=np.float32).reshape((h, w))
        return arr


class DepthGeometryNode(Node):
    """ROS 2 Node for 3D depth geometry processing and hazard extraction."""

    def __init__(self) -> None:
        if HAS_RCLPY:
            super().__init__("depth_geometry_node")
            self._init_params_and_publishers()
        else:
            super().__init__("depth_geometry_node_mock")
            self._init_engine_only()

    def _init_params_and_publishers(self) -> None:
        # Declare ROS parameters
        self.declare_parameter("fx", 385.0)
        self.declare_parameter("fy", 385.0)
        self.declare_parameter("cx", 320.0)
        self.declare_parameter("cy", 240.0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("camera_height_m", 0.45)
        self.declare_parameter("camera_pitch_rad", 0.209)
        self.declare_parameter("step_threshold_m", 0.15)
        self.declare_parameter("lethal_height_m", 0.35)

        # Build CameraIntrinsics
        fx = float(self.get_parameter("fx").value)
        fy = float(self.get_parameter("fy").value)
        cx = float(self.get_parameter("cx").value)
        cy = float(self.get_parameter("cy").value)
        w = int(self.get_parameter("width").value)
        h = int(self.get_parameter("height").value)
        cam_h = float(self.get_parameter("camera_height_m").value)
        cam_p = float(self.get_parameter("camera_pitch_rad").value)

        intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=w, height=h,
            camera_height_m=cam_h,
            camera_pitch_rad=cam_p
        )
        self.engine = DepthGeometryEngine(intrinsics)

        # QoS Profiles
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        reliable_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        # Subscription
        self.sub_depth = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            sensor_qos,
        )

        # Publishers
        self.pub_cost = self.create_publisher(Image, "/depth/geometric_cost", reliable_qos)
        self.pub_pos_obs = self.create_publisher(Image, "/depth/positive_obstacles", reliable_qos)
        self.pub_neg_obs = self.create_publisher(Image, "/depth/negative_obstacles", reliable_qos)
        self.pub_disc = self.create_publisher(Image, "/depth/discontinuities", reliable_qos)
        self.pub_quality = self.create_publisher(Float32, "/depth/quality_metric", reliable_qos)

        # Telemetry counters
        self.frame_count = 0
        self.total_latency_ms = 0.0

        self.get_logger().info(
            f"DepthGeometryNode initialized. Intrinsics: {w}x{h}, fx={fx}, h={cam_h}m, pitch={cam_p:.3f}rad. "
            f"Enforcing Rule 12 invariant: invalid depth != free space."
        )

    def _init_engine_only(self) -> None:
        """Fallback for running without active ROS 2 environment."""
        intrinsics = CameraIntrinsics()
        self.engine = DepthGeometryEngine(intrinsics)
        self.frame_count = 0
        self.total_latency_ms = 0.0

    def process_frame(self, depth_m: np.ndarray) -> DepthGeometryResult:
        """Synchronously processes a metric depth frame."""
        return self.engine.process_depth(depth_m)

    def depth_callback(self, msg: Any) -> None:
        """Callback for incoming /camera/depth/image_raw."""
        if not HAS_RCLPY:
            return

        t_start = time.perf_counter()

        # 1. Decode depth image safely
        try:
            depth_m = image_msg_to_numpy(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to decode depth frame: {e}")
            return

        # 2. Process depth geometry
        result = self.engine.process_depth(depth_m)
        dt_ms = (time.perf_counter() - t_start) * 1000.0

        self.frame_count += 1
        self.total_latency_ms += dt_ms

        # 3. Publish geometric cost map (mono8: 0..100)
        cost_u8 = (np.clip(result.geometric_cost, 0.0, 1.0) * 100.0).astype(np.uint8)
        cost_msg = numpy_to_image_msg(cost_u8, "mono8", msg.header)
        self.pub_cost.publish(cost_msg)

        # 4. Publish positive obstacles (mono8: 0 or 255)
        pos_u8 = (result.positive_obstacle_mask.astype(np.uint8)) * 255
        pos_msg = numpy_to_image_msg(pos_u8, "mono8", msg.header)
        self.pub_pos_obs.publish(pos_msg)

        # 5. Publish negative obstacles (mono8: 0 or 255)
        neg_u8 = (result.negative_obstacle_mask.astype(np.uint8)) * 255
        neg_msg = numpy_to_image_msg(neg_u8, "mono8", msg.header)
        self.pub_neg_obs.publish(neg_msg)

        # 6. Publish discontinuities (mono8: 0 or 255)
        if result.discontinuity_mask is not None:
            disc_u8 = (result.discontinuity_mask.astype(np.uint8)) * 255
        else:
            disc_u8 = np.zeros(depth_m.shape, dtype=np.uint8)
        disc_msg = numpy_to_image_msg(disc_u8, "mono8", msg.header)
        self.pub_disc.publish(disc_msg)

        # 7. Publish quality metric
        quality_msg = Float32()
        quality_msg.data = float(result.confidence)
        self.pub_quality.publish(quality_msg)

        # 8. Periodic telemetry logging (every 30 frames)
        if self.frame_count % 30 == 0:
            avg_lat = self.total_latency_ms / self.frame_count
            n_pos = int(np.sum(result.positive_obstacle_mask))
            n_neg = int(np.sum(result.negative_obstacle_mask))
            self.get_logger().info(
                f"[DepthNode #{self.frame_count}] Latency: {dt_ms:.1f}ms (Avg: {avg_lat:.1f}ms, {1000.0/max(avg_lat, 0.001):.1f} FPS) | "
                f"Quality: {result.confidence:.2f} | PosObs: {n_pos}px | NegObs: {n_neg}px"
            )


def main(args: Optional[list] = None) -> None:
    if not HAS_RCLPY:
        print("rclpy not installed. Run in a ROS 2 environment.")
        return

    rclpy.init(args=args)
    node = DepthGeometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("DepthGeometryNode received KeyboardInterrupt, shutting down cleanly...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
