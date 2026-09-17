#!/usr/bin/env python3
"""ROS 2 Traversability Classifier Node for SIH 26126 Autonomous UGV.

Subscribes to:
  /camera/image_raw (sensor_msgs/msg/Image)

Publishes:
  /perception/segmentation_mask (sensor_msgs/msg/Image, encoding: mono8, 0..8)
  /perception/terrain_cost (sensor_msgs/msg/Image, encoding: mono8, 0..100)
  /perception/confidence (std_msgs/msg/Float32, [0.0, 1.0])
  /perception/colored_mask (sensor_msgs/msg/Image, encoding: rgb8)

Enforces strict timestamp preservation, runtime timing telemetry, and invalid-frame safeguards.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
candidate_roots = [
    os.environ.get("SIH_ROOT", ""),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")),
]
for root in candidate_roots:
    if root and os.path.isdir(os.path.join(root, "src", "perception")):
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

from src.perception.ros_perception_node import PerceptionInferenceEngine, PerceptionOutput


def numpy_to_image_msg(img_np: np.ndarray, encoding: str, header: Any) -> Any:
    """Converts a numpy array directly into a sensor_msgs/Image message."""
    msg = Image()
    msg.header = header
    msg.height = img_np.shape[0]
    msg.width = img_np.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = False
    bytes_per_pixel = img_np.shape[2] if img_np.ndim == 3 else 1
    msg.step = img_np.shape[1] * bytes_per_pixel
    msg.data = img_np.tobytes()
    return msg


def image_msg_to_numpy(msg: Any) -> Optional[np.ndarray]:
    """Converts a sensor_msgs/Image message into an RGB numpy array."""
    if not msg.data or msg.height == 0 or msg.width == 0:
        return None

    dtype = np.uint8
    channels = 3
    if msg.encoding in ("mono8", "8UC1"):
        channels = 1
    elif msg.encoding in ("rgb8", "bgr8"):
        channels = 3
    elif msg.encoding in ("rgba8", "bgra8"):
        channels = 4
    else:
        # Fallback raw byte interpretation
        channels = 3

    try:
        raw_arr = np.frombuffer(msg.data, dtype=dtype)
        if channels == 1:
            frame = raw_arr.reshape((msg.height, msg.width))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif channels == 4:
            frame = raw_arr.reshape((msg.height, msg.width, 4))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB if "bgr" in msg.encoding else cv2.COLOR_RGBA2RGB)
        else:
            frame = raw_arr.reshape((msg.height, msg.width, 3))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if "bgr" in msg.encoding else frame
        return frame_rgb
    except Exception:
        return None


class TraversabilityClassifierNode(Node):
    """ROS 2 Node executing real-time terrain perception and cost generation."""

    def __init__(self) -> None:
        if not HAS_RCLPY:
            raise RuntimeError("rclpy is not installed in the active environment.")

        super().__init__("traversability_classifier_node")

        # 1. Declare Configurable Parameters
        self.declare_parameter("model_path", "")
        self.declare_parameter("model_backend", "pytorch")
        self.declare_parameter("input_width", 512)
        self.declare_parameter("input_height", 512)
        self.declare_parameter("device", "auto")
        self.declare_parameter("confidence_threshold", 0.60)
        self.declare_parameter("publish_colored_mask", True)

        # 2. Retrieve Parameters
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        model_backend = self.get_parameter("model_backend").get_parameter_value().string_value
        input_w = self.get_parameter("input_width").get_parameter_value().integer_value
        input_h = self.get_parameter("input_height").get_parameter_value().integer_value
        device = self.get_parameter("device").get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.publish_colored = self.get_parameter("publish_colored_mask").get_parameter_value().bool_value

        # 3. Initialize Core Inference Engine
        self.engine = PerceptionInferenceEngine(
            model_path=model_path if model_path else None,
            model_backend=model_backend,
            input_size=(input_w, input_h),
            device=device,
            confidence_threshold=self.confidence_threshold,
        )

        # 4. QoS Profile (Best Effort, queue=5 for real-time sensor streams)
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # 5. Subscribers & Publishers
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            sensor_qos,
        )

        self.mask_pub = self.create_publisher(Image, "/perception/segmentation_mask", 10)
        self.cost_pub = self.create_publisher(Image, "/perception/terrain_cost", 10)
        self.conf_pub = self.create_publisher(Float32, "/perception/confidence", 10)

        if self.publish_colored:
            self.colored_pub = self.create_publisher(Image, "/perception/colored_mask", 10)
        else:
            self.colored_pub = None

        self.get_logger().info(
            f"TraversabilityClassifierNode initialized successfully. "
            f"Backend: {self.engine.model_backend}, Device: {self.engine.device}, "
            f"Input Size: {input_w}x{input_h}, Confidence Threshold: {self.confidence_threshold}"
        )

    def image_callback(self, msg: Image) -> None:
        """Processes incoming camera frame and publishes terrain costmaps."""
        header = msg.header
        timestamp = header.stamp.sec + header.stamp.nanosec * 1e-9

        # Convert ROS message to RGB numpy array
        rgb = image_msg_to_numpy(msg)

        # Process through inference engine (handles null/corrupt frames gracefully)
        result: PerceptionOutput = self.engine.process_frame(rgb, timestamp=timestamp)

        # Publish Confidence
        conf_msg = Float32()
        conf_msg.data = float(result.confidence)
        self.conf_pub.publish(conf_msg)

        # Publish Segmentation Mask (mono8)
        mask_msg = numpy_to_image_msg(result.segmentation_mask, encoding="mono8", header=header)
        self.mask_pub.publish(mask_msg)

        # Publish Terrain Cost Map (mono8)
        cost_msg = numpy_to_image_msg(result.terrain_cost, encoding="mono8", header=header)
        self.cost_pub.publish(cost_msg)

        # Publish Colorized Visualization if enabled
        if self.colored_pub is not None and result.colored_mask is not None:
            color_msg = numpy_to_image_msg(result.colored_mask, encoding="rgb8", header=header)
            self.colored_pub.publish(color_msg)

        # Telemetry & Diagnostics logging
        if not result.is_valid:
            self.get_logger().warn(f"Degraded frame received at t={timestamp:.3f}: {result.diagnostics.get('error')}")
        elif result.confidence < self.confidence_threshold:
            self.get_logger().warn(
                f"Low perception confidence ({result.confidence:.2f} < {self.confidence_threshold:.2f}) "
                f"at t={timestamp:.3f}. Costmap marked cautious."
            )

    def destroy_node(self) -> None:
        """Clean lifecycle shutdown: outputs final telemetry summary."""
        summary = self.engine.get_telemetry_summary()
        if hasattr(self, "get_logger"):
            self.get_logger().info(
                f"Shutting down TraversabilityClassifierNode. Lifetime Telemetry: "
                f"Processed: {summary['total_frames']} frames | "
                f"Mean Latency: {summary['mean_latency_ms']:.2f} ms | "
                f"Throughput: {summary['continuous_fps']:.1f} FPS"
            )
        super().destroy_node()


def main(args: Optional[list] = None) -> None:
    """ROS 2 Entrypoint."""
    if not HAS_RCLPY:
        print("rclpy not available on this host. Run inside ROS 2 environment or use scripts/benchmark_perception_node.py.", file=sys.stderr)
        sys.exit(1)

    rclpy.init(args=args)
    node = TraversabilityClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
