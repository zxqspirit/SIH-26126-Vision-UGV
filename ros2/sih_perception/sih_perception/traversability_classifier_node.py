#!/usr/bin/env python3
"""Traversability classifier node for terrain segmentation and confidence estimation."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class TraversabilityClassifierNode(Node):
    """Subscribes to camera frames, infers terrain traversability, and outputs masks & confidence."""

    def __init__(self):
        super().__init__('traversability_classifier_node')

        # Declare parameters
        self.declare_parameter('confidence_threshold', 0.65)
        self.declare_parameter('model_path', 'models/traversability/baseline_cnn.onnx')

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.mask_pub = self.create_publisher(Image, '/perception/traversability_mask', 10)
        self.confidence_pub = self.create_publisher(Float32, '/perception/confidence', 10)

        self.get_logger().info('TraversabilityClassifierNode initialized successfully.')

    def image_callback(self, msg: Image):
        """Processes incoming camera frame and publishes traversability mask."""
        # Minimal skeleton callback: publishes nominal confidence
        conf_msg = Float32()
        conf_msg.data = 0.95
        self.confidence_pub.publish(conf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
