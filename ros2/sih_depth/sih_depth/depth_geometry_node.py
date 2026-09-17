#!/usr/bin/env python3
"""Depth geometry node for pointcloud and terrain hazard computation."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid


class DepthGeometryNode(Node):
    """Processes stereo/RGB-D depth images, projects normals, and detects slope hazards."""

    def __init__(self):
        super().__init__('depth_geometry_node')

        # Declare parameters
        self.declare_parameter('max_slope_deg', 25.0)
        self.declare_parameter('step_threshold_m', 0.15)

        # Subscriptions
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        # Publishers
        self.hazard_pub = self.create_publisher(Image, '/depth/geometry_hazards', 10)
        self.slope_pub = self.create_publisher(OccupancyGrid, '/depth/slope_grid', 10)

        self.get_logger().info('DepthGeometryNode initialized successfully.')

    def depth_callback(self, msg: Image):
        """Processes depth frame and extracts geometry slope/step hazards."""
        # Minimal skeleton callback: confirms depth stream receipt
        pass


def main(args=None):
    rclpy.init(args=args)
    node = DepthGeometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
