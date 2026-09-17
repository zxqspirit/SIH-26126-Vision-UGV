#!/usr/bin/env python3
"""Visual odometry node for 6-DoF vehicle pose estimation and TF2 broadcasting."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class VisualOdometryNode(Node):
    """Tracks keypoints across camera frames and broadcasts odom -> base_link transforms."""

    def __init__(self):
        super().__init__('visual_odometry_node')

        # Declare parameters
        self.declare_parameter('max_features', 1000)
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/localization/odometry', 10)
        self.status_pub = self.create_publisher(String, '/localization/status', 10)

        self.get_logger().info('VisualOdometryNode initialized successfully.')

    def image_callback(self, msg: Image):
        pass


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
