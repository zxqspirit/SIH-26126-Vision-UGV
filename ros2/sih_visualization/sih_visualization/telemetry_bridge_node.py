#!/usr/bin/env python3
"""Telemetry bridge node for aggregating topic states and streaming to web dashboard and RViz2."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class TelemetryBridgeNode(Node):
    """Aggregates UGV runtime states and exposes telemetry streams for operator interface."""

    def __init__(self):
        super().__init__('telemetry_bridge_node')

        # Subscriptions
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.status_sub = self.create_subscription(String, '/safety/status', self.status_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/localization/odometry', self.odom_callback, 10)

        # Publishers
        self.bridge_heartbeat_pub = self.create_publisher(String, '/visualization/bridge_status', 10)

        self.get_logger().info('TelemetryBridgeNode initialized successfully.')

    def cmd_callback(self, msg: Twist):
        pass

    def status_callback(self, msg: String):
        pass

    def odom_callback(self, msg: Odometry):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
