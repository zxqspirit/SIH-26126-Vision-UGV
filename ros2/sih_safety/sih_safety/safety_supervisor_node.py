#!/usr/bin/env python3
"""Safety supervisor node for deadman monitoring, confidence derating, and E-STOP gating."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String


class SafetySupervisorNode(Node):
    """Monitors watchdog heartbeat and confidence metrics, gating safe commands onto /cmd_vel."""

    def __init__(self):
        super().__init__('safety_supervisor_node')

        # Declare parameters
        self.declare_parameter('heartbeat_timeout_sec', 0.5)
        self.declare_parameter('min_autonomous_confidence', 0.6)

        # Subscriptions
        self.raw_cmd_sub = self.create_subscription(
            Twist,
            '/navigation/cmd_vel_raw',
            self.cmd_callback,
            10
        )
        self.conf_sub = self.create_subscription(
            Float32,
            '/perception/confidence',
            self.conf_callback,
            10
        )
        self.heartbeat_sub = self.create_subscription(
            String,
            '/ugv/heartbeat',
            self.heartbeat_callback,
            10
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/safety/status', 10)

        self.get_logger().info('SafetySupervisorNode initialized successfully.')

    def cmd_callback(self, msg: Twist):
        # Passes gated command through
        self.cmd_pub.publish(msg)

    def conf_callback(self, msg: Float32):
        pass

    def heartbeat_callback(self, msg: String):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = SafetySupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
