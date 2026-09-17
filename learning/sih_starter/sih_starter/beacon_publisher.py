"""Beginner ROS 2 Publisher Node: Beacon Publisher.

Publishes periodic heartbeat string messages to demonstrate:
- Node creation with rclpy
- Declaring and reading parameters
- Creating a publisher on a named topic
- Timer-based callback loops
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class BeaconPublisher(Node):
    """Publishes periodic system status heartbeats."""

    def __init__(self) -> None:
        super().__init__('beacon_publisher')

        # 1. Declare and read parameter
        self.declare_parameter('publish_period_sec', 1.0)
        period = self.get_parameter('publish_period_sec').get_parameter_value().double_value

        # 2. Create publisher on topic '/ugv/heartbeat' with queue depth 10
        self.publisher_ = self.create_publisher(String, '/ugv/heartbeat', 10)

        # 3. Create timer callback
        self.count = 0
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f'BeaconPublisher initialized. Broadcasting to /ugv/heartbeat every {period:.1f}s'
        )

    def timer_callback(self) -> None:
        """Periodic callback invoked by the ROS 2 executor."""
        msg = String()
        msg.data = f'UGV System Alive | sequence: {self.count} | status: NOMINAL'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published beacon #{self.count}: "{msg.data}"')
        self.count += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BeaconPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
