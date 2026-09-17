"""Beginner ROS 2 Subscriber Node: Beacon Subscriber.

Listens for heartbeat messages on /ugv/heartbeat to demonstrate:
- Creating a subscriber with a callback function
- Event-driven message handling
- Informative logging with get_logger()
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class BeaconSubscriber(Node):
    """Subscribes to system heartbeat messages."""

    def __init__(self) -> None:
        super().__init__('beacon_subscriber')

        # Create subscriber on topic '/ugv/heartbeat' with queue depth 10
        self.subscription = self.create_subscription(
            String,
            '/ugv/heartbeat',
            self.listener_callback,
            10
        )
        self.get_logger().info('BeaconSubscriber initialized. Listening on /ugv/heartbeat...')

    def listener_callback(self, msg: String) -> None:
        """Event callback invoked automatically whenever a message arrives on the topic."""
        self.get_logger().info(f'Received beacon: "{msg.data}"')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BeaconSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
