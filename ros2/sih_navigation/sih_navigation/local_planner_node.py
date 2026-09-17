#!/usr/bin/env python3
"""Local planner node for trajectory evaluation across traversability costmaps."""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped


class LocalPlannerNode(Node):
    """Generates collision-free trajectories on the traversability costmap toward goals."""

    def __init__(self):
        super().__init__('local_planner_node')

        # Declare parameters
        self.declare_parameter('max_linear_speed', 1.2)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('goal_tolerance_m', 0.3)

        # Subscriptions
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/costmap/traversability_grid',
            self.costmap_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/localization/odometry',
            self.odom_callback,
            10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/ugv/goal_pose',
            self.goal_callback,
            10
        )

        # Publishers
        self.path_pub = self.create_publisher(Path, '/navigation/planned_path', 10)
        self.cmd_pub = self.create_publisher(Twist, '/navigation/cmd_vel_raw', 10)

        self.get_logger().info('LocalPlannerNode initialized successfully.')

    def costmap_callback(self, msg: OccupancyGrid):
        pass

    def odom_callback(self, msg: Odometry):
        pass

    def goal_callback(self, msg: PoseStamped):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
