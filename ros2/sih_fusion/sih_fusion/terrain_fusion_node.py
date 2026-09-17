#!/usr/bin/env python3
"""Terrain fusion node for combining semantic traversability and 3D geometric hazards."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32


class TerrainFusionNode(Node):
    """Synchronizes semantic masks and geometric hazards, computing a fused 2.5D costmap."""

    def __init__(self):
        super().__init__('terrain_fusion_node')

        # Declare parameters
        self.declare_parameter('semantic_weight', 0.6)
        self.declare_parameter('geometry_weight', 0.4)
        self.declare_parameter('inflation_radius_m', 0.45)

        # Subscriptions
        self.mask_sub = self.create_subscription(
            Image,
            '/perception/traversability_mask',
            self.mask_callback,
            10
        )
        self.slope_sub = self.create_subscription(
            OccupancyGrid,
            '/depth/slope_grid',
            self.slope_callback,
            10
        )

        # Publishers
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/costmap/traversability_grid', 10)
        self.fusion_conf_pub = self.create_publisher(Float32, '/costmap/fusion_confidence', 10)

        self.get_logger().info('TerrainFusionNode initialized successfully.')

    def mask_callback(self, msg: Image):
        pass

    def slope_callback(self, msg: OccupancyGrid):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = TerrainFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
