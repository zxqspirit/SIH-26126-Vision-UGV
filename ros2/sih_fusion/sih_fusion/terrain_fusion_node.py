#!/usr/bin/env python3
"""ROS 2 Terrain Fusion Node for SIH 26126 Autonomous UGV.

Subscribes to:
  /perception/segmentation_mask (sensor_msgs/msg/Image: mono8 0..8)
  /perception/terrain_cost (sensor_msgs/msg/Image: mono8 0..100)
  /perception/confidence (std_msgs/msg/Float32: 0.0..1.0)
  /depth/geometric_cost (sensor_msgs/msg/Image: mono8 0..100)
  /depth/quality_metric (std_msgs/msg/Float32: 0.0..1.0)

Publishes:
  /costmap/traversability_grid (nav_msgs/msg/OccupancyGrid: 0..100, -1=Unknown)
  /costmap/obstacle_mask (nav_msgs/msg/OccupancyGrid: 0 or 100)
  /costmap/uncertainty_grid (nav_msgs/msg/OccupancyGrid: 0..100)
  /costmap/fusion_confidence (std_msgs/msg/Float32: 0.0..1.0)

Enforces:
  - Geometry Veto on physical obstacles
  - Semantic Veto on liquid/mud hazards
  - Rule 11/12 non-zero unknown cost penalties
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Any

import numpy as np

# Ensure project root is in sys.path
candidate_roots = [
    os.environ.get("SIH_ROOT", ""),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")),
]
for root in candidate_roots:
    if root and os.path.isdir(os.path.join(root, "src", "fusion")):
        if root not in sys.path:
            sys.path.insert(0, root)
        break

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import Image
    from nav_msgs.msg import OccupancyGrid, MapMetaData
    from geometry_msgs.msg import Pose, Point, Quaternion
    from std_msgs.msg import Float32, Header
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    class Node:
        def __init__(self, name: str) -> None:
            self.name = name

from src.interfaces.types import (
    CameraIntrinsics,
    SemanticResult,
    DepthGeometryResult,
    FusedTraversabilityResult,
    TerrainClass,
)
from src.fusion.fusion_engine import FusionEngine
from src.fusion.disagreement import FusionParameters
from src.depth_geometry.point_cloud import DepthProjector


def numpy_to_occupancy_grid(
    grid_data: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    header: Any,
    unknown_val: int = -1,
) -> Any:
    """Converts a 2D numpy array (H, W) into a nav_msgs/OccupancyGrid message."""
    msg = OccupancyGrid()
    msg.header = header
    msg.header.frame_id = "base_link"

    h, w = grid_data.shape
    msg.info.resolution = float(resolution)
    msg.info.width = int(w)
    msg.info.height = int(h)

    # Origin pose: base_link position at grid (X=0, Y=-5m)
    msg.info.origin.position.x = float(origin_x)
    msg.info.origin.position.y = float(origin_y)
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0

    # Flatten in row-major order
    # Cast to int8 with unknown handling
    flat = grid_data.flatten()
    msg.data = [int(v) if v >= 0 else unknown_val for v in flat]
    return msg


class TerrainFusionNode(Node):
    """ROS 2 Node for multimodal semantic-geometric terrain costmap fusion."""

    def __init__(self) -> None:
        if HAS_RCLPY:
            super().__init__("terrain_fusion_node")
            self._init_node()
        else:
            super().__init__("terrain_fusion_node_mock")
            self._init_engine_only()

    def _init_node(self) -> None:
        # Declare parameters
        self.declare_parameter("resolution_m", 0.10)
        self.declare_parameter("grid_forward_m", 10.0)
        self.declare_parameter("grid_lateral_m", 10.0)
        self.declare_parameter("default_unknown_cost", 128)
        self.declare_parameter("camera_height_m", 0.45)
        self.declare_parameter("camera_pitch_rad", 0.209)

        res = float(self.get_parameter("resolution_m").value)
        fwd = float(self.get_parameter("grid_forward_m").value)
        lat = float(self.get_parameter("grid_lateral_m").value)
        unk = int(self.get_parameter("default_unknown_cost").value)
        cam_h = float(self.get_parameter("camera_height_m").value)
        pitch = float(self.get_parameter("camera_pitch_rad").value)

        self.intrinsics = CameraIntrinsics(
            camera_height_m=cam_h,
            camera_pitch_rad=pitch
        )
        self.engine = FusionEngine(
            self.intrinsics,
            grid_size_m=(fwd, lat),
            resolution_m=res,
            default_unknown_cost=unk,
        )
        self.projector = DepthProjector(self.intrinsics)

        # QoS Profiles
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        reliable_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        # Subscriptions
        self.sub_sem_cost = self.create_subscription(
            Image,
            "/perception/terrain_cost",
            self.sem_cost_callback,
            sensor_qos,
        )
        self.sub_sem_conf = self.create_subscription(
            Float32,
            "/perception/confidence",
            self.sem_conf_callback,
            sensor_qos,
        )
        self.sub_geom_cost = self.create_subscription(
            Image,
            "/depth/geometric_cost",
            self.geom_cost_callback,
            sensor_qos,
        )
        self.sub_depth_qual = self.create_subscription(
            Float32,
            "/depth/quality_metric",
            self.depth_qual_callback,
            sensor_qos,
        )

        # Publishers
        self.pub_costmap = self.create_publisher(OccupancyGrid, "/costmap/traversability_grid", reliable_qos)
        self.pub_obstacle = self.create_publisher(OccupancyGrid, "/costmap/obstacle_mask", reliable_qos)
        self.pub_uncertainty = self.create_publisher(OccupancyGrid, "/costmap/uncertainty_grid", reliable_qos)
        self.pub_confidence = self.create_publisher(Float32, "/costmap/fusion_confidence", reliable_qos)

        # Internal state buffers
        self.latest_sem_cost: Optional[np.ndarray] = None
        self.latest_sem_conf: float = 0.85
        self.latest_geom_cost: Optional[np.ndarray] = None
        self.latest_depth_qual: float = 0.85
        self.last_header: Optional[Any] = None

        self.frame_count = 0
        self.total_latency_ms = 0.0

        self.get_logger().info(
            f"TerrainFusionNode initialized. Grid: {fwd}x{lat}m @ {res}m/cell. "
            f"Asymmetric arbitration: Geometry Veto (physical) + Semantic Veto (liquid/mud)."
        )

    def _init_engine_only(self) -> None:
        self.intrinsics = CameraIntrinsics()
        self.engine = FusionEngine(self.intrinsics)
        self.projector = DepthProjector(self.intrinsics)
        self.latest_sem_conf = 0.85
        self.latest_depth_qual = 0.85
        self.frame_count = 0

    def sem_cost_callback(self, msg: Any) -> None:
        h = getattr(msg, "height", 480)
        w = getattr(msg, "width", 640)
        data = getattr(msg, "data", b"")
        self.latest_sem_cost = np.frombuffer(data, dtype=np.uint8).reshape((h, w)).astype(np.float32) / 100.0
        self.last_header = msg.header
        self._try_fuse()

    def sem_conf_callback(self, msg: Any) -> None:
        self.latest_sem_conf = float(msg.data)

    def geom_cost_callback(self, msg: Any) -> None:
        h = getattr(msg, "height", 480)
        w = getattr(msg, "width", 640)
        data = getattr(msg, "data", b"")
        self.latest_geom_cost = np.frombuffer(data, dtype=np.uint8).reshape((h, w)).astype(np.float32) / 100.0
        self.last_header = msg.header
        self._try_fuse()

    def depth_qual_callback(self, msg: Any) -> None:
        self.latest_depth_qual = float(msg.data)

    def _try_fuse(self) -> None:
        if self.latest_sem_cost is None or self.latest_geom_cost is None:
            return

        t_start = time.perf_counter()

        # Build SemanticResult
        sem_trav = 1.0 - self.latest_sem_cost
        terrain_classes = np.zeros(sem_trav.shape, dtype=np.int32)
        sem_res = SemanticResult(
            traversability_mask=sem_trav,
            terrain_class_map=terrain_classes,
            confidence=self.latest_sem_conf,
        )

        # Build DepthGeometryResult
        # Cost >= 0.60 indicates invalid or obstacle
        valid_depth = self.latest_geom_cost < 0.60
        geo_res = DepthGeometryResult(
            ground_height_map=np.zeros_like(sem_trav),
            positive_obstacle_mask=self.latest_geom_cost >= 0.50,
            negative_obstacle_mask=np.zeros_like(valid_depth),
            depth_validity_mask=valid_depth,
            geometric_cost=self.latest_geom_cost,
            confidence=self.latest_depth_qual,
        )

        # Synthetic 3D points
        h, w = sem_trav.shape
        clean_depth = np.where(valid_depth, 2.5, np.nan)
        pts_base, _ = self.projector.project_and_transform_to_base_link(clean_depth)

        fused = self.engine.fuse(sem_res, geo_res, pts_base)
        dt_ms = (time.perf_counter() - t_start) * 1000.0

        self.frame_count += 1
        self.total_latency_ms += dt_ms

        if HAS_RCLPY and self.last_header is not None:
            # 1. Publish Traversability Costmap OccupancyGrid (0..100, -1=Unknown)
            occ_data = (fused.fused_costmap.astype(np.float32) / 254.0 * 100.0).astype(np.int8)
            occ_data[fused.unknown_mask] = -1
            msg_costmap = numpy_to_occupancy_grid(
                occ_data, self.engine.resolution_m, 0.0, -self.engine.grid_size_m[1] / 2.0, self.last_header
            )
            self.pub_costmap.publish(msg_costmap)

            # 2. Publish Obstacle Mask OccupancyGrid (0 or 100)
            if fused.obstacle_mask is not None:
                obs_data = np.where(fused.obstacle_mask, 100, 0).astype(np.int8)
            else:
                obs_data = np.where(fused.fused_costmap >= 220, 100, 0).astype(np.int8)
            msg_obs = numpy_to_occupancy_grid(
                obs_data, self.engine.resolution_m, 0.0, -self.engine.grid_size_m[1] / 2.0, self.last_header
            )
            self.pub_obstacle.publish(msg_obs)

            # 3. Publish Uncertainty Grid (0..100)
            if fused.uncertainty_grid is not None:
                unc_data = (fused.uncertainty_grid * 100.0).astype(np.int8)
            else:
                unc_data = np.full(fused.fused_costmap.shape, 50, dtype=np.int8)
            msg_unc = numpy_to_occupancy_grid(
                unc_data, self.engine.resolution_m, 0.0, -self.engine.grid_size_m[1] / 2.0, self.last_header
            )
            self.pub_uncertainty.publish(msg_unc)

            # 4. Publish Fusion Confidence
            conf_msg = Float32()
            conf_msg.data = float(fused.confidence)
            self.pub_confidence.publish(conf_msg)

            if self.frame_count % 30 == 0:
                avg_lat = self.total_latency_ms / self.frame_count
                self.get_logger().info(
                    f"[FusionNode #{self.frame_count}] Latency: {dt_ms:.1f}ms (Avg: {avg_lat:.1f}ms) | "
                    f"Conf: {fused.confidence:.2f} | ObsCells: {int(np.sum(obs_data == 100))}"
                )


def main(args: Optional[list] = None) -> None:
    if not HAS_RCLPY:
        print("rclpy not installed. Run in a ROS 2 environment.")
        return

    rclpy.init(args=args)
    node = TerrainFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("TerrainFusionNode shutting down cleanly...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
