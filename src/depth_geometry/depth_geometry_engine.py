"""Depth geometry coordinator engine."""

from __future__ import annotations

import time
import numpy as np

from ..interfaces.types import CameraIntrinsics, DepthGeometryResult
from .point_cloud import DepthProjector
from .ground_estimator import GroundEstimator
from .obstacle_detector import ObstacleDetector


class DepthGeometryEngine:
    """Coordinates point projection, ground modeling, and obstacle detection."""

    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        self.intrinsics = intrinsics
        self.projector = DepthProjector(intrinsics)
        self.ground_estimator = GroundEstimator()
        self.obstacle_detector = ObstacleDetector()

    def process_depth(self, depth_m: np.ndarray) -> DepthGeometryResult:
        """Process metric depth map into 3D geometric traversability."""
        start_time = time.perf_counter()

        # 1. Directly project and transform to robot base_link frame (< 6 ms)
        points_base, valid_mask = self.projector.project_and_transform_to_base_link(depth_m)

        # 2. Ground plane estimation (< 2 ms)
        ground_map, height_diff, plane_coeffs = self.ground_estimator.estimate_ground(points_base, valid_mask)

        # 3. Obstacle detection & Rule 12 enforcement (< 5 ms)
        result = self.obstacle_detector.detect(points_base, valid_mask, ground_map, height_diff)
        result.plane_coeffs = plane_coeffs
        result.latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Subsample point cloud for visualization (e.g. 1 in every 16 points)
        step = 4
        sub_pts = points_base[::step, ::step, :].reshape(-1, 3)
        sub_valid = valid_mask[::step, ::step].flatten()
        result.points_3d = sub_pts[sub_valid]

        return result
