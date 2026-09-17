"""Positive and negative obstacle detection from metric 3D depth geometry.

Enforces Rule 12: Invalid depth is never treated as free space.
Detects:
- Positive obstacles: points rising above the traversable step threshold (e.g. > 15 cm)
- Negative obstacles: unexpected drops / ditches / holes
- Geometric traversability cost: normalized metric obstacle hazard [0.0..1.0]
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import cv2

from ..interfaces.types import DepthGeometryResult


class ObstacleDetector:
    """Detects geometric hazards and computes metric traversability costs."""

    def __init__(
        self,
        step_threshold_m: float = 0.15,
        lethal_height_m: float = 0.35,
        drop_threshold_m: float = 0.20,
        max_traversable_range_m: float = 12.0,
    ) -> None:
        self.step_threshold = step_threshold_m
        self.lethal_height = lethal_height_m
        self.drop_threshold = drop_threshold_m
        self.max_range = max_traversable_range_m

    def detect(
        self,
        points_base_link: np.ndarray,
        valid_mask: np.ndarray,
        ground_height_map: np.ndarray,
        height_diff_map: np.ndarray,
    ) -> DepthGeometryResult:
        """Detect geometric obstacles and compute geometric traversability costs."""
        h, w = valid_mask.shape
        x = points_base_link[..., 0]
        y = points_base_link[..., 1]
        z = points_base_link[..., 2]

        in_range = valid_mask & (x > 0.3) & (x <= self.max_range)

        # 1. Positive obstacles: points protruding above ground plane
        positive_obstacle = in_range & (height_diff_map > self.step_threshold)

        # 2. Negative obstacles: points significantly below expected ground (ditches, holes)
        negative_obstacle = in_range & (height_diff_map < -self.drop_threshold)

        # 3. Geometric traversability cost in [0.0, 1.0] (1.0 = lethal hazard)
        # Smoothly scales from step_threshold (0.0 cost) to lethal_height (1.0 cost)
        geom_cost = np.zeros((h, w), dtype=np.float32)

        # Cost for positive obstacles
        pos_scale = np.clip(
            (height_diff_map - self.step_threshold) / (self.lethal_height - self.step_threshold),
            0.0, 1.0
        )
        geom_cost = np.where(positive_obstacle, np.maximum(geom_cost, pos_scale), geom_cost)

        # Negative obstacles are lethal hazards
        geom_cost = np.where(negative_obstacle, 1.0, geom_cost)

        # Rule 12: CRITICAL - Invalid depth must NEVER be treated as free space.
        # Mark invalid depth regions with a high uncertainty penalty (0.6) so the planner
        # does NOT blindly steer into sensor-blind spots.
        invalid_depth = ~valid_mask
        geom_cost = np.where(invalid_depth, 0.60, geom_cost)

        # 4. Geometry confidence score C_geom
        # Evaluates the percentage of valid depth readings in the critical forward corridor (x in [0.5, 4.0m], |y| < 1.0m)
        corridor_mask = (x >= 0.5) & (x <= 4.0) & (abs(y) <= 1.0)
        corridor_pts = np.sum(corridor_mask)
        if corridor_pts > 50:
            valid_ratio = float(np.sum(corridor_mask & valid_mask) / corridor_pts)
            confidence = float(np.clip(valid_ratio, 0.1, 0.98))
        else:
            # If no points are in corridor, confidence is low
            confidence = 0.30

        return DepthGeometryResult(
            ground_height_map=ground_height_map,
            positive_obstacle_mask=positive_obstacle,
            negative_obstacle_mask=negative_obstacle,
            depth_validity_mask=valid_mask,
            geometric_cost=geom_cost,
            confidence=confidence,
        )
