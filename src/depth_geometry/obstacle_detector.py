"""Positive and negative obstacle detection from metric 3D depth geometry.

Enforces Rule 12: Invalid depth is never treated as free space.
Detects:
- Positive obstacles: points rising above traversable step threshold (e.g. > 15 cm)
- Negative obstacles: unexpected drops / ditches / holes
- Height discontinuities: sharp vertical steps/cliffs via spatial surface gradients
- Occlusion shadow indicators: abrupt ground termination into invalid depth
- Geometric traversability cost: normalized metric obstacle hazard [0.0..1.0]
- Depth quality metric: quantitative reliability score [0.0..1.0]
"""

from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np

from ..interfaces.types import DepthGeometryResult


class ObstacleDetector:
    """Detects geometric hazards, discontinuities, and computes traversability costs."""

    def __init__(
        self,
        step_threshold_m: float = 0.15,
        lethal_height_m: float = 0.35,
        drop_threshold_m: float = 0.20,
        discontinuity_threshold_m: float = 0.18,
        max_traversable_range_m: float = 12.0,
    ) -> None:
        self.step_threshold = step_threshold_m
        self.lethal_height = lethal_height_m
        self.drop_threshold = drop_threshold_m
        self.discontinuity_threshold = discontinuity_threshold_m
        self.max_range = max_traversable_range_m

    def detect_discontinuities(
        self,
        z_base: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Computes spatial height gradient to detect sharp vertical step edges/cliffs."""
        # Replace invalid with 0 for gradient computation
        z_safe = np.where(valid_mask, z_base, 0.0).astype(np.float32)

        # Sobel gradients in horizontal and vertical directions
        grad_x = cv2.Sobel(z_safe, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(z_safe, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Discontinuity occurs where gradient magnitude exceeds threshold on valid pixels
        discontinuity_mask = valid_mask & (grad_mag > self.discontinuity_threshold)
        return discontinuity_mask

    def detect_occlusion_shadows(
        self,
        points_base_link: np.ndarray,
        valid_mask: np.ndarray,
        positive_obstacle: np.ndarray,
    ) -> np.ndarray:
        """Detects negative-obstacle occlusion shadows where ground abruptly vanishes into invalid depth.

        Vectorized GPU/CPU friendly implementation (< 1 ms latency).
        """
        h, w = valid_mask.shape
        col_start = int(w * 0.25)
        col_end = int(w * 0.75)
        row_start = int(h * 0.35)

        shadow_mask = np.zeros((h, w), dtype=bool)

        # Transition where lower pixel is valid ground, and upper pixel is invalid depth
        # and NOT obstructed by a positive obstacle
        ground_to_invalid = (
            valid_mask[1:, col_start:col_end] &
            ~valid_mask[:-1, col_start:col_end] &
            ~positive_obstacle[1:, col_start:col_end]
        )

        # Upward morphological dilation into the invalid space
        kernel = np.ones((8, 1), dtype=np.uint8)
        trans_slice = np.zeros((h, col_end - col_start), dtype=np.uint8)
        trans_slice[:-1, :] = ground_to_invalid.astype(np.uint8)

        dilated = cv2.dilate(trans_slice, kernel, iterations=1, anchor=(0, 7))
        shadow_mask[:, col_start:col_end] = (dilated > 0) & ~valid_mask[:, col_start:col_end]
        return shadow_mask

    def compute_depth_quality(
        self,
        points_base_link: np.ndarray,
        valid_mask: np.ndarray,
        height_diff_map: np.ndarray,
    ) -> float:
        """Calculates quantitative depth quality metric in [0.0, 1.0]."""
        h, w = valid_mask.shape
        # Forward corridor in camera FOV: central 60% width, lower 65% height
        corridor_mask = np.zeros((h, w), dtype=bool)
        corridor_mask[int(h * 0.35):, int(w * 0.2):int(w * 0.8)] = True
        total_corridor_pixels = np.sum(corridor_mask)

        # Valid density in forward corridor
        valid_corridor = corridor_mask & valid_mask
        density = float(np.sum(valid_corridor) / max(total_corridor_pixels, 1))

        # Planar ground residual smoothness
        if np.sum(valid_corridor) > 50:
            dz_corridor = height_diff_map[valid_corridor]
            residual_std = float(np.std(dz_corridor))
            smoothness = float(np.clip(1.0 - (residual_std / 0.20), 0.1, 1.0))
        else:
            smoothness = 0.20

        quality = density * smoothness
        return float(np.clip(quality, 0.05, 0.98))

    def detect(
        self,
        points_base_link: np.ndarray,
        valid_mask: np.ndarray,
        ground_height_map: np.ndarray,
        height_diff_map: np.ndarray,
    ) -> DepthGeometryResult:
        """Detect geometric obstacles, step discontinuities, and compute geometric costs."""
        h, w = valid_mask.shape
        x = points_base_link[..., 0]
        y = points_base_link[..., 1]
        z = points_base_link[..., 2]

        in_range = valid_mask & (x >= 0.35) & (x <= self.max_range)

        # 1. Positive obstacles: points protruding above traversable step threshold
        positive_obstacle = in_range & (height_diff_map > self.step_threshold)

        # 2. Negative obstacles: points below drop threshold
        negative_obstacle_direct = in_range & (height_diff_map < -self.drop_threshold)

        # 3. Occlusion shadow negative-obstacle indicators
        occlusion_shadow = self.detect_occlusion_shadows(points_base_link, valid_mask, positive_obstacle)
        negative_obstacle = negative_obstacle_direct | occlusion_shadow

        # 4. Height discontinuity detection (steep step edges)
        discontinuity_mask = self.detect_discontinuities(z, valid_mask)

        # 5. Geometric traversability cost in [0.0, 1.0] (1.0 = lethal hazard)
        geom_cost = np.zeros((h, w), dtype=np.float32)

        # Cost scaling for positive obstacles (0.15m to 0.35m)
        pos_scale = np.clip(
            (height_diff_map - self.step_threshold) / (self.lethal_height - self.step_threshold),
            0.0, 1.0
        )
        geom_cost = np.where(positive_obstacle, np.maximum(geom_cost, pos_scale), geom_cost)

        # Negative obstacles and step discontinuities are lethal hazards
        geom_cost = np.where(negative_obstacle, 1.0, geom_cost)
        geom_cost = np.where(discontinuity_mask, np.maximum(geom_cost, 0.85), geom_cost)

        # ---------------------------------------------------------------------------
        # CRITICAL SAFETY INVARIANT (Rule 12):
        # Invalid depth must NEVER be treated as free space.
        # Assign high uncertainty penalty (0.60) so planners never steer blindly
        # into unobserved sensor blind spots.
        # ---------------------------------------------------------------------------
        invalid_depth = ~valid_mask
        geom_cost = np.where(invalid_depth, 0.60, geom_cost)

        # 6. Depth Quality Metric
        depth_quality = self.compute_depth_quality(points_base_link, valid_mask, height_diff_map)

        return DepthGeometryResult(
            ground_height_map=ground_height_map,
            positive_obstacle_mask=positive_obstacle,
            negative_obstacle_mask=negative_obstacle,
            depth_validity_mask=valid_mask,
            geometric_cost=geom_cost,
            confidence=depth_quality,
            points_3d=None,
            latency_ms=0.0,
            discontinuity_mask=discontinuity_mask,
        )
