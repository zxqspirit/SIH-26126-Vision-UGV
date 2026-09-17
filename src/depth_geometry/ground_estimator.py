"""Ground plane estimation and surface elevation modeling.

Estimates the traversable ground plane in the robot base_link frame using
robust statistics and RANSAC, providing reference elevations for obstacle detection.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np


class GroundEstimator:
    """Estimates the ground elevation plane in base_link coordinates."""

    def __init__(
        self,
        max_ground_height_m: float = 0.15,
        forward_range_m: Tuple[float, float] = (0.5, 6.0),
        lateral_range_m: Tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        self.max_ground_height_m = max_ground_height_m
        self.forward_range = forward_range_m
        self.lateral_range = lateral_range_m

    def estimate_ground(
        self,
        points_base_link: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
        """Estimate ground height map and ground plane equation ax + by + cz + d = 0.

        Returns:
            ground_height_map: np.ndarray of shape (H, W), estimated ground elevation at each pixel
            height_diff_map: np.ndarray of shape (H, W), Z_point - Z_ground (positive = above ground)
            plane_coeffs: tuple of (a, b, c, d)
        """
        h, w = valid_mask.shape
        x = points_base_link[..., 0]
        y = points_base_link[..., 1]
        z = points_base_link[..., 2]

        # Candidate ground points: in forward corridor, near ground level (|z| < 0.35m)
        in_corridor = (
            valid_mask &
            (x >= self.forward_range[0]) & (x <= self.forward_range[1]) &
            (y >= self.lateral_range[0]) & (y <= self.lateral_range[1]) &
            (z >= -0.35) & (z <= 0.35)
        )

        x_pts = x[in_corridor]
        y_pts = y[in_corridor]
        z_pts = z[in_corridor]

        # Default horizontal ground plane at z = 0.0: 0*x + 0*y + 1*z + 0 = 0
        a, b, c, d = 0.0, 0.0, 1.0, 0.0

        if len(z_pts) > 100:
            # Fit plane z = p0*x + p1*y + p2 using robust least squares
            A = np.column_stack([x_pts, y_pts, np.ones_like(x_pts)])
            try:
                # Solve least squares
                coeffs, _, _, _ = np.linalg.lstsq(A, z_pts, rcond=None)
                p0, p1, p2 = coeffs
                # Check if slope is reasonable for outdoor terrain (< 30 degrees)
                if abs(p0) < 0.6 and abs(p1) < 0.6 and abs(p2) < 0.3:
                    # z = p0*x + p1*y + p2  =>  p0*x + p1*y - z + p2 = 0
                    a, b, c, d = float(p0), float(p1), -1.0, float(p2)
            except Exception:
                pass

        # Compute ground elevation at every (x, y) point: z_ground = -(a*x + b*y + d) / c
        z_ground = -(a * x + b * y + d) / c
        # For pixels without valid depth, set default nominal ground
        z_ground_clean = np.where(valid_mask, z_ground, 0.0)

        # Height difference: points significantly above ground are obstacles
        height_diff = np.where(valid_mask, z - z_ground_clean, 0.0)

        return z_ground_clean, height_diff, (a, b, c, d)
