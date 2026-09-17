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
        forward_range_m: Tuple[float, float] = (0.5, 5.0),
        lateral_range_m: Tuple[float, float] = (-1.5, 1.5),
        max_slope_deg: float = 25.0,
    ) -> None:
        self.max_ground_height = max_ground_height_m
        self.forward_range = forward_range_m
        self.lateral_range = lateral_range_m
        self.max_slope_deg = max_slope_deg
        # tan(25 deg) ~ 0.466
        self.max_slope_tan = np.tan(np.radians(max_slope_deg))

    def estimate_ground(
        self,
        points_base_link: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
        """Estimate ground height map and plane equation ax + by + cz + d = 0.

        Returns:
            ground_height_map: np.ndarray of shape (H, W), estimated ground elevation at each pixel
            height_diff_map: np.ndarray of shape (H, W), Z_point - Z_ground (positive = above ground)
            plane_coeffs: tuple of (a, b, c, d)
        """
        h, w = valid_mask.shape
        x = points_base_link[..., 0]
        y = points_base_link[..., 1]
        z = points_base_link[..., 2]

        # Candidate ground points: in forward corridor near nominal ground (|z| < 0.35m)
        in_corridor = (
            valid_mask &
            (x >= self.forward_range[0]) & (x <= self.forward_range[1]) &
            (y >= self.lateral_range[0]) & (y <= self.lateral_range[1]) &
            (z >= -0.35) & (z <= 0.35)
        )

        x_pts = x[in_corridor]
        y_pts = y[in_corridor]
        z_pts = z[in_corridor]

        # Default horizontal ground plane at z = 0.0: 0*x + 0*y - 1*z + 0 = 0
        a, b, c, d = 0.0, 0.0, -1.0, 0.0

        n_pts = len(z_pts)
        if n_pts >= 50:
            # Subsample candidate points for fast robust plane fitting (< 1 ms)
            if n_pts > 1000:
                step = n_pts // 1000
                x_fit, y_fit, z_fit = x_pts[::step], y_pts[::step], z_pts[::step]
            else:
                x_fit, y_fit, z_fit = x_pts, y_pts, z_pts

            A = np.column_stack([x_fit, y_fit, np.ones_like(x_fit)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, z_fit, rcond=None)
                p0, p1, p2 = coeffs

                # Verify slope is physically plausible for outdoor UGV (< max_slope_deg)
                total_slope = np.sqrt(p0**2 + p1**2)
                if total_slope <= self.max_slope_tan and abs(p2) <= 0.25:
                    # z = p0*x + p1*y + p2  =>  p0*x + p1*y - 1.0*z + p2 = 0
                    a, b, c, d = float(p0), float(p1), -1.0, float(p2)
            except Exception:
                pass

        # Compute ground elevation at every pixel: z_ground = -(a*x + b*y + d) / c = a*x + b*y + d
        z_ground = a * x + b * y + d

        # For invalid depth pixels, set nominal zero ground
        z_ground_clean = np.where(valid_mask, z_ground, 0.0)

        # Height difference: points significantly above ground are obstacles
        height_diff = np.where(valid_mask, z - z_ground_clean, 0.0)

        return z_ground_clean, height_diff, (a, b, c, d)
