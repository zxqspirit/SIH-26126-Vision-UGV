"""3D point cloud back-projection and coordinate frame transformation.

Projects 2D depth pixels into 3D metric coordinates in the camera optical frame,
and transforms them into the robot base_link frame (X forward, Y left, Z up).
Strictly adheres to Rule 12: Invalid depth is never treated as free space.
"""

from __future__ import annotations

from typing import Tuple, Optional
import cv2
import numpy as np

from ..interfaces.types import CameraIntrinsics


class DepthProjector:
    """Projects metric depth into 3D coordinates in camera and base_link frames."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        min_range_m: float = 0.35,
        max_range_m: float = 12.0,
        apply_filter: bool = True,
    ) -> None:
        self.intrinsics = intrinsics
        self.min_range = min_range_m
        self.max_range = max_range_m
        self.apply_filter = apply_filter
        self._init_pixel_grids()

    def _init_pixel_grids(self) -> None:
        """Precompute pixel coordinate meshgrids and trigonometric factors."""
        w, h = self.intrinsics.width, self.intrinsics.height
        u_coords, v_coords = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32)
        )
        self.u_norm = (u_coords - self.intrinsics.cx) / self.intrinsics.fx
        self.v_norm = (v_coords - self.intrinsics.cy) / self.intrinsics.fy

        pitch = self.intrinsics.camera_pitch_rad
        cos_p = float(np.cos(pitch))
        sin_p = float(np.sin(pitch))
        self.cos_p = cos_p
        self.sin_p = sin_p
        self.h_cam = float(self.intrinsics.camera_height_m)

        # Precomputed combined projection & SE(3) rotation factors:
        # X_robot = Z_c * cos(pitch) - Y_c * sin(pitch) = Z_c * (cos_p - v_norm * sin_p)
        # Y_robot = -X_c = - (u_norm * Z_c)
        # Z_robot = h_cam - (Z_c * sin(pitch) + Y_c * cos(pitch)) = h_cam - Z_c * (sin_p + v_norm * cos_p)
        self.rx_factor = (cos_p - self.v_norm * sin_p).astype(np.float32)
        self.rz_factor = (sin_p + self.v_norm * cos_p).astype(np.float32)

    def filter_depth(self, depth_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies range gating and median speckle denoising.

        Returns:
            clean_depth: np.ndarray of shape (H, W), filtered depth in meters.
            valid_mask: np.ndarray of shape (H, W), boolean True where depth is valid.
        """
        # Range gating
        valid_mask = np.isfinite(depth_m) & (depth_m >= self.min_range) & (depth_m <= self.max_range)

        if not self.apply_filter:
            return depth_m, valid_mask

        # Median filter on valid depth to suppress isolated speckles while preserving step edges
        depth_copy = np.where(valid_mask, depth_m, 0.0).astype(np.float32)
        filtered = cv2.medianBlur(depth_copy, 3)

        # Retain validity after filtering
        clean_valid = valid_mask & (filtered >= self.min_range) & (filtered <= self.max_range)
        clean_depth = np.where(clean_valid, filtered, np.nan)

        return clean_depth, clean_valid

    def project_to_camera_frame(self, depth_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project depth to 3D points in camera optical frame.

        Returns:
            points_optical: np.ndarray of shape (H, W, 3), [X_right, Y_down, Z_forward]
            valid_mask: np.ndarray of shape (H, W), boolean True where depth is strictly valid
        """
        h, w = depth_m.shape[:2]
        if (w, h) != (self.intrinsics.width, self.intrinsics.height):
            u_norm = (np.arange(w, dtype=np.float32) - self.intrinsics.cx) / self.intrinsics.fx
            v_norm = (np.arange(h, dtype=np.float32) - self.intrinsics.cy) / self.intrinsics.fy
            u_grid, v_grid = np.meshgrid(u_norm, v_norm)
        else:
            u_grid = self.u_norm
            v_grid = self.v_norm

        clean_depth, valid_mask = self.filter_depth(depth_m)

        x_c = u_grid * clean_depth
        y_c = v_grid * clean_depth
        z_c = clean_depth

        points_optical = np.stack([x_c, y_c, z_c], axis=-1)
        return points_optical, valid_mask

    def transform_to_base_link(self, points_optical: np.ndarray) -> np.ndarray:
        """Transform camera optical frame points to robot base_link frame.

        Optical frame: X right, Y down, Z forward.
        Base_link frame: X forward, Y left, Z up from ground contact.
        Camera mounted at height h_cam, tilted pitch_rad downwards.
        """
        x_c = points_optical[..., 0]
        y_c = points_optical[..., 1]
        z_c = points_optical[..., 2]

        x_robot = z_c * self.cos_p - y_c * self.sin_p
        y_robot = -x_c
        z_robot = self.h_cam - (z_c * self.sin_p + y_c * self.cos_p)

        return np.stack([x_robot, y_robot, z_robot], axis=-1)

    def project_and_transform_to_base_link(self, depth_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Directly projects filtered depth to robot base_link frame without intermediate stack.

        Optimized for real-time performance (>60 FPS).
        """
        clean_depth, valid_mask = self.filter_depth(depth_m)

        x_rob = clean_depth * self.rx_factor
        y_rob = - (self.u_norm * clean_depth)
        z_rob = self.h_cam - clean_depth * self.rz_factor

        points_base = np.empty((*clean_depth.shape, 3), dtype=np.float32)
        points_base[..., 0] = x_rob
        points_base[..., 1] = y_rob
        points_base[..., 2] = z_rob

        return points_base, valid_mask
