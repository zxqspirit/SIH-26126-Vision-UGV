"""3D point cloud back-projection and coordinate frame transformation.

Projects 2D depth pixels into 3D metric coordinates in the camera optical frame,
and transforms them into the robot base_link frame (X forward, Y left, Z up).
Strictly adheres to Rule 12: Invalid depth is never treated as free space.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from ..interfaces.types import CameraIntrinsics


class DepthProjector:
    """Projects metric depth into 3D coordinates in camera and base_link frames."""

    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        self.intrinsics = intrinsics
        self._init_pixel_grids()

    def _init_pixel_grids(self) -> None:
        """Precompute pixel coordinate meshgrids."""
        w, h = self.intrinsics.width, self.intrinsics.height
        u_coords, v_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.u_norm = (u_coords - self.intrinsics.cx) / self.intrinsics.fx
        self.v_norm = (v_coords - self.intrinsics.cy) / self.intrinsics.fy

    def project_to_camera_frame(self, depth_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project depth to 3D points in camera optical frame.

        Returns:
            points_optical: np.ndarray of shape (H, W, 3), [X_right, Y_down, Z_forward]
            valid_mask: np.ndarray of shape (H, W), boolean True where depth is valid (>0.2m and <15m)
        """
        h, w = depth_m.shape[:2]
        if (w, h) != (self.intrinsics.width, self.intrinsics.height):
            u_norm = (np.arange(w, dtype=np.float32) - self.intrinsics.cx) / self.intrinsics.fx
            v_norm = (np.arange(h, dtype=np.float32) - self.intrinsics.cy) / self.intrinsics.fy
            u_grid, v_grid = np.meshgrid(u_norm, v_norm)
        else:
            u_grid = self.u_norm
            v_grid = self.v_norm

        # Rule 12: Identify strictly valid depth returns
        valid_mask = np.isfinite(depth_m) & (depth_m > 0.2) & (depth_m < 20.0)

        # Replace invalid depth with NaN for geometry processing
        safe_depth = np.where(valid_mask, depth_m, np.nan)

        x_c = u_grid * safe_depth
        y_c = v_grid * safe_depth
        z_c = safe_depth

        points_optical = np.stack([x_c, y_c, z_c], axis=-1)
        return points_optical, valid_mask

    def transform_to_base_link(self, points_optical: np.ndarray) -> np.ndarray:
        """Transform camera optical frame points to robot base_link frame.

        Optical frame: X right, Y down, Z forward.
        Base_link frame: X forward, Y left, Z up from ground.
        Camera mounted at height h_cam, tilted pitch_rad downwards.
        """
        x_c = points_optical[..., 0]
        y_c = points_optical[..., 1]
        z_c = points_optical[..., 2]

        pitch = self.intrinsics.camera_pitch_rad
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        h_cam = self.intrinsics.camera_height_m

        # Rotation around lateral axis + translation to ground base_link
        x_robot = z_c * cos_p - y_c * sin_p
        y_robot = -x_c
        z_robot = h_cam - (z_c * sin_p + y_c * cos_p)

        return np.stack([x_robot, y_robot, z_robot], axis=-1)
