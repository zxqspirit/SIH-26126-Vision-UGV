"""Visual Odometry engine for outdoor GPS-denied navigation.

Tracks visual features across consecutive real outdoor camera frames,
estimates 6-DOF relative motion via PnP-RANSAC, and integrates ego-motion dead-reckoning.
Emits real-time tracking diagnostics and confidence.
"""

from __future__ import annotations

import time
from typing import Optional, List, Tuple
import numpy as np
import cv2

from ..interfaces.types import (
    CameraIntrinsics,
    VisualOdometryResult,
    TrackingStatus,
)
from .tracking_diagnostics import TrackingDiagnostics


class VisualOdometry:
    """Estimates camera/UGV motion visually from RGB and depth streams."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        max_features: int = 400,
        min_feature_dist: float = 12.0,
    ) -> None:
        self.intrinsics = intrinsics
        self.max_features = max_features
        self.min_feature_dist = min_feature_dist
        self.diagnostics = TrackingDiagnostics()

        # Feature detector: Shi-Tomasi corners / FAST
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=0.015,
            minDistance=min_feature_dist,
            blockSize=7
        )

        # LK optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # State
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_depth: Optional[np.ndarray] = None

        # Cumulative odometry in global frame
        self.global_x: float = 0.0
        self.global_y: float = 0.0
        self.global_z: float = 0.0
        self.global_yaw: float = 0.0
        self.trajectory_history: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]

    def reset(self) -> None:
        """Reset odometry state."""
        self.prev_gray = None
        self.prev_pts = None
        self.prev_depth = None
        self.global_x = 0.0
        self.global_y = 0.0
        self.global_z = 0.0
        self.global_yaw = 0.0
        self.trajectory_history = [(0.0, 0.0, 0.0)]

    def process_frame(self, rgb_image: np.ndarray, depth_m: np.ndarray) -> VisualOdometryResult:
        """Process incoming frame and compute visual ego-motion."""
        start_time = time.perf_counter()

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        delta_x = 0.0
        delta_y = 0.0
        delta_yaw = 0.0
        inlier_count = 0
        total_tracked = 0

        # First frame initialization
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 30:
            corners = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_gray = gray
            self.prev_pts = corners
            self.prev_depth = depth_m
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return VisualOdometryResult(
                x=self.global_x,
                y=self.global_y,
                z=self.global_z,
                yaw=self.global_yaw,
                delta_x=0.0,
                delta_y=0.0,
                delta_yaw=0.0,
                inlier_count=len(corners) if corners is not None else 0,
                tracking_status=TrackingStatus.TRACKING_OK,
                confidence=0.90,
                latency_ms=latency_ms,
            )

        # 1. Track features forward with pyramidal Lucas-Kanade optical flow
        tracked_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        valid_flow = (status.flatten() == 1)
        pts_prev = self.prev_pts[valid_flow].reshape(-1, 2)
        pts_curr = tracked_pts[valid_flow].reshape(-1, 2)
        total_tracked = len(pts_curr)

        if total_tracked >= 8 and self.prev_depth is not None:
            # 2. Extract 3D coordinates of tracked features in previous frame
            u_coords = np.clip(pts_prev[:, 0].astype(np.int32), 0, w - 1)
            v_coords = np.clip(pts_prev[:, 1].astype(np.int32), 0, h - 1)
            depth_vals = self.prev_depth[v_coords, u_coords]

            valid_depth_mask = np.isfinite(depth_vals) & (depth_vals > 0.4) & (depth_vals < 15.0)

            pts3d = []
            pts2d = []

            for (u, v), z, is_val in zip(pts_prev, depth_vals, valid_depth_mask):
                if is_val:
                    # Camera optical frame: X right, Y down, Z forward
                    x3 = (u - self.intrinsics.cx) * z / self.intrinsics.fx
                    y3 = (v - self.intrinsics.cy) * z / self.intrinsics.fy
                    pts3d.append([x3, y3, z])
            pts2d = pts_curr[valid_depth_mask]

            if len(pts3d) >= 6:
                obj_pts = np.array(pts3d, dtype=np.float32)
                img_pts = np.array(pts2d, dtype=np.float32)
                camera_matrix = self.intrinsics.matrix

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=obj_pts,
                    imagePoints=img_pts,
                    cameraMatrix=camera_matrix,
                    distCoeffs=None,
                    iterationsCount=150,
                    reprojectionError=2.5,
                    flags=cv2.SOLVEPNP_EPNP,
                )

                if success and inliers is not None:
                    inlier_count = len(inliers)
                    R, _ = cv2.Rodrigues(rvec)

                    # Transform motion to robot base_link frame
                    # Optical frame motion: tvec = [tx, ty, tz]
                    # In robot frame: Forward dx = tz, Lateral dy = -tx, Yaw around vertical
                    dx = float(tvec[2, 0])
                    dy = float(-tvec[0, 0])
                    # Yaw angle from rotation matrix
                    dyaw = float(np.arctan2(R[0, 2], R[2, 2]))

                    # Sanity bounds check for outdoor UGV frame-to-frame motion
                    if abs(dx) < 1.5 and abs(dy) < 1.0 and abs(dyaw) < 0.6:
                        delta_x = dx
                        delta_y = dy
                        delta_yaw = dyaw

        # 3. Evaluate tracking health and confidence
        tracking_status, vo_confidence = self.diagnostics.evaluate(inlier_count, total_tracked)

        # 4. Integrate into cumulative odometry if tracking is acceptable
        if tracking_status != TrackingStatus.TRACKING_LOST:
            # Rotate local motion by current global yaw
            cos_h = np.cos(self.global_yaw)
            sin_h = np.sin(self.global_yaw)
            self.global_x += delta_x * cos_h - delta_y * sin_h
            self.global_y += delta_x * sin_h + delta_y * cos_h
            self.global_yaw += delta_yaw
            # Wrap yaw to [-pi, pi]
            self.global_yaw = (self.global_yaw + np.pi) % (2.0 * np.pi) - np.pi
            self.trajectory_history.append((self.global_x, self.global_y, self.global_yaw))

        # 5. Redetect features if point density has thinned out
        if inlier_count < 50:
            corners = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_pts = corners
        else:
            self.prev_pts = pts_curr.reshape(-1, 1, 2)

        self.prev_gray = gray
        self.prev_depth = depth_m

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return VisualOdometryResult(
            x=round(self.global_x, 4),
            y=round(self.global_y, 4),
            z=round(self.global_z, 4),
            yaw=round(self.global_yaw, 4),
            delta_x=round(delta_x, 4),
            delta_y=round(delta_y, 4),
            delta_yaw=round(delta_yaw, 4),
            inlier_count=inlier_count,
            tracking_status=tracking_status,
            confidence=vo_confidence,
            latency_ms=latency_ms,
        )
