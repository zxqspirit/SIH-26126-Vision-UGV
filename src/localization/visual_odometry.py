"""Visual Odometry engine for outdoor GPS-denied navigation.

Tracks visual features across consecutive real outdoor camera frames,
estimates 6-DOF relative motion via PnP-RANSAC, and integrates ego-motion dead-reckoning.
Emits real-time tracking diagnostics, confidence, and relocalization state.

Relocalization State Machine:
    INITIALIZING -> TRACKING -> TRACKING_LOST -> RELOCALIZATION_ATTEMPT -> RECOVERED -> TRACKING
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
    RelocalizationState,
)
from .tracking_diagnostics import TrackingDiagnostics


class VisualOdometry:
    """Estimates camera/UGV motion visually from RGB and depth streams."""

    # Relocalization constants
    RELOC_ATTEMPT_AFTER_N_LOST = 3  # Enter RELOCALIZATION_ATTEMPT after N consecutive losses
    RECOVERY_QUALITY_LEVEL = 0.008  # More aggressive feature detection during recovery
    RECOVERY_LK_WIN_SIZE = (31, 31)  # Wider search window during recovery

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

        # Recovery feature params (relaxed for aggressive redetection)
        self.recovery_feature_params = dict(
            maxCorners=max_features,
            qualityLevel=self.RECOVERY_QUALITY_LEVEL,
            minDistance=max(min_feature_dist * 0.6, 5.0),
            blockSize=9
        )

        # LK optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Recovery LK params (wider window for aggressive search)
        self.recovery_lk_params = dict(
            winSize=self.RECOVERY_LK_WIN_SIZE,
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.008)
        )

        # CLAHE for histogram equalization during recovery
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

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

        # Relocalization state machine
        self._reloc_state: RelocalizationState = RelocalizationState.INITIALIZING
        self._consecutive_lost: int = 0
        self._frames_since_recovery: int = 0
        self._total_recovery_count: int = 0

    @property
    def relocalization_state(self) -> RelocalizationState:
        """Current relocalization state."""
        return self._reloc_state

    @property
    def consecutive_lost_frames(self) -> int:
        """Number of consecutive frames with TRACKING_LOST."""
        return self._consecutive_lost

    @property
    def total_recoveries(self) -> int:
        """Total number of successful tracking recoveries."""
        return self._total_recovery_count

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
        self._reloc_state = RelocalizationState.INITIALIZING
        self._consecutive_lost = 0
        self._frames_since_recovery = 0
        self._total_recovery_count = 0

    def _preprocess_gray(self, gray: np.ndarray) -> np.ndarray:
        """Apply CLAHE histogram equalization during recovery for better feature detection."""
        if self._reloc_state in (RelocalizationState.TRACKING_LOST,
                                  RelocalizationState.RELOCALIZATION_ATTEMPT):
            return self._clahe.apply(gray)
        return gray

    def _get_feature_params(self) -> dict:
        """Return feature detection params based on current relocalization state."""
        if self._reloc_state in (RelocalizationState.TRACKING_LOST,
                                  RelocalizationState.RELOCALIZATION_ATTEMPT,
                                  RelocalizationState.RECOVERED):
            return self.recovery_feature_params
        return self.feature_params

    def _get_lk_params(self) -> dict:
        """Return LK optical flow params based on current relocalization state."""
        if self._reloc_state in (RelocalizationState.TRACKING_LOST,
                                  RelocalizationState.RELOCALIZATION_ATTEMPT):
            return self.recovery_lk_params
        return self.lk_params

    def _update_reloc_state(self, tracking_status: TrackingStatus, inlier_count: int) -> None:
        """Update relocalization state machine based on current tracking result."""
        was_lost = self._reloc_state in (
            RelocalizationState.TRACKING_LOST,
            RelocalizationState.RELOCALIZATION_ATTEMPT,
        )

        if tracking_status == TrackingStatus.TRACKING_LOST:
            self._consecutive_lost += 1
            self._frames_since_recovery = 0

            if self._consecutive_lost >= self.RELOC_ATTEMPT_AFTER_N_LOST:
                self._reloc_state = RelocalizationState.RELOCALIZATION_ATTEMPT
            else:
                self._reloc_state = RelocalizationState.TRACKING_LOST

        elif tracking_status in (TrackingStatus.TRACKING_OK, TrackingStatus.TRACKING_DEGRADED):
            if was_lost:
                # Successfully recovered from tracking loss
                self._reloc_state = RelocalizationState.RECOVERED
                self._total_recovery_count += 1
                self._frames_since_recovery = 1
            elif self._reloc_state == RelocalizationState.RECOVERED:
                self._frames_since_recovery += 1
                # Transition to stable TRACKING after 3 consecutive good frames post-recovery
                if self._frames_since_recovery >= 3:
                    self._reloc_state = RelocalizationState.TRACKING
                    self._frames_since_recovery = 0
            elif self._reloc_state == RelocalizationState.INITIALIZING:
                self._reloc_state = RelocalizationState.TRACKING
            else:
                self._reloc_state = RelocalizationState.TRACKING

            self._consecutive_lost = 0

    def process_frame(self, rgb_image: np.ndarray, depth_m: np.ndarray) -> VisualOdometryResult:
        """Process incoming frame and compute visual ego-motion."""
        start_time = time.perf_counter()

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        # Apply CLAHE during recovery phases
        gray_processed = self._preprocess_gray(gray)
        h, w = gray.shape

        delta_x = 0.0
        delta_y = 0.0
        delta_yaw = 0.0
        inlier_count = 0
        total_tracked = 0

        # First frame initialization
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 30:
            feat_params = self._get_feature_params()
            corners = cv2.goodFeaturesToTrack(gray_processed, mask=None, **feat_params)
            self.prev_gray = gray_processed
            self.prev_pts = corners
            self.prev_depth = depth_m
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            init_inliers = len(corners) if corners is not None else 0
            tracking_status = TrackingStatus.TRACKING_OK if init_inliers >= 30 else TrackingStatus.TRACKING_LOST
            _, init_confidence = self.diagnostics.evaluate(init_inliers, init_inliers)
            self._update_reloc_state(tracking_status, init_inliers)

            return VisualOdometryResult(
                x=self.global_x,
                y=self.global_y,
                z=self.global_z,
                yaw=self.global_yaw,
                delta_x=0.0,
                delta_y=0.0,
                delta_yaw=0.0,
                inlier_count=init_inliers,
                tracking_status=tracking_status,
                confidence=init_confidence,
                latency_ms=latency_ms,
                relocalization_state=self._reloc_state.value,
                consecutive_lost_frames=self._consecutive_lost,
                recovery_frame_count=self._frames_since_recovery,
            )

        # 1. Track features forward with pyramidal Lucas-Kanade optical flow
        lk_params = self._get_lk_params()
        tracked_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_processed, self.prev_pts, None, **lk_params
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

        # 4. Update relocalization state machine
        self._update_reloc_state(tracking_status, inlier_count)

        # 5. Integrate into cumulative odometry if tracking is acceptable
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

        # 6. Redetect features if point density has thinned out
        if inlier_count < 50:
            feat_params = self._get_feature_params()
            corners = cv2.goodFeaturesToTrack(gray_processed, mask=None, **feat_params)
            self.prev_pts = corners
        else:
            self.prev_pts = pts_curr.reshape(-1, 1, 2)

        self.prev_gray = gray_processed
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
            relocalization_state=self._reloc_state.value,
            consecutive_lost_frames=self._consecutive_lost,
            recovery_frame_count=self._frames_since_recovery,
        )
