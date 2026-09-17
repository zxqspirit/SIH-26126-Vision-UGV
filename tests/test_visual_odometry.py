"""Tests for Visual Odometry and tracking diagnostics."""

import pytest
import numpy as np
from src.interfaces.types import CameraIntrinsics, TrackingStatus
from src.localization.visual_odometry import VisualOdometry
from src.localization.tracking_diagnostics import TrackingDiagnostics


def test_tracking_diagnostics_states():
    """Diagnostics properly maps inlier counts to OK, DEGRADED, and LOST states."""
    diag = TrackingDiagnostics(nominal_inliers=40, degraded_inliers=18, lost_inliers=8)

    status_ok, conf_ok = diag.evaluate(inlier_count=50, total_tracked=60)
    assert status_ok == TrackingStatus.TRACKING_OK
    assert conf_ok >= 0.75

    status_deg, conf_deg = diag.evaluate(inlier_count=25, total_tracked=60)
    assert status_deg == TrackingStatus.TRACKING_DEGRADED
    assert 0.35 <= conf_deg < 0.70

    status_lost, conf_lost = diag.evaluate(inlier_count=5, total_tracked=60)
    assert status_lost == TrackingStatus.TRACKING_LOST
    assert conf_lost < 0.30


def test_visual_odometry_initialization():
    """VO initializes at origin and emits valid first-frame odometry."""
    intrinsics = CameraIntrinsics()
    vo = VisualOdometry(intrinsics)

    dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_depth = np.full((480, 640), 2.5, dtype=np.float32)

    res = vo.process_frame(dummy_rgb, dummy_depth)

    assert res.x == 0.0
    assert res.y == 0.0
    assert res.yaw == 0.0
    assert res.tracking_status == TrackingStatus.TRACKING_OK
