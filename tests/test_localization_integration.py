"""Pytest unit tests for the upgraded visual localization subsystem.

Tests:
  - Relocalization state machine transitions
  - Recovery heuristic behavior (CLAHE + wider window)
  - Localization logger format validation
  - Confidence bounding invariants
  - Tracking on real outdoor scenarios
"""

import json
import os
import tempfile

import numpy as np
import pytest

from src.interfaces.types import CameraIntrinsics, TrackingStatus, RelocalizationState
from src.localization.visual_odometry import VisualOdometry
from src.localization.localization_logger import LocalizationLogger
from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader


def _make_vo() -> VisualOdometry:
    """Create a VisualOdometry instance with default intrinsics."""
    return VisualOdometry(CameraIntrinsics())


def _make_textured_frame(seed: int = 0) -> tuple:
    """Generate a synthetic textured RGB frame and depth map."""
    rng = np.random.RandomState(seed)
    rgb = rng.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 3.0, dtype=np.float32)
    depth += rng.uniform(-0.5, 0.5, (480, 640)).astype(np.float32)
    return rgb, depth


def _make_blank_frame() -> tuple:
    """Generate a featureless uniform frame (worst case for tracking)."""
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    depth = np.full((480, 640), 3.0, dtype=np.float32)
    return rgb, depth


# ========================================================================
# Relocalization State Machine Tests
# ========================================================================

class TestRelocalizationStateMachine:

    def test_initial_state_is_initializing(self):
        vo = _make_vo()
        assert vo.relocalization_state == RelocalizationState.INITIALIZING

    def test_transitions_to_tracking_on_first_good_frame(self):
        vo = _make_vo()
        rgb, depth = _make_textured_frame(0)
        result = vo.process_frame(rgb, depth)
        # After first frame with sufficient features, should be TRACKING
        assert result.relocalization_state in ("INITIALIZING", "TRACKING")

    def test_consecutive_lost_counter_increments(self):
        vo = _make_vo()
        # Initialize with textured frame
        rgb1, depth1 = _make_textured_frame(0)
        vo.process_frame(rgb1, depth1)

        # Feed blank frames to trigger tracking loss
        for i in range(5):
            rgb_blank, depth_blank = _make_blank_frame()
            result = vo.process_frame(rgb_blank, depth_blank)

        assert vo.consecutive_lost_frames >= 3

    def test_relocalization_attempt_after_n_losses(self):
        vo = _make_vo()
        rgb1, depth1 = _make_textured_frame(0)
        vo.process_frame(rgb1, depth1)

        for i in range(vo.RELOC_ATTEMPT_AFTER_N_LOST + 1):
            rgb_blank, depth_blank = _make_blank_frame()
            result = vo.process_frame(rgb_blank, depth_blank)

        assert result.relocalization_state in ("TRACKING_LOST", "RELOCALIZATION_ATTEMPT")

    def test_recovery_after_tracking_loss(self):
        vo = _make_vo()
        # Good frame
        rgb1, depth1 = _make_textured_frame(0)
        vo.process_frame(rgb1, depth1)

        # Feed a few good frames to establish tracking
        for i in range(3):
            rgb, depth = _make_textured_frame(i + 1)
            vo.process_frame(rgb, depth)

        # Force loss with blank frames
        for i in range(4):
            rgb_blank, depth_blank = _make_blank_frame()
            vo.process_frame(rgb_blank, depth_blank)

        assert vo.consecutive_lost_frames > 0

        # Recover with textured frames
        for i in range(5):
            rgb_rec, depth_rec = _make_textured_frame(100 + i)
            result = vo.process_frame(rgb_rec, depth_rec)

        # Should have recovered
        assert vo.consecutive_lost_frames == 0

    def test_reset_clears_relocalization_state(self):
        vo = _make_vo()
        rgb, depth = _make_textured_frame(0)
        vo.process_frame(rgb, depth)
        vo.reset()
        assert vo.relocalization_state == RelocalizationState.INITIALIZING
        assert vo.consecutive_lost_frames == 0
        assert vo.total_recoveries == 0


# ========================================================================
# Localization Logger Tests
# ========================================================================

class TestLocalizationLogger:

    def test_logger_creates_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = f.name

        try:
            logger = LocalizationLogger(log_path=log_path)
            vo = _make_vo()
            rgb, depth = _make_textured_frame(0)
            result = vo.process_frame(rgb, depth)
            logger.log_frame(result, frame_timestamp=0.1)
            logger.close()

            assert os.path.exists(log_path)
            with open(log_path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 1

            record = json.loads(lines[0])
            assert "timestamp" in record
            assert "tracking_status" in record
            assert "relocalization_state" in record
            assert "confidence" in record
            assert "inlier_count" in record
            assert "latency_ms" in record
            assert "consecutive_lost_frames" in record
        finally:
            os.unlink(log_path)

    def test_logger_frame_count(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = f.name

        try:
            logger = LocalizationLogger(log_path=log_path)
            vo = _make_vo()
            for i in range(5):
                rgb, depth = _make_textured_frame(i)
                result = vo.process_frame(rgb, depth)
                logger.log_frame(result, frame_timestamp=i * 0.1)

            assert logger.total_logged == 5
            logger.close()
        finally:
            os.unlink(log_path)


# ========================================================================
# Confidence and Invariant Tests
# ========================================================================

class TestConfidenceInvariants:

    def test_confidence_bounded_0_to_1(self):
        vo = _make_vo()
        for i in range(10):
            rgb, depth = _make_textured_frame(i)
            result = vo.process_frame(rgb, depth)
            assert 0.0 <= result.confidence <= 1.0

    def test_confidence_low_on_blank_frames(self):
        vo = _make_vo()
        rgb1, depth1 = _make_textured_frame(0)
        vo.process_frame(rgb1, depth1)

        rgb_blank, depth_blank = _make_blank_frame()
        result = vo.process_frame(rgb_blank, depth_blank)
        assert result.confidence < 0.5

    def test_vo_result_has_relocalization_fields(self):
        vo = _make_vo()
        rgb, depth = _make_textured_frame(0)
        result = vo.process_frame(rgb, depth)
        assert hasattr(result, "relocalization_state")
        assert hasattr(result, "consecutive_lost_frames")
        assert hasattr(result, "recovery_frame_count")


# ========================================================================
# Real Outdoor Scenario Tests
# ========================================================================

class TestRealOutdoorScenarios:

    def test_scenario_1_high_texture_sustained_tracking(self):
        """Scenario 1 (open path): Expect sustained TRACKING_OK across all frames."""
        loader = OutdoorDatasetLoader("datasets/processed/scenario_1_open_path")
        vo = _make_vo()
        lost_frames = 0
        for frame in loader.iter_frames():
            result = vo.process_frame(frame.rgb, frame.depth_m)
            if result.tracking_status == TrackingStatus.TRACKING_LOST:
                lost_frames += 1
        assert lost_frames == 0, f"Unexpected tracking loss in high-texture scenario: {lost_frames} frames"

    def test_scenario_5_visual_degradation_detects_loss(self):
        """Scenario 5 (visual degradation): VO must correctly detect tracking loss."""
        loader = OutdoorDatasetLoader("datasets/processed/scenario_5_visual_degradation")
        vo = _make_vo()
        lost_frames = 0
        for frame in loader.iter_frames():
            result = vo.process_frame(frame.rgb, frame.depth_m)
            if result.tracking_status == TrackingStatus.TRACKING_LOST:
                lost_frames += 1
        # Most frames should be lost in this degraded scenario
        assert lost_frames >= 10, f"Expected >=10 lost frames in degraded scenario, got {lost_frames}"

    def test_latency_acceptable_on_real_data(self):
        """VO must process at >20 FPS (< 50ms per frame) on real data."""
        loader = OutdoorDatasetLoader("datasets/processed/scenario_1_open_path")
        vo = _make_vo()
        latencies = []
        for frame in loader.iter_frames():
            result = vo.process_frame(frame.rgb, frame.depth_m)
            latencies.append(result.latency_ms)
        mean_lat = float(np.mean(latencies))
        assert mean_lat < 50.0, f"Mean latency {mean_lat:.1f}ms exceeds 50ms threshold"
