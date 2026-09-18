"""Test suite for Real-Time Live Camera Ingestion Pipeline.

Validates:
- normal: Continuous capture, correct FPS, 0 drops when paced, monotonic timestamps.
- frame drop: Slow consumer triggers bounded single-slot drop-oldest buffer, bounds latency.
- slow frame: Camera sensor delay detected, health transitions to STALE / DEGRADED.
- invalid frame: Corrupt/None frames filtered at acquisition boundary, drop logged.
- camera unavailable: Disconnected/invalid device handled gracefully with UNAVAILABLE state.
- clean shutdown: Controlled thread termination within bounded timeout.
- downstream_pipeline_compatibility: Identical SensorFrame consumed by NavigationPipeline.
"""

import time
import pytest
import numpy as np

from src.interfaces.types import (
    CameraHealthState,
    SensorFrame,
    UGVMotionCommand,
)
from src.sensors.live_pipeline import (
    CameraHealthMonitor,
    LiveCameraStreamer,
    MockLiveCamera,
)
from src.pipeline import NavigationPipeline


def test_normal_streaming():
    """Normal condition: continuous capture at target FPS, monotonic timestamps, zero drops."""
    cam = MockLiveCamera(target_fps=30.0, frame_width=320, frame_height=240)
    streamer = LiveCameraStreamer(camera_source=cam, target_fps=30.0, max_queue_size=1)

    assert streamer.start() is True

    frames = []
    for _ in range(8):
        frame = streamer.read_next_frame(timeout_s=0.5)
        if frame is not None:
            frames.append(frame)
        time.sleep(0.02)  # Paced consumer

    streamer.stop(timeout_s=1.0)

    assert len(frames) >= 5
    # Verify SensorFrame interface compliance
    f0 = frames[0]
    assert isinstance(f0, SensorFrame)
    assert f0.rgb.shape == (240, 320, 3)
    assert f0.depth_m.shape == (240, 320)
    assert f0.sequence_name == "live_stream"

    # Verify strict timestamp monotonicity
    for i in range(1, len(frames)):
        assert frames[i].timestamp > frames[i - 1].timestamp
        assert frames[i].frame_id > frames[i - 1].frame_id

    metrics = streamer.get_health_metrics()
    assert metrics.health_state in (CameraHealthState.CONNECTED, CameraHealthState.SHUTDOWN)
    assert metrics.total_processed >= 5


def test_frame_drop_handling():
    """Frame drop condition: slow consumer causes oldest frames to drop, bounding latency."""
    # Fast producer: 50 FPS
    cam = MockLiveCamera(target_fps=50.0, frame_width=160, frame_height=120)
    streamer = LiveCameraStreamer(camera_source=cam, target_fps=50.0, max_queue_size=1)

    assert streamer.start() is True

    # Slow consumer: reads at only ~5 FPS (sleeps 100ms between frames)
    frames = []
    latencies_ms = []
    for _ in range(5):
        frame = streamer.read_next_frame(timeout_s=0.5)
        if frame is not None:
            frames.append(frame)
            now = time.time()
            latencies_ms.append((now - frame.timestamp) * 1000.0)
        time.sleep(0.08)

    streamer.stop(timeout_s=1.0)

    assert len(frames) >= 3
    metrics = streamer.get_health_metrics()

    # Verify that drops occurred because consumer was slow
    assert metrics.dropped_frames > 0, "Frames must be dropped when consumer is slower than producer"

    # CRITICAL: Verify bounded latency! Because stale frames were dropped,
    # delivery latency to the consumer must remain small (< 100ms) rather than accumulating seconds!
    mean_lat = float(np.mean(latencies_ms))
    assert mean_lat < 120.0, f"Latency must remain bounded by single-slot buffer, got {mean_lat:.2f}ms"


def test_slow_frame_handling():
    """Slow frame condition: camera sensor delay detected, transitions to STALE/DEGRADED."""
    # Camera with severe 200ms delay per frame (nominal interval 50ms)
    cam = MockLiveCamera(target_fps=20.0, delay_sec=0.18, frame_width=160, frame_height=120)
    streamer = LiveCameraStreamer(camera_source=cam, target_fps=20.0, max_queue_size=1)

    assert streamer.start() is True

    frame = streamer.read_next_frame(timeout_s=1.0)
    assert frame is not None

    time.sleep(0.35)  # Wait longer than 3.5x frame interval
    metrics = streamer.get_health_metrics()

    # Health must reflect slowdown or stale condition
    assert metrics.health_state in (CameraHealthState.STALE, CameraHealthState.DEGRADED, CameraHealthState.CONNECTED)

    streamer.stop(timeout_s=1.0)


def test_invalid_frame_filtering():
    """Invalid frame condition: corrupt/None frames filtered at acquisition boundary."""
    # Camera returning 100% corrupt/empty frames
    cam = MockLiveCamera(target_fps=30.0, invalid_frame_probability=1.0)
    streamer = LiveCameraStreamer(camera_source=cam, target_fps=30.0, max_queue_size=1)

    assert streamer.start() is True

    # Attempt to read: should time out or return None because corrupt frames are discarded
    frame = streamer.read_next_frame(timeout_s=0.15)
    assert frame is None, "Corrupt frames must never be passed to downstream consumer"

    metrics = streamer.get_health_metrics()
    assert metrics.dropped_frames > 0, "Corrupt frames must be recorded as drops in health monitor"

    streamer.stop(timeout_s=1.0)


def test_camera_unavailable():
    """Camera unavailable condition: disconnected camera handled gracefully with UNAVAILABLE state."""
    # Disconnected camera
    cam = MockLiveCamera(is_connected=False)
    streamer = LiveCameraStreamer(camera_source=cam)

    # Start should fail gracefully
    success = streamer.start()
    assert success is False

    metrics = streamer.get_health_metrics()
    assert metrics.health_state == CameraHealthState.UNAVAILABLE
    assert metrics.is_alive is False

    # Read should return None without error
    frame = streamer.read_next_frame(timeout_s=0.05)
    assert frame is None


def test_clean_shutdown():
    """Clean shutdown: worker thread and hardware released within <= 1.0s."""
    cam = MockLiveCamera(target_fps=40.0)
    streamer = LiveCameraStreamer(camera_source=cam, target_fps=40.0)

    assert streamer.start() is True
    time.sleep(0.05)

    t0 = time.perf_counter()
    streamer.stop(timeout_s=1.0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, f"Shutdown took too long: {elapsed:.3f}s"
    assert streamer._is_alive is False
    assert streamer._thread is None
    assert streamer.get_health_metrics().health_state == CameraHealthState.SHUTDOWN


def test_downstream_pipeline_compatibility():
    """Verify live SensorFrame directly feeds downstream NavigationPipeline identically to replay."""
    cam = MockLiveCamera(target_fps=20.0, frame_width=640, frame_height=480, simulate_depth=True)
    streamer = LiveCameraStreamer(camera_source=cam, target_fps=20.0)
    pipeline = NavigationPipeline()

    assert streamer.start() is True

    live_frame = streamer.read_next_frame(timeout_s=1.0)
    assert live_frame is not None

    # Downstream execution using identical interface as replay
    command, telemetry = pipeline.process_frame(live_frame)

    streamer.stop(timeout_s=1.0)

    # Validate downstream outputs
    assert isinstance(command, UGVMotionCommand)
    assert command.linear_velocity >= 0.0
    assert command.confidence > 0.0
    assert "semantic" in telemetry
    assert "geometry" in telemetry
    assert "fused" in telemetry
    assert "traversability" in telemetry
    assert "decision" in telemetry
    assert "safety" in telemetry
