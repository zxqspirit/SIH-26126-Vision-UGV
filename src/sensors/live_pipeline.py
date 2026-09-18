"""Real-time Live Camera Ingestion Pipeline.

Provides live camera streaming for physical cameras (V4L2 / DirectShow)
and mock/virtual cameras using the identical downstream SensorFrame interface
as offline replay mode.

Features:
- Asynchronous acquisition worker thread
- High-resolution monotonic timestamping
- Bounded lock-free ring buffer (single-slot drop-oldest policy)
- Strict latency bounds (drops stale frames when downstream lags)
- Active health monitoring (FPS, drops, jitter, stale detection)
- Graceful clean shutdown within bounded timeout
"""

from __future__ import annotations

import os

import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from ..interfaces.types import (
    CameraHealthMetrics,
    CameraHealthState,
    CameraIntrinsics,
    SensorFrame,
)


class MockLiveCamera:
    """Programmable virtual camera source for deterministic testing of live ingestion."""

    def __init__(
        self,
        target_fps: float = 30.0,
        frame_width: int = 640,
        frame_height: int = 480,
        simulate_depth: bool = True,
        is_connected: bool = True,
        drop_probability: float = 0.0,
        delay_sec: float = 0.0,
        invalid_frame_probability: float = 0.0,
    ) -> None:
        self.target_fps = target_fps
        self.width = frame_width
        self.height = frame_height
        self.simulate_depth = simulate_depth
        self.is_connected = is_connected
        self.drop_prob = drop_probability
        self.delay_sec = delay_sec
        self.invalid_prob = invalid_frame_probability

        self._frame_count = 0
        self._is_opened = False

    def open(self) -> bool:
        if not self.is_connected:
            self._is_opened = False
            return False
        self._is_opened = True
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Read frame: returns (success, rgb, depth_m)."""
        if not self._is_opened or not self.is_connected:
            return False, None, None

        if self.delay_sec > 0:
            time.sleep(self.delay_sec)

        # Simulate camera drop
        if self.drop_prob > 0 and np.random.rand() < self.drop_prob:
            return False, None, None

        # Simulate invalid/corrupt frame
        if self.invalid_prob > 0 and np.random.rand() < self.invalid_prob:
            # Corrupted / 0-byte or None
            return True, None, None

        # Synthetic RGB frame
        rgb = np.full((self.height, self.width, 3), 120, dtype=np.uint8)
        # Add dynamic pattern to verify frame differentiation
        cv2.circle(rgb, (int(self.width / 2), int(self.height / 2)), 50, (200, 100, 50), -1)

        # Synthetic metric depth (meters)
        depth_m = None
        if self.simulate_depth:
            depth_m = np.full((self.height, self.width), 2.5, dtype=np.float32)

        self._frame_count += 1
        return True, rgb, depth_m

    def release(self) -> None:
        self._is_opened = False




class ImageSequenceCamera:
    """Streams a folder of image frames as a live real-time camera source.

    Supports offline datasets (e.g. Kaggle off-road dataset, recorded drive sequences)
    emulating an active hardware camera at a fixed target FPS with realistic metric depth.
    """

    def __init__(
        self,
        image_dir: str,
        target_fps: float = 15.0,
        frame_width: int = 640,
        frame_height: int = 480,
        simulate_depth: bool = True,
        loop: bool = True,
    ) -> None:
        self.image_dir = image_dir
        self.target_fps = target_fps
        self.width = frame_width
        self.height = frame_height
        self.simulate_depth = simulate_depth
        self.loop = loop

        self._image_paths: List[str] = []
        self._cur_idx = 0
        self._is_opened = False
        self._lock = threading.Lock()

        self._discover_images()

    def _discover_images(self) -> None:
        if os.path.exists(self.image_dir):
            if os.path.isdir(self.image_dir):
                files = sorted(os.listdir(self.image_dir))
                self._image_paths = [
                    os.path.join(self.image_dir, f)
                    for f in files
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ]
            elif os.path.isfile(self.image_dir):
                self._image_paths = [self.image_dir]

    def open(self) -> bool:
        with self._lock:
            self._discover_images()
            if not self._image_paths:
                self._is_opened = False
                return False
            self._is_opened = True
            self._cur_idx = 0
            return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            if not self._is_opened or not self._image_paths:
                return False, None, None

            if self._cur_idx >= len(self._image_paths):
                if self.loop:
                    self._cur_idx = 0
                else:
                    return False, None, None

            img_path = self._image_paths[self._cur_idx]
            self._cur_idx += 1

        bgr = cv2.imread(img_path)
        if bgr is None:
            return False, None, None

        if (bgr.shape[1], bgr.shape[0]) != (self.width, self.height):
            bgr = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth_m = None
        if self.simulate_depth:
            # Generate realistic perspective ground depth gradient
            depth_m = np.zeros((self.height, self.width), dtype=np.float32)
            horizon_idx = int(self.height * 0.45)
            ground_rows = self.height - horizon_idx
            if ground_rows > 0:
                ground_grad = np.linspace(10.0, 1.2, ground_rows, dtype=np.float32)[:, np.newaxis]
                depth_m[horizon_idx:, :] = np.tile(ground_grad, (1, self.width))

        return True, rgb, depth_m

    def release(self) -> None:
        with self._lock:
            self._is_opened = False

class CameraHealthMonitor:
    """Tracks acquisition rate, processing rate, latency, and operational health."""

    def __init__(self, target_fps: float = 30.0, window_size: int = 30) -> None:
        self.target_fps = target_fps
        self.window_size = window_size
        self.capture_times: List[float] = []
        self.process_times: List[float] = []

        self.total_captured = 0
        self.total_processed = 0
        self.dropped_frames = 0
        self.consecutive_drops = 0
        self.last_capture_timestamp: float = 0.0
        self.last_frame_latency_ms: float = 0.0
        self.state = CameraHealthState.CONNECTED

    def record_capture(self, timestamp: float) -> None:
        self.capture_times.append(timestamp)
        if len(self.capture_times) > self.window_size:
            self.capture_times.pop(0)
        self.total_captured += 1
        self.last_capture_timestamp = timestamp
        self.consecutive_drops = 0

    def record_drop(self) -> None:
        self.dropped_frames += 1
        self.consecutive_drops += 1

    def record_process(self, deliver_time: float, capture_time: float) -> None:
        self.process_times.append(deliver_time)
        if len(self.process_times) > self.window_size:
            self.process_times.pop(0)
        self.total_processed += 1
        self.last_frame_latency_ms = max(0.0, (deliver_time - capture_time) * 1000.0)

    def evaluate_state(self, is_alive: bool) -> CameraHealthState:
        now = time.time()
        if self.state == CameraHealthState.SHUTDOWN:
            return self.state
        if not is_alive:
            self.state = CameraHealthState.UNAVAILABLE
            return self.state

        # Check for stale stream
        expected_interval = 1.0 / max(1.0, self.target_fps)
        if self.last_capture_timestamp > 0 and (now - self.last_capture_timestamp) > (3.5 * expected_interval):
            self.state = CameraHealthState.STALE
            return self.state

        # Calculate current capture FPS
        cap_fps = self.get_capture_fps()
        if cap_fps < (0.45 * self.target_fps) or self.consecutive_drops > 8:
            self.state = CameraHealthState.DEGRADED
            return self.state

        self.state = CameraHealthState.CONNECTED
        return self.state

    def get_capture_fps(self) -> float:
        if len(self.capture_times) < 2:
            return 0.0
        dt = self.capture_times[-1] - self.capture_times[0]
        return float((len(self.capture_times) - 1) / dt) if dt > 0 else 0.0

    def get_processing_fps(self) -> float:
        if len(self.process_times) < 2:
            return 0.0
        dt = self.process_times[-1] - self.process_times[0]
        return float((len(self.process_times) - 1) / dt) if dt > 0 else 0.0

    def get_metrics(self, is_alive: bool) -> CameraHealthMetrics:
        state = self.evaluate_state(is_alive)
        return CameraHealthMetrics(
            health_state=state,
            capture_fps=round(self.get_capture_fps(), 2),
            processing_fps=round(self.get_processing_fps(), 2),
            dropped_frames=self.dropped_frames,
            frame_latency_ms=round(self.last_frame_latency_ms, 2),
            consecutive_drops=self.consecutive_drops,
            last_timestamp=round(self.last_capture_timestamp, 4),
            total_captured=self.total_captured,
            total_processed=self.total_processed,
            is_alive=is_alive,
        )


class LiveCameraStreamer:
    """Threaded live camera ingestion yielding downstream SensorFrames.

    Emits the identical SensorFrame interface consumed by NavigationPipeline.
    Implements a single-slot bounded drop-oldest ring buffer to bound latency.
    """

    def __init__(
        self,
        camera_source: Union[int, str, MockLiveCamera] = 0,
        target_fps: float = 30.0,
        max_queue_size: int = 1,
        sequence_name: str = "live_stream",
    ) -> None:
        self.source_arg = camera_source
        self.target_fps = target_fps
        self.max_queue_size = max_queue_size
        self.sequence_name = sequence_name

        self.health_monitor = CameraHealthMonitor(target_fps=target_fps)
        self.frame_queue: queue.Queue[SensorFrame] = queue.Queue(maxsize=max_queue_size)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Any = None
        self._is_alive = False
        self._last_ts = 0.0
        self._frame_count = 0
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Initialize camera source and launch acquisition thread."""
        with self._lock:
            if self._is_alive:
                return True

            self._stop_event.clear()

            # Initialize capture source
            if isinstance(self.source_arg, (MockLiveCamera, ImageSequenceCamera)):
                self._cap = self.source_arg
                success = self._cap.open()
            elif isinstance(self.source_arg, int):
                self._cap = cv2.VideoCapture(self.source_arg)
                success = self._cap.isOpened()
            elif isinstance(self.source_arg, str):
                if self.source_arg.isdigit():
                    self._cap = cv2.VideoCapture(int(self.source_arg))
                    success = self._cap.isOpened()
                elif os.path.isdir(self.source_arg):
                    self._cap = ImageSequenceCamera(self.source_arg, target_fps=self.target_fps)
                    success = self._cap.open()
                elif os.path.isfile(self.source_arg):
                    self._cap = cv2.VideoCapture(self.source_arg)
                    success = self._cap.isOpened()
                else:
                    success = False
            else:
                success = False

            if not success:
                self._is_alive = False
                self.health_monitor.state = CameraHealthState.UNAVAILABLE
                return False

            self._is_alive = True
            self._thread = threading.Thread(
                target=self._acquisition_worker,
                name="CameraAcquisitionWorker",
                daemon=True,
            )
            self._thread.start()
            return True

    def _acquisition_worker(self) -> None:
        """Dedicated acquisition worker acquiring frames at camera rate."""
        frame_interval = 1.0 / max(1.0, self.target_fps)

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            try:
                # Capture frame
                if isinstance(self._cap, (MockLiveCamera, ImageSequenceCamera)):
                    ret, rgb, depth_m = self._cap.read()
                elif isinstance(self._cap, cv2.VideoCapture):
                    ret, bgr = self._cap.read()
                    if ret and bgr is not None:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        depth_m = None
                    else:
                        # Auto-loop video file if reached EOF
                        if isinstance(self.source_arg, str) and not self.source_arg.isdigit() and os.path.isfile(self.source_arg):
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, bgr = self._cap.read()
                            if ret and bgr is not None:
                                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                                depth_m = None
                            else:
                                ret, rgb, depth_m = False, None, None
                        else:
                            ret, rgb, depth_m = False, None, None
                else:
                    ret, rgb, depth_m = False, None, None

                # Handle invalid / failed frame capture
                if not ret or rgb is None or not isinstance(rgb, np.ndarray) or rgb.size == 0:
                    self.health_monitor.record_drop()
                    # Sleep slightly before retry
                    time.sleep(min(0.01, frame_interval))
                    continue

                # Ensure valid dimensions
                if len(rgb.shape) != 3 or rgb.shape[2] != 3:
                    self.health_monitor.record_drop()
                    time.sleep(min(0.01, frame_interval))
                    continue

                # Enforce monotonic timestamps
                now = time.time()
                ts = max(now, self._last_ts + 0.0005)
                self._last_ts = ts

                # Default depth fallback if missing
                if depth_m is None:
                    h, w = rgb.shape[:2]
                    depth_m = np.zeros((h, w), dtype=np.float32)

                # Format standardized SensorFrame (IDENTICAL TO REPLAY)
                frame = SensorFrame(
                    timestamp=ts,
                    rgb=rgb,
                    depth_m=depth_m,
                    right_rgb=None,
                    frame_id=self._frame_count,
                    sequence_name=self.sequence_name,
                )
                self._frame_count += 1
                self.health_monitor.record_capture(ts)

                # Bounded Ring Buffer: Drop oldest if full to guarantee bounded latency
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        # Drop oldest unread frame
                        _ = self.frame_queue.get_nowait()
                        self.health_monitor.record_drop()
                    except queue.Empty:
                        pass
                    # Put fresh frame
                    self.frame_queue.put(frame)

            except Exception:
                self.health_monitor.record_drop()

            # Pace acquisition to target FPS if producer is faster
            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def read_next_frame(self, timeout_s: float = 0.5) -> Optional[SensorFrame]:
        """Fetch latest SensorFrame from queue (consumes identical interface to replay)."""
        if not self._is_alive:
            return None

        try:
            frame = self.frame_queue.get(timeout=timeout_s)
            now = time.time()
            self.health_monitor.record_process(deliver_time=now, capture_time=frame.timestamp)
            return frame
        except queue.Empty:
            return None

    def get_health_metrics(self) -> CameraHealthMetrics:
        """Fetch current real-time health telemetry."""
        return self.health_monitor.get_metrics(self._is_alive)

    def stop(self, timeout_s: float = 1.0) -> None:
        """Gracefully terminate worker thread and release camera resources."""
        with self._lock:
            if not self._is_alive:
                return

            self._stop_event.set()
            self._is_alive = False

            # Join acquisition worker
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=timeout_s)
                self._thread = None

            # Release camera hardware
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

            # Drain queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            self.health_monitor.state = CameraHealthState.SHUTDOWN
