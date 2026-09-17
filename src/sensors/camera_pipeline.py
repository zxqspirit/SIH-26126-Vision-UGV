"""Real sensor data pipeline and common camera input abstraction.

Supports:
1. Recorded RGB Video (.mp4, .avi, .mkv)
2. Image Sequences (.png, .jpg)
3. Live RGB Camera (V4L2 / DirectShow device index)
4. Real Stereo / RGB-D (RGB + Float32/16UC1 Depth)

Strictly adheres to ROS coordinate standards (REP-103/104) and enforces
timestamp monotonicity, FPS pacing, and camera metadata formatting.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
import glob
import os
import re
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..interfaces.types import CameraIntrinsics


class CameraSourceType(str, Enum):
    """Enumeration of supported camera ingestion modalities."""
    VIDEO_FILE = "VIDEO_FILE"
    IMAGE_SEQUENCE = "IMAGE_SEQUENCE"
    LIVE_CAMERA = "LIVE_CAMERA"
    STEREO_RGBD = "STEREO_RGBD"


@dataclass
class SensorPacket:
    """Strongly-typed standardized camera frame packet across all modalities."""
    frame_index: int
    timestamp: float  # Fractional epoch seconds
    rgb: np.ndarray  # Shape (H, W, 3), uint8 in RGB color space
    depth: Optional[np.ndarray] = None  # Shape (H, W), float32 in meters
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    frame_id: str = "camera_optical_frame"
    is_live: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]


@dataclass
class BenchmarkMetrics:
    """Performance metrics captured during camera source ingestion testing."""
    total_frames_read: int
    total_elapsed_sec: float
    input_fps: float
    processing_fps: float
    dropped_frames: int
    mean_dt_sec: float
    min_dt_sec: float
    max_dt_sec: float
    jitter_ms: float
    monotonic_violations: int

    def summary(self) -> str:
        return (
            f"Frames Read: {self.total_frames_read} | "
            f"Input FPS: {self.input_fps:.2f} | "
            f"Processing FPS: {self.processing_fps:.2f} | "
            f"Dropped Frames: {self.dropped_frames} | "
            f"Mean dt: {self.mean_dt_sec * 1000.0:.2f} ms | "
            f"Jitter: {self.jitter_ms:.2f} ms | "
            f"Monotonic Violations: {self.monotonic_violations}"
        )


class BaseCameraSource(abc.ABC):
    """Abstract Base Class for all physical and recorded camera streams."""

    def __init__(
        self,
        target_fps: float = 10.0,
        frame_id: str = "camera_optical_frame",
        intrinsics: Optional[CameraIntrinsics] = None,
        loop: bool = False,
    ) -> None:
        self.target_fps = target_fps
        self.frame_id = frame_id
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.loop = loop

        self._frame_count = 0
        self._is_opened = False
        self._last_timestamp: float = 0.0
        self._dropped_frames: int = 0

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @abc.abstractmethod
    def open(self) -> bool:
        """Initializes hardware device or opens media file."""
        pass

    @abc.abstractmethod
    def read(self) -> Optional[SensorPacket]:
        """Reads the next synchronized sensor packet."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Releases hardware resources or file handles."""
        pass

    def __enter__(self) -> "BaseCameraSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def pace(self, cycle_start_time: float) -> None:
        """Paces reading to match target_fps if streaming from recorded storage."""
        if self.target_fps <= 0:
            return
        target_period = 1.0 / self.target_fps
        elapsed = time.time() - cycle_start_time
        sleep_needed = target_period - elapsed
        if sleep_needed > 0.001:
            time.sleep(sleep_needed)
        elif sleep_needed < -target_period:
            # More than one full frame behind schedule
            self._dropped_frames += 1


class VideoFileSource(BaseCameraSource):
    """Ingests recorded outdoor RGB video files (.mp4, .avi, .mkv)."""

    def __init__(
        self,
        video_path: str,
        target_fps: Optional[float] = None,
        frame_id: str = "camera_optical_frame",
        loop: bool = False,
    ) -> None:
        super().__init__(target_fps=target_fps or 10.0, frame_id=frame_id, loop=loop)
        self.video_path = video_path
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if not os.path.isfile(self.video_path):
            return False
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            return False
        native_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if native_fps and native_fps > 0 and self.target_fps == 10.0:
            self.target_fps = float(native_fps)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            self.intrinsics.width = w
            self.intrinsics.height = h
            self.intrinsics.cx = w / 2.0
            self.intrinsics.cy = h / 2.0
        self._is_opened = True
        return True

    def read(self) -> Optional[SensorPacket]:
        if not self._is_opened or self._cap is None:
            return None

        cycle_start = time.time()
        ret, bgr = self._cap.read()

        if not ret or bgr is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, bgr = self._cap.read()
                if not ret or bgr is None:
                    return None
            else:
                return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        now = time.time()
        ts = max(now, self._last_timestamp + 0.001)
        self._last_timestamp = ts

        packet = SensorPacket(
            frame_index=self._frame_count,
            timestamp=ts,
            rgb=rgb,
            depth=None,
            intrinsics=self.intrinsics,
            frame_id=self.frame_id,
            is_live=False,
            metadata={"source": "video_file", "path": self.video_path},
        )
        self._frame_count += 1
        self.pace(cycle_start)
        return packet

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False


class ImageSequenceSource(BaseCameraSource):
    """Ingests sorted outdoor image sequences (.png, .jpg) from a directory."""

    def __init__(
        self,
        sequence_dir: str,
        target_fps: float = 10.0,
        frame_id: str = "camera_optical_frame",
        loop: bool = False,
    ) -> None:
        super().__init__(target_fps=target_fps, frame_id=frame_id, loop=loop)
        self.sequence_dir = sequence_dir
        self.image_files: List[str] = []
        self._current_idx = 0

    def open(self) -> bool:
        if not os.path.isdir(self.sequence_dir):
            return False
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        self.image_files = []
        for pat in patterns:
            self.image_files.extend(glob.glob(os.path.join(self.sequence_dir, pat)))
        self.image_files.sort(key=lambda t: [int(c) if c.isdigit() else 0 for c in re.split(r'(\d+)', t)])
        if not self.image_files:
            return False
        self._current_idx = 0
        self._is_opened = True
        return True

    def read(self) -> Optional[SensorPacket]:
        if not self._is_opened or not self.image_files:
            return None

        if self._current_idx >= len(self.image_files):
            if self.loop:
                self._current_idx = 0
            else:
                return None

        cycle_start = time.time()
        img_path = self.image_files[self._current_idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.intrinsics.width = w
        self.intrinsics.height = h
        self.intrinsics.cx = w / 2.0
        self.intrinsics.cy = h / 2.0

        now = time.time()
        ts = max(now, self._last_timestamp + 0.001)
        self._last_timestamp = ts

        packet = SensorPacket(
            frame_index=self._frame_count,
            timestamp=ts,
            rgb=rgb,
            depth=None,
            intrinsics=self.intrinsics,
            frame_id=self.frame_id,
            is_live=False,
            metadata={"source": "image_sequence", "path": img_path},
        )
        self._current_idx += 1
        self._frame_count += 1
        self.pace(cycle_start)
        return packet

    def close(self) -> None:
        self.image_files.clear()
        self._is_opened = False


class LiveCameraSource(BaseCameraSource):
    """Captures live camera feeds from hardware device index (USB / V4L2)."""

    def __init__(
        self,
        device_index: int = 0,
        target_fps: float = 30.0,
        width: int = 640,
        height: int = 480,
        frame_id: str = "camera_optical_frame",
    ) -> None:
        super().__init__(target_fps=target_fps, frame_id=frame_id, loop=False)
        self.device_index = device_index
        self.requested_width = width
        self.requested_height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.intrinsics.width = w
        self.intrinsics.height = h
        self.intrinsics.cx = w / 2.0
        self.intrinsics.cy = h / 2.0
        self._is_opened = True
        return True

    def read(self) -> Optional[SensorPacket]:
        if not self._is_opened or self._cap is None:
            return None

        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        now = time.time()
        ts = max(now, self._last_timestamp + 0.0001)
        self._last_timestamp = ts

        packet = SensorPacket(
            frame_index=self._frame_count,
            timestamp=ts,
            rgb=rgb,
            depth=None,
            intrinsics=self.intrinsics,
            frame_id=self.frame_id,
            is_live=True,
            metadata={"source": "live_camera", "device_index": self.device_index},
        )
        self._frame_count += 1
        return packet

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False


class StereoRGBDSource(BaseCameraSource):
    """Ingests paired RGB and metric depth frames from directory structures."""

    def __init__(
        self,
        dataset_dir: str,
        target_fps: float = 10.0,
        depth_scale: float = 1.0,
        frame_id: str = "camera_optical_frame",
        loop: bool = False,
    ) -> None:
        super().__init__(target_fps=target_fps, frame_id=frame_id, loop=loop)
        self.dataset_dir = dataset_dir
        self.depth_scale = depth_scale
        self.rgb_files: List[str] = []
        self.depth_files: List[str] = []
        self._current_idx = 0

    def open(self) -> bool:
        rgb_dir = os.path.join(self.dataset_dir, "rgb")
        depth_dir = os.path.join(self.dataset_dir, "depth")
        if not os.path.isdir(rgb_dir):
            return False

        patterns_img = ["*.png", "*.jpg", "*.jpeg"]
        self.rgb_files = []
        for pat in patterns_img:
            self.rgb_files.extend(glob.glob(os.path.join(rgb_dir, pat)))
        self.rgb_files.sort(key=lambda t: [int(c) if c.isdigit() else 0 for c in re.split(r'(\d+)', t)])

        self.depth_files = []
        if os.path.isdir(depth_dir):
            patterns_depth = ["*.npy", "*.png", "*.tiff"]
            for pat in patterns_depth:
                self.depth_files.extend(glob.glob(os.path.join(depth_dir, pat)))
            self.depth_files.sort(key=lambda t: [int(c) if c.isdigit() else 0 for c in re.split(r'(\d+)', t)])

        if not self.rgb_files:
            return False
        self._current_idx = 0
        self._is_opened = True
        return True

    def read(self) -> Optional[SensorPacket]:
        if not self._is_opened or not self.rgb_files:
            return None

        if self._current_idx >= len(self.rgb_files):
            if self.loop:
                self._current_idx = 0
            else:
                return None

        cycle_start = time.time()
        rgb_path = self.rgb_files[self._current_idx]
        bgr = cv2.imread(rgb_path)
        if bgr is None:
            return None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.intrinsics.width = w
        self.intrinsics.height = h
        self.intrinsics.cx = w / 2.0
        self.intrinsics.cy = h / 2.0

        # Load matching depth if available
        depth: Optional[np.ndarray] = None
        if self._current_idx < len(self.depth_files):
            depth_path = self.depth_files[self._current_idx]
            if depth_path.endswith(".npy"):
                raw_depth = np.load(depth_path).astype(np.float32)
                depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if raw is not None:
                    if raw.dtype == np.uint16:
                        depth = raw.astype(np.float32) / 1000.0  # mm to meters
                    else:
                        depth = raw.astype(np.float32) * self.depth_scale

        if depth is None:
            depth = np.zeros((h, w), dtype=np.float32)

        now = time.time()
        ts = max(now, self._last_timestamp + 0.001)
        self._last_timestamp = ts

        packet = SensorPacket(
            frame_index=self._frame_count,
            timestamp=ts,
            rgb=rgb,
            depth=depth,
            intrinsics=self.intrinsics,
            frame_id=self.frame_id,
            is_live=False,
            metadata={"source": "stereo_rgbd", "dataset": self.dataset_dir},
        )
        self._current_idx += 1
        self._frame_count += 1
        self.pace(cycle_start)
        return packet

    def close(self) -> None:
        self.rgb_files.clear()
        self.depth_files.clear()
        self._is_opened = False


def create_camera_source(
    uri: Union[str, int],
    target_fps: float = 10.0,
    loop: bool = False,
    frame_id: str = "camera_optical_frame",
) -> BaseCameraSource:
    """Factory creating the appropriate camera source from URI or path."""
    if isinstance(uri, int) or (isinstance(uri, str) and uri.isdigit()):
        return LiveCameraSource(device_index=int(uri), target_fps=target_fps, frame_id=frame_id)

    uri_str = str(uri)
    if os.path.isfile(uri_str) and uri_str.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
        return VideoFileSource(video_path=uri_str, target_fps=target_fps, loop=loop, frame_id=frame_id)

    if os.path.isdir(uri_str):
        if os.path.isdir(os.path.join(uri_str, "rgb")):
            return StereoRGBDSource(dataset_dir=uri_str, target_fps=target_fps, loop=loop, frame_id=frame_id)
        return ImageSequenceSource(sequence_dir=uri_str, target_fps=target_fps, loop=loop, frame_id=frame_id)

    raise ValueError(f"Unable to resolve camera source for URI: {uri}")


def benchmark_source(
    source: BaseCameraSource,
    max_frames: int = 30,
) -> BenchmarkMetrics:
    """Runs throughput and timestamp benchmark on a camera source."""
    if not source.is_opened:
        if not source.open():
            raise RuntimeError("Failed to open camera source for benchmarking.")

    timestamps: List[float] = []
    read_times: List[float] = []
    start_wall = time.perf_counter()

    for _ in range(max_frames):
        t0 = time.perf_counter()
        packet = source.read()
        t1 = time.perf_counter()
        if packet is None:
            break
        timestamps.append(packet.timestamp)
        read_times.append(t1 - t0)

    total_wall_elapsed = time.perf_counter() - start_wall
    n_frames = len(timestamps)

    if n_frames < 2:
        return BenchmarkMetrics(
            total_frames_read=n_frames,
            total_elapsed_sec=total_wall_elapsed,
            input_fps=0.0,
            processing_fps=0.0,
            dropped_frames=source.dropped_frames,
            mean_dt_sec=0.0,
            min_dt_sec=0.0,
            max_dt_sec=0.0,
            jitter_ms=0.0,
            monotonic_violations=0,
        )

    # Calculate inter-frame deltas
    dts = [timestamps[i] - timestamps[i - 1] for i in range(1, n_frames)]
    mean_dt = float(np.mean(dts))
    min_dt = float(np.min(dts))
    max_dt = float(np.max(dts))
    jitter_ms = float(np.std(dts) * 1000.0)

    monotonic_violations = sum(1 for dt in dts if dt <= 0)
    input_fps = float(n_frames / total_wall_elapsed) if total_wall_elapsed > 0 else 0.0
    mean_read_time = float(np.mean(read_times))
    processing_fps = float(1.0 / mean_read_time) if mean_read_time > 0 else 0.0

    return BenchmarkMetrics(
        total_frames_read=n_frames,
        total_elapsed_sec=total_wall_elapsed,
        input_fps=input_fps,
        processing_fps=processing_fps,
        dropped_frames=source.dropped_frames,
        mean_dt_sec=mean_dt,
        min_dt_sec=min_dt,
        max_dt_sec=max_dt,
        jitter_ms=jitter_ms,
        monotonic_violations=monotonic_violations,
    )
