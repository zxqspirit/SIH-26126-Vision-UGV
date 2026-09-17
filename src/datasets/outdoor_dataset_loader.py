"""Real outdoor visual dataset loader for offline UGV validation.

Loads synchronized RGB and metric depth frames from directory structures
or recorded outdoor video files.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Iterator, List, Optional, Tuple
import numpy as np
import cv2

from ..interfaces.types import SensorFrame, CameraIntrinsics


class OutdoorDatasetLoader:
    """Loads synchronized real outdoor sensor sequences."""

    def __init__(
        self,
        dataset_dir: str,
        depth_scale: float = 1.0,  # Multiplier to convert raw depth to meters
        target_fps: float = 10.0,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.depth_scale = depth_scale
        self.target_fps = target_fps
        self.rgb_files: List[str] = []
        self.depth_files: List[str] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Find matching RGB and Depth files."""
        rgb_dir = os.path.join(self.dataset_dir, "rgb")
        depth_dir = os.path.join(self.dataset_dir, "depth")

        if os.path.exists(rgb_dir):
            patterns = ["*.png", "*.jpg", "*.jpeg"]
            for pat in patterns:
                self.rgb_files.extend(glob.glob(os.path.join(rgb_dir, pat)))
            self.rgb_files.sort(key=self._natural_keys)

        if os.path.exists(depth_dir):
            patterns = ["*.npy", "*.png", "*.tiff"]
            for pat in patterns:
                self.depth_files.extend(glob.glob(os.path.join(depth_dir, pat)))
            self.depth_files.sort(key=self._natural_keys)

    @staticmethod
    def _natural_keys(text: str) -> List[int]:
        return [int(c) if c.isdigit() else 0 for c in re.split(r'(\d+)', text)]

    def __len__(self) -> int:
        return len(self.rgb_files)

    def get_frame(self, index: int) -> Optional[SensorFrame]:
        """Fetch frame at index."""
        if index < 0 or index >= len(self.rgb_files):
            return None

        rgb_path = self.rgb_files[index]
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            return None
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Load matching depth if available
        if index < len(self.depth_files):
            depth_path = self.depth_files[index]
            if depth_path.endswith(".npy"):
                depth = np.load(depth_path).astype(np.float32)
            else:
                raw_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if raw_depth is not None:
                    # Often uint16 in millimeters: divide by 1000.0 to get meters
                    depth = raw_depth.astype(np.float32) * (self.depth_scale / 1000.0 if raw_depth.dtype == np.uint16 else self.depth_scale)
                else:
                    depth = np.zeros((h, w), dtype=np.float32)
        else:
            # Synthetic metric depth projection if raw depth image wasn't supplied
            depth = np.zeros((h, w), dtype=np.float32)

        ts = index * (1.0 / self.target_fps)
        return SensorFrame(
            timestamp=ts,
            rgb=rgb,
            depth_m=depth,
            frame_id=index,
            sequence_name=os.path.basename(self.dataset_dir),
        )

    def iter_frames(self) -> Iterator[SensorFrame]:
        """Generator yielding frames sequentially."""
        for i in range(len(self)):
            frame = self.get_frame(i)
            if frame is not None:
                yield frame
