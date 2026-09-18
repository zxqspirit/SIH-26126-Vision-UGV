"""Structured JSON-lines logger for per-frame localization telemetry.

Captures every visual odometry frame result as a single JSON line for
post-hoc analysis, debugging, and integration testing.
Does NOT log to stdout to avoid polluting ROS 2 console output.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from ..interfaces.types import VisualOdometryResult


class LocalizationLogger:
    """Appends one JSON line per processed frame to a log file."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        if log_path is None:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join("logs", "localization")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"vo_log_{timestamp_str}.jsonl")

        self.log_path = log_path
        self._file = open(log_path, "a", encoding="utf-8")
        self._frame_count = 0

    def log_frame(self, result: VisualOdometryResult, frame_timestamp: float = 0.0) -> None:
        """Write a single JSON line for this frame."""
        record = {
            "frame_index": self._frame_count,
            "timestamp": round(frame_timestamp, 6),
            "x": result.x,
            "y": result.y,
            "z": result.z,
            "yaw": result.yaw,
            "delta_x": result.delta_x,
            "delta_y": result.delta_y,
            "delta_yaw": result.delta_yaw,
            "inlier_count": result.inlier_count,
            "tracking_status": result.tracking_status.value if hasattr(result.tracking_status, 'value') else str(result.tracking_status),
            "relocalization_state": result.relocalization_state,
            "confidence": round(result.confidence, 4),
            "latency_ms": round(result.latency_ms, 3),
            "consecutive_lost_frames": result.consecutive_lost_frames,
            "recovery_frame_count": result.recovery_frame_count,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        self._frame_count += 1

    def close(self) -> None:
        """Flush and close the log file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def total_logged(self) -> int:
        return self._frame_count
