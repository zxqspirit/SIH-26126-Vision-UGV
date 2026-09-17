"""Visual odometry tracking health diagnostics and confidence assessment."""

from __future__ import annotations

from typing import Tuple
import numpy as np

from ..interfaces.types import TrackingStatus


class TrackingDiagnostics:
    """Evaluates feature tracking health and computes localization confidence."""

    def __init__(
        self,
        nominal_inliers: int = 40,
        degraded_inliers: int = 18,
        lost_inliers: int = 8,
    ) -> None:
        self.nominal_inliers = nominal_inliers
        self.degraded_inliers = degraded_inliers
        self.lost_inliers = lost_inliers

    def evaluate(self, inlier_count: int, total_tracked: int) -> Tuple[TrackingStatus, float]:
        """Evaluate tracking status and visual odometry confidence score.

        Returns:
            status: TrackingStatus enum
            confidence: float in [0.0..1.0]
        """
        if inlier_count >= self.nominal_inliers:
            status = TrackingStatus.TRACKING_OK
            # Inlier ratio
            ratio = float(inlier_count / max(total_tracked, 1))
            confidence = float(np.clip(0.70 + 0.30 * min(ratio, 1.0), 0.70, 0.98))
        elif inlier_count >= self.degraded_inliers:
            status = TrackingStatus.TRACKING_DEGRADED
            frac = (inlier_count - self.degraded_inliers) / (self.nominal_inliers - self.degraded_inliers)
            confidence = float(np.clip(0.35 + 0.30 * frac, 0.35, 0.65))
        else:
            status = TrackingStatus.TRACKING_LOST
            confidence = float(np.clip(inlier_count / max(self.lost_inliers, 1) * 0.25, 0.05, 0.25))

        return status, confidence
