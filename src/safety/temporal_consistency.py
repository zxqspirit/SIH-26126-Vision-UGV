"""Temporal Consistency Tracker for Confidence-to-Behavior Safety.

Tracks multi-frame perception stability:
- Exponential moving average (EMA) smoothing
- Rolling window variance for perception flicker detection
- Sudden confidence drop detector (delta > 0.30 in single cycle)
- Re-observation coherence counter for conservative recovery
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional
import numpy as np


class TemporalConsistencyTracker:
    """Monitors temporal stability and jitter across successive perception cycles."""

    def __init__(
        self,
        window_size: int = 5,
        ema_alpha: float = 0.40,
        sudden_drop_threshold: float = 0.30,
        high_variance_threshold: float = 0.04,
        reobserve_frames_required: int = 3,
    ) -> None:
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.drop_thresh = sudden_drop_threshold
        self.var_thresh = high_variance_threshold
        self.reobserve_required = reobserve_frames_required

        self.conf_history: Deque[float] = deque(maxlen=window_size)
        self.disagree_history: Deque[float] = deque(maxlen=window_size)
        self.timestamps: Deque[float] = deque(maxlen=window_size)

        self.ema_confidence: Optional[float] = None
        self.reobserve_counter: int = 0
        self.is_reobserve_holding: bool = False
        self.last_drop_magnitude: float = 0.0

    def reset(self) -> None:
        """Reset history and internal state."""
        self.conf_history.clear()
        self.disagree_history.clear()
        self.timestamps.clear()
        self.ema_confidence = None
        self.reobserve_counter = 0
        self.is_reobserve_holding = False
        self.last_drop_magnitude = 0.0

    def update(
        self,
        timestamp: float,
        raw_confidence: float,
        disagreement_ratio: float = 0.0,
    ) -> Dict[str, float]:
        """Update temporal history with current cycle values.

        Returns:
            Dictionary with computed temporal metrics.
        """
        c = float(np.clip(raw_confidence, 0.0, 1.0))
        d = float(np.clip(disagreement_ratio, 0.0, 1.0))

        # Check single-cycle drop
        if len(self.conf_history) > 0:
            prev_c = self.conf_history[-1]
            drop = prev_c - c
            self.last_drop_magnitude = max(0.0, drop)
        else:
            self.last_drop_magnitude = 0.0

        # Update histories
        self.conf_history.append(c)
        self.disagree_history.append(d)
        self.timestamps.append(timestamp)

        # Update EMA
        if self.ema_confidence is None:
            self.ema_confidence = c
        else:
            self.ema_confidence = self.ema_alpha * c + (1.0 - self.ema_alpha) * self.ema_confidence

        # Compute rolling variance
        if len(self.conf_history) >= 3:
            conf_var = float(np.var(list(self.conf_history)))
        else:
            conf_var = 0.0

        # Calculate temporal consistency factor in [0.0, 1.0]
        # Starts at 1.0, penalized by high variance and sudden drops
        temporal_score = 1.0

        # Penalty for sudden drop
        if self.last_drop_magnitude >= self.drop_thresh:
            temporal_score -= 0.35
            # Trigger re-observation hold
            self.is_reobserve_holding = True
            self.reobserve_counter = 0

        # Penalty for jitter / variance
        if conf_var > self.var_thresh:
            jitter_penalty = min(0.30, (conf_var - self.var_thresh) * 5.0)
            temporal_score -= jitter_penalty

        temporal_score = float(np.clip(temporal_score, 0.10, 1.0))

        # Manage re-observation coherence
        if self.is_reobserve_holding:
            if c >= 0.45 and self.last_drop_magnitude < self.drop_thresh and conf_var <= self.var_thresh:
                self.reobserve_counter += 1
                if self.reobserve_counter >= self.reobserve_required:
                    self.is_reobserve_holding = False
            else:
                self.reobserve_counter = 0

        return {
            "ema_confidence": round(self.ema_confidence, 4),
            "rolling_variance": round(conf_var, 5),
            "sudden_drop": round(self.last_drop_magnitude, 4),
            "temporal_consistency": round(temporal_score, 4),
            "reobserve_active": self.is_reobserve_holding,
            "reobserve_counter": self.reobserve_counter,
        }
