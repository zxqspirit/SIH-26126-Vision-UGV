"""Deterministic Confidence Safety Gate for outdoor UGV.

Enforces:
- Rule 9: Safety behavior must be deterministic and inspectable.
- Rule 10: Never let an LLM directly control a physical robot.
- Rule 13: Confidence must produce an actual behavior change.

Inspects:
1. CNN confidence
2. Depth quality
3. Semantic/geometric disagreement
4. Visual localization quality
5. Temporal consistency

States:
- HIGH: Normal recommendation
- MEDIUM: Lower speed / larger margin
- LOW: Conservative recommendation / re-observe
- CRITICAL: Safe-stop recommendation

Every decision logs:
- timestamp
- confidence
- reason
- action
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np

from ..interfaces.types import (
    SafetyAction,
    SafetyDecisionLog,
    SafetyResult,
    SafetyState,
    TrackingStatus,
)
from .safety_logger import SafetyDecisionLogger
from .temporal_consistency import TemporalConsistencyTracker


class SafetyGate:
    """Deterministic arbiter regulating motion commands based on multi-source confidence."""

    def __init__(
        self,
        high_conf_thresh: float = 0.75,
        medium_conf_thresh: float = 0.45,
        low_conf_thresh: float = 0.25,
        disagreement_thresh_medium: float = 0.35,
        disagreement_thresh_critical: float = 0.60,
        emergency_stopping_dist_m: float = 0.35,
        max_cautious_velocity: float = 0.35,  # m/s
        max_crawl_velocity: float = 0.15,  # m/s
        inscribed_radius_m: float = 0.35,  # m
        log_file_path: Optional[str] = None,
    ) -> None:
        self.th_high = high_conf_thresh
        self.th_med = medium_conf_thresh
        self.th_low = low_conf_thresh
        self.th_disagree_med = disagreement_thresh_medium
        self.th_disagree_crit = disagreement_thresh_critical
        self.emergency_dist = emergency_stopping_dist_m
        self.v_cautious_max = max_cautious_velocity
        self.v_crawl_max = max_crawl_velocity
        self.inscribed_radius = inscribed_radius_m

        self.temporal_tracker = TemporalConsistencyTracker()
        self.logger = SafetyDecisionLogger(log_file_path=log_file_path)

    def arbitrate(
        self,
        nominal_v: float,
        nominal_w: float,
        c_perc: float,
        c_geom: float,
        c_vo: float,
        c_fusion: float,
        disagreement_ratio: float = 0.0,
        tracking_status: TrackingStatus = TrackingStatus.TRACKING_OK,
        min_obstacle_dist_m: float = 5.0,
        timestamp: Optional[float] = None,
    ) -> SafetyResult:
        """Arbitrate nominal commands against multi-source confidence and clearance.

        Inspects:
        - CNN confidence (c_perc)
        - Depth quality (c_geom)
        - Semantic/geometric disagreement (disagreement_ratio)
        - Visual localization quality (c_vo, tracking_status)
        - Temporal consistency (tracked via TemporalConsistencyTracker)

        Every decision logs: timestamp, confidence, reason, action.
        """
        ts = float(timestamp) if timestamp is not None else time.time()
        reasons: List[str] = []

        # 1. Inspect Individual Signals
        c_p = float(np.clip(c_perc, 0.0, 1.0))
        c_g = float(np.clip(c_geom, 0.0, 1.0))
        c_v = float(np.clip(c_vo, 0.0, 1.0))
        c_f = float(np.clip(c_fusion, 0.0, 1.0))
        d_sg = float(np.clip(disagreement_ratio, 0.0, 1.0))

        conf_sources = {
            "perception_cnn": c_p,
            "depth_geometry": c_g,
            "visual_odometry": c_v,
            "fusion": c_f,
        }
        bottleneck_source = min(conf_sources, key=conf_sources.get)
        raw_weakest = min(conf_sources.values())

        # Disagreement penalty: nominal border misalignment <= 0.10 incurs no penalty;
        # higher conflict dampens confidence progressively.
        excess_disagree = max(0.0, d_sg - 0.10)
        disagreement_factor = max(0.0, 1.0 - 0.50 * excess_disagree)
        raw_confidence = raw_weakest * disagreement_factor

        # 2. Temporal Consistency Update
        temporal_metrics = self.temporal_tracker.update(
            timestamp=ts,
            raw_confidence=raw_confidence,
            disagreement_ratio=d_sg,
        )
        temporal_score = temporal_metrics["temporal_consistency"]
        reobserve_active = temporal_metrics["reobserve_active"]

        # Aggregate overall confidence
        overall_conf = float(np.clip(raw_confidence * temporal_score, 0.0, 1.0))

        signal_telemetry = {
            "cnn_confidence": c_p,
            "depth_quality": c_g,
            "vo_quality": c_v,
            "fusion_confidence": c_f,
            "disagreement_ratio": d_sg,
            "temporal_consistency": temporal_score,
            "rolling_variance": temporal_metrics["rolling_variance"],
            "sudden_drop": temporal_metrics["sudden_drop"],
        }

        # 3. State & Behavior Determination

        # Priority 1: Visual Odometry Tracking Loss -> CRITICAL (Safe Stop)
        if tracking_status == TrackingStatus.TRACKING_LOST or c_v < 0.20:
            reasons.append(
                f"Visual odometry lost (status={tracking_status.value}, C_vo={c_v:.2f}). Holding position."
            )
            state = SafetyState.CRITICAL
            action = SafetyAction.SAFE_STOP.value
            v_cmd = 0.0
            w_cmd = 0.0
            scale = 0.0
            inflation = 2.0
            is_estop = True

        # Priority 2: Imminent Obstacle Collision within Braking Zone -> CRITICAL (Safe Stop)
        elif min_obstacle_dist_m <= self.emergency_dist:
            reasons.append(
                f"Imminent collision: obstacle at {min_obstacle_dist_m:.2f}m <= {self.emergency_dist:.2f}m."
            )
            state = SafetyState.CRITICAL
            action = SafetyAction.SAFE_STOP.value
            v_cmd = 0.0
            w_cmd = 0.0
            scale = 0.0
            inflation = 2.0
            is_estop = True

        # Priority 3: Severe Disagreement or Low System Confidence -> CRITICAL (Safe Stop)
        elif overall_conf < self.th_low or d_sg >= self.th_disagree_crit:
            if d_sg >= self.th_disagree_crit:
                reasons.append(
                    f"Critical semantic/geometric disagreement {d_sg:.2f} >= {self.th_disagree_crit:.2f}. Emergency safety stop."
                )
            else:
                reasons.append(
                    f"System confidence {overall_conf:.2f} < {self.th_low:.2f} (bottleneck: {bottleneck_source}). Emergency safety stop."
                )
            state = SafetyState.CRITICAL
            action = SafetyAction.SAFE_STOP.value
            v_cmd = 0.0
            w_cmd = 0.0
            scale = 0.0
            inflation = 1.8
            is_estop = True

        # Priority 4: Low Confidence or Re-Observe Active -> LOW (Conservative Recommendation / Re-Observe)
        elif overall_conf < self.th_med or d_sg >= self.th_disagree_med or reobserve_active:
            state = SafetyState.LOW
            action = SafetyAction.CONSERVATIVE_REOBSERVE.value
            inflation = 1.6
            is_estop = False

            if reobserve_active:
                scale = 0.0
                v_cmd = 0.0
                w_cmd = float(np.clip(nominal_w, -0.15, 0.15))
                reasons.append(
                    f"Temporal instability / sudden drop detected ({temporal_metrics['sudden_drop']:.2f}). Re-observe hold active ({temporal_metrics['reobserve_counter']}/{self.temporal_tracker.reobserve_required} coherent frames)."
                )
            else:
                scale = 0.25
                v_cmd = min(nominal_v * scale, self.v_crawl_max)
                w_cmd = float(np.clip(nominal_w, -0.30, 0.30))
                reasons.append(
                    f"Low confidence {overall_conf:.2f} (bottleneck: {bottleneck_source}): conservative crawl at max {self.v_crawl_max} m/s."
                )

        # Priority 5: Medium Confidence or Degraded VO -> MEDIUM (Lower Speed / Larger Margin)
        elif overall_conf < self.th_high or tracking_status == TrackingStatus.TRACKING_DEGRADED:
            scale = 0.55
            v_cmd = min(nominal_v * scale, self.v_cautious_max)
            w_cmd = float(np.clip(nominal_w, -0.50, 0.50))
            inflation = 1.3
            is_estop = False
            state = SafetyState.MEDIUM
            action = SafetyAction.LOWER_SPEED_EXPAND_MARGIN.value
            reasons.append(
                f"Medium confidence {overall_conf:.2f}: reduced speed to {v_cmd:.2f} m/s with {inflation}x clearance margin."
            )

        # Priority 6: High Confidence Nominal Operation -> HIGH (Normal Recommendation)
        else:
            scale = 1.0
            v_cmd = nominal_v
            w_cmd = nominal_w
            inflation = 1.0
            is_estop = False
            state = SafetyState.HIGH
            action = SafetyAction.NORMAL.value
            reasons.append("High confidence: all systems nominal, full commanded speed permitted.")

        primary_reason = reasons[0] if reasons else "Nominal operation"
        clearance_margin = self.inscribed_radius * inflation

        # 4. Mandatory Decision Logging: timestamp, confidence, reason, action
        decision_log = self.logger.log(
            timestamp=ts,
            confidence=overall_conf,
            reason=primary_reason,
            action=action,
            state=state,
            signals=signal_telemetry,
            nominal_v=nominal_v,
            recommended_v=v_cmd,
            recommended_w=w_cmd,
            clearance_margin_m=clearance_margin,
            reobserve_active=reobserve_active,
        )

        return SafetyResult(
            overall_confidence=overall_conf,
            safety_state=state,
            commanded_linear_velocity=round(v_cmd, 3),
            commanded_angular_velocity=round(w_cmd, 3),
            speed_scale_factor=scale,
            clearance_inflation_factor=inflation,
            is_emergency_stop=is_estop,
            audit_reasons=reasons,
            action=action,
            decision_log=decision_log,
        )
