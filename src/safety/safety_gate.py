"""Deterministic Confidence Safety Gate for outdoor UGV.

Enforces:
- Rule 9: Safety behavior must be deterministic and inspectable.
- Rule 10: Never let an LLM directly control a physical robot.
- Rule 13: Confidence must produce an actual behavior change.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from ..interfaces.types import (
    SafetyState,
    SafetyResult,
    TrackingStatus,
    PlanningResult,
)


class SafetyGate:
    """Deterministic arbiter regulating motion commands based on sensor confidence."""

    def __init__(
        self,
        high_conf_thresh: float = 0.75,
        medium_conf_thresh: float = 0.45,
        low_conf_thresh: float = 0.25,
        emergency_stopping_dist_m: float = 0.45,
        max_cautious_velocity: float = 0.35,  # m/s
        max_crawl_velocity: float = 0.15,  # m/s
    ) -> None:
        self.th_high = high_conf_thresh
        self.th_med = medium_conf_thresh
        self.th_low = low_conf_thresh
        self.emergency_dist = emergency_stopping_dist_m
        self.v_cautious_max = max_cautious_velocity
        self.v_crawl_max = max_crawl_velocity

    def arbitrate(
        self,
        nominal_v: float,
        nominal_w: float,
        c_perc: float,
        c_geom: float,
        c_vo: float,
        c_fusion: float,
        tracking_status: TrackingStatus = TrackingStatus.TRACKING_OK,
        min_obstacle_dist_m: float = 5.0,
    ) -> SafetyResult:
        """Arbitrate nominal commands against multi-source confidence and clearance."""
        reasons: List[str] = []

        # 1. Multi-source confidence aggregation (weakest-link principle)
        conf_sources = {
            "perception": float(np.clip(c_perc, 0.0, 1.0)),
            "depth_geometry": float(np.clip(c_geom, 0.0, 1.0)),
            "visual_odometry": float(np.clip(c_vo, 0.0, 1.0)),
            "fusion": float(np.clip(c_fusion, 0.0, 1.0)),
        }
        overall_conf = min(conf_sources.values())

        # 2. Priority 1: Visual Odometry Tracking Loss
        if tracking_status == TrackingStatus.TRACKING_LOST or c_vo < 0.20:
            reasons.append(f"Visual odometry lost (status={tracking_status.value}, C_vo={c_vo:.2f}). Holding position.")
            return SafetyResult(
                overall_confidence=overall_conf,
                safety_state=SafetyState.LOCALIZATION_LOST,
                commanded_linear_velocity=0.0,
                commanded_angular_velocity=0.0,
                speed_scale_factor=0.0,
                clearance_inflation_factor=2.0,
                is_emergency_stop=True,
                audit_reasons=reasons,
            )

        # 3. Priority 2: Imminent Obstacle Collision within Braking Zone
        if min_obstacle_dist_m <= self.emergency_dist:
            reasons.append(f"Imminent collision: obstacle at {min_obstacle_dist_m:.2f}m <= {self.emergency_dist:.2f}m.")
            return SafetyResult(
                overall_confidence=overall_conf,
                safety_state=SafetyState.SAFETY_STOP,
                commanded_linear_velocity=0.0,
                commanded_angular_velocity=0.0,
                speed_scale_factor=0.0,
                clearance_inflation_factor=2.0,
                is_emergency_stop=True,
                audit_reasons=reasons,
            )

        # 4. Priority 3: Low System Confidence Emergency Stop (Rule 13)
        if overall_conf < self.th_low:
            min_source = min(conf_sources, key=conf_sources.get)
            reasons.append(f"System confidence {overall_conf:.2f} < {self.th_low:.2f} (bottleneck: {min_source}). Emergency safety stop.")
            return SafetyResult(
                overall_confidence=overall_conf,
                safety_state=SafetyState.SAFETY_STOP,
                commanded_linear_velocity=0.0,
                commanded_angular_velocity=0.0,
                speed_scale_factor=0.0,
                clearance_inflation_factor=1.8,
                is_emergency_stop=True,
                audit_reasons=reasons,
            )

        # 5. Priority 4: Low Confidence Crawl (0.25 <= Conf < 0.45)
        if overall_conf < self.th_med:
            scale = 0.25
            v_cmd = min(nominal_v * scale, self.v_crawl_max)
            # Limit rotational rate during low confidence
            w_cmd = float(np.clip(nominal_w, -0.30, 0.30))
            reasons.append(f"Low confidence {overall_conf:.2f}: crawling at max {self.v_crawl_max} m/s.")
            return SafetyResult(
                overall_confidence=overall_conf,
                safety_state=SafetyState.LOW_CONFIDENCE_SLOW,
                commanded_linear_velocity=round(v_cmd, 3),
                commanded_angular_velocity=round(w_cmd, 3),
                speed_scale_factor=scale,
                clearance_inflation_factor=1.6,
                is_emergency_stop=False,
                audit_reasons=reasons,
            )

        # 6. Priority 5: Cautious Degraded Speed (0.45 <= Conf < 0.75)
        if overall_conf < self.th_high or tracking_status == TrackingStatus.TRACKING_DEGRADED:
            scale = 0.55
            v_cmd = min(nominal_v * scale, self.v_cautious_max)
            w_cmd = float(np.clip(nominal_w, -0.55, 0.55))
            reasons.append(f"Medium confidence {overall_conf:.2f}: reduced speed to {v_cmd:.2f} m/s with 1.3x clearance.")
            return SafetyResult(
                overall_confidence=overall_conf,
                safety_state=SafetyState.CAUTIOUS_DEGRADED,
                commanded_linear_velocity=round(v_cmd, 3),
                commanded_angular_velocity=round(w_cmd, 3),
                speed_scale_factor=scale,
                clearance_inflation_factor=1.3,
                is_emergency_stop=False,
                audit_reasons=reasons,
            )

        # 7. Priority 6: High Confidence Nominal Operation
        reasons.append("High confidence: all systems nominal, full commanded speed permitted.")
        return SafetyResult(
            overall_confidence=overall_conf,
            safety_state=SafetyState.HIGH_CONFIDENCE,
            commanded_linear_velocity=round(nominal_v, 3),
            commanded_angular_velocity=round(nominal_w, 3),
            speed_scale_factor=1.0,
            clearance_inflation_factor=1.0,
            is_emergency_stop=False,
            audit_reasons=reasons,
        )
