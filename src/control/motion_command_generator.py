"""UGV motion command generator.

Transforms planned trajectories and safety gate outputs into the standardized
software-only motion command interface.
Important: The current project does NOT physically actuate motors.
"""

from __future__ import annotations

import time
from typing import Optional

from ..interfaces.types import (
    PlanningResult,
    SafetyResult,
    SafetyState,
    SteeringDirection,
    UGVMotionCommand,
)


class MotionCommandGenerator:
    """Generates the inspectable UGV motion command dictionary/dataclass."""

    def __init__(self) -> None:
        pass

    def generate(
        self,
        planning_result: PlanningResult,
        safety_result: SafetyResult,
        timestamp: Optional[float] = None,
    ) -> UGVMotionCommand:
        """Combine planning and safety arbitrations into the final motion command."""
        ts = timestamp if timestamp is not None else time.time()

        # Commanded velocities are strictly regulated by the safety gate
        v_cmd = safety_result.commanded_linear_velocity
        w_cmd = safety_result.commanded_angular_velocity

        # Determine steering direction from commanded w
        if safety_result.is_emergency_stop or v_cmd <= 0.001:
            steering = SteeringDirection.STOP.value
        elif abs(w_cmd) < 0.10:
            steering = SteeringDirection.FORWARD.value
        elif w_cmd > 0.35:
            steering = SteeringDirection.HARD_LEFT.value
        elif w_cmd > 0.10:
            steering = SteeringDirection.SLIGHT_LEFT.value
        elif w_cmd < -0.35:
            steering = SteeringDirection.HARD_RIGHT.value
        else:
            steering = SteeringDirection.SLIGHT_RIGHT.value

        # Determine overall navigation state
        if safety_result.safety_state == SafetyState.SAFETY_STOP:
            nav_state = "ESTOP"
        elif safety_result.safety_state == SafetyState.LOCALIZATION_LOST:
            nav_state = "RECOVERY_HOLD"
        elif planning_result.status == "OBSTACLE_BLOCKED":
            nav_state = "AVOIDING_BLOCKED"
        elif safety_result.safety_state == SafetyState.LOW_CONFIDENCE_SLOW:
            nav_state = "CAUTIOUS_CRAWL"
        elif safety_result.safety_state == SafetyState.CAUTIOUS_DEGRADED:
            nav_state = "CAUTIOUS_EXPLORATION"
        elif abs(w_cmd) > 0.15:
            nav_state = "AVOIDING"
        else:
            nav_state = "TRACKING"

        return UGVMotionCommand(
            linear_velocity=round(v_cmd, 3),
            angular_velocity=round(w_cmd, 3),
            steering_direction=steering,
            navigation_state=nav_state,
            confidence=round(safety_result.overall_confidence, 3),
            safety_state=safety_result.safety_state.value,
            timestamp=ts,
        )
