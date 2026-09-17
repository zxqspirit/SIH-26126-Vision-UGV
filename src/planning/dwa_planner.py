"""Dynamic Window Approach (DWA) local trajectory planner.

Evaluates kinematically feasible forward rollouts in the local costmap,
avoids obstacles, optimizes for forward traversability, and selects recommended velocities.
"""

from __future__ import annotations

import time
from typing import List, Tuple, Optional
import numpy as np

from ..interfaces.types import (
    Trajectory,
    PlanningResult,
    SteeringDirection,
)
from .costmap_2d import Costmap2D


class DWAPlanner:
    """Dynamic Window Approach local trajectory planner."""

    def __init__(
        self,
        max_linear_velocity: float = 0.80,  # m/s
        min_linear_velocity: float = 0.15,  # m/s
        max_angular_velocity: float = 0.85,  # rad/s
        max_linear_accel: float = 0.50,  # m/s^2
        max_angular_accel: float = 1.00,  # rad/s^2
        sim_time_s: float = 2.5,  # Lookahead horizon
        dt_s: float = 0.15,  # Step size along trajectory
        num_v_samples: int = 6,
        num_w_samples: int = 17,
        w_progress: float = 0.35,
        w_clearance: float = 0.30,
        w_cost: float = 0.20,
        w_heading: float = 0.15,
    ) -> None:
        self.v_max = max_linear_velocity
        self.v_min = min_linear_velocity
        self.w_max = max_angular_velocity
        self.a_v_max = max_linear_accel
        self.a_w_max = max_angular_accel
        self.sim_time = sim_time_s
        self.dt = dt_s
        self.num_v_samples = num_v_samples
        self.num_w_samples = num_w_samples

        self.w_progress = w_progress
        self.w_clearance = w_clearance
        self.w_cost = w_cost
        self.w_heading = w_heading

    def plan(
        self,
        costmap: Costmap2D,
        current_v: float = 0.0,
        current_w: float = 0.0,
        target_heading_rad: float = 0.0,
    ) -> PlanningResult:
        """Evaluate candidate trajectory rollouts and pick optimal collision-free command."""
        start_time = time.perf_counter()

        v_samples = np.linspace(self.v_min, self.v_max, self.num_v_samples)
        w_samples = np.linspace(-self.w_max, self.w_max, self.num_w_samples)

        candidates: List[Trajectory] = []
        best_traj: Optional[Trajectory] = None
        best_score = -float('inf')

        steps = int(self.sim_time / self.dt)
        max_possible_dist = self.v_max * self.sim_time

        for v in v_samples:
            for w in w_samples:
                x = 0.0
                y = 0.0
                yaw = 0.0
                pts: List[Tuple[float, float, float]] = [(x, y, yaw)]
                is_valid = True
                min_clearance = float('inf')
                accum_cell_cost = 0.0

                for _ in range(steps):
                    x += v * np.cos(yaw) * self.dt
                    y += v * np.sin(yaw) * self.dt
                    yaw += w * self.dt
                    pts.append((x, y, yaw))

                    is_clear, clearance, cell_cost = costmap.check_footprint_clearance(x, y)
                    if not is_clear:
                        is_valid = False
                        break

                    accum_cell_cost += cell_cost
                    if clearance < min_clearance:
                        min_clearance = clearance

                if not is_valid or min_clearance == float('inf'):
                    candidates.append(Trajectory(
                        points=pts,
                        linear_velocity=float(v),
                        angular_velocity=float(w),
                        cost=999.0,
                        clearance_m=0.0,
                        is_valid=False,
                    ))
                    continue

                avg_cost = accum_cell_cost / max(len(pts) - 1, 1)

                # Objective scoring:
                # 1. Forward progress along target heading [0..1]
                final_x = pts[-1][0]
                final_y = pts[-1][1]
                forward_progress = final_x * np.cos(target_heading_rad) + final_y * np.sin(target_heading_rad)
                progress_score = max(0.0, min(1.0, forward_progress / max_possible_dist))

                # 2. Heading alignment [0..1]
                final_heading = pts[-1][2]
                heading_err = abs(final_heading - target_heading_rad)
                heading_score = max(0.0, 1.0 - (heading_err / np.pi))

                # 3. Clearance score [0..1]
                clearance_score = min(1.0, min_clearance / 3.0)

                # 4. Traversability cost score [0..1]
                cost_score = max(0.0, 1.0 - (avg_cost / 254.0))

                total_score = (
                    self.w_progress * progress_score +
                    self.w_clearance * clearance_score +
                    self.w_cost * cost_score +
                    self.w_heading * heading_score
                )

                traj = Trajectory(
                    points=pts,
                    linear_velocity=float(v),
                    angular_velocity=float(w),
                    cost=float(avg_cost),
                    clearance_m=float(min_clearance),
                    is_valid=True,
                )
                candidates.append(traj)

                if total_score > best_score:
                    best_score = total_score
                    best_traj = traj

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if best_traj is not None:
            rec_v = best_traj.linear_velocity
            rec_w = best_traj.angular_velocity
            status = "PATH_FOUND"
        else:
            rec_v = 0.0
            rec_w = 0.0
            status = "OBSTACLE_BLOCKED"

        if status == "OBSTACLE_BLOCKED" or rec_v == 0.0:
            steering = SteeringDirection.STOP
        elif abs(rec_w) < 0.12:
            steering = SteeringDirection.FORWARD
        elif rec_w > 0.35:
            steering = SteeringDirection.HARD_LEFT
        elif rec_w > 0.12:
            steering = SteeringDirection.SLIGHT_LEFT
        elif rec_w < -0.35:
            steering = SteeringDirection.HARD_RIGHT
        else:
            steering = SteeringDirection.SLIGHT_RIGHT

        return PlanningResult(
            selected_trajectory=best_traj,
            candidate_trajectories=candidates,
            recommended_linear_velocity=round(rec_v, 3),
            recommended_angular_velocity=round(rec_w, 3),
            recommended_steering=steering,
            status=status,
            target_heading_rad=target_heading_rad,
            latency_ms=latency_ms,
        )
