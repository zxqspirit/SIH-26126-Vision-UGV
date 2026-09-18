"""Navigation Decision Engine: Software-Only Decision Architecture.

Orchestrates:
1. Global A* Path Planning toward GoalPose
2. Local DWA Trajectory Rollout Evaluation tracking global path
3. Vehicle Footprint and Metric Clearance Verification
4. Auditable Path Cost Decomposition & Explanation
5. Deterministic Navigation State Machine Transitions

Important: All commands are software recommendations only. No physical UGV exists.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from ..interfaces.types import (
    GoalPose,
    NavigationPath,
    Trajectory,
    PlanningResult,
    PathCostBreakdown,
    NavigationDecisionResult,
    NavigationState,
    SteeringDirection,
    TrackingStatus,
    TraversabilityLevel,
    VisualOdometryResult,
    TraversabilityMapResult,
)
from .costmap_2d import Costmap2D
from .global_planner import AStarGlobalPlanner
from .dwa_planner import DWAPlanner


class NavigationDecisionEngine:
    """Deterministic navigation planner producing explainable software recommendations."""

    def __init__(
        self,
        inscribed_radius_m: float = 0.35,
        max_linear_velocity: float = 0.80,
        min_linear_velocity: float = 0.15,
        max_angular_velocity: float = 0.85,
        goal_tolerance_m: float = 0.50,
    ) -> None:
        self.inscribed_radius = inscribed_radius_m
        self.max_v = max_linear_velocity
        self.min_v = min_linear_velocity
        self.max_w = max_angular_velocity
        self.goal_tolerance = goal_tolerance_m

        self.global_planner = AStarGlobalPlanner()
        self.local_planner = DWAPlanner(
            max_linear_velocity=max_linear_velocity,
            min_linear_velocity=min_linear_velocity,
            max_angular_velocity=max_angular_velocity,
        )

        self.current_state = NavigationState.IDLE
        self.active_goal: Optional[GoalPose] = None

    def set_goal(self, goal: Optional[GoalPose]) -> None:
        """Set or update target navigation goal."""
        self.active_goal = goal
        if goal is not None:
            self.current_state = NavigationState.NAVIGATING_TO_GOAL
        else:
            self.current_state = NavigationState.IDLE

    def decide(
        self,
        costmap: Costmap2D,
        traversability: TraversabilityMapResult,
        odometry: VisualOdometryResult,
        goal: Optional[GoalPose] = None,
        current_v: float = 0.0,
        current_w: float = 0.0,
    ) -> NavigationDecisionResult:
        """Compute recommended path, trajectory, velocities, and state explanation."""
        start_time = time.perf_counter()

        # Update active goal if passed explicitly
        if goal is not None:
            self.active_goal = goal

        effective_goal = self.active_goal

        # 1. Compute distance to goal (if goal exists)
        goal_dist_m = 0.0
        target_heading = 0.0
        is_goal_reached = False

        if effective_goal is not None:
            goal_dist_m = math.hypot(effective_goal.x, effective_goal.y)
            target_heading = math.atan2(effective_goal.y, max(effective_goal.x, 0.01))
            if goal_dist_m <= effective_goal.tolerance_m:
                is_goal_reached = True

        # Check if Goal is already reached
        if is_goal_reached:
            self.current_state = NavigationState.GOAL_REACHED
            latency = (time.perf_counter() - start_time) * 1000.0
            return NavigationDecisionResult(
                recommended_path=NavigationPath(waypoints=[(0.0, 0.0)], is_valid=True),
                recommended_trajectory=Trajectory(
                    points=[(0.0, 0.0, 0.0)],
                    linear_velocity=0.0,
                    angular_velocity=0.0,
                    cost=0.0,
                    clearance_m=costmap.get_obstacle_distance(0.0, 0.0),
                    is_valid=True,
                ),
                candidate_trajectories=[],
                recommended_linear_velocity=0.0,
                recommended_angular_velocity=0.0,
                recommended_steering=SteeringDirection.STOP,
                navigation_state=NavigationState.GOAL_REACHED,
                cost_explanation=PathCostBreakdown(
                    total_score=1.0,
                    progress_score=1.0,
                    clearance_score=1.0,
                    traversability_score=1.0,
                    heading_score=1.0,
                    average_cost=0.0,
                    min_clearance_m=costmap.get_obstacle_distance(0.0, 0.0),
                    primary_terrain="GOAL_REACHED",
                    explanation_text=f"Goal reached within {effective_goal.tolerance_m}m tolerance.",
                ),
                min_clearance_m=costmap.get_obstacle_distance(0.0, 0.0),
                status="GOAL_REACHED",
                target_heading_rad=target_heading,
                goal=effective_goal,
                goal_distance_m=round(goal_dist_m, 2),
                latency_ms=round(latency, 2),
            )

        # 2. Check Visual Odometry localization status
        if odometry.tracking_status == TrackingStatus.TRACKING_LOST:
            self.current_state = NavigationState.RECOVERY_HOLD
            latency = (time.perf_counter() - start_time) * 1000.0
            return NavigationDecisionResult(
                recommended_path=None,
                recommended_trajectory=None,
                candidate_trajectories=[],
                recommended_linear_velocity=0.0,
                recommended_angular_velocity=0.0,
                recommended_steering=SteeringDirection.STOP,
                navigation_state=NavigationState.RECOVERY_HOLD,
                cost_explanation=PathCostBreakdown(
                    total_score=0.0,
                    progress_score=0.0,
                    clearance_score=0.0,
                    traversability_score=0.0,
                    heading_score=0.0,
                    average_cost=255.0,
                    min_clearance_m=0.0,
                    primary_terrain="UNKNOWN",
                    explanation_text="Visual Odometry tracking lost. Holding position for recovery.",
                ),
                min_clearance_m=0.0,
                status="LOCALIZATION_LOST",
                target_heading_rad=target_heading,
                goal=effective_goal,
                goal_distance_m=round(goal_dist_m, 2),
                latency_ms=round(latency, 2),
            )

        # 3. Global Topological Path Planning (A*)
        global_path = self.global_planner.plan(
            costmap=costmap,
            start_x_m=0.0,
            start_y_m=0.0,
            goal=effective_goal,
        )

        # Derive local lookahead heading from global path if waypoints exist
        if global_path.is_valid and len(global_path.waypoints) > 1:
            # Pick a lookahead waypoint ~1.5m to 2.5m ahead
            lookahead_pt = global_path.waypoints[min(3, len(global_path.waypoints) - 1)]
            path_heading = math.atan2(lookahead_pt[1], max(lookahead_pt[0], 0.01))
            target_heading = path_heading

        # 4. Local Kinematic Trajectory Rollout Evaluation (DWA)
        local_plan: PlanningResult = self.local_planner.plan(
            costmap=costmap,
            current_v=current_v,
            current_w=current_w,
            target_heading_rad=target_heading,
        )

        selected_traj = local_plan.selected_trajectory

        # 5. Clearance Calculation & Verification
        if selected_traj is not None:
            min_clearance = selected_traj.clearance_m
            avg_cost = selected_traj.cost
            rec_v = local_plan.recommended_linear_velocity
            rec_w = local_plan.recommended_angular_velocity
            rec_steering = local_plan.recommended_steering
            status = local_plan.status
        else:
            min_clearance = 0.0
            avg_cost = 255.0
            rec_v = 0.0
            rec_w = 0.0
            rec_steering = SteeringDirection.STOP
            status = "NO_VALID_TRAJECTORY"

        # 6. Auditable Path Cost Breakdown
        cost_expl = self._explain_trajectory_cost(
            costmap=costmap,
            traversability=traversability,
            selected_traj=selected_traj,
            target_heading=target_heading,
            global_path=global_path,
        )

        # 7. Deterministic State Machine Transition
        if status == "OBSTACLE_BLOCKED" or min_clearance <= self.inscribed_radius:
            new_state = NavigationState.BLOCKED
        elif abs(rec_w) > 0.15:
            new_state = NavigationState.AVOIDING_OBSTACLE
        elif odometry.tracking_status == TrackingStatus.TRACKING_DEGRADED:
            new_state = NavigationState.CAUTIOUS_EXPLORATION
        elif effective_goal is not None:
            new_state = NavigationState.NAVIGATING_TO_GOAL
        elif global_path.is_valid and len(global_path.waypoints) > 1:
            new_state = NavigationState.TRACKING_PATH
        else:
            new_state = NavigationState.CAUTIOUS_EXPLORATION

        self.current_state = new_state
        latency = (time.perf_counter() - start_time) * 1000.0

        return NavigationDecisionResult(
            recommended_path=global_path,
            recommended_trajectory=selected_traj,
            candidate_trajectories=local_plan.candidate_trajectories,
            recommended_linear_velocity=round(rec_v, 3),
            recommended_angular_velocity=round(rec_w, 3),
            recommended_steering=rec_steering,
            navigation_state=self.current_state,
            cost_explanation=cost_expl,
            min_clearance_m=round(min_clearance, 3),
            status=status,
            target_heading_rad=round(target_heading, 3),
            goal=effective_goal,
            goal_distance_m=round(goal_dist_m, 2),
            latency_ms=round(latency, 2),
        )

    def _explain_trajectory_cost(
        self,
        costmap: Costmap2D,
        traversability: TraversabilityMapResult,
        selected_traj: Optional[Trajectory],
        target_heading: float,
        global_path: NavigationPath,
    ) -> PathCostBreakdown:
        """Decompose rollout score into inspectable mathematical components."""
        if selected_traj is None or not selected_traj.is_valid:
            return PathCostBreakdown(
                total_score=0.0,
                progress_score=0.0,
                clearance_score=0.0,
                traversability_score=0.0,
                heading_score=0.0,
                average_cost=999.0,
                min_clearance_m=0.0,
                primary_terrain="BLOCKED",
                explanation_text="Trajectory rejected: penetrates vehicle footprint or lethal obstacle zone.",
            )

        pts = selected_traj.points
        final_pt = pts[-1]
        max_dist = self.max_v * self.local_planner.sim_time

        # 1. Forward progress score [0..1]
        fwd = final_pt[0] * math.cos(target_heading) + final_pt[1] * math.sin(target_heading)
        prog_score = max(0.0, min(1.0, fwd / max_dist))

        # 2. Clearance score [0..1]
        clr_score = min(1.0, selected_traj.clearance_m / 3.0)

        # 3. Traversability cost score [0..1]
        trav_score = max(0.0, 1.0 - (selected_traj.cost / 254.0))

        # 4. Heading alignment score [0..1]
        heading_err = abs(final_pt[2] - target_heading)
        head_score = max(0.0, 1.0 - (heading_err / math.pi))

        total_score = (
            self.local_planner.w_progress * prog_score +
            self.local_planner.w_clearance * clr_score +
            self.local_planner.w_cost * trav_score +
            self.local_planner.w_heading * head_score
        )

        # Identify dominant terrain under trajectory
        sampled_levels = []
        for x, y, _ in pts:
            r, c = costmap.world_to_grid(x, y)
            if costmap.is_in_bounds(r, c):
                sampled_levels.append(int(traversability.level_grid[r, c]))

        if sampled_levels:
            mode_level = max(set(sampled_levels), key=sampled_levels.count)
            try:
                terrain_desc = TraversabilityLevel(mode_level).name
            except Exception:
                terrain_desc = "UNKNOWN"
        else:
            terrain_desc = "UNKNOWN"

        explanation = (
            f"Rollout (v={selected_traj.linear_velocity:.2f}m/s, w={selected_traj.angular_velocity:+.2f}rad/s) selected. "
            f"Score: {total_score:.3f} (progress={prog_score:.2f}, clearance={clr_score:.2f}, "
            f"terrain={trav_score:.2f}, heading={head_score:.2f}). "
            f"Min clearance: {selected_traj.clearance_m:.2f}m over {terrain_desc} terrain."
        )

        return PathCostBreakdown(
            total_score=round(total_score, 3),
            progress_score=round(prog_score, 3),
            clearance_score=round(clr_score, 3),
            traversability_score=round(trav_score, 3),
            heading_score=round(head_score, 3),
            average_cost=round(selected_traj.cost, 1),
            min_clearance_m=round(selected_traj.clearance_m, 3),
            primary_terrain=terrain_desc,
            explanation_text=explanation,
        )
