"""Pytest unit and replay tests for Navigation Planning and Decision Architecture.

Validates:
- A* Global pathfinding toward GoalPose
- Avoidance of lethal obstacles (cost >= 220)
- Risk-penalized path selection through safer terrain
- Exact metric clearance calculation and vehicle footprint respect
- Explainable path cost decomposition
- Navigation state machine transitions
- Full end-to-end replay on real recorded outdoor sequences
"""

import math
import numpy as np
import pytest

from src.interfaces.types import (
    GoalPose,
    NavigationPath,
    NavigationState,
    SteeringDirection,
    TrackingStatus,
    TraversabilityLevel,
    VisualOdometryResult,
    TraversabilityMapResult,
)
from src.planning.costmap_2d import Costmap2D
from src.planning.global_planner import AStarGlobalPlanner
from src.planning.navigation_decision_engine import NavigationDecisionEngine
from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
from src.pipeline import NavigationPipeline


def _make_dummy_trav_result(costmap: Costmap2D) -> TraversabilityMapResult:
    h, w = costmap.grid_h, costmap.grid_w
    level_grid = np.full((h, w), TraversabilityLevel.FREE.value, dtype=np.uint8)
    level_grid[costmap.inflated_grid <= 19] = TraversabilityLevel.PREFERRED.value
    level_grid[costmap.inflated_grid >= 220] = TraversabilityLevel.BLOCKED.value
    return TraversabilityMapResult(
        level_grid=level_grid,
        cost_grid=costmap.inflated_grid,
        obstacle_grid=(costmap.inflated_grid >= 220),
        uncertainty_grid=np.full((h, w), 0.1, dtype=np.float32),
        visual_map=np.zeros((h, w, 4), dtype=np.uint8),
        level_counts={"FREE": int(np.sum(level_grid == 1))},
        confidence=0.90,
        latency_ms=1.0,
        resolution_m=costmap.resolution_m,
        grid_size_m=costmap.grid_size_m,
    )


def _make_dummy_vo(tracking_status=TrackingStatus.TRACKING_OK) -> VisualOdometryResult:
    return VisualOdometryResult(
        x=0.0, y=0.0, z=0.0, yaw=0.0,
        tracking_status=tracking_status,
        confidence=0.95 if tracking_status == TrackingStatus.TRACKING_OK else 0.05,
    )


def test_global_path_reaches_goal():
    costmap = Costmap2D()
    costmap.grid.fill(0)
    costmap.inflated_grid.fill(0)
    planner = AStarGlobalPlanner()
    goal = GoalPose(x=5.0, y=1.0, tolerance_m=0.40)
    path = planner.plan(costmap, start_x_m=0.0, start_y_m=0.0, goal=goal)
    assert path.is_valid
    assert len(path.waypoints) >= 2
    assert path.total_length_m > 4.5
    end_pt = path.waypoints[-1]
    dist_to_goal = math.hypot(end_pt[0] - goal.x, end_pt[1] - goal.y)
    assert dist_to_goal <= goal.tolerance_m + 0.15


def test_global_path_avoids_blocked_cells():
    costmap = Costmap2D()
    costmap.grid.fill(0)
    costmap.inflated_grid.fill(0)
    costmap.inflated_grid[23:28, 40:60] = 254
    planner = AStarGlobalPlanner()
    goal = GoalPose(x=5.0, y=0.0, tolerance_m=0.50)
    path = planner.plan(costmap, start_x_m=0.0, start_y_m=0.0, goal=goal)
    assert path.is_valid
    for x, y in path.waypoints:
        r, c = costmap.world_to_grid(x, y)
        if costmap.is_in_bounds(r, c):
            assert costmap.inflated_grid[r, c] < 220


def test_penalize_risky_terrain():
    costmap = Costmap2D()
    costmap.grid.fill(10)
    costmap.inflated_grid.fill(10)
    costmap.inflated_grid[15:35, 45:55] = 160
    planner = AStarGlobalPlanner(risk_weight=4.0)
    goal = GoalPose(x=5.0, y=0.0, tolerance_m=0.40)
    path = planner.plan(costmap, start_x_m=0.0, start_y_m=0.0, goal=goal)
    assert path.is_valid
    max_lateral_dev = max(abs(y) for x, y in path.waypoints)
    assert max_lateral_dev > 0.30


def test_decision_engine_goal_reached():
    costmap = Costmap2D()
    costmap.grid.fill(0)
    costmap.inflated_grid.fill(0)
    trav = _make_dummy_trav_result(costmap)
    vo = _make_dummy_vo()
    engine = NavigationDecisionEngine()
    goal = GoalPose(x=0.20, y=0.10, tolerance_m=0.40)
    res = engine.decide(costmap, trav, vo, goal=goal)
    assert res.navigation_state == NavigationState.GOAL_REACHED
    assert res.recommended_linear_velocity == 0.0
    assert res.recommended_angular_velocity == 0.0
    assert res.recommended_steering == SteeringDirection.STOP
    assert res.status == "GOAL_REACHED"
    assert "Goal reached" in res.cost_explanation.explanation_text


def test_decision_engine_localization_lost():
    costmap = Costmap2D()
    trav = _make_dummy_trav_result(costmap)
    vo_lost = _make_dummy_vo(tracking_status=TrackingStatus.TRACKING_LOST)
    engine = NavigationDecisionEngine()
    goal = GoalPose(x=5.0, y=0.0)
    res = engine.decide(costmap, trav, vo_lost, goal=goal)
    assert res.navigation_state == NavigationState.RECOVERY_HOLD
    assert res.recommended_linear_velocity == 0.0
    assert res.status == "LOCALIZATION_LOST"
    assert "holding position" in res.cost_explanation.explanation_text.lower()


def test_clearance_and_vehicle_footprint():
    costmap = Costmap2D(inscribed_radius_m=0.35)
    costmap.grid.fill(0)
    costmap.inflated_grid.fill(0)
    r, c = costmap.world_to_grid(2.0, 0.8)
    costmap.grid[r, c] = 254
    costmap._inflate_obstacles()
    trav = _make_dummy_trav_result(costmap)
    vo = _make_dummy_vo()
    engine = NavigationDecisionEngine(inscribed_radius_m=0.35)
    res = engine.decide(costmap, trav, vo, goal=GoalPose(x=4.0, y=0.0))
    assert res.recommended_trajectory is not None
    assert res.recommended_trajectory.is_valid
    assert res.min_clearance_m > 0.35
    assert res.cost_explanation.min_clearance_m == res.min_clearance_m


def test_explainable_path_cost_breakdown():
    costmap = Costmap2D()
    costmap.grid.fill(12)
    costmap.inflated_grid.fill(12)
    trav = _make_dummy_trav_result(costmap)
    vo = _make_dummy_vo()
    engine = NavigationDecisionEngine()
    res = engine.decide(costmap, trav, vo, goal=GoalPose(x=5.0, y=0.0))
    expl = res.cost_explanation
    assert expl is not None
    assert 0.0 <= expl.progress_score <= 1.0
    assert 0.0 <= expl.clearance_score <= 1.0
    assert 0.0 <= expl.traversability_score <= 1.0
    assert 0.0 <= expl.heading_score <= 1.0
    assert expl.total_score > 0.0
    assert len(expl.explanation_text) > 20
    assert "Rollout" in expl.explanation_text


def test_real_sequence_navigation_open_path():
    loader = OutdoorDatasetLoader("datasets/processed/scenario_1_open_path")
    assert len(loader) > 0
    pipeline = NavigationPipeline()
    goal = GoalPose(x=7.0, y=0.0, tolerance_m=0.6)
    for i in range(len(loader)):
        frame = loader.get_frame(i)
        cmd, tel = pipeline.process_frame(frame, goal=goal)
        decision = tel["decision"]
        assert decision is not None
        assert decision.recommended_trajectory is not None
        assert decision.recommended_trajectory.is_valid
        assert decision.min_clearance_m >= 0.35
        assert decision.cost_explanation is not None
        assert len(decision.cost_explanation.explanation_text) > 0
        assert decision.navigation_state in (
            NavigationState.NAVIGATING_TO_GOAL,
            NavigationState.TRACKING_PATH,
            NavigationState.AVOIDING_OBSTACLE,
        )


def test_real_sequence_sudden_obstacle_avoidance():
    loader = OutdoorDatasetLoader("datasets/processed/scenario_2_sudden_obstacle")
    assert len(loader) > 0
    pipeline = NavigationPipeline()
    for i in range(len(loader)):
        frame = loader.get_frame(i)
        cmd, tel = pipeline.process_frame(frame)
        decision = tel["decision"]
        assert decision is not None
        if decision.status == "PATH_FOUND":
            assert decision.min_clearance_m >= 0.35
        else:
            assert decision.navigation_state == NavigationState.BLOCKED
            assert cmd.linear_velocity == 0.0
