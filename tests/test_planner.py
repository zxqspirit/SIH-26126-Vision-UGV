"""Tests for Dynamic Window Approach (DWA) local trajectory planner."""

import pytest
import numpy as np
from src.planning.costmap_2d import Costmap2D
from src.planning.dwa_planner import DWAPlanner
from src.interfaces.types import SteeringDirection


@pytest.fixture
def planner_and_costmap():
    costmap = Costmap2D()
    planner = DWAPlanner(max_linear_velocity=0.80)
    return planner, costmap


def test_dwa_chooses_open_path(planner_and_costmap):
    """When path straight ahead is completely free, planner commands forward velocity."""
    planner, costmap = planner_and_costmap
    costmap.grid.fill(0)
    costmap.inflated_grid.fill(0)

    res = planner.plan(costmap=costmap, current_v=0.4, current_w=0.0, target_heading_rad=0.0)

    assert res.status == "PATH_FOUND"
    assert res.recommended_linear_velocity > 0.30
    assert abs(res.recommended_angular_velocity) < 0.20
    assert res.recommended_steering in (SteeringDirection.FORWARD, SteeringDirection.SLIGHT_LEFT, SteeringDirection.SLIGHT_RIGHT)


def test_dwa_avoids_obstacle(planner_and_costmap):
    """When straight path has an obstacle, planner reacts by steering away or halting."""
    planner, costmap = planner_and_costmap
    costmap.grid.fill(0)
    costmap.inflated_grid.fill(0)

    # Place lethal obstacle directly ahead in center corridor (rows 6 to 18, cols 46 to 54)
    costmap.inflated_grid[6:18, 46:54] = 254

    res = planner.plan(costmap=costmap, current_v=0.4, current_w=0.0, target_heading_rad=0.0)

    # If an evasive path is found, it must steer; if completely impassable, it halts
    if res.selected_trajectory is not None and res.recommended_linear_velocity > 0.0:
        assert abs(res.recommended_angular_velocity) > 0.10, "Planner failed to steer around center obstacle!"
        assert res.recommended_steering != SteeringDirection.FORWARD
    else:
        assert res.status == "OBSTACLE_BLOCKED" or res.recommended_steering == SteeringDirection.STOP
