"""Tests for 3D depth geometry processing and Rule 12."""

import pytest
import numpy as np
from src.interfaces.types import CameraIntrinsics
from src.depth_geometry.depth_geometry_engine import DepthGeometryEngine
from scripts.generate_sample_scenarios import make_realistic_outdoor_scene


@pytest.fixture
def depth_engine():
    intrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    return DepthGeometryEngine(intrinsics)


def test_invalid_depth_is_not_free_space(depth_engine):
    """Rule 12: Invalid depth (0.0, NaN, Inf) must NEVER be treated as free space."""
    # Generate realistic ground frame
    _, depth_m, _ = make_realistic_outdoor_scene(0, 15, "scenario_1_open_path")
    # Force central ground corridor to invalid (0.0)
    depth_m[250:350, 250:350] = 0.0

    result = depth_engine.process_depth(depth_m)

    # Check that invalid pixels are properly flagged in validity mask
    assert not np.any(result.depth_validity_mask[250:350, 250:350])

    # Rule 12: Invalid depth must carry a non-zero hazard / uncertainty penalty
    invalid_costs = result.geometric_cost[250:350, 250:350]
    assert np.all(invalid_costs > 0.0), "Rule 12 violation: invalid depth assigned 0 cost (free space)!"
    assert np.all(invalid_costs >= 0.50), "Invalid depth must carry an uncertainty penalty >= 0.50"


def test_positive_obstacle_detection(depth_engine):
    """Points rising > 15 cm above ground plane must be detected as positive obstacles."""
    # Generate scenario with positive obstacle (crate / barrier) on path
    _, depth_m, _ = make_realistic_outdoor_scene(0, 15, "scenario_2_sudden_obstacle")

    result = depth_engine.process_depth(depth_m)

    # Obstacle region must be flagged as positive obstacle
    assert np.sum(result.positive_obstacle_mask) > 100, "Obstacle points were not detected!"
    # Obstacle area must reach lethal cost (1.0) and significant mean hazard (>= 0.50)
    obs_costs = result.geometric_cost[result.positive_obstacle_mask]
    assert np.max(obs_costs) == 1.00
    assert np.mean(obs_costs) >= 0.50


def test_ground_plane_estimation(depth_engine):
    """Smooth flat ground produces near-zero positive obstacle detections."""
    _, depth_m, _ = make_realistic_outdoor_scene(0, 15, "scenario_1_open_path")

    result = depth_engine.process_depth(depth_m)

    # Nominal clear ground should have zero positive obstacle points
    assert np.sum(result.positive_obstacle_mask) == 0
    # Average geometric cost on valid ground should be low
    valid = result.depth_validity_mask
    assert np.mean(result.geometric_cost[valid]) < 0.20
