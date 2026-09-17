"""Comprehensive unit and regression tests for 3D depth geometry processing and Rule 12.

Validates the 4 required conditions:
1. Flat ground: ground plane fitting, residual tolerance, corridor traversability.
2. Obstacle: positive obstacle detection, lethal cost assignment.
3. Invalid depth: CRITICAL invariant (invalid depth must NEVER become free space).
4. Depth discontinuity: sharp height gradient / step-edge detection.
Additionally tests negative obstacles, depth quality metric, and real recorded sensor frames.
"""

import glob
import os
import pytest
import numpy as np
from src.interfaces.types import CameraIntrinsics, DepthGeometryResult
from src.depth_geometry.depth_geometry_engine import DepthGeometryEngine
from scripts.generate_sample_scenarios import make_realistic_outdoor_scene


@pytest.fixture
def depth_engine():
    intrinsics = CameraIntrinsics(
        fx=385.0, fy=385.0, cx=320.0, cy=240.0,
        width=640, height=480,
        camera_height_m=0.45, camera_pitch_rad=0.209
    )
    return DepthGeometryEngine(intrinsics)


def test_flat_ground(depth_engine):
    """Condition 1: Flat ground must produce low traversability cost and zero positive obstacles."""
    _, depth_m, _ = make_realistic_outdoor_scene(0, 15, "scenario_1_open_path")

    result = depth_engine.process_depth(depth_m)

    # 1. Zero positive obstacle false positives on nominal clear path
    assert np.sum(result.positive_obstacle_mask) == 0, "False positive obstacles detected on flat ground!"

    # 2. Corridor ground points must have very low geometric cost (<= 0.10)
    # Forward corridor: central columns (240:400) and lower ground rows (300:440)
    corridor_valid = result.depth_validity_mask[300:440, 240:400]
    corridor_costs = result.geometric_cost[300:440, 240:400][corridor_valid]
    assert len(corridor_costs) > 100, "Insufficient valid corridor points!"
    assert np.mean(corridor_costs) < 0.10, f"Corridor mean cost {np.mean(corridor_costs)} exceeds 0.10!"

    # 3. Ground plane fit must be near-horizontal (slope < 5 degrees)
    if result.plane_coeffs is not None:
        a, b, c, d = result.plane_coeffs
        slope_rad = np.arctan(np.sqrt(a**2 + b**2))
        slope_deg = np.degrees(slope_rad)
        assert slope_deg < 5.0, f"Estimated ground slope {slope_deg:.2f} deg is too steep for flat ground!"

    # 4. Depth quality confidence must be high (> 0.70) on pristine flat ground
    assert result.confidence >= 0.70, f"Confidence {result.confidence} is too low on clear ground!"


def test_obstacle(depth_engine):
    """Condition 2: Positive obstacles rising > 15 cm must be detected with lethal cost (1.0)."""
    _, depth_m, _ = make_realistic_outdoor_scene(0, 15, "scenario_2_sudden_obstacle")

    result = depth_engine.process_depth(depth_m)

    # 1. Obstacle points must be detected
    assert np.sum(result.positive_obstacle_mask) > 100, "Obstacle points were not detected!"

    # 2. Obstacle area must reach lethal cost (1.0) and high mean hazard (>= 0.50)
    obs_costs = result.geometric_cost[result.positive_obstacle_mask]
    assert np.max(obs_costs) == 1.00, "Lethal obstacle failed to reach max cost 1.0!"
    assert np.mean(obs_costs) >= 0.50, f"Obstacle mean cost {np.mean(obs_costs)} too low!"

    # 3. Subsampled 3D pointcloud should be populated
    assert result.points_3d is not None and len(result.points_3d) > 0


def test_invalid_depth_never_free_space(depth_engine):
    """Condition 3: CRITICAL RULE 12 - Invalid depth must NEVER become free space (cost != 0.0)."""
    _, depth_m, _ = make_realistic_outdoor_scene(0, 15, "scenario_1_open_path")

    # Inject diverse invalid depth patterns:
    # A. Dropout / blind zone (0.0)
    depth_m[200:260, 200:300] = 0.0
    # B. NaN values (specular glare)
    depth_m[280:320, 200:300] = np.nan
    # C. Infinite values (sky / sensor overflow)
    depth_m[340:380, 200:300] = np.inf
    # D. Below minimum sensor range (e.g. 0.10 m)
    depth_m[400:430, 200:300] = 0.10
    # E. Beyond maximum sensor range (e.g. 25.0 m)
    depth_m[440:470, 200:300] = 25.0

    result = depth_engine.process_depth(depth_m)

    # Validity mask must flag all corrupted pixels as invalid
    assert not np.any(result.depth_validity_mask[200:260, 200:300]), "0.0 depth was marked valid!"
    assert not np.any(result.depth_validity_mask[280:320, 200:300]), "NaN depth was marked valid!"
    assert not np.any(result.depth_validity_mask[340:380, 200:300]), "Inf depth was marked valid!"
    assert not np.any(result.depth_validity_mask[400:430, 200:300]), "< min_range depth was marked valid!"
    assert not np.any(result.depth_validity_mask[440:470, 200:300]), "> max_range depth was marked valid!"

    # CRITICAL INVARIANT: Check entire frame for Rule 12 violations
    invalid_mask = ~result.depth_validity_mask
    free_space_violations = invalid_mask & (result.geometric_cost == 0.0)
    num_violations = np.sum(free_space_violations)
    assert num_violations == 0, f"RULE 12 VIOLATION: {num_violations} invalid depth pixels were treated as free space!"

    # All invalid pixels must receive the uncertainty penalty (>= 0.60)
    invalid_costs = result.geometric_cost[invalid_mask]
    assert np.all(invalid_costs >= 0.60), "Invalid depth must carry an uncertainty penalty >= 0.60!"


def test_depth_discontinuity(depth_engine):
    """Condition 4: Height discontinuity detection on sharp step edges/cliffs."""
    # Construct a synthetic depth image with a sharp vertical step curb at X=2.0m
    h, w = 480, 640
    depth_m = np.full((h, w), 2.5, dtype=np.float32)
    # Step edge across rows 280:350 (closer object creates steep depth & height step)
    depth_m[280:350, 200:440] = 1.8

    result = depth_engine.process_depth(depth_m)

    # Discontinuity mask must detect the sharp step boundary
    assert result.discontinuity_mask is not None, "Discontinuity mask is None!"
    discontinuity_count = np.sum(result.discontinuity_mask)
    assert discontinuity_count > 20, f"Discontinuity detector found only {discontinuity_count} points!"

    # The discontinuity points must have high hazard cost (>= 0.85)
    disc_costs = result.geometric_cost[result.discontinuity_mask]
    assert np.all(disc_costs >= 0.85), "Discontinuity edge cost below 0.85!"


def test_negative_obstacle_and_depth_quality(depth_engine):
    """Verifies negative obstacle indicator (ditches/drops) and depth quality scoring."""
    # Scenario 1 (clear path) vs Scenario 4 (depth degradation / dropouts)
    _, d_clean, _ = make_realistic_outdoor_scene(0, 15, "scenario_1_open_path")
    _, d_degraded, _ = make_realistic_outdoor_scene(0, 15, "scenario_4_depth_degradation")

    res_clean = depth_engine.process_depth(d_clean)
    res_degraded = depth_engine.process_depth(d_degraded)

    # Clean frame should have significantly higher depth quality score than degraded frame
    assert res_clean.confidence > res_degraded.confidence
    assert res_clean.confidence - res_degraded.confidence > 0.25, "Quality score did not drop under degradation!"
    assert res_degraded.confidence < 0.70, f"Degraded confidence {res_degraded.confidence} too high!"

    # Negative obstacle test: inject a trench (drop of 30 cm)
    d_trench = d_clean.copy()
    d_trench[350:380, 280:360] = 0.0
    res_trench = depth_engine.process_depth(d_trench)
    assert np.sum(res_trench.negative_obstacle_mask) > 0, "Negative obstacle occlusion shadow was not detected!"


def test_real_recorded_depth_files(depth_engine):
    """Verifies pipeline on real recorded outdoor depth files from datasets/processed/."""
    depth_files = glob.glob('datasets/processed/*/depth/frame_*.npy')
    assert len(depth_files) > 0, "No real depth files found in datasets/processed/!"

    # Test sample of real frames
    for filepath in depth_files[:10]:
        depth_m = np.load(filepath)
        result = depth_engine.process_depth(depth_m)

        # Invariant check: NO invalid depth pixel is ever free space
        invalid = ~result.depth_validity_mask
        assert not np.any(invalid & (result.geometric_cost == 0.0)), f"Rule 12 failed on {filepath}!"

        # Check latency is real-time (< 35 ms, > 28 FPS throughput)
        assert result.latency_ms < 35.0, f"Latency {result.latency_ms:.2f} ms exceeds real-time budget!"
