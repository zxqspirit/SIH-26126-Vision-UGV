#!/usr/bin/env python3
"""Comprehensive test and validation script for Depth Geometry Engine.

Tests:
1. Flat Ground: ground estimation, planar residual tolerance, low corridor cost.
2. Obstacle: positive obstacle detection (>15cm step), lethal cost assignment (1.0).
3. Invalid Depth: CRITICAL RULE 12 (invalid depth must NEVER become free space).
4. Depth Discontinuity: sharp height gradient step edges / cliffs.
5. Negative Obstacles: ditches and occlusion shadow detection.
6. Real Recorded Data Benchmark: 75 frames across 5 outdoor scenarios.
"""

from __future__ import annotations

import glob
import os
import sys
import time
import numpy as np

# Ensure repository root is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.interfaces.types import CameraIntrinsics
from src.depth_geometry.depth_geometry_engine import DepthGeometryEngine
from scripts.generate_sample_scenarios import make_realistic_outdoor_scene


def run_all_tests() -> bool:
    print("=" * 80)
    print("      SIH 26126 - 3D DEPTH GEOMETRY COMPREHENSIVE VERIFICATION SUITE       ")
    print("=" * 80)

    intrinsics = CameraIntrinsics(
        fx=385.0, fy=385.0, cx=320.0, cy=240.0,
        width=640, height=480,
        camera_height_m=0.45, camera_pitch_rad=0.209
    )
    engine = DepthGeometryEngine(intrinsics)

    all_passed = True

    # -------------------------------------------------------------------------
    # TEST 1: Flat Ground
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Flat Ground Verification...")
    _, d_flat, _ = make_realistic_outdoor_scene(0, 15, "scenario_1_open_path")
    res_flat = engine.process_depth(d_flat)

    # 1. Zero positive obstacle false positives
    pos_obs_cnt = int(np.sum(res_flat.positive_obstacle_mask))
    corridor_valid = res_flat.depth_validity_mask[300:440, 240:400]
    corridor_cost = float(np.mean(res_flat.geometric_cost[300:440, 240:400][corridor_valid]))
    slope_deg = 0.0
    if res_flat.plane_coeffs:
        a, b, _, _ = res_flat.plane_coeffs
        slope_deg = float(np.degrees(np.arctan(np.sqrt(a**2 + b**2))))

    t1_pass = (pos_obs_cnt == 0) and (corridor_cost < 0.10) and (slope_deg < 5.0) and (res_flat.confidence >= 0.70)
    print(f"  - Positive Obstacle False Positives: {pos_obs_cnt} px (Expected: 0)")
    print(f"  - Corridor Mean Traversability Cost: {corridor_cost:.4f} (Expected: < 0.10)")
    print(f"  - Ground Plane Slope: {slope_deg:.2f} deg (Expected: < 5.0 deg)")
    print(f"  - Depth Quality Confidence: {res_flat.confidence:.3f} (Expected: >= 0.70)")
    print(f"  -> Result: {'[PASS]' if t1_pass else '[FAIL]'}")
    all_passed = all_passed and t1_pass

    # -------------------------------------------------------------------------
    # TEST 2: Obstacle Extraction
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Obstacle Extraction Verification...")
    _, d_obs, _ = make_realistic_outdoor_scene(0, 15, "scenario_2_sudden_obstacle")
    res_obs = engine.process_depth(d_obs)

    obs_pixels = int(np.sum(res_obs.positive_obstacle_mask))
    obs_costs = res_obs.geometric_cost[res_obs.positive_obstacle_mask]
    max_cost = float(np.max(obs_costs)) if len(obs_costs) > 0 else 0.0
    mean_cost = float(np.mean(obs_costs)) if len(obs_costs) > 0 else 0.0
    pts_count = len(res_obs.points_3d) if res_obs.points_3d is not None else 0

    t2_pass = (obs_pixels > 100) and (max_cost == 1.0) and (mean_cost >= 0.50) and (pts_count > 0)
    print(f"  - Obstacle Region Pixels: {obs_pixels} px (Expected: > 100)")
    print(f"  - Max Obstacle Cost: {max_cost:.2f} (Expected: 1.00)")
    print(f"  - Mean Obstacle Hazard: {mean_cost:.3f} (Expected: >= 0.50)")
    print(f"  - Subsampled 3D Pointcloud Size: {pts_count} points")
    print(f"  -> Result: {'[PASS]' if t2_pass else '[FAIL]'}")
    all_passed = all_passed and t2_pass

    # -------------------------------------------------------------------------
    # TEST 3: Invalid Depth Safeguard (CRITICAL RULE 12)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Invalid Depth Safeguard (Rule 12 Invariant)...")
    d_corrupt = d_flat.copy()
    d_corrupt[200:250, 200:300] = 0.0
    d_corrupt[260:300, 200:300] = np.nan
    d_corrupt[310:350, 200:300] = np.inf
    d_corrupt[360:400, 200:300] = 0.10  # Below 0.35m
    d_corrupt[410:450, 200:300] = 20.0  # Above 12.0m

    res_corrupt = engine.process_depth(d_corrupt)
    invalid_mask = ~res_corrupt.depth_validity_mask
    free_space_violations = invalid_mask & (res_corrupt.geometric_cost == 0.0)
    num_violations = int(np.sum(free_space_violations))
    invalid_costs = res_corrupt.geometric_cost[invalid_mask]
    min_penalty = float(np.min(invalid_costs))

    t3_pass = (num_violations == 0) and (min_penalty >= 0.60)
    print(f"  - Corrupted Invalid Pixels In Frame: {int(np.sum(invalid_mask))}")
    print(f"  - Invalid Treated As Free Space (Cost == 0.0): {num_violations} violations (Expected: 0)")
    print(f"  - Minimum Uncertainty Penalty on Invalid: {min_penalty:.2f} (Expected: >= 0.60)")
    print(f"  -> Result: {'[PASS]' if t3_pass else '[FAIL]'}")
    all_passed = all_passed and t3_pass

    # -------------------------------------------------------------------------
    # TEST 4: Depth Discontinuity Detection
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Depth Discontinuity Detection...")
    h, w = 480, 640
    d_step = np.full((h, w), 2.5, dtype=np.float32)
    d_step[280:350, 200:440] = 1.8  # 0.7m sharp depth step curb

    res_step = engine.process_depth(d_step)
    disc_count = int(np.sum(res_step.discontinuity_mask)) if res_step.discontinuity_mask is not None else 0
    disc_costs = res_step.geometric_cost[res_step.discontinuity_mask] if disc_count > 0 else np.array([0.0])
    min_disc_cost = float(np.min(disc_costs))

    t4_pass = (disc_count > 20) and (min_disc_cost >= 0.85)
    print(f"  - Discontinuity Edge Points Detected: {disc_count} px (Expected: > 20)")
    print(f"  - Discontinuity Minimum Hazard Cost: {min_disc_cost:.2f} (Expected: >= 0.85)")
    print(f"  -> Result: {'[PASS]' if t4_pass else '[FAIL]'}")
    all_passed = all_passed and t4_pass

    # -------------------------------------------------------------------------
    # TEST 5: Negative Obstacles & Occlusion Shadows
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Negative Obstacle & Quality Metric...")
    _, d_deg, _ = make_realistic_outdoor_scene(0, 15, "scenario_4_depth_degradation")
    res_deg = engine.process_depth(d_deg)

    d_trench = d_flat.copy()
    d_trench[350:380, 280:360] = 0.0
    res_trench = engine.process_depth(d_trench)
    neg_obs_count = int(np.sum(res_trench.negative_obstacle_mask))

    t5_pass = (res_flat.confidence > res_deg.confidence) and (res_deg.confidence < 0.70) and (neg_obs_count > 0)
    print(f"  - Pristine Depth Quality: {res_flat.confidence:.3f}")
    print(f"  - Degraded Depth Quality: {res_deg.confidence:.3f}")
    print(f"  - Quality Drop: {res_flat.confidence - res_deg.confidence:.3f} (Significant degradation flagged)")
    print(f"  - Negative Obstacle Pixels (Occlusion Shadow): {neg_obs_count} px (Expected: > 0)")
    print(f"  -> Result: {'[PASS]' if t5_pass else '[FAIL]'}")
    all_passed = all_passed and t5_pass

    # -------------------------------------------------------------------------
    # TEST 6: Real Recorded Outdoor Datasets Benchmark
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Real Recorded Depth Benchmark (75 Frames across 5 Scenarios)...")
    scenarios = glob.glob("datasets/processed/*")
    print(f"{'Scenario':<32} | {'Frames':<6} | {'Avg Lat (ms)':<12} | {'FPS':<6} | {'Quality':<8} | {'Rule 12 Violations':<20}")
    print("-" * 95)

    scenario_latencies = []
    total_rule12_violations = 0

    for sc in sorted(scenarios):
        sc_name = os.path.basename(sc)
        frame_paths = sorted(glob.glob(os.path.join(sc, "depth", "frame_*.npy")))
        if not frame_paths:
            continue

        lats = []
        quals = []
        vios = 0
        for fp in frame_paths:
            depth_arr = np.load(fp)
            res = engine.process_depth(depth_arr)
            lats.append(res.latency_ms)
            quals.append(res.confidence)
            # Rule 12 check
            inv = ~res.depth_validity_mask
            vios += int(np.sum(inv & (res.geometric_cost == 0.0)))

        scenario_latencies.extend(lats)
        total_rule12_violations += vios
        avg_lat = float(np.mean(lats))
        fps = 1000.0 / max(avg_lat, 0.001)
        avg_q = float(np.mean(quals))

        print(f"{sc_name:<32} | {len(lats):<6} | {avg_lat:<12.2f} | {fps:<6.1f} | {avg_q:<8.2f} | {vios:<20}")

    print("-" * 95)
    overall_lat = float(np.mean(scenario_latencies))
    overall_fps = 1000.0 / max(overall_lat, 0.001)
    print(f"Overall Benchmark: Avg Latency = {overall_lat:.2f} ms ({overall_fps:.1f} FPS) | Total Rule 12 Violations = {total_rule12_violations}")

    t6_pass = (overall_lat < 30.0) and (total_rule12_violations == 0)
    print(f"  -> Result: {'[PASS]' if t6_pass else '[FAIL]'}")
    all_passed = all_passed and t6_pass

    print("\n" + "=" * 80)
    print(f"SUMMARY: ALL 6 TESTS {'PASSED [SUCCESS]' if all_passed else 'FAILED [ERROR]'}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
