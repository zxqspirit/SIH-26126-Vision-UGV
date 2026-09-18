"""Comprehensive localization integration test harness.

Replays all 5 outdoor scenarios through the upgraded visual odometry engine with
relocalization state tracking, recovery heuristics, and structured logging.

Test Conditions Covered:
  LOC-01: High texture (grass + structures)
  LOC-02: Sudden obstacle (mixed terrain)
  LOC-03: Vegetation boundary (low/high texture transition)
  LOC-04: Depth degradation (lighting change analog)
  LOC-05: Visual degradation (low texture + partial obstruction)
  LOC-06: Camera movement simulation (synthetic jitter on scenario_1)
  LOC-07: Full sequence replay (chained trajectory + metrics)

Measures:
  - Tracking duration (consecutive OK frames before first loss)
  - Tracking loss events per scenario
  - Recovery time (frames from TRACKING_LOST to first OK/RECOVERED)
  - Runtime (mean, p95, p99 latency in ms; FPS)
  - Trajectory error (ATE RMSE via Umeyama alignment)

Does NOT claim physical UGV localization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
from src.interfaces.types import CameraIntrinsics, TrackingStatus, RelocalizationState
from src.localization.visual_odometry import VisualOdometry
from src.localization.localization_logger import LocalizationLogger


@dataclass
class ScenarioMetrics:
    """Metrics collected from a single scenario replay."""
    scenario_name: str
    test_id: str
    test_condition: str
    total_frames: int = 0
    total_distance_m: float = 0.0
    tracking_ok_frames: int = 0
    tracking_degraded_frames: int = 0
    tracking_lost_frames: int = 0
    tracking_duration_before_first_loss: int = 0  # consecutive OK frames before first loss
    tracking_loss_events: int = 0  # number of distinct loss episodes
    recovery_events: int = 0
    recovery_times_frames: List[int] = field(default_factory=list)  # frames per recovery
    mean_recovery_time: float = 0.0
    relocalization_states: Dict[str, int] = field(default_factory=dict)
    mean_inlier_count: float = 0.0
    min_confidence: float = 1.0
    mean_confidence: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    fps: float = 0.0
    ate_rmse_m: float = 0.0
    final_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    status: str = "PASS"


def compute_ate_rmse(estimated_xyz: np.ndarray, reference_xyz: np.ndarray) -> float:
    """Compute ATE RMSE via Umeyama rigid-body alignment."""
    n = len(estimated_xyz)
    if n < 3:
        return 0.0

    mu_est = np.mean(estimated_xyz, axis=0)
    mu_ref = np.mean(reference_xyz, axis=0)
    p = estimated_xyz - mu_est
    q = reference_xyz - mu_ref

    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_ref - R @ mu_est
    aligned = (R @ estimated_xyz.T).T + t
    errors = np.linalg.norm(aligned - reference_xyz, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def apply_synthetic_jitter(rgb: np.ndarray, frame_idx: int, seed: int = 42) -> np.ndarray:
    """Apply deterministic synthetic affine perturbations to simulate camera vibration."""
    rng = np.random.RandomState(seed + frame_idx)
    h, w = rgb.shape[:2]
    # Random translation +-15px, rotation +-3 degrees
    tx = rng.uniform(-15, 15)
    ty = rng.uniform(-15, 15)
    angle = rng.uniform(-3.0, 3.0)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def run_scenario(
    scenario_dir: str,
    test_id: str,
    test_condition: str,
    apply_jitter: bool = False,
    log_dir: str = "logs/localization",
) -> ScenarioMetrics:
    """Run VO on a single scenario and collect metrics."""
    loader = OutdoorDatasetLoader(scenario_dir, target_fps=10.0)
    intrinsics = CameraIntrinsics()
    vo = VisualOdometry(intrinsics)

    scenario_name = os.path.basename(scenario_dir.rstrip("/\\"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{test_id}_{scenario_name}.jsonl")
    logger = LocalizationLogger(log_path=log_path)

    latencies = []
    inliers = []
    confidences = []
    reloc_states: Dict[str, int] = {}
    ok_count = 0
    deg_count = 0
    lost_count = 0

    # Tracking duration / loss tracking
    first_loss_occurred = False
    consecutive_ok_before_first_loss = 0
    loss_episodes = 0
    in_loss_episode = False
    current_loss_duration = 0
    recovery_times = []

    t_start = time.perf_counter()

    for idx, frame in enumerate(loader.iter_frames()):
        rgb = frame.rgb
        if apply_jitter:
            rgb = apply_synthetic_jitter(rgb, idx)

        result = vo.process_frame(rgb, frame.depth_m)
        logger.log_frame(result, frame.timestamp)

        latencies.append(result.latency_ms)
        inliers.append(result.inlier_count)
        confidences.append(result.confidence)

        # Count relocalization states
        rs = result.relocalization_state
        reloc_states[rs] = reloc_states.get(rs, 0) + 1

        if result.tracking_status == TrackingStatus.TRACKING_OK:
            ok_count += 1
            if not first_loss_occurred:
                consecutive_ok_before_first_loss += 1
            if in_loss_episode:
                # Recovery from loss
                recovery_times.append(current_loss_duration)
                current_loss_duration = 0
                in_loss_episode = False
        elif result.tracking_status == TrackingStatus.TRACKING_DEGRADED:
            deg_count += 1
            if not first_loss_occurred:
                consecutive_ok_before_first_loss += 1
            if in_loss_episode:
                recovery_times.append(current_loss_duration)
                current_loss_duration = 0
                in_loss_episode = False
        else:
            lost_count += 1
            if not first_loss_occurred:
                first_loss_occurred = True
            if not in_loss_episode:
                loss_episodes += 1
                in_loss_episode = True
                current_loss_duration = 1
            else:
                current_loss_duration += 1

    total_time = time.perf_counter() - t_start
    total_frames = len(latencies)
    logger.close()

    # Compute trajectory distance
    traj = np.array(vo.trajectory_history)
    total_dist = 0.0
    if len(traj) > 1:
        diffs = np.diff(traj[:, :2], axis=0)
        total_dist = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # Compute ATE against synthetic reference
    est_xyz = np.array([[p[0], p[1], 0.0] for p in vo.trajectory_history])
    ref_xyz = np.zeros_like(est_xyz)
    for i in range(len(ref_xyz)):
        ref_xyz[i, 0] = i * 0.15  # nominal forward progress
    ate_rmse = compute_ate_rmse(est_xyz, ref_xyz)

    mean_recovery = float(np.mean(recovery_times)) if recovery_times else 0.0

    metrics = ScenarioMetrics(
        scenario_name=scenario_name,
        test_id=test_id,
        test_condition=test_condition,
        total_frames=total_frames,
        total_distance_m=round(total_dist, 4),
        tracking_ok_frames=ok_count,
        tracking_degraded_frames=deg_count,
        tracking_lost_frames=lost_count,
        tracking_duration_before_first_loss=consecutive_ok_before_first_loss,
        tracking_loss_events=loss_episodes,
        recovery_events=len(recovery_times),
        recovery_times_frames=recovery_times,
        mean_recovery_time=round(mean_recovery, 2),
        relocalization_states=reloc_states,
        mean_inlier_count=round(float(np.mean(inliers)), 1),
        min_confidence=round(min(confidences), 4) if confidences else 0.0,
        mean_confidence=round(float(np.mean(confidences)), 4) if confidences else 0.0,
        mean_latency_ms=round(float(np.mean(latencies)), 2),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
        p99_latency_ms=round(float(np.percentile(latencies, 99)), 2),
        fps=round(total_frames / total_time, 1) if total_time > 0 else 0.0,
        ate_rmse_m=round(ate_rmse, 4),
        final_position=(round(vo.global_x, 4), round(vo.global_y, 4), round(vo.global_yaw, 4)),
        status="PASS" if lost_count == 0 else ("DEGRADED" if loss_episodes <= 2 else "FAIL"),
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Localization Integration Test Harness")
    parser.add_argument("--dataset-root", default="datasets/processed")
    parser.add_argument("--output-json", default="docs/research/localization_integration_results.json")
    parser.add_argument("--log-dir", default="logs/localization")
    args = parser.parse_args()

    test_cases = [
        ("LOC-01", "scenario_1_open_path", "High texture (grass + structures)", False),
        ("LOC-02", "scenario_2_sudden_obstacle", "Sudden obstacle (mixed terrain)", False),
        ("LOC-03", "scenario_3_terrain_boundary", "Vegetation boundary (texture transition)", False),
        ("LOC-04", "scenario_4_depth_degradation", "Depth degradation (lighting change analog)", False),
        ("LOC-05", "scenario_5_visual_degradation", "Visual degradation (low texture + obstruction)", False),
        ("LOC-06", "scenario_1_open_path", "Camera vibration (synthetic jitter)", True),
    ]

    print("=" * 90)
    print("SIH 26126: LOCALIZATION INTEGRATION TEST HARNESS (GPS-DENIED)")
    print("=" * 90)

    all_results = []

    for test_id, scenario, condition, jitter in test_cases:
        sc_dir = os.path.join(args.dataset_root, scenario)
        if not os.path.isdir(sc_dir):
            print(f"[SKIP] {test_id}: {sc_dir} not found")
            continue

        metrics = run_scenario(sc_dir, test_id, condition, apply_jitter=jitter, log_dir=args.log_dir)
        all_results.append(metrics)

        status_icon = {"PASS": "OK", "DEGRADED": "WARN", "FAIL": "FAIL"}[metrics.status]
        print(
            f"[{status_icon:4s}] {test_id}: {metrics.test_condition:<45s} | "
            f"OK={metrics.tracking_ok_frames:2d} DEG={metrics.tracking_degraded_frames:1d} LOST={metrics.tracking_lost_frames:2d} | "
            f"LossEp={metrics.tracking_loss_events} Recov={metrics.recovery_events} | "
            f"Lat={metrics.mean_latency_ms:4.1f}ms ({metrics.fps:5.1f} FPS) | "
            f"Reloc: {list(metrics.relocalization_states.keys())}"
        )

    # LOC-07: Chained full-sequence metrics
    print("-" * 90)
    total_ok = sum(m.tracking_ok_frames for m in all_results)
    total_lost = sum(m.tracking_lost_frames for m in all_results)
    total_frames = sum(m.total_frames for m in all_results)
    total_recovery = sum(m.recovery_events for m in all_results)
    avg_latency = float(np.mean([m.mean_latency_ms for m in all_results])) if all_results else 0.0
    print(
        f"[TOTAL] LOC-07: Full sequence replay                             | "
        f"Frames={total_frames} OK={total_ok} LOST={total_lost} | "
        f"Recoveries={total_recovery} | "
        f"Avg Lat={avg_latency:.1f}ms"
    )

    # Save results
    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    serializable = []
    for m in all_results:
        d = asdict(m)
        d["final_position"] = list(d["final_position"])
        serializable.append(d)

    with open(args.output_json, "w") as f:
        json.dump(serializable, f, indent=2)

    print("=" * 90)
    print(f"Results written to: {args.output_json}")
    print(f"Per-frame logs in: {args.log_dir}/")
    print("=" * 90)


if __name__ == "__main__":
    main()
