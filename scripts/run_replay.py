#!/usr/bin/env python3
"""
scripts/run_replay.py
Standalone Replay Execution Script for SIH 26126.

Workflow:
recorded data -> replay -> perception -> depth -> fusion -> localization -> navigation -> safety -> metrics

Usage:
    python scripts/run_replay.py --scenario scenario_1_open_path [--limit 10] [--output replay_output.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
from src.pipeline import NavigationPipeline


def run_replay(
    scenario: str,
    dataset_root: str = "datasets/processed",
    frame_limit: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute frame-by-frame replay through the full navigation pipeline."""
    scenario_path = os.path.join(dataset_root, scenario)
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario dataset not found at: {scenario_path}")

    loader = OutdoorDatasetLoader(scenario_path)
    pipeline = NavigationPipeline()

    # Apply configuration overrides if provided
    if config:
        if "safety" in config:
            s_cfg = config["safety"]
            for k in ["th_high", "th_med", "th_low", "emergency_dist", "v_cautious_max", "v_crawl_max"]:
                if k in s_cfg and hasattr(pipeline.safety_gate, k):
                    setattr(pipeline.safety_gate, k, s_cfg[k])
        if "navigation" in config:
            n_cfg = config["navigation"]
            for k in ["max_v", "max_w", "min_v"]:
                if k in n_cfg and hasattr(pipeline.decision_engine, k):
                    setattr(pipeline.decision_engine, k, n_cfg[k])

    total_frames = len(loader) if frame_limit is None else min(frame_limit, len(loader))
    if verbose:
        print(f"=== Replaying '{scenario}' ({total_frames} frames) ===")
        print(f"Frame | Commanded (v, w) | Steering     | Safety State | C_total | Clearance | Latency")
        print("-" * 80)

    records: List[Dict[str, Any]] = []
    latencies: List[float] = []
    start_time = time.perf_counter()

    for idx in range(total_frames):
        frame = loader.get_frame(idx)
        t0 = time.perf_counter()
        cmd, tele = pipeline.process_frame(frame)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        safety = tele["safety"]
        decision = tele["decision"]
        odometry = tele["odometry"]

        rec = {
            "frame_id": idx,
            "timestamp": frame.timestamp,
            "latency_ms": round(lat_ms, 2),
            "command": {
                "linear_velocity": round(cmd.linear_velocity, 4),
                "angular_velocity": round(cmd.angular_velocity, 4),
                "steering_direction": cmd.steering_direction,
                "navigation_state": cmd.navigation_state,
            },
            "safety": {
                "state": safety.safety_state.value if hasattr(safety.safety_state, "value") else str(safety.safety_state),
                "confidence": round(safety.overall_confidence, 4),
                "speed_scale_factor": round(safety.speed_scale_factor, 4),
                "is_emergency_stop": safety.is_emergency_stop,
                "primary_reason": safety.audit_reasons[0] if safety.audit_reasons else "Nominal",
            },
            "planning": {
                "status": decision.status,
                "min_clearance_m": round(decision.min_clearance_m, 4) if decision.min_clearance_m is not None else None,
                "recommended_steering": str(decision.recommended_steering),
            },
            "odometry": {
                "tracking_status": odometry.tracking_status.value if hasattr(odometry.tracking_status, "value") else str(odometry.tracking_status),
                "confidence": round(odometry.confidence, 4),
                "inliers": odometry.inlier_count,
            }
        }
        records.append(rec)

        if verbose:
            print(
                f"{idx:05d} | v={cmd.linear_velocity:5.2f} w={cmd.angular_velocity:+5.2f} | "
                f"{cmd.steering_direction:12s} | {rec['safety']['state']:12s} | "
                f"{rec['safety']['confidence']:7.2f} | {rec['planning']['min_clearance_m'] or 0.0:6.2f}m   | {lat_ms:6.1f} ms"
            )

    total_wall_time = time.perf_counter() - start_time
    fps = total_frames / total_wall_time if total_wall_time > 0 else 0.0

    summary = {
        "scenario": scenario,
        "total_frames": total_frames,
        "total_wall_time_s": round(total_wall_time, 3),
        "mean_fps": round(fps, 2),
        "mean_latency_ms": round(float(sum(latencies) / len(latencies)), 2) if latencies else 0.0,
        "frames": records,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Replay outdoor dataset sequence through navigation stack.")
    parser.add_argument("--scenario", required=True, help="Scenario folder name (e.g. scenario_1_open_path)")
    parser.add_argument("--dataset-root", default="datasets/processed", help="Root directory containing scenarios")
    parser.add_argument("--limit", type=int, default=None, help="Optional max frames to process")
    parser.add_argument("--output", default=None, help="Path to write JSON replay telemetry output")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-frame printout")
    args = parser.parse_args()

    result = run_replay(
        scenario=args.scenario,
        dataset_root=args.dataset_root,
        frame_limit=args.limit,
        verbose=not args.quiet,
    )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nReplay telemetry saved to: {args.output}")


if __name__ == "__main__":
    main()
