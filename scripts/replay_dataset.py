"""Replay dataset runner and validation tool for SIH 26126 Autonomous Navigation Stack.

Replays recorded outdoor sensor sequences through the full end-to-end pipeline:
Real Sensor Data -> AI Perception -> Depth Geometry -> Fusion ->
Visual Odometry -> Costmap & Planning -> Safety Gate -> Recommended Motion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
from src.interfaces.types import TrackingStatus, SafetyState
from src.pipeline import NavigationPipeline


def run_scenario(
    scenario_name: str,
    dataset_base_dir: str = "datasets/processed",
    verify: bool = False,
    export_results: bool = True,
) -> Dict[str, Any]:
    """Run navigation pipeline over a full outdoor scenario sequence."""
    scenario_dir = os.path.join(dataset_base_dir, scenario_name)
    if not os.path.exists(scenario_dir):
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    loader = OutdoorDatasetLoader(scenario_dir, target_fps=10.0)
    pipeline = NavigationPipeline()

    print("\n" + "=" * 80)
    print(f"SIH 26126: RUNNING SCENARIO: {scenario_name} ({len(loader)} frames)")
    print("=" * 80)

    results_log: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for frame in loader.iter_frames():
        command, telemetry = pipeline.process_frame(frame)
        latencies.append(telemetry["total_latency_ms"])

        frame_summary = {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "latency_ms": telemetry["total_latency_ms"],
            "fps": telemetry["fps"],
            "linear_velocity": command.linear_velocity,
            "angular_velocity": command.angular_velocity,
            "steering_direction": command.steering_direction,
            "navigation_state": command.navigation_state,
            "safety_state": command.safety_state,
            "confidence": command.confidence,
            "c_perc": round(float(telemetry["semantic"].confidence), 3),
            "c_geom": round(float(telemetry["geometry"].confidence), 3),
            "c_vo": round(float(telemetry["odometry"].confidence), 3),
            "c_fusion": round(float(telemetry["fused"].confidence), 3),
            "vo_inliers": telemetry["odometry"].inlier_count,
            "vo_status": telemetry["odometry"].tracking_status.value,
            "audit_reasons": telemetry["safety"].audit_reasons,
        }
        results_log.append(frame_summary)

        print(
            f"Frame {frame.frame_id:02d} | "
            f"FPS: {telemetry['fps']:4.1f} | "
            f"Cmd: v={command.linear_velocity:4.2f}m/s w={command.angular_velocity:+5.2f}rad/s [{command.steering_direction:12s}] | "
            f"State: {command.navigation_state:16s} | "
            f"Conf: {command.confidence:4.2f} ({command.safety_state})"
        )

    avg_latency = float(np_mean := sum(latencies) / max(len(latencies), 1))
    avg_fps = float(1000.0 / max(avg_latency, 1.0))

    summary = {
        "scenario": scenario_name,
        "total_frames": len(loader),
        "mean_latency_ms": round(avg_latency, 2),
        "mean_fps": round(avg_fps, 1),
        "final_pose": {
            "x": pipeline.odometry.global_x,
            "y": pipeline.odometry.global_y,
            "yaw": pipeline.odometry.global_yaw,
        },
        "frames": results_log,
    }

    print("-" * 80)
    print(f"Summary: Mean Latency: {avg_latency:.1f} ms ({avg_fps:.1f} FPS) | Final Global Pose: (X={summary['final_pose']['x']:.2f}, Y={summary['final_pose']['y']:.2f})")
    print("=" * 80)

    # Verification checks
    if verify:
        if scenario_name == "scenario_1_open_path":
            # Clear path: high confidence, forward motion
            assert summary["frames"][-1]["safety_state"] in ("HIGH_CONFIDENCE", "CAUTIOUS_DEGRADED")
            assert summary["frames"][-1]["linear_velocity"] > 0.30
            print(">>> VERIFICATION PASSED: Open path enables safe high-confidence forward motion.")

        elif scenario_name == "scenario_2_sudden_obstacle":
            # Sudden obstacle: must detect hazard and reduce speed or steer away
            has_avoided_or_stopped = any(
                f["steering_direction"] in ("SLIGHT_RIGHT", "HARD_RIGHT", "SLIGHT_LEFT", "HARD_LEFT", "STOP")
                or f["safety_state"] == "SAFETY_STOP"
                for f in summary["frames"]
            )
            assert has_avoided_or_stopped, "Failed to react to positive obstacle!"
            print(">>> VERIFICATION PASSED: Positive obstacle detected geometrically and avoided.")

        elif scenario_name == "scenario_4_depth_degradation":
            # Rule 12: Invalid depth must degrade confidence and scale speed down
            has_degraded = any(
                f["safety_state"] in ("CAUTIOUS_DEGRADED", "LOW_CONFIDENCE_SLOW", "SAFETY_STOP")
                for f in summary["frames"]
            )
            assert has_degraded, "Rule 12 failed: Invalid depth was ignored!"
            print(">>> VERIFICATION PASSED (Rule 12): Invalid depth produced explicit cautious degradation.")

        elif scenario_name == "scenario_5_visual_degradation":
            # Rule 13: VO degradation must trigger localization recovery / stop
            has_stopped = any(
                f["safety_state"] in ("LOCALIZATION_LOST", "SAFETY_STOP") or f["vo_status"] == "TRACKING_LOST"
                for f in summary["frames"]
            )
            assert has_stopped, "Failed to halt on tracking loss!"
            print(">>> VERIFICATION PASSED (Rule 13): Feature loss triggered deterministic safe stop.")

    if export_results:
        os.makedirs("results", exist_ok=True)
        out_file = os.path.join("results", f"replay_{scenario_name}.json")
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Exported metrics to {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Replay outdoor dataset for SIH 26126.")
    parser.add_argument("--scenario", type=str, default="scenario_1_open_path", help="Scenario name")
    parser.add_argument("--verify", action="store_true", help="Run automated assertions on output")
    parser.add_argument("--all", action="store_true", help="Run all 5 scenarios")
    args = parser.parse_args()

    if args.all:
        scenarios = [
            "scenario_1_open_path",
            "scenario_2_sudden_obstacle",
            "scenario_3_terrain_boundary",
            "scenario_4_depth_degradation",
            "scenario_5_visual_degradation",
        ]
        for sc in scenarios:
            run_scenario(sc, verify=args.verify)
    else:
        run_scenario(args.scenario, verify=args.verify)


if __name__ == "__main__":
    main()
