#!/usr/bin/env python3
"""Benchmark UGV 10-stage pipeline performance and latency.

Measures:
1. input
2. preprocessing
3. cnn
4. depth
5. fusion
6. slam
7. planning
8. safety
9. visualization
10. end-to-end latency
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
from src.interfaces.types import GoalPose
from src.pipeline import NavigationPipeline
from src.visualization.dashboard_server import serialize_frame_telemetry


def run_benchmark(scenario_name: str = "scenario_1_open_path", max_frames: int = 15, output_json: str = None):
    loader = OutdoorDatasetLoader(f"datasets/processed/{scenario_name}")
    pipeline = NavigationPipeline()
    pipeline.reset()
    goal = GoalPose(x=8.0, y=0.0, yaw=0.0)

    stages = ["input", "preprocessing", "cnn", "depth", "fusion", "slam", "planning", "safety", "visualization", "e2e"]
    timings = {k: [] for k in stages}

    # Warmup 2 frames
    warmup_n = min(2, len(loader))
    for i in range(warmup_n):
        f = loader.get_frame(i)
        c, t = pipeline.process_frame(f, goal=goal)
        _ = serialize_frame_telemetry(f, c, t, pipeline, i)

    # Benchmark loop
    limit = min(len(loader), max_frames)
    for i in range(limit):
        t_start = time.perf_counter()

        # 1. Input
        t0 = time.perf_counter()
        frame = loader.get_frame(i)
        t_input = (time.perf_counter() - t0) * 1000.0
        timings["input"].append(t_input)

        # 2. Preprocessing
        t0 = time.perf_counter()
        _ = frame.rgb.astype(np.float32) / 255.0
        _ = np.nan_to_num(frame.depth_m, nan=0.0, posinf=15.0, neginf=0.0)
        t_prep = (time.perf_counter() - t0) * 1000.0
        timings["preprocessing"].append(t_prep)

        # 3. CNN
        t0 = time.perf_counter()
        sem = pipeline.perception.process_frame(frame)
        t_cnn = (time.perf_counter() - t0) * 1000.0
        timings["cnn"].append(t_cnn)

        # 4. Depth
        t0 = time.perf_counter()
        geom = pipeline.geometry.process_depth(frame.depth_m)
        t_depth = (time.perf_counter() - t0) * 1000.0
        timings["depth"].append(t_depth)

        # 5. Fusion (Optimization 3: Re-use cached points_base)
        t0 = time.perf_counter()
        pts_base = getattr(geom, "points_base", None)
        if pts_base is None:
            pts_opt, _ = pipeline.geometry.projector.project_to_camera_frame(frame.depth_m)
            pts_base = pipeline.geometry.projector.transform_to_base_link(pts_opt)
        fused = pipeline.fusion.fuse(sem, geom, pts_base)
        t_fuse = (time.perf_counter() - t0) * 1000.0
        timings["fusion"].append(t_fuse)

        # 6. SLAM
        t0 = time.perf_counter()
        odom = pipeline.odometry.process_frame(frame.rgb, frame.depth_m)
        t_slam = (time.perf_counter() - t0) * 1000.0
        timings["slam"].append(t_slam)

        trav_map = pipeline.traversability.process(fused=fused, semantic=sem, geometry=geom, odometry=odom)
        pipeline.costmap.update_from_traversability_result(trav_map)

        # 7. Planning
        t0 = time.perf_counter()
        dec = pipeline.decision_engine.decide(
            costmap=pipeline.costmap,
            traversability=trav_map,
            odometry=odom,
            goal=goal,
            current_v=pipeline.last_command.linear_velocity,
            current_w=pipeline.last_command.angular_velocity,
        )
        t_plan = (time.perf_counter() - t0) * 1000.0
        timings["planning"].append(t_plan)

        planning = pipeline.planner.plan(pipeline.costmap, (0.0, 0.0, 0.0), (dec.recommended_linear_velocity, dec.recommended_angular_velocity))

        # 8. Safety
        t0 = time.perf_counter()
        min_obstacle_dist = pipeline.costmap.get_obstacle_distance(0.0, 0.0)
        disagreement_ratio = float(np.mean(fused.disagreement_mask)) if getattr(fused, "disagreement_mask", None) is not None else 0.0
        safety = pipeline.safety_gate.arbitrate(
            nominal_v=dec.recommended_linear_velocity,
            nominal_w=dec.recommended_angular_velocity,
            c_perc=sem.confidence,
            c_geom=geom.confidence,
            c_vo=odom.confidence,
            c_fusion=fused.confidence,
            disagreement_ratio=disagreement_ratio,
            tracking_status=odom.tracking_status,
            min_obstacle_dist_m=min_obstacle_dist,
            timestamp=frame.timestamp,
        )
        t_safe = (time.perf_counter() - t0) * 1000.0
        timings["safety"].append(t_safe)

        cmd = pipeline.command_generator.generate(
            planning_result=planning,
            safety_result=safety,
            timestamp=frame.timestamp,
        )
        pipeline.last_command = cmd

        telemetry = {
            "semantic": sem, "geometry": geom, "fused": fused, "odometry": odom,
            "traversability": trav_map, "decision": dec, "safety": safety,
            "planning": planning, "fps": 10.0, "total_latency_ms": 100.0,
        }

        # 9. Visualization
        t0 = time.perf_counter()
        _ = serialize_frame_telemetry(frame, cmd, telemetry, pipeline, i)
        t_vis = (time.perf_counter() - t0) * 1000.0
        timings["visualization"].append(t_vis)

        # 10. End-to-end
        t_e2e = (time.perf_counter() - t_start) * 1000.0
        timings["e2e"].append(t_e2e)

    stage_keys = ["input", "preprocessing", "cnn", "depth", "fusion", "slam", "planning", "safety", "visualization"]
    total_staged = sum(float(np.mean(timings[k])) for k in stage_keys)
    e2e_mean = float(np.mean(timings["e2e"]))

    results = {}
    print(f"\n{'=' * 75}")
    print(f"     10-STAGE LATENCY PROFILING REPORT ({scenario_name}, {limit} frames)")
    print(f"{'=' * 75}")
    print(f"{'Stage':<16} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Share (%)':<10}")
    print("-" * 75)
    for k in stage_keys + ["e2e"]:
        arr = np.array(timings[k])
        mean_val = float(np.mean(arr))
        p95_val = float(np.percentile(arr, 95))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        share = (mean_val / total_staged) * 100.0 if k != "e2e" else 100.0

        results[k] = {
            "mean_ms": round(mean_val, 2),
            "p95_ms": round(p95_val, 2),
            "min_ms": round(min_val, 2),
            "max_ms": round(max_val, 2),
            "share_pct": round(share, 1),
        }
        print(f"{k:<16} | {mean_val:<10.2f} | {p95_val:<10.2f} | {min_val:<10.2f} | {max_val:<10.2f} | {share:<10.1f}")
    print("=" * 75)
    print(f"Total Staged Latency: {total_staged:.2f} ms | Measured E2E: {e2e_mean:.2f} ms | Effective FPS: {1000.0/e2e_mean:.1f}")

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        payload = {
            "scenario": scenario_name,
            "frames": limit,
            "total_staged_ms": round(total_staged, 2),
            "e2e_mean_ms": round(e2e_mean, 2),
            "fps": round(1000.0 / e2e_mean, 1),
            "results": results,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved benchmark results to {output_json}")

    return results, e2e_mean, 1000.0 / e2e_mean


def main():
    parser = argparse.ArgumentParser(description="Profile 10 pipeline stages on real data.")
    parser.add_argument("--scenario", type=str, default="scenario_1_open_path", help="Scenario name")
    parser.add_argument("--frames", type=int, default=15, help="Number of frames to benchmark")
    parser.add_argument("--output", type=str, default="experiments/performance/latest_benchmark.json", help="Output JSON path")
    args = parser.parse_args()

    run_benchmark(scenario_name=args.scenario, max_frames=args.frames, output_json=args.output)


if __name__ == "__main__":
    main()
