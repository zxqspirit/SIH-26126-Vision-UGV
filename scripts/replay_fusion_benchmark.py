#!/usr/bin/env python3
"""Replay fusion benchmark for SIH 26126 Autonomous Navigation Stack.

Replays all 75 recorded outdoor sensor frames across 5 scenarios through the
Multimodal Fusion Engine, validating:
- Real-time latency (< 15 ms, > 60 FPS)
- Disagreement rate and veto activations
- Obstacle extraction in 2D BEV costmap
- Rule 11 & Rule 12 non-zero unknown safety compliance
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import Dict, Any, List
import numpy as np

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
from src.interfaces.types import CameraIntrinsics
from src.perception.perception_engine import PerceptionEngine
from src.depth_geometry.depth_geometry_engine import DepthGeometryEngine
from src.fusion.fusion_engine import FusionEngine


def run_fusion_replay():
    print("=" * 80)
    print("      SIH 26126 - MULTIMODAL FUSION FULL REPLAY BENCHMARK (75 FRAMES)      ")
    print("=" * 80)

    intrinsics = CameraIntrinsics()
    perception = PerceptionEngine()
    geometry = DepthGeometryEngine(intrinsics)
    fusion = FusionEngine(intrinsics)

    scenarios = sorted(glob.glob("datasets/processed/*"))

    overall_latencies = []
    total_rule11_violations = 0
    total_rule12_violations = 0
    total_disagreements = 0

    print(f"{'Scenario':<32} | {'Frames':<6} | {'Avg Lat':<10} | {'FPS':<6} | {'Conf':<6} | {'Disag%':<8} | {'ObsCells':<9} | {'Vio':<5}")
    print("-" * 95)

    for sc in scenarios:
        sc_name = os.path.basename(sc)
        loader = OutdoorDatasetLoader(sc, target_fps=10.0)

        latencies = []
        confs = []
        disags = []
        obs_counts = []
        vios = 0

        for frame in loader.iter_frames():
            sem = perception.process_frame(frame)
            geo = geometry.process_depth(frame.depth_m)
            points_base = geo.points_3d

            # Run fusion
            clean_depth = np.where(geo.depth_validity_mask, frame.depth_m, np.nan)
            pts_base, _ = geometry.projector.project_and_transform_to_base_link(clean_depth)

            fused_res = fusion.fuse(sem, geo, pts_base)

            latencies.append(fused_res.latency_ms)
            confs.append(fused_res.confidence)
            disags.append(float(np.mean(fused_res.disagreement_mask)))
            obs_counts.append(int(np.sum(fused_res.obstacle_mask)))

            # Check Rule 11: unobserved cells must never be 0 (free space)
            unobs = fused_res.unknown_mask
            if np.any(unobs & (fused_res.fused_costmap == 0)):
                vios += 1
                total_rule11_violations += 1

            # Check Rule 12: invalid depth must never be 1.0 traversability
            inv_d = ~geo.depth_validity_mask
            if fused_res.fused_traversability is not None:
                if np.any(inv_d & (fused_res.fused_traversability > 0.50)):
                    vios += 1
                    total_rule12_violations += 1

        overall_latencies.extend(latencies)
        avg_lat = float(np.mean(latencies))
        fps = 1000.0 / max(avg_lat, 0.001)
        avg_conf = float(np.mean(confs))
        avg_disag = float(np.mean(disags)) * 100.0
        avg_obs = int(np.mean(obs_counts))

        print(f"{sc_name:<32} | {len(latencies):<6} | {avg_lat:<8.2f}ms | {fps:<6.1f} | {avg_conf:<6.2f} | {avg_disag:<7.2f}% | {avg_obs:<9} | {vios:<5}")

    print("-" * 95)
    mean_lat = float(np.mean(overall_latencies))
    mean_fps = 1000.0 / max(mean_lat, 0.001)
    print(f"Overall Fusion Engine Performance:")
    print(f"  - Mean Total Latency:       {mean_lat:.2f} ms ({mean_fps:.1f} FPS)")
    print(f"  - Real-Time Target (<15ms): {'[PASSED]' if mean_lat < 15.0 else '[EXCEEDED]'}")
    print(f"  - Rule 11 Violations:       {total_rule11_violations} (Zero unobserved cells marked free)")
    print(f"  - Rule 12 Violations:       {total_rule12_violations} (Zero invalid depth marked free)")
    print("=" * 80)


if __name__ == "__main__":
    run_fusion_replay()
