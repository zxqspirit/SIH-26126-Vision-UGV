"""Optimized sensitivity sweep tool for Traversability Base Costs on DWA Trajectory Selection.

Pre-caches upstream perception/geometry/fusion/odometry once, allowing 35 perturbation
evaluations over 75 frames to complete in ~5-10 seconds total.
"""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.interfaces.types import (
    TerrainClass,
    TraversabilityCostConfig,
    PlanningResult,
    TraversabilityMapResult,
)
from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
from src.pipeline import NavigationPipeline
from src.planning.costmap_2d import Costmap2D
from src.traversability.traversability_map_engine import TraversabilityMapEngine
from src.planning.dwa_planner import DWAPlanner


@dataclass
class SweepPointResult:
    """Evaluation metrics for a single base cost perturbation."""
    terrain_class: int
    terrain_name: str
    multiplier: float
    tested_cost: int
    frames_evaluated: int
    path_found_rate: float
    mean_v: float
    std_v: float
    mean_abs_w: float
    mean_cost: float
    mean_clearance_m: float
    min_clearance_m: float
    trajectory_shift_rate: float
    level_counts_mean: Dict[str, float] = field(default_factory=dict)


@dataclass
class TerrainSensitivitySummary:
    """Summary of sensitivity analysis for one terrain class."""
    terrain_class: int
    terrain_name: str
    nominal_cost: int
    points: List[SweepPointResult]
    cost_sensitivity_index: float
    velocity_sensitivity_index: float
    max_trajectory_shift_rate: float
    safe_operating_range: Tuple[int, int]
    recommendation: str


class TraversabilitySensitivitySweeper:
    """Runs systematic ±50% sensitivity sweeps across outdoor scenarios."""

    SCENARIOS = [
        "scenario_1_open_path",
        "scenario_2_sudden_obstacle",
        "scenario_3_terrain_boundary",
        "scenario_4_depth_degradation",
        "scenario_5_visual_degradation",
    ]

    SWEEP_MULTIPLIERS = [0.50, 0.75, 1.00, 1.25, 1.50]

    def __init__(self, dataset_root: str = "datasets/processed") -> None:
        self.dataset_root = dataset_root
        self.nominal_config = TraversabilityCostConfig()
        self.cached_upstream = self._precompute_upstream()

    @property
    def cached_frames(self):
        return self.cached_upstream

    def _precompute_upstream(self) -> List[Dict[str, Any]]:
        """Precompute and cache upstream outputs once across all 75 frames."""
        print(f"Pre-caching upstream outputs across {len(self.SCENARIOS)} scenarios...", flush=True)
        pipeline = NavigationPipeline()
        cached = []
        for sc in self.SCENARIOS:
            loader = OutdoorDatasetLoader(f"{self.dataset_root}/{sc}")
            pipeline.odometry.reset()
            for i in range(len(loader)):
                frame = loader.get_frame(i)
                if frame is None:
                    continue

                semantic = pipeline.perception.process_frame(frame)
                geometry = pipeline.geometry.process_depth(frame.depth_m)
                pts_opt, _ = pipeline.geometry.projector.project_to_camera_frame(frame.depth_m)
                pts_base = pipeline.geometry.projector.transform_to_base_link(pts_opt)
                fused = pipeline.fusion.fuse(semantic, geometry, pts_base)
                odometry = pipeline.odometry.process_frame(frame.rgb, frame.depth_m)

                cached.append({
                    "scenario": sc,
                    "frame_id": i,
                    "semantic": semantic,
                    "geometry": geometry,
                    "fused": fused,
                    "odometry": odometry,
                })
        print(f"Pre-cached {len(cached)} frames successfully.", flush=True)
        return cached

    def evaluate_config(
        self,
        config: TraversabilityCostConfig,
        baseline_trajs: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[Dict[str, float], List[Tuple[float, float]], Dict[str, float]]:
        """Evaluate a configuration over the pre-cached upstream outputs."""
        trav_engine = TraversabilityMapEngine(config)
        costmap = Costmap2D()
        planner = DWAPlanner()

        v_list = []
        w_list = []
        cost_list = []
        clr_list = []
        paths_found = 0
        trajs = []
        shifts = 0
        accum_levels: Dict[str, float] = {}

        last_v = 0.0
        last_w = 0.0

        for idx, item in enumerate(self.cached_upstream):
            trav_result = trav_engine.process(
                fused=item["fused"],
                semantic=item["semantic"],
                geometry=item["geometry"],
                odometry=item["odometry"],
            )

            costmap.update_from_traversability_result(trav_result)

            p: PlanningResult = planner.plan(
                costmap=costmap,
                current_v=last_v,
                current_w=last_w,
                target_heading_rad=0.0,
            )

            is_found = (p.status == "PATH_FOUND")
            if is_found:
                paths_found += 1
                v_list.append(p.recommended_linear_velocity)
                w_list.append(abs(p.recommended_angular_velocity))
                if p.selected_trajectory is not None:
                    cost_list.append(p.selected_trajectory.cost)
                    clr_list.append(p.selected_trajectory.clearance_m)
                    curr_traj = (round(p.recommended_linear_velocity, 2), round(p.recommended_angular_velocity, 2))
                else:
                    curr_traj = (0.0, 0.0)
                last_v = p.recommended_linear_velocity
                last_w = p.recommended_angular_velocity
            else:
                v_list.append(0.0)
                w_list.append(0.0)
                curr_traj = (0.0, 0.0)
                last_v = 0.0
                last_w = 0.0

            trajs.append(curr_traj)

            if baseline_trajs is not None:
                base_v, base_w = baseline_trajs[idx]
                if abs(curr_traj[0] - base_v) > 0.05 or abs(curr_traj[1] - base_w) > 0.05:
                    shifts += 1

            for lvl_name, cnt in trav_result.level_counts.items():
                accum_levels[lvl_name] = accum_levels.get(lvl_name, 0.0) + cnt

        total_frames = len(self.cached_upstream)
        metrics = {
            "path_found_rate": round(paths_found / max(total_frames, 1), 4),
            "mean_v": round(float(np.mean(v_list)), 4) if v_list else 0.0,
            "std_v": round(float(np.std(v_list)), 4) if v_list else 0.0,
            "mean_abs_w": round(float(np.mean(w_list)), 4) if w_list else 0.0,
            "mean_cost": round(float(np.mean(cost_list)), 2) if cost_list else 0.0,
            "mean_clearance_m": round(float(np.mean(clr_list)), 3) if clr_list else 0.0,
            "min_clearance_m": round(float(np.min(clr_list)), 3) if clr_list else 0.0,
            "trajectory_shift_rate": round(shifts / max(total_frames, 1), 4),
        }

        mean_levels = {k: round(v / max(total_frames, 1), 1) for k, v in accum_levels.items()}
        return metrics, trajs, mean_levels

    def run_full_sweep(self) -> Dict[str, TerrainSensitivitySummary]:
        """Execute ±50% sensitivity sweep for all terrain classes."""
        t_start = time.perf_counter()
        print(f"Running Full Sensitivity Sweep...", flush=True)

        # 1. Baseline run with nominal configuration
        nominal_metrics, baseline_trajs, nominal_levels = self.evaluate_config(self.nominal_config)
        print(f"Nominal Baseline: PathFound={nominal_metrics['path_found_rate']*100:.1f}%, "
              f"v_mean={nominal_metrics['mean_v']:.3f} m/s, Cost={nominal_metrics['mean_cost']:.1f}, "
              f"Clearance={nominal_metrics['mean_clearance_m']:.3f}m", flush=True)

        results: Dict[str, TerrainSensitivitySummary] = {}

        target_classes = [
            (TerrainClass.PAVED_ROAD.value, "PAVED_ROAD"),
            (TerrainClass.TRAVERSABLE_DIRT.value, "TRAVERSABLE_DIRT"),
            (TerrainClass.LOW_GRASS.value, "LOW_GRASS"),
            (TerrainClass.GRAVEL.value, "GRAVEL"),
            (TerrainClass.HIGH_VEGETATION.value, "HIGH_VEGETATION"),
            (TerrainClass.OBSTACLE_SOLID.value, "OBSTACLE_SOLID"),
            (TerrainClass.WATER_PUDDLE.value, "WATER_PUDDLE"),
        ]

        for tc_val, tc_name in target_classes:
            nom_cost = self.nominal_config.terrain_base_costs.get(tc_val, 100)
            point_results: List[SweepPointResult] = []

            for mult in self.SWEEP_MULTIPLIERS:
                perturbed_cost = int(round(nom_cost * mult))
                perturbed_cost = max(1, min(254, perturbed_cost))

                cfg = copy.deepcopy(self.nominal_config)
                cfg.terrain_base_costs[tc_val] = perturbed_cost

                metrics, _, levels = self.evaluate_config(cfg, baseline_trajs=baseline_trajs)

                pt = SweepPointResult(
                    terrain_class=tc_val,
                    terrain_name=tc_name,
                    multiplier=mult,
                    tested_cost=perturbed_cost,
                    frames_evaluated=len(self.cached_upstream),
                    path_found_rate=metrics["path_found_rate"],
                    mean_v=metrics["mean_v"],
                    std_v=metrics["std_v"],
                    mean_abs_w=metrics["mean_abs_w"],
                    mean_cost=metrics["mean_cost"],
                    mean_clearance_m=metrics["mean_clearance_m"],
                    min_clearance_m=metrics["min_clearance_m"],
                    trajectory_shift_rate=metrics["trajectory_shift_rate"],
                    level_counts_mean=levels,
                )
                point_results.append(pt)

            p_low = point_results[0]
            p_high = point_results[-1]
            nom = point_results[2]

            d_param = 1.0
            d_cost = abs(p_high.mean_cost - p_low.mean_cost) / max(nom.mean_cost, 1.0)
            d_v = abs(p_high.mean_v - p_low.mean_v) / max(nom.mean_v, 0.01)

            cost_sens = round(d_cost / d_param, 4)
            v_sens = round(d_v / d_param, 4)
            max_shift = max(pt.trajectory_shift_rate for pt in point_results)

            safe_range, rec = self._determine_recommendation(tc_name, nom_cost, point_results, cost_sens, v_sens)

            summary = TerrainSensitivitySummary(
                terrain_class=tc_val,
                terrain_name=tc_name,
                nominal_cost=nom_cost,
                points=point_results,
                cost_sensitivity_index=cost_sens,
                velocity_sensitivity_index=v_sens,
                max_trajectory_shift_rate=max_shift,
                safe_operating_range=safe_range,
                recommendation=rec,
            )
            results[tc_name] = summary
            print(f"  {tc_name:18s} (nom={nom_cost:3d}): CostSens={cost_sens:.3f}, "
                  f"VSens={v_sens:.3f}, MaxShift={max_shift*100:.1f}%, SafeRange={safe_range}, Rec: {rec}", flush=True)

        duration = time.perf_counter() - t_start
        print(f"\nAll 35 sweep evaluations completed in {duration:.2f}s.", flush=True)
        return results

    def _determine_recommendation(
        self,
        name: str,
        nominal_cost: int,
        points: List[SweepPointResult],
        cost_sens: float,
        v_sens: float,
    ) -> Tuple[Tuple[int, int], str]:
        """Synthesize engineering recommendation for initial value locking."""
        min_cost = min(p.tested_cost for p in points)
        max_cost = max(p.tested_cost for p in points)

        clearances = [p.min_clearance_m for p in points]
        if min(clearances) < 0.35:
            rec = f"SAFETY CRITICAL: Clearance dropped to {min(clearances):.2f}m. Must lock at or above nominal {nominal_cost}."
            return (nominal_cost, max_cost), rec

        if v_sens < 0.05 and cost_sens < 0.15:
            rec = f"ROBUST: Highly stable (S_v={v_sens:.2f}, S_c={cost_sens:.2f}). Safe to lock initial value at {nominal_cost}."
            return (min_cost, max_cost), rec
        elif v_sens >= 0.05 and cost_sens < 0.30:
            rec = f"MODERATE: Measurable speed adaptation (S_v={v_sens:.2f}). Nominal {nominal_cost} is well-calibrated."
            return (int(nominal_cost * 0.8), int(nominal_cost * 1.2)), rec
        else:
            rec = f"HIGH SENSITIVITY: S_c={cost_sens:.2f}. Trajectory selection responds strongly; lock calibrated value {nominal_cost}."
            return (int(nominal_cost * 0.9), int(nominal_cost * 1.1)), rec


if __name__ == "__main__":
    sweeper = TraversabilitySensitivitySweeper()
    results = sweeper.run_full_sweep()
