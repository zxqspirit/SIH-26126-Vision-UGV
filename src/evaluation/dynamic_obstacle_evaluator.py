"""Dynamic Obstacle Evaluation Engine.

Traces and evaluates the 5-stage reactivity chain:
New Observation -> Obstacle Update -> Traversability Update -> Path Update -> Recommended Command Update

Measures:
- Observation-to-decision latency (mean, p95, per-stage breakdown)
- Path-change latency (reaction delay from obstacle entry to evasive action)
- Minimum recommended clearance along rollout trajectories
- False obstacle events and missed obstacle events

Important: All commands are software recommendations only. No physical collision avoidance is claimed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.interfaces.types import (
    SensorFrame,
    SemanticResult,
    DepthGeometryResult,
    FusedTraversabilityResult,
    VisualOdometryResult,
    TraversabilityMapResult,
    NavigationDecisionResult,
    NavigationState,
    SafetyResult,
    UGVMotionCommand,
    GoalPose,
)
from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
from src.pipeline import NavigationPipeline


@dataclass
class FrameEvaluationTelemetry:
    """Detailed telemetry recorded for a single frame evaluation."""
    frame_id: int
    timestamp_s: float
    # Latencies (ms)
    t_obstacle_update_ms: float
    t_traversability_update_ms: float
    t_path_update_ms: float
    t_command_update_ms: float
    t_total_decision_ms: float
    # Obstacle metrics
    positive_obstacle_px: int
    bev_obstacle_cells: int
    bev_blocked_cells: int
    # Path & Command metrics
    path_valid: bool
    path_waypoints_count: int
    path_length_m: float
    recommended_v: float
    recommended_w: float
    steering: str
    navigation_state: str
    safety_state: str
    min_clearance_m: float
    is_evasive_action: bool
    explanation_summary: str


@dataclass
class ScenarioEvaluationSummary:
    """Aggregated dynamic obstacle evaluation metrics for a complete scenario."""
    scenario_name: str
    total_frames: int
    # Latency Metrics
    mean_decision_latency_ms: float
    p95_decision_latency_ms: float
    max_decision_latency_ms: float
    stage_breakdown_ms: Dict[str, float]
    # Path Reaction & Dynamic Metrics
    obstacle_detected_frame: Optional[int]
    obstacle_detected_time_s: Optional[float]
    path_reaction_frame: Optional[int]
    path_reaction_time_s: Optional[float]
    path_change_delay_frames: Optional[int]
    path_change_latency_ms: Optional[float]
    # Clearance & Safety Metrics
    min_clearance_m: float
    mean_clearance_m: float
    footprint_violations: int
    # Detection Reliability
    false_obstacle_events: int
    missed_obstacle_events: int
    frames: List[FrameEvaluationTelemetry] = field(default_factory=list)
    safety_notice: str = "Commands are software recommendations only. No physical collision avoidance is claimed."


class DynamicObstacleEvaluator:
    """Systematic evaluator of dynamic obstacle responsiveness on recorded sequences."""

    OBSTACLE_PIXEL_THRESHOLD = 500  # Minimum positive obstacle px to declare physical obstacle presence

    def __init__(self, dataset_root: str = "datasets/processed") -> None:
        self.dataset_root = dataset_root

    def evaluate_scenario(
        self,
        scenario_name: str,
        goal: Optional[GoalPose] = None,
    ) -> ScenarioEvaluationSummary:
        """Run complete 5-stage dynamic evaluation on a recorded scenario."""
        loader = OutdoorDatasetLoader(f"{self.dataset_root}/{scenario_name}")
        pipeline = NavigationPipeline()

        frame_telemetries: List[FrameEvaluationTelemetry] = []
        obs_detected_frame = None
        obs_detected_time = None
        path_reacted_frame = None
        path_reacted_time = None

        baseline_w = 0.0

        for i in range(len(loader)):
            frame = loader.get_frame(i)
            if frame is None:
                continue

            t_cycle_start = time.perf_counter()

            # -------------------------------------------------------------
            # Stage 1 & 2: New Observation -> Metric Obstacle Update
            # -------------------------------------------------------------
            t_stage2_start = time.perf_counter()
            geometry: DepthGeometryResult = pipeline.geometry.process_depth(frame.depth_m)
            points_opt, _ = pipeline.geometry.projector.project_to_camera_frame(frame.depth_m)
            points_base = pipeline.geometry.projector.transform_to_base_link(points_opt)
            t_stage2 = (time.perf_counter() - t_stage2_start) * 1000.0

            # -------------------------------------------------------------
            # Stage 3: Traversability Update
            # -------------------------------------------------------------
            t_stage3_start = time.perf_counter()
            semantic: SemanticResult = pipeline.perception.process_frame(frame)
            fused: FusedTraversabilityResult = pipeline.fusion.fuse(semantic, geometry, points_base)
            odometry: VisualOdometryResult = pipeline.odometry.process_frame(frame.rgb, frame.depth_m)
            trav_map: TraversabilityMapResult = pipeline.traversability.process(
                fused=fused, semantic=semantic, geometry=geometry, odometry=odometry,
            )
            t_stage3 = (time.perf_counter() - t_stage3_start) * 1000.0

            # -------------------------------------------------------------
            # Stage 4: Path Update
            # -------------------------------------------------------------
            t_stage4_start = time.perf_counter()
            pipeline.costmap.update_from_traversability_result(trav_map)
            # A* Global Path planning to goal
            global_path = pipeline.decision_engine.global_planner.plan(
                costmap=pipeline.costmap,
                start_x_m=0.0,
                start_y_m=0.0,
                goal=goal,
            )
            t_stage4 = (time.perf_counter() - t_stage4_start) * 1000.0

            # -------------------------------------------------------------
            # Stage 5: Recommended Command Update
            # -------------------------------------------------------------
            t_stage5_start = time.perf_counter()
            decision: NavigationDecisionResult = pipeline.decision_engine.decide(
                costmap=pipeline.costmap,
                traversability=trav_map,
                odometry=odometry,
                goal=goal,
                current_v=pipeline.last_command.linear_velocity,
                current_w=pipeline.last_command.angular_velocity,
            )

            # Safety gate arbitration
            safety: SafetyResult = pipeline.safety_gate.arbitrate(
                nominal_v=decision.recommended_linear_velocity,
                nominal_w=decision.recommended_angular_velocity,
                c_perc=semantic.confidence,
                c_geom=geometry.confidence,
                c_vo=odometry.confidence,
                c_fusion=fused.confidence,
                tracking_status=odometry.tracking_status,
                min_obstacle_dist_m=decision.min_clearance_m,
            )

            cmd: UGVMotionCommand = pipeline.command_generator.generate(
                planning_result=pipeline.decision_engine.local_planner.plan(
                    costmap=pipeline.costmap,
                    current_v=pipeline.last_command.linear_velocity,
                    current_w=pipeline.last_command.angular_velocity,
                    target_heading_rad=decision.target_heading_rad,
                ),
                safety_result=safety,
                timestamp=frame.timestamp,
            )
            pipeline.last_command = cmd
            t_stage5 = (time.perf_counter() - t_stage5_start) * 1000.0

            t_total = (time.perf_counter() - t_cycle_start) * 1000.0

            # Detect obstacle appearance
            pos_obs_px = int(geometry.positive_obstacle_mask.sum())
            has_obstacle = pos_obs_px >= self.OBSTACLE_PIXEL_THRESHOLD
            if has_obstacle and obs_detected_frame is None:
                obs_detected_frame = i
                obs_detected_time = frame.timestamp

            # Detect evasive action (steering shift or crawl/halt)
            is_evasive = (abs(cmd.angular_velocity) > 0.15) or (cmd.linear_velocity <= 0.15) or (decision.navigation_state == NavigationState.AVOIDING_OBSTACLE)
            if has_obstacle and is_evasive and path_reacted_frame is None:
                path_reacted_frame = i
                path_reacted_time = frame.timestamp

            expl_summary = decision.cost_explanation.explanation_text if decision.cost_explanation else ""

            tel = FrameEvaluationTelemetry(
                frame_id=i,
                timestamp_s=round(frame.timestamp, 3),
                t_obstacle_update_ms=round(t_stage2, 2),
                t_traversability_update_ms=round(t_stage3, 2),
                t_path_update_ms=round(t_stage4, 2),
                t_command_update_ms=round(t_stage5, 2),
                t_total_decision_ms=round(t_total, 2),
                positive_obstacle_px=pos_obs_px,
                bev_obstacle_cells=int(fused.obstacle_mask.sum()) if fused.obstacle_mask is not None else 0,
                bev_blocked_cells=trav_map.level_counts.get("BLOCKED", 0),
                path_valid=global_path.is_valid,
                path_waypoints_count=len(global_path.waypoints),
                path_length_m=global_path.total_length_m,
                recommended_v=cmd.linear_velocity,
                recommended_w=cmd.angular_velocity,
                steering=cmd.steering_direction,
                navigation_state=cmd.navigation_state,
                safety_state=cmd.safety_state,
                min_clearance_m=decision.min_clearance_m,
                is_evasive_action=is_evasive,
                explanation_summary=expl_summary,
            )
            frame_telemetries.append(tel)

        # Aggregate metrics
        latencies = [t.t_total_decision_ms for t in frame_telemetries]
        clearances = [t.min_clearance_m for t in frame_telemetries if t.min_clearance_m > 0.0]

        # Calculate path change latency
        if obs_detected_frame is not None and path_reacted_frame is not None:
            delay_frames = max(0, path_reacted_frame - obs_detected_frame)
            delay_ms = round((path_reacted_time - obs_detected_time) * 1000.0, 2)
        else:
            delay_frames = None
            delay_ms = None

        # Footprint violations: clearance < 0.35m
        footprint_violations = sum(1 for t in frame_telemetries if 0.0 < t.min_clearance_m < 0.35)

        # Detection reliability:
        # False obstacle event: open path scenario detecting persistent large obstacles (>1000 px)
        # Missed obstacle event: scenario with physical obstacle failing to detect obstacles
        if scenario_name == "scenario_1_open_path":
            false_events = sum(1 for t in frame_telemetries if t.positive_obstacle_px >= self.OBSTACLE_PIXEL_THRESHOLD)
            missed_events = 0
        elif scenario_name == "scenario_2_sudden_obstacle":
            false_events = 0
            missed_events = sum(1 for t in frame_telemetries if t.positive_obstacle_px < self.OBSTACLE_PIXEL_THRESHOLD)
        else:
            false_events = 0
            missed_events = 0

        breakdown = {
            "obstacle_update_ms": round(float(np.mean([t.t_obstacle_update_ms for t in frame_telemetries])), 2),
            "traversability_update_ms": round(float(np.mean([t.t_traversability_update_ms for t in frame_telemetries])), 2),
            "path_update_ms": round(float(np.mean([t.t_path_update_ms for t in frame_telemetries])), 2),
            "command_update_ms": round(float(np.mean([t.t_command_update_ms for t in frame_telemetries])), 2),
        }

        return ScenarioEvaluationSummary(
            scenario_name=scenario_name,
            total_frames=len(frame_telemetries),
            mean_decision_latency_ms=round(float(np.mean(latencies)), 2),
            p95_decision_latency_ms=round(float(np.percentile(latencies, 95)), 2),
            max_decision_latency_ms=round(float(np.max(latencies)), 2),
            stage_breakdown_ms=breakdown,
            obstacle_detected_frame=obs_detected_frame,
            obstacle_detected_time_s=round(obs_detected_time, 3) if obs_detected_time else None,
            path_reaction_frame=path_reacted_frame,
            path_reaction_time_s=round(path_reacted_time, 3) if path_reacted_time else None,
            path_change_delay_frames=delay_frames,
            path_change_latency_ms=delay_ms,
            min_clearance_m=round(float(np.min(clearances)), 3) if clearances else 0.0,
            mean_clearance_m=round(float(np.mean(clearances)), 3) if clearances else 0.0,
            footprint_violations=footprint_violations,
            false_obstacle_events=false_events,
            missed_obstacle_events=missed_events,
            frames=frame_telemetries,
        )

    def run_all_scenarios(self) -> Dict[str, ScenarioEvaluationSummary]:
        """Run dynamic obstacle evaluation across all 5 outdoor scenarios."""
        scenarios = [
            "scenario_1_open_path",
            "scenario_2_sudden_obstacle",
            "scenario_3_terrain_boundary",
            "scenario_4_depth_degradation",
            "scenario_5_visual_degradation",
        ]
        results = {}
        for sc in scenarios:
            summary = self.evaluate_scenario(sc)
            results[sc] = summary
            print(f"=== {sc} ===")
            print(f"  Latency: mean={summary.mean_decision_latency_ms:.1f}ms (p95={summary.p95_decision_latency_ms:.1f}ms)")
            if summary.path_change_latency_ms is not None:
                print(f"  Reaction Delay: {summary.path_change_latency_ms:.1f}ms ({summary.path_change_delay_frames} frames)")
            print(f"  Clearance: min={summary.min_clearance_m:.2f}m, mean={summary.mean_clearance_m:.2f}m, violations={summary.footprint_violations}")
            print(f"  Reliability: FalseEvents={summary.false_obstacle_events}, MissedEvents={summary.missed_obstacle_events}")
        return results


if __name__ == "__main__":
    evaluator = DynamicObstacleEvaluator()
    evaluator.run_all_scenarios()
