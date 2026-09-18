"""End-to-end Vision-Based Autonomous Navigation Pipeline for Outdoor UGV.

Orchestrates:
Real Sensor Data -> AI Perception -> Depth Geometry -> Fusion ->
Visual Odometry -> Costmap & Planning -> Safety Gate -> UGV Motion Command.
"""

from __future__ import annotations
import math

import time
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .interfaces.types import (
    CameraIntrinsics,
    SensorFrame,
    SemanticResult,
    DepthGeometryResult,
    FusedTraversabilityResult,
    VisualOdometryResult,
    PlanningResult,
    SafetyResult,
    UGVMotionCommand,
    TraversabilityMapResult,
    GoalPose,
    NavigationDecisionResult,
)
from .perception.perception_engine import PerceptionEngine
from .depth_geometry.depth_geometry_engine import DepthGeometryEngine
from .fusion.fusion_engine import FusionEngine
from .localization.visual_odometry import VisualOdometry
from .planning.costmap_2d import Costmap2D
from .traversability.traversability_map_engine import TraversabilityMapEngine
from .planning.dwa_planner import DWAPlanner
from .planning.navigation_decision_engine import NavigationDecisionEngine
from .safety.safety_gate import SafetyGate
from .control.motion_command_generator import MotionCommandGenerator


class NavigationPipeline:
    """Full modular software navigation stack for outdoor GPS-denied UGV."""

    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        model_path: Optional[str] = None,
    ) -> None:
        self.intrinsics = intrinsics if intrinsics is not None else CameraIntrinsics()

        # Initialize all modules
        self.perception = PerceptionEngine(model_path=model_path)
        self.geometry = DepthGeometryEngine(self.intrinsics)
        self.fusion = FusionEngine(self.intrinsics)
        self.odometry = VisualOdometry(self.intrinsics)
        self.costmap = Costmap2D()
        self.traversability = TraversabilityMapEngine()
        self.planner = DWAPlanner()
        self.decision_engine = NavigationDecisionEngine()
        self.safety_gate = SafetyGate()
        self.command_generator = MotionCommandGenerator()

        # Last executed state
        self.last_command = UGVMotionCommand(0.0, 0.0, "STOP", "IDLE", 1.0, "INITIALIZING", 0.0)

    def reset(self) -> None:
        """Reset stateful components (odometry, costmap, decision engine)."""
        self.odometry.reset()
        self.costmap = Costmap2D()
        self.traversability = TraversabilityMapEngine()
        self.decision_engine = NavigationDecisionEngine()
        self.last_command = UGVMotionCommand(0.0, 0.0, "STOP", "IDLE", 1.0, "INITIALIZING", 0.0)

    def process_frame(
        self,
        frame: SensorFrame,
        target_heading_rad: float = 0.0,
        goal: Optional[GoalPose] = None,
    ) -> Tuple[UGVMotionCommand, Dict[str, Any]]:
        """Run complete navigation cycle on a synchronized outdoor sensor frame.

        Returns:
            command: Recommended UGVMotionCommand
            debug_telemetry: Dictionary containing intermediate representations
        """
        cycle_start = time.perf_counter()

        # 1. AI Perception: Semantic traversability
        semantic: SemanticResult = self.perception.process_frame(frame)

        # 2. Metric Depth Geometry: 3D obstacles and ground modeling
        geometry: DepthGeometryResult = self.geometry.process_depth(frame.depth_m)

        # 3. 3D Points in base_link (Optimization 3: Re-use cached points_base without redundant projection)
        points_base = getattr(geometry, "points_base", None)
        if points_base is None:
            points_opt, _ = self.geometry.projector.project_to_camera_frame(frame.depth_m)
            points_base = self.geometry.projector.transform_to_base_link(points_opt)

        # 4. Semantic + Geometric Evidence Fusion into BEV Costmap
        fused: FusedTraversabilityResult = self.fusion.fuse(semantic, geometry, points_base)

        # 5. Visual Odometry / SLAM: Visual ego-motion
        odometry: VisualOdometryResult = self.odometry.process_frame(frame.rgb, frame.depth_m)

        # 5.5 Traversability Map: Unified classification with configurable costs
        trav_map: TraversabilityMapResult = self.traversability.process(
            fused=fused, semantic=semantic, geometry=geometry, odometry=odometry,
        )

        # 6. Update 2D local costmap from unified traversability map
        self.costmap.update_from_traversability_result(trav_map)

        # 7. Navigation Decision Engine: A* Global Path + DWA Local Rollouts + Cost Explanation
        decision: NavigationDecisionResult = self.decision_engine.decide(
            costmap=self.costmap,
            traversability=trav_map,
            odometry=odometry,
            goal=goal,
            current_v=self.last_command.linear_velocity,
            current_w=self.last_command.angular_velocity,
        )

        # Construct legacy planning result for backwards compatibility
        planning = PlanningResult(
            selected_trajectory=decision.recommended_trajectory,
            candidate_trajectories=decision.candidate_trajectories,
            recommended_linear_velocity=decision.recommended_linear_velocity,
            recommended_angular_velocity=decision.recommended_angular_velocity,
            recommended_steering=decision.recommended_steering,
            status=decision.status,
            target_heading_rad=decision.target_heading_rad,
            latency_ms=decision.latency_ms,
        )

        # Compute minimum obstacle distance in immediate vehicle braking zone (first 0.45m of path)
        min_obstacle_dist = self.costmap.get_obstacle_distance(0.0, 0.0)
        if decision.recommended_trajectory is not None and getattr(decision.recommended_trajectory, "points", None):
            braking_points = [
                pt for pt in decision.recommended_trajectory.points
                if math.hypot(pt[0], pt[1]) <= 0.45
            ]
            if braking_points:
                min_obstacle_dist = min(self.costmap.get_obstacle_distance(pt[0], pt[1]) for pt in braking_points)

        # Compute spatial semantic/geometric disagreement ratio
        disagreement_ratio = 0.0
        if getattr(fused, "disagreement_mask", None) is not None:
            disagreement_ratio = float(np.mean(fused.disagreement_mask))

        # 8. Deterministic Confidence-Based Safety Gate
        safety: SafetyResult = self.safety_gate.arbitrate(
            nominal_v=decision.recommended_linear_velocity,
            nominal_w=decision.recommended_angular_velocity,
            c_perc=semantic.confidence,
            c_geom=geometry.confidence,
            c_vo=odometry.confidence,
            c_fusion=fused.confidence,
            disagreement_ratio=disagreement_ratio,
            tracking_status=odometry.tracking_status,
            min_obstacle_dist_m=min_obstacle_dist,
            timestamp=frame.timestamp,
        )

        # 9. Format final UGV Motion Command
        command: UGVMotionCommand = self.command_generator.generate(
            planning_result=planning,
            safety_result=safety,
            timestamp=frame.timestamp,
        )
        self.last_command = command

        total_latency_ms = (time.perf_counter() - cycle_start) * 1000.0

        telemetry = {
            "timestamp": frame.timestamp,
            "frame_id": frame.frame_id,
            "total_latency_ms": round(total_latency_ms, 2),
            "fps": round(1000.0 / max(total_latency_ms, 1.0), 1),
            "semantic": semantic,
            "geometry": geometry,
            "fused": fused,
            "odometry": odometry,
            "traversability": trav_map,
            "planning": planning,
            "decision": decision,
            "safety": safety,
            "safety_action": safety.action,
            "safety_decision_log": safety.decision_log.to_dict() if safety.decision_log else None,
            "command": command,
        }

        return command, telemetry
