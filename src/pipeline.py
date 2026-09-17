"""End-to-end Vision-Based Autonomous Navigation Pipeline for Outdoor UGV.

Orchestrates:
Real Sensor Data -> AI Perception -> Depth Geometry -> Fusion ->
Visual Odometry -> Costmap & Planning -> Safety Gate -> UGV Motion Command.
"""

from __future__ import annotations

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
)
from .perception.perception_engine import PerceptionEngine
from .depth_geometry.depth_geometry_engine import DepthGeometryEngine
from .fusion.fusion_engine import FusionEngine
from .localization.visual_odometry import VisualOdometry
from .planning.costmap_2d import Costmap2D
from .planning.dwa_planner import DWAPlanner
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
        self.planner = DWAPlanner()
        self.safety_gate = SafetyGate()
        self.command_generator = MotionCommandGenerator()

        # Last executed state
        self.last_command = UGVMotionCommand(0.0, 0.0, "STOP", "IDLE", 1.0, "INITIALIZING", 0.0)

    def reset(self) -> None:
        """Reset stateful components (odometry, costmap)."""
        self.odometry.reset()
        self.costmap = Costmap2D()
        self.last_command = UGVMotionCommand(0.0, 0.0, "STOP", "IDLE", 1.0, "INITIALIZING", 0.0)

    def process_frame(
        self,
        frame: SensorFrame,
        target_heading_rad: float = 0.0,
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

        # 3. 3D Points in base_link
        points_opt, _ = self.geometry.projector.project_to_camera_frame(frame.depth_m)
        points_base = self.geometry.projector.transform_to_base_link(points_opt)

        # 4. Semantic + Geometric Evidence Fusion into BEV Costmap
        fused: FusedTraversabilityResult = self.fusion.fuse(semantic, geometry, points_base)

        # 5. Visual Odometry / SLAM: Visual ego-motion
        odometry: VisualOdometryResult = self.odometry.process_frame(frame.rgb, frame.depth_m)

        # 6. Update 2D local costmap
        self.costmap.update_from_fused_result(fused)

        # 7. Local Trajectory Planning (DWA)
        planning: PlanningResult = self.planner.plan(
            costmap=self.costmap,
            current_v=self.last_command.linear_velocity,
            current_w=self.last_command.angular_velocity,
            target_heading_rad=target_heading_rad,
        )

        # Compute minimum obstacle distance in forward path for safety checking
        min_obstacle_dist = 5.0
        if planning.selected_trajectory is not None:
            min_obstacle_dist = planning.selected_trajectory.clearance_m

        # 8. Deterministic Confidence-Based Safety Gate
        safety: SafetyResult = self.safety_gate.arbitrate(
            nominal_v=planning.recommended_linear_velocity,
            nominal_w=planning.recommended_angular_velocity,
            c_perc=semantic.confidence,
            c_geom=geometry.confidence,
            c_vo=odometry.confidence,
            c_fusion=fused.confidence,
            tracking_status=odometry.tracking_status,
            min_obstacle_dist_m=min_obstacle_dist,
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
            "planning": planning,
            "safety": safety,
            "command": command,
        }

        return command, telemetry
