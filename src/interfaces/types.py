"""Strongly-typed data contracts and interfaces for SIH 26126 Autonomous Navigation Stack.

Enforces modularity, inspectability, and replaceability across the perception,
depth geometry, fusion, visual odometry, costmap planning, and safety gate stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TrackingStatus(str, Enum):
    """Visual tracking status states."""
    TRACKING_OK = "TRACKING_OK"
    TRACKING_DEGRADED = "TRACKING_DEGRADED"
    TRACKING_LOST = "TRACKING_LOST"


class RelocalizationState(str, Enum):
    """State machine for visual relocalization tracking."""
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    TRACKING_LOST = "TRACKING_LOST"
    RELOCALIZATION_ATTEMPT = "RELOCALIZATION_ATTEMPT"
    RECOVERED = "RECOVERED"


class SafetyState(str, Enum):
    """Deterministic safety states for confidence-based degradation."""
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    CAUTIOUS_DEGRADED = "CAUTIOUS_DEGRADED"
    LOW_CONFIDENCE_SLOW = "LOW_CONFIDENCE_SLOW"
    SAFETY_STOP = "SAFETY_STOP"
    LOCALIZATION_LOST = "LOCALIZATION_LOST"


class SteeringDirection(str, Enum):
    """Discrete human-readable steering recommendation."""
    FORWARD = "FORWARD"
    SLIGHT_LEFT = "SLIGHT_LEFT"
    SLIGHT_RIGHT = "SLIGHT_RIGHT"
    HARD_LEFT = "HARD_LEFT"
    HARD_RIGHT = "HARD_RIGHT"
    STOP = "STOP"


class TerrainClass(int, Enum):
    """Semantic terrain classification categories."""
    UNKNOWN = 0
    PAVED_ROAD = 1
    TRAVERSABLE_DIRT = 2
    LOW_GRASS = 3
    GRAVEL = 4
    HIGH_VEGETATION = 5
    OBSTACLE_SOLID = 6
    WATER_PUDDLE = 7


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters and extrinsics."""
    fx: float = 385.0
    fy: float = 385.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    baseline_m: float = 0.05  # Stereo baseline in meters if stereo camera
    camera_height_m: float = 0.45  # Height above ground in base_link
    camera_pitch_rad: float = 0.209  # ~12 degrees downward tilt

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)


@dataclass
class SensorFrame:
    """Synchronized real outdoor sensor payload."""
    timestamp: float
    rgb: np.ndarray  # Shape (H, W, 3), uint8
    depth_m: np.ndarray  # Shape (H, W), float32 in meters (0 or NaN for invalid)
    right_rgb: Optional[np.ndarray] = None  # Shape (H, W, 3) if stereo
    frame_id: int = 0
    sequence_name: str = "outdoor_run"


@dataclass
class SemanticResult:
    """Output from lightweight CNN perception."""
    traversability_mask: np.ndarray  # Shape (H, W), float32 in [0.0, 1.0]
    terrain_class_map: np.ndarray  # Shape (H, W), int32 matching TerrainClass
    confidence: float  # Scalar in [0.0, 1.0]
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthGeometryResult:
    """Output from 3D metric depth geometry processing."""
    ground_height_map: np.ndarray  # Shape (H, W), estimated ground plane height
    positive_obstacle_mask: np.ndarray  # Shape (H, W), bool (True where height > step threshold)
    negative_obstacle_mask: np.ndarray  # Shape (H, W), bool (True where drop / ditch detected)
    depth_validity_mask: np.ndarray  # Shape (H, W), bool (True where valid depth was measured)
    geometric_cost: np.ndarray  # Shape (H, W), float32 in [0.0, 1.0] (1.0 = lethal)
    confidence: float  # Scalar in [0.0, 1.0] based on depth coverage and validity
    points_3d: Optional[np.ndarray] = None  # Subsampled 3D point cloud in base_link (N, 3)
    latency_ms: float = 0.0
    discontinuity_mask: Optional[np.ndarray] = None  # Shape (H, W), bool (True where step discontinuity detected)
    plane_coeffs: Optional[Tuple[float, float, float, float]] = None  # Ground plane (a, b, c, d)


@dataclass
class FusedTraversabilityResult:
    """Output from semantic + geometric evidence fusion."""
    fused_costmap: np.ndarray  # Shape (grid_h, grid_w), uint8 in [0..255] (254=lethal, 255=unknown)
    disagreement_mask: np.ndarray  # Shape (grid_h, grid_w), bool (True where semantics & geometry conflict)
    unknown_mask: np.ndarray  # Shape (grid_h, grid_w), bool (True where terrain is unobserved/invalid)
    confidence: float  # Scalar in [0.0, 1.0]
    resolution_m: float = 0.1  # meters per grid cell
    origin_x_m: float = 0.0  # UGV base_link position in grid coords
    origin_y_m: float = 5.0  # Center offset
    grid_size_m: Tuple[float, float] = (10.0, 10.0)  # (length, width) forward-facing
    latency_ms: float = 0.0
    fused_traversability: Optional[np.ndarray] = None  # Shape (H, W), float32 in [0.0..1.0]
    obstacle_mask: Optional[np.ndarray] = None  # Shape (grid_h, grid_w), bool (True where lethal hazard)
    uncertainty_grid: Optional[np.ndarray] = None  # Shape (grid_h, grid_w), float32 in [0.0..1.0]
    uncertainty_px: Optional[np.ndarray] = None  # Shape (H, W), float32 in [0.0..1.0]


@dataclass
class VisualOdometryResult:
    """Output from visual motion estimation."""
    x: float = 0.0  # Global X in meters
    y: float = 0.0  # Global Y in meters
    z: float = 0.0  # Global Z in meters
    yaw: float = 0.0  # Yaw heading in radians
    delta_x: float = 0.0  # Frame-to-frame forward translation
    delta_y: float = 0.0  # Frame-to-frame lateral translation
    delta_yaw: float = 0.0  # Frame-to-frame angular rotation
    inlier_count: int = 0
    tracking_status: TrackingStatus = TrackingStatus.TRACKING_OK
    confidence: float = 1.0  # Scalar in [0.0, 1.0] based on inlier count & optical flow
    latency_ms: float = 0.0
    relocalization_state: str = "INITIALIZING"  # RelocalizationState value
    consecutive_lost_frames: int = 0
    recovery_frame_count: int = 0  # Frames since last recovery (0 if never recovered)


@dataclass
class Trajectory:
    """Candidate rollout trajectory."""
    points: List[Tuple[float, float, float]]  # List of (x, y, yaw)
    linear_velocity: float
    angular_velocity: float
    cost: float
    clearance_m: float
    is_valid: bool = True


@dataclass
class PlanningResult:
    """Output from local trajectory planner."""
    selected_trajectory: Optional[Trajectory]
    candidate_trajectories: List[Trajectory]
    recommended_linear_velocity: float  # m/s
    recommended_angular_velocity: float  # rad/s
    recommended_steering: SteeringDirection
    status: str  # "PATH_FOUND", "OBSTACLE_BLOCKED", "NO_TRAVERSABLE_PATH"
    target_heading_rad: float = 0.0
    latency_ms: float = 0.0


@dataclass
class SafetyResult:
    """Output from the deterministic confidence safety gate."""
    overall_confidence: float  # Aggregated confidence in [0.0, 1.0]
    safety_state: SafetyState
    commanded_linear_velocity: float  # Safe scaled velocity (0.0 if e-stop)
    commanded_angular_velocity: float
    speed_scale_factor: float  # Multiplier in [0.0, 1.0]
    clearance_inflation_factor: float  # Multiplier >= 1.0
    is_emergency_stop: bool
    audit_reasons: List[str] = field(default_factory=list)


@dataclass
class UGVMotionCommand:
    """Final software output emitted by the autonomy stack."""
    linear_velocity: float  # Commanded forward velocity in m/s
    angular_velocity: float  # Commanded turn rate in rad/s
    steering_direction: str  # e.g. "FORWARD", "SLIGHT_LEFT", "HARD_RIGHT", "STOP"
    navigation_state: str  # "TRACKING", "AVOIDING", "CAUTIOUS_EXPLORATION", "RECOVERY", "ESTOP"
    confidence: float  # Overall system confidence in [0.0, 1.0]
    safety_state: str  # "HIGH_CONFIDENCE", "CAUTIOUS_DEGRADED", "SAFETY_STOP", etc.
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "linear_velocity": round(float(self.linear_velocity), 3),
            "angular_velocity": round(float(self.angular_velocity), 3),
            "steering_direction": str(self.steering_direction),
            "navigation_state": str(self.navigation_state),
            "confidence": round(float(self.confidence), 3),
            "safety_state": str(self.safety_state),
            "timestamp": float(self.timestamp),
        }
