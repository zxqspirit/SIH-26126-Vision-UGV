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
    """Deterministic safety states for confidence-to-behavior safety design."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    CRITICAL = "CRITICAL"

    # Backward compatibility aliases
    HIGH_CONFIDENCE = "HIGH"
    CAUTIOUS_DEGRADED = "MEDIUM"
    LOW_CONFIDENCE_SLOW = "LOW"
    SAFETY_STOP = "CRITICAL"
    LOCALIZATION_LOST = "CRITICAL"


class SafetyAction(str, Enum):
    """Deterministic action recommendations emitted by the safety gate."""
    NORMAL = "NORMAL_RECOMMENDATION"
    LOWER_SPEED_EXPAND_MARGIN = "LOWER_SPEED_EXPAND_MARGIN"
    CONSERVATIVE_REOBSERVE = "CONSERVATIVE_REOBSERVE"
    SAFE_STOP = "SAFE_STOP_RECOMMENDATION"


@dataclass
class SafetyDecisionLog:
    """Logged decision record for auditable safety tracking."""
    timestamp: float
    confidence: float
    reason: str
    action: str
    state: str = "HIGH"
    signals: Dict[str, float] = field(default_factory=dict)
    nominal_v: float = 0.0
    recommended_v: float = 0.0
    recommended_w: float = 0.0
    clearance_margin_m: float = 0.35
    reobserve_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "action": self.action,
            "state": self.state,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "nominal_v": round(self.nominal_v, 3),
            "recommended_v": round(self.recommended_v, 3),
            "recommended_w": round(self.recommended_w, 3),
            "clearance_margin_m": round(self.clearance_margin_m, 3),
            "reobserve_active": self.reobserve_active,
        }


class SteeringDirection(str, Enum):
    """Discrete human-readable steering recommendation."""
    FORWARD = "FORWARD"
    SLIGHT_LEFT = "SLIGHT_LEFT"
    SLIGHT_RIGHT = "SLIGHT_RIGHT"
    HARD_LEFT = "HARD_LEFT"
    HARD_RIGHT = "HARD_RIGHT"
    STOP = "STOP"


class NavigationState(str, Enum):
    """High-level software navigation states."""
    IDLE = "IDLE"
    NAVIGATING_TO_GOAL = "NAVIGATING_TO_GOAL"
    TRACKING_PATH = "TRACKING_PATH"
    AVOIDING_OBSTACLE = "AVOIDING_OBSTACLE"
    CAUTIOUS_EXPLORATION = "CAUTIOUS_EXPLORATION"
    RECOVERY_HOLD = "RECOVERY_HOLD"
    GOAL_REACHED = "GOAL_REACHED"
    BLOCKED = "BLOCKED"


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


class TraversabilityLevel(int, Enum):
    """Six-level traversability classification for navigation cost mapping."""
    PREFERRED   = 0   # cost 0-19: optimal surface, full speed
    FREE        = 1   # cost 20-79: traversable with minor roughness
    MEDIUM_RISK = 2   # cost 80-179: caution required
    HIGH_RISK   = 3   # cost 180-219: traversable only if necessary
    BLOCKED     = 4   # cost 220-254: impassable obstacle
    UNKNOWN     = 5   # cost 255: no observation available


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
    terrain_class_grid: Optional[np.ndarray] = None  # Shape (grid_h, grid_w), uint8 TerrainClass values


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
class TraversabilityCostConfig:
    """Configurable cost thresholds and parameters for traversability classification.

    All values are initial empirical estimates — they must be validated
    experimentally. Do NOT hard-code final weights without validation.
    """
    # Level boundary thresholds (upper bound inclusive for each level)
    preferred_max: int = 19
    free_max: int = 79
    medium_risk_max: int = 179
    high_risk_max: int = 219
    blocked_max: int = 254  # 255 reserved for UNKNOWN

    # Per-TerrainClass nominal base costs (TerrainClass.value -> cost)
    terrain_base_costs: Dict[int, int] = field(default_factory=lambda: {
        0: 255,   # UNKNOWN -> maximum cost
        1: 5,     # PAVED_ROAD -> preferred
        2: 12,    # TRAVERSABLE_DIRT -> preferred
        3: 35,    # LOW_GRASS -> free
        4: 50,    # GRAVEL -> free
        5: 120,   # HIGH_VEGETATION -> medium risk
        6: 240,   # OBSTACLE_SOLID -> blocked
        7: 190,   # WATER_PUDDLE -> high risk
    })

    # Uncertainty-based cost inflation
    uncertainty_cost_weight: float = 0.3     # Blend factor: cost += weight * uncertainty * 254
    high_uncertainty_thresh: float = 0.7     # Above this, shift cost toward HIGH_RISK
    disagreement_cost_penalty: int = 40      # Added to cost in disagreement zones

    # Unknown cell cost (Rule 11)
    unknown_cost: int = 255

    # Pose-dependent cost scaling
    localization_lost_inflation: float = 1.5    # Multiply non-preferred costs when VO lost
    degraded_tracking_inflation: float = 1.15   # Multiply when VO degraded


@dataclass
class TraversabilityMapResult:
    """Complete traversability map output for navigation and visualization."""
    level_grid: np.ndarray          # (grid_h, grid_w) uint8, TraversabilityLevel values
    cost_grid: np.ndarray           # (grid_h, grid_w) uint8, navigation-ready [0..255]
    obstacle_grid: np.ndarray       # (grid_h, grid_w) bool, lethal obstacles
    uncertainty_grid: np.ndarray    # (grid_h, grid_w) float32 [0..1]
    visual_map: np.ndarray          # (grid_h, grid_w, 4) uint8 RGBA color-coded overlay
    level_counts: Dict[str, int] = field(default_factory=dict)  # per-level cell count
    confidence: float = 0.0
    latency_ms: float = 0.0
    resolution_m: float = 0.1
    grid_size_m: Tuple[float, float] = (10.0, 10.0)


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
class GoalPose:
    """Target waypoint in local odom / map coordinates."""
    x: float
    y: float
    yaw: float = 0.0
    tolerance_m: float = 0.50
    heading_tolerance_rad: float = 0.25


@dataclass
class NavigationPath:
    """Recommended global / topological path waypoints."""
    waypoints: List[Tuple[float, float]]  # List of (x, y) in meters
    total_length_m: float = 0.0
    accumulated_cost: float = 0.0
    is_valid: bool = True


@dataclass
class PathCostBreakdown:
    """Explainable multi-objective cost decomposition."""
    total_score: float
    progress_score: float        # Forward progress toward goal [0..1]
    clearance_score: float       # Margin to nearest obstacle [0..1]
    traversability_score: float  # Smoothness / safety of ground [0..1]
    heading_score: float         # Alignment with goal / path heading [0..1]
    average_cost: float          # Raw uint8 average along rollout [0..255]
    min_clearance_m: float       # Minimum metric clearance along rollout
    primary_terrain: str = "UNKNOWN"
    explanation_text: str = ""


@dataclass
class NavigationDecisionResult:
    """Complete software output from the navigation decision engine."""
    recommended_path: Optional[NavigationPath]
    recommended_trajectory: Optional[Trajectory]
    candidate_trajectories: List[Trajectory]
    recommended_linear_velocity: float
    recommended_angular_velocity: float
    recommended_steering: SteeringDirection
    navigation_state: NavigationState
    cost_explanation: Optional[PathCostBreakdown]
    min_clearance_m: float
    status: str
    target_heading_rad: float = 0.0
    goal: Optional[GoalPose] = None
    goal_distance_m: float = 0.0
    latency_ms: float = 0.0


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
    action: str = "NORMAL_RECOMMENDATION"
    decision_log: Optional[SafetyDecisionLog] = None


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


class CameraHealthState(str, Enum):
    """Health states for real-time camera ingestion."""
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class CameraHealthMetrics:
    """Real-time performance and health metrics for camera stream."""
    health_state: CameraHealthState
    capture_fps: float
    processing_fps: float
    dropped_frames: int
    frame_latency_ms: float
    consecutive_drops: int = 0
    last_timestamp: float = 0.0
    total_captured: int = 0
    total_processed: int = 0
    is_alive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "health_state": self.health_state.value,
            "capture_fps": round(self.capture_fps, 2),
            "processing_fps": round(self.processing_fps, 2),
            "dropped_frames": self.dropped_frames,
            "frame_latency_ms": round(self.frame_latency_ms, 2),
            "consecutive_drops": self.consecutive_drops,
            "last_timestamp": round(self.last_timestamp, 4),
            "total_captured": self.total_captured,
            "total_processed": self.total_processed,
            "is_alive": self.is_alive,
        }
