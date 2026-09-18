"""Pytest unit tests for the TraversabilityMapEngine.

Tests:
  - Correct level classification under ideal, obstacle, and unknown inputs
  - Rule 11 enforcement: unknown cells never classified as free
  - Uncertainty-based cost inflation
  - Disagreement penalty application
  - Localization-dependent cost scaling
  - Configurable thresholds
  - Level count consistency
  - Obstacle grid superset invariant
  - Visual map shape and dtype
  - End-to-end on real outdoor scenario
"""

import numpy as np
import pytest

from src.interfaces.types import (
    CameraIntrinsics,
    SemanticResult,
    DepthGeometryResult,
    FusedTraversabilityResult,
    VisualOdometryResult,
    TraversabilityLevel,
    TraversabilityCostConfig,
    TrackingStatus,
)
from src.traversability.traversability_map_engine import TraversabilityMapEngine


# ============================================================================
# Helpers: Create synthetic upstream outputs for controlled testing
# ============================================================================

def _make_semantic(h=480, w=640, trav_val=0.9, confidence=0.85) -> SemanticResult:
    """Create a uniform semantic result."""
    return SemanticResult(
        traversability_mask=np.full((h, w), trav_val, dtype=np.float32),
        terrain_class_map=np.full((h, w), 2, dtype=np.int32),  # TRAVERSABLE_DIRT
        confidence=confidence,
    )


def _make_geometry(h=480, w=640, cost_val=0.1, confidence=0.9,
                   has_obstacle=False) -> DepthGeometryResult:
    """Create a uniform geometry result."""
    pos_obs = np.zeros((h, w), dtype=bool)
    if has_obstacle:
        # Place obstacle in center
        pos_obs[200:280, 280:360] = True
    return DepthGeometryResult(
        ground_height_map=np.zeros((h, w), dtype=np.float32),
        positive_obstacle_mask=pos_obs,
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=np.ones((h, w), dtype=bool),
        geometric_cost=np.full((h, w), cost_val, dtype=np.float32),
        confidence=confidence,
    )


def _make_fused(grid_h=100, grid_w=100, cost_val=10, unknown_pct=0.0,
                obstacle_pct=0.0, disagreement_pct=0.0,
                uncertainty_val=0.1) -> FusedTraversabilityResult:
    """Create a synthetic fused result with controllable properties."""
    costmap = np.full((grid_h, grid_w), cost_val, dtype=np.uint8)
    unknown_mask = np.zeros((grid_h, grid_w), dtype=bool)
    obstacle_mask = np.zeros((grid_h, grid_w), dtype=bool)
    disagreement_mask = np.zeros((grid_h, grid_w), dtype=bool)
    uncertainty_grid = np.full((grid_h, grid_w), uncertainty_val, dtype=np.float32)

    total_cells = grid_h * grid_w

    # Set unknown cells
    n_unknown = int(total_cells * unknown_pct)
    if n_unknown > 0:
        flat_idx = np.arange(total_cells)
        np.random.seed(42)
        unk_idx = np.random.choice(flat_idx, n_unknown, replace=False)
        unknown_mask.flat[unk_idx] = True
        costmap.flat[unk_idx] = 128
        uncertainty_grid.flat[unk_idx] = 1.0

    # Set obstacle cells
    n_obs = int(total_cells * obstacle_pct)
    if n_obs > 0:
        known_idx = np.where(~unknown_mask.ravel())[0]
        obs_idx = known_idx[:n_obs]
        obstacle_mask.flat[obs_idx] = True
        costmap.flat[obs_idx] = 240

    # Set disagreement cells
    n_dis = int(total_cells * disagreement_pct)
    if n_dis > 0:
        known_idx = np.where(~unknown_mask.ravel() & ~obstacle_mask.ravel())[0]
        dis_idx = known_idx[:n_dis]
        disagreement_mask.flat[dis_idx] = True

    return FusedTraversabilityResult(
        fused_costmap=costmap,
        disagreement_mask=disagreement_mask,
        unknown_mask=unknown_mask,
        confidence=0.8,
        uncertainty_grid=uncertainty_grid,
        obstacle_mask=obstacle_mask,
        fused_traversability=np.full((480, 640), 0.9, dtype=np.float32),
    )


def _make_odometry(tracking=TrackingStatus.TRACKING_OK, confidence=0.95) -> VisualOdometryResult:
    """Create a VO result with specified tracking status."""
    return VisualOdometryResult(
        tracking_status=tracking,
        confidence=confidence,
    )


# ============================================================================
# Tests
# ============================================================================

class TestTraversabilityLevelClassification:

    def test_all_preferred_on_ideal_input(self):
        """Fully traversable, low-cost input -> all observed cells PREFERRED."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=5, unknown_pct=0.0)
        result = engine.process(
            fused=fused,
            semantic=_make_semantic(),
            geometry=_make_geometry(),
            odometry=_make_odometry(),
        )
        assert np.all(result.level_grid == TraversabilityLevel.PREFERRED.value)
        assert result.level_counts["PREFERRED"] == 100 * 100

    def test_blocked_on_obstacle(self):
        """Cells with cost >= 220 must be classified BLOCKED."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=240, unknown_pct=0.0, obstacle_pct=0.0)
        result = engine.process(
            fused=fused,
            semantic=_make_semantic(),
            geometry=_make_geometry(),
            odometry=_make_odometry(),
        )
        assert np.all(result.level_grid == TraversabilityLevel.BLOCKED.value)
        assert result.level_counts["BLOCKED"] == 100 * 100

    def test_unknown_never_free(self):
        """Unknown cells must NEVER be classified as FREE or PREFERRED (Rule 11)."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=5, unknown_pct=0.5)
        result = engine.process(
            fused=fused,
            semantic=_make_semantic(),
            geometry=_make_geometry(),
            odometry=_make_odometry(),
        )
        unknown_levels = result.level_grid[fused.unknown_mask]
        assert np.all(unknown_levels == TraversabilityLevel.UNKNOWN.value),             "Rule 11 violated: unknown cells classified as non-UNKNOWN"
        assert result.cost_grid[fused.unknown_mask].min() == 255

    def test_free_level_correct_range(self):
        """Cost in [20, 79] -> FREE level."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=50, unknown_pct=0.0)
        result = engine.process(
            fused=fused,
            semantic=_make_semantic(),
            geometry=_make_geometry(),
            odometry=_make_odometry(),
        )
        assert np.all(result.level_grid == TraversabilityLevel.FREE.value)

    def test_medium_risk_level(self):
        """Cost in [80, 179] -> MEDIUM_RISK level."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=120, unknown_pct=0.0)
        result = engine.process(
            fused=fused,
            semantic=_make_semantic(),
            geometry=_make_geometry(),
            odometry=_make_odometry(),
        )
        assert np.all(result.level_grid == TraversabilityLevel.MEDIUM_RISK.value)

    def test_high_risk_level(self):
        """Cost in [180, 219] -> HIGH_RISK level."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=200, unknown_pct=0.0)
        result = engine.process(
            fused=fused,
            semantic=_make_semantic(),
            geometry=_make_geometry(),
            odometry=_make_odometry(),
        )
        assert np.all(result.level_grid == TraversabilityLevel.HIGH_RISK.value)


class TestUncertaintyAndDisagreement:

    def test_uncertainty_inflates_cost(self):
        """High uncertainty should shift costs upward."""
        engine = TraversabilityMapEngine()
        # Low uncertainty baseline
        fused_low = _make_fused(cost_val=50, uncertainty_val=0.1)
        result_low = engine.process(
            fused=fused_low, semantic=_make_semantic(),
            geometry=_make_geometry(), odometry=_make_odometry(),
        )
        # High uncertainty
        fused_high = _make_fused(cost_val=50, uncertainty_val=0.9)
        result_high = engine.process(
            fused=fused_high, semantic=_make_semantic(),
            geometry=_make_geometry(), odometry=_make_odometry(),
        )
        # Cost should be higher with high uncertainty
        assert result_high.cost_grid.mean() > result_low.cost_grid.mean()

    def test_disagreement_adds_penalty(self):
        """Disagreement zones should have +40 cost penalty."""
        engine = TraversabilityMapEngine()
        # No disagreement
        fused_clean = _make_fused(cost_val=60, disagreement_pct=0.0)
        result_clean = engine.process(
            fused=fused_clean, semantic=_make_semantic(),
            geometry=_make_geometry(), odometry=_make_odometry(),
        )
        # With disagreement
        fused_dis = _make_fused(cost_val=60, disagreement_pct=0.3)
        result_dis = engine.process(
            fused=fused_dis, semantic=_make_semantic(),
            geometry=_make_geometry(), odometry=_make_odometry(),
        )
        # Disagreement cells should have higher cost
        dis_cells = fused_dis.disagreement_mask
        clean_cost = result_clean.cost_grid[~dis_cells].mean()
        dis_cost = result_dis.cost_grid[dis_cells].mean()
        assert dis_cost > clean_cost


class TestPoseDependentScaling:

    def test_localization_lost_inflates_costs(self):
        """When VO is TRACKING_LOST, non-preferred costs should be inflated by 1.5x."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=100, unknown_pct=0.0)

        result_ok = engine.process(
            fused=fused, semantic=_make_semantic(),
            geometry=_make_geometry(),
            odometry=_make_odometry(TrackingStatus.TRACKING_OK),
        )
        result_lost = engine.process(
            fused=fused, semantic=_make_semantic(),
            geometry=_make_geometry(),
            odometry=_make_odometry(TrackingStatus.TRACKING_LOST, confidence=0.05),
        )
        # Lost should have higher costs
        assert result_lost.cost_grid.mean() > result_ok.cost_grid.mean()

    def test_degraded_tracking_minor_inflation(self):
        """Degraded tracking should inflate less than lost."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=100, unknown_pct=0.0)

        result_ok = engine.process(
            fused=fused, semantic=_make_semantic(),
            geometry=_make_geometry(),
            odometry=_make_odometry(TrackingStatus.TRACKING_OK),
        )
        result_deg = engine.process(
            fused=fused, semantic=_make_semantic(),
            geometry=_make_geometry(),
            odometry=_make_odometry(TrackingStatus.TRACKING_DEGRADED, confidence=0.5),
        )
        result_lost = engine.process(
            fused=fused, semantic=_make_semantic(),
            geometry=_make_geometry(),
            odometry=_make_odometry(TrackingStatus.TRACKING_LOST, confidence=0.05),
        )
        assert result_deg.cost_grid.mean() > result_ok.cost_grid.mean()
        assert result_lost.cost_grid.mean() > result_deg.cost_grid.mean()


class TestConfigurableThresholds:

    def test_custom_thresholds_change_classification(self):
        """Custom config with different boundaries should produce different level counts."""
        # Default: preferred_max=19, a cost of 15 is PREFERRED
        engine_default = TraversabilityMapEngine()
        fused = _make_fused(cost_val=15, unknown_pct=0.0)
        result_default = engine_default.process(
            fused=fused, semantic=_make_semantic(),
            geometry=_make_geometry(), odometry=_make_odometry(),
        )
        assert result_default.level_counts["PREFERRED"] == 10000

        # Custom: preferred_max=10, cost 15 is now FREE
        custom_cfg = TraversabilityCostConfig(preferred_max=10)
        engine_custom = TraversabilityMapEngine(config=custom_cfg)
        result_custom = engine_custom.process(
            fused=fused, semantic=_make_semantic(),
            geometry=_make_geometry(), odometry=_make_odometry(),
        )
        assert result_custom.level_counts["FREE"] == 10000

    def test_hot_swap_config(self):
        """Engine should allow config hot-swap without recreation."""
        engine = TraversabilityMapEngine()
        new_cfg = TraversabilityCostConfig(preferred_max=5)
        engine.update_config(new_cfg)
        assert engine.config.preferred_max == 5


class TestOutputInvariants:

    def test_level_counts_sum_to_total(self):
        """Sum of per-level counts must equal total grid cells."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=50, unknown_pct=0.3, obstacle_pct=0.05)
        result = engine.process(
            fused=fused, semantic=_make_semantic(),
            geometry=_make_geometry(), odometry=_make_odometry(),
        )
        total = sum(result.level_counts.values())
        assert total == 100 * 100

    def test_obstacle_grid_superset_of_fusion(self):
        """Traversability obstacle grid must be a superset of fusion obstacle mask."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=50, obstacle_pct=0.1)
        result = engine.process(
            fused=fused, semantic=_make_semantic(),
            geometry=_make_geometry(), odometry=_make_odometry(),
        )
        # Every cell that is obstacle in fusion must be obstacle in traversability
        fusion_obs = fused.obstacle_mask
        trav_obs = result.obstacle_grid
        assert np.all(trav_obs[fusion_obs]),             "Traversability obstacle grid is not a superset of fusion obstacle mask"

    def test_visual_map_shape_and_dtype(self):
        """Visual RGBA map must be (H, W, 4) uint8."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=50)
        result = engine.process(
            fused=fused, semantic=_make_semantic(),
            geometry=_make_geometry(), odometry=_make_odometry(),
        )
        assert result.visual_map.shape == (100, 100, 4)
        assert result.visual_map.dtype == np.uint8

    def test_confidence_bounded(self):
        """Output confidence must be in [0.0, 1.0]."""
        engine = TraversabilityMapEngine()
        fused = _make_fused(cost_val=50, unknown_pct=0.8)
        result = engine.process(
            fused=fused, semantic=_make_semantic(confidence=0.3),
            geometry=_make_geometry(confidence=0.2),
            odometry=_make_odometry(confidence=0.1),
        )
        assert 0.0 <= result.confidence <= 1.0


class TestRealScenarioReplay:

    def test_real_scenario_1_produces_valid_output(self):
        """End-to-end on scenario_1: traversability map must produce valid output."""
        from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
        from src.pipeline import NavigationPipeline

        pipeline = NavigationPipeline()
        loader = OutdoorDatasetLoader("datasets/processed/scenario_1_open_path")

        frame = loader.get_frame(7)
        assert frame is not None

        cmd, telemetry = pipeline.process_frame(frame)

        trav = telemetry.get("traversability")
        assert trav is not None, "Traversability map missing from telemetry"
        assert trav.level_grid.shape == (100, 100)
        assert trav.cost_grid.shape == (100, 100)
        assert trav.obstacle_grid.shape == (100, 100)
        assert trav.visual_map.shape == (100, 100, 4)

        # Rule 11: unknown cells must not be PREFERRED or FREE
        unknown_levels = trav.level_grid[telemetry["fused"].unknown_mask]
        assert np.all(unknown_levels == TraversabilityLevel.UNKNOWN.value)

        # Must have some traversable cells
        assert trav.level_counts["PREFERRED"] + trav.level_counts["FREE"] > 0

        # Latency should be sub-millisecond (just classification, no heavy compute)
        assert trav.latency_ms < 10.0, f"Traversability latency {trav.latency_ms}ms too high"
