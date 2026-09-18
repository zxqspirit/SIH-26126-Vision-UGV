"""Comprehensive unit tests for multimodal semantic and geometric fusion.

Tests the 6 core operating modes:
1. Agreement (both safe / both obstacle) -> weighted fusion, low uncertainty.
2. Mode 2: CNN safe / Depth obstacle conflict -> GEOMETRY VETO (forced lethal obstacle).
3. Mode 3: CNN obstacle / Depth clear conflict -> SEMANTIC VETO (liquid/mud hazard overrides flat depth).
4. Mode 4: Missing depth -> RULE 12 GUARD (clamped crawl, uncertainty 1.0, non-zero cost).
5. Mode 5: Low CNN confidence -> Dynamic trust shift to Depth Geometry (85%).
6. Mode 6: Both uncertain -> Failsafe unknown state (confidence < 0.30).
7. BEV Costmap & Rule 11 Invariant: Unobserved cells assigned default_unknown_cost (128).
"""

import pytest
import numpy as np
from src.interfaces.types import (
    CameraIntrinsics,
    SemanticResult,
    DepthGeometryResult,
    TerrainClass,
    FusedTraversabilityResult,
)
from src.fusion.disagreement import DisagreementDetector, FusionParameters
from src.fusion.fusion_engine import FusionEngine
from src.depth_geometry.point_cloud import DepthProjector


@pytest.fixture
def detector():
    return DisagreementDetector()


@pytest.fixture
def fusion_engine():
    intrinsics = CameraIntrinsics(fx=385.0, fy=385.0, cx=320.0, cy=240.0, width=640, height=480)
    return FusionEngine(intrinsics)


def test_mode1_agreement_high_confidence(detector):
    """Mode 1: Both modalities agree on clear traversable ground."""
    h, w = 100, 100
    p_sem = np.full((h, w), 0.95, dtype=np.float32)
    classes = np.full((h, w), TerrainClass.TRAVERSABLE_DIRT, dtype=np.int32)
    semantic = SemanticResult(p_sem, classes, confidence=0.92)

    geometry = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=np.zeros((h, w), dtype=bool),
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=np.ones((h, w), dtype=bool),
        geometric_cost=np.zeros((h, w), dtype=np.float32),
        confidence=0.95,
    )

    fused_trav, disagree, unknown, conf, unc = detector.analyze(semantic, geometry, return_uncertainty=True)

    # Both agree -> No disagreement
    assert not np.any(disagree), "Disagreement falsely flagged on agreed terrain!"
    # Traversability should be very high (> 0.90)
    assert np.all(fused_trav >= 0.90), "Fused traversability lower than expected on agreed safe ground!"
    # Spatial uncertainty should be low (< 0.25)
    assert np.all(unc <= 0.25), "Uncertainty elevated on agreed safe ground!"
    # Fusion confidence should be high
    assert conf >= 0.85, f"Fusion confidence {conf} too low on clean agreeing data!"


def test_mode2_geometry_veto_physical_obstacle(detector):
    """Mode 2: Geometry Veto - Physical obstacle overrides semantic traversability claims."""
    h, w = 100, 100
    # Semantics mistakenly claims high traversability (0.90 dirt/grass)
    p_sem = np.full((h, w), 0.90, dtype=np.float32)
    classes = np.full((h, w), TerrainClass.LOW_GRASS, dtype=np.int32)
    semantic = SemanticResult(p_sem, classes, confidence=0.88)

    # But depth geometry detects a 0.85 positive obstacle hazard in center
    p_geom_cost = np.zeros((h, w), dtype=np.float32)
    p_geom_cost[40:60, 40:60] = 0.85
    pos_obs = np.zeros((h, w), dtype=bool)
    pos_obs[40:60, 40:60] = True
    valid_depth = np.ones((h, w), dtype=bool)

    geometry = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=pos_obs,
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=valid_depth,
        geometric_cost=p_geom_cost,
        confidence=0.92,
    )

    fused_trav, disagree, unknown, conf, unc = detector.analyze(semantic, geometry, return_uncertainty=True)

    # Disagreement must be detected in the conflicting region
    assert np.any(disagree[40:60, 40:60]), "Conflict between semantics and geometry was not detected!"
    # Geometry Veto: Physical obstacle overrides semantic claims -> fused traversability forced to 0.0
    assert np.all(fused_trav[40:60, 40:60] == 0.0), "Geometry veto failed: obstacle was not marked impassable!"
    # Uncertainty in conflict zone must be elevated
    assert np.all(unc[40:60, 40:60] >= 0.70), "Uncertainty was not elevated in conflict zone!"


def test_mode3_semantic_veto_liquid_mud_hazard(detector):
    """Mode 3: Semantic Veto - Water puddle / mud hazard overrides geometrically flat surface."""
    h, w = 100, 100
    # Semantics warns of water puddle (Class 7) with very low traversability (0.10)
    p_sem = np.full((h, w), 0.90, dtype=np.float32)
    p_sem[40:60, 40:60] = 0.10
    classes = np.full((h, w), TerrainClass.PAVED_ROAD, dtype=np.int32)
    classes[40:60, 40:60] = TerrainClass.WATER_PUDDLE
    semantic = SemanticResult(p_sem, classes, confidence=0.90)

    # But depth sensor sees perfectly flat geometric ground (cost = 0.0)
    geometry = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=np.zeros((h, w), dtype=bool),
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=np.ones((h, w), dtype=bool),
        geometric_cost=np.zeros((h, w), dtype=np.float32),
        confidence=0.95,
    )

    fused_trav, disagree, unknown, conf, unc = detector.analyze(semantic, geometry, return_uncertainty=True)

    # Disagreement must be detected in puddle region
    assert np.any(disagree[40:60, 40:60]), "Semantic veto conflict was not detected!"
    # Traversability in puddle must be severely penalized (<= 0.15)
    assert np.all(fused_trav[40:60, 40:60] <= 0.15), "Semantic veto failed: puddle traversability was not penalized!"
    # Uncertainty in puddle region must be elevated
    assert np.all(unc[40:60, 40:60] >= 0.60), "Uncertainty was not elevated in puddle region!"


def test_mode4_missing_depth_rule12_guard(detector):
    """Mode 4: Missing depth must NEVER become free space (Rule 12)."""
    h, w = 100, 100
    semantic = SemanticResult(
        traversability_mask=np.full((h, w), 0.95, dtype=np.float32),
        terrain_class_map=np.full((h, w), TerrainClass.TRAVERSABLE_DIRT, dtype=np.int32),
        confidence=0.85,
    )

    # Left half has valid depth, right half has missing depth (e.g. solar glare)
    valid_depth = np.ones((h, w), dtype=bool)
    valid_depth[:, 50:] = False

    geometry = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=np.zeros((h, w), dtype=bool),
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=valid_depth,
        geometric_cost=np.where(valid_depth, 0.0, 0.60).astype(np.float32),
        confidence=0.65,
    )

    fused_trav, disagree, unknown, conf, unc = detector.analyze(semantic, geometry, return_uncertainty=True)

    # Missing depth region must be marked unknown
    assert np.all(unknown[:, 50:]), "Missing depth was not marked unknown!"
    # RULE 12 INVARIANT: Traversability must NEVER reach free space (> 0.50)
    assert not np.any(fused_trav[:, 50:] > 0.40), "Rule 12 violation: missing depth allowed > 0.40 traversability!"
    # Uncertainty must be maximum (1.0)
    assert np.all(unc[:, 50:] == 1.0), "Missing depth did not assign 1.0 uncertainty!"


def test_mode5_low_cnn_confidence_trust_shift(detector):
    """Mode 5: When CNN confidence is low (dust/blur), authority shifts 85% to Depth Geometry."""
    h, w = 100, 100
    # Degraded CNN confidence (0.25)
    semantic = SemanticResult(
        traversability_mask=np.full((h, w), 0.50, dtype=np.float32),
        terrain_class_map=np.full((h, w), TerrainClass.UNKNOWN, dtype=np.int32),
        confidence=0.25,
    )

    # Clear Depth Geometry sees flat traversable ground (cost = 0.0, trav = 1.0)
    geometry = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=np.zeros((h, w), dtype=bool),
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=np.ones((h, w), dtype=bool),
        geometric_cost=np.zeros((h, w), dtype=np.float32),
        confidence=0.92,
    )

    fused_trav, _, _, conf, unc = detector.analyze(semantic, geometry, return_uncertainty=True)

    # Since geometry weight is 85%, fused traversability should be:
    # 0.15 * 0.50 + 0.85 * 1.0 = 0.075 + 0.85 = 0.925
    assert np.all(fused_trav >= 0.90), "Authority did not shift to Depth Geometry under low CNN confidence!"


def test_mode6_both_uncertain_failsafe_trip(detector):
    """Mode 6: When both CNN and Depth are degraded, failsafe triggers (conf < 0.30)."""
    h, w = 100, 100
    semantic = SemanticResult(
        traversability_mask=np.full((h, w), 0.50, dtype=np.float32),
        terrain_class_map=np.full((h, w), TerrainClass.UNKNOWN, dtype=np.int32),
        confidence=0.20,
    )
    geometry = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=np.zeros((h, w), dtype=bool),
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=np.ones((h, w), dtype=bool),
        geometric_cost=np.full((h, w), 0.50, dtype=np.float32),
        confidence=0.20,
    )

    fused_trav, _, _, conf, unc = detector.analyze(semantic, geometry, return_uncertainty=True)

    # Failsafe trip: traversability forced to 0.0
    assert np.all(fused_trav == 0.0), "Double uncertainty did not force 0.0 traversability!"
    # Confidence must drop < 0.30 (tripping deterministic SafetyGate to SAFETY_STOP)
    assert conf < 0.30, f"Confidence {conf} did not drop below 0.30 under double sensor degradation!"
    # Uncertainty must be 1.0
    assert np.all(unc == 1.0)


def test_bev_costmap_uncertainty_and_rule11(fusion_engine):
    """Verifies BEV costmap accumulation, obstacle mask, uncertainty grid, and Rule 11."""
    h, w = 480, 640
    p_sem = np.full((h, w), 0.90, dtype=np.float32)
    classes = np.full((h, w), TerrainClass.TRAVERSABLE_DIRT, dtype=np.int32)
    semantic = SemanticResult(p_sem, classes, confidence=0.88)

    # Obstacle located in center of image at ~3m
    p_geom = np.zeros((h, w), dtype=np.float32)
    p_geom[200:260, 280:360] = 0.90
    pos_obs = np.zeros((h, w), dtype=bool)
    pos_obs[200:260, 280:360] = True

    valid_depth = np.ones((h, w), dtype=bool)
    # Beyond 6m depth is missing
    valid_depth[:150, :] = False

    geometry = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=pos_obs,
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=valid_depth,
        geometric_cost=p_geom,
        confidence=0.85,
    )

    # Compute 3D points
    depth_m = np.full((h, w), 3.0, dtype=np.float32)
    depth_m[:150, :] = 15.0  # Out of range
    proj = DepthProjector(fusion_engine.intrinsics)
    points_base, _ = proj.project_and_transform_to_base_link(depth_m)

    fused = fusion_engine.fuse(semantic, geometry, points_base)

    # 1. Output structures exist
    assert fused.obstacle_mask is not None, "Obstacle mask not generated in BEV grid!"
    assert fused.uncertainty_grid is not None, "Uncertainty grid not generated in BEV grid!"
    assert fused.fused_traversability is not None

    # 2. Obstacle detected in grid
    assert np.any(fused.obstacle_mask), "Obstacle not reflected in BEV obstacle mask!"

    # 3. Rule 11 Invariant: Unobserved cells must have non-zero cost (default_unknown_cost = 128)
    unobserved = fused.unknown_mask
    assert np.any(unobserved), "Expected unobserved cells in 10x10m BEV grid!"
    unobs_costs = fused.fused_costmap[unobserved]
    assert np.all(unobs_costs == 128), "Rule 11 violation: unobserved cells assigned 0 cost instead of 128!"
