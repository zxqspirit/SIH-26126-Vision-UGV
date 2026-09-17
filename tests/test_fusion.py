"""Tests for multimodal semantic and geometric fusion."""

import pytest
import numpy as np
from src.fusion.disagreement import DisagreementDetector
from src.interfaces.types import SemanticResult, DepthGeometryResult, TerrainClass


@pytest.fixture
def detector():
    return DisagreementDetector()


def test_disagreement_detected(detector):
    """When semantics and geometry disagree on hazard vs traversable, conflict is flagged."""
    h, w = 100, 100
    # Semantics says high traversability (0.90)
    p_sem = np.full((h, w), 0.90, dtype=np.float32)
    classes = np.full((h, w), TerrainClass.TRAVERSABLE_DIRT, dtype=np.int32)
    semantic = SemanticResult(p_sem, classes, confidence=0.88)

    # But depth geometry detects a 0.85 positive obstacle hazard in center
    p_geom_cost = np.zeros((h, w), dtype=np.float32)
    p_geom_cost[40:60, 40:60] = 0.85
    valid_depth = np.ones((h, w), dtype=bool)

    geometry = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=(p_geom_cost >= 0.5),
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=valid_depth,
        geometric_cost=p_geom_cost,
        confidence=0.92,
    )

    fused_trav, disagree, unknown, conf = detector.analyze(semantic, geometry)

    # Disagreement must be detected in the conflicting region
    assert np.any(disagree[40:60, 40:60]), "Conflict between semantics and geometry was not detected!"
    # Geometry Veto: Physical obstacle overrides semantic claims -> fused traversability forced to 0.0
    assert np.all(fused_trav[40:60, 40:60] == 0.0), "Geometry veto failed: obstacle was not marked impassable!"


def test_invalid_depth_lowers_confidence(detector):
    """Invalid depth triggers uncertainty and reduces overall fusion confidence."""
    h, w = 100, 100
    semantic = SemanticResult(
        traversability_mask=np.full((h, w), 0.85, dtype=np.float32),
        terrain_class_map=np.full((h, w), TerrainClass.TRAVERSABLE_DIRT, dtype=np.int32),
        confidence=0.90,
    )

    # Scenario with 50% invalid depth
    valid_depth = np.ones((h, w), dtype=bool)
    valid_depth[:, :50] = False

    geometry = DepthGeometryResult(
        ground_height_map=np.zeros((h, w)),
        positive_obstacle_mask=np.zeros((h, w), dtype=bool),
        negative_obstacle_mask=np.zeros((h, w), dtype=bool),
        depth_validity_mask=valid_depth,
        geometric_cost=np.zeros((h, w), dtype=np.float32),
        confidence=0.90,
    )

    _, _, unknown, conf = detector.analyze(semantic, geometry)

    # Region with invalid depth must be marked unknown
    assert np.all(unknown[:, :50])
    # Confidence must be penalized compared to baseline
    assert conf < 0.90
