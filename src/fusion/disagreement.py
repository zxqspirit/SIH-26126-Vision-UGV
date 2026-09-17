"""Disagreement detection between semantic perception and metric 3D depth geometry.

Enforces:
1. Geometry Veto: Physical obstacle overrides semantic traversability claims.
2. Semantic Veto: Visual hazards (water puddle, mud) override geometric flatness.
3. Sensor Degradation: Invalid depth triggers unverified uncertainty (Rule 12).
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from ..interfaces.types import SemanticResult, DepthGeometryResult, TerrainClass


class DisagreementDetector:
    """Identifies and resolves conflicts between semantic and geometric evidence."""

    def __init__(
        self,
        semantic_traversable_thresh: float = 0.70,
        geometric_obstacle_thresh: float = 0.50,
        semantic_hazard_thresh: float = 0.30,
        geometric_flat_thresh: float = 0.20,
    ) -> None:
        self.sem_trav_th = semantic_traversable_thresh
        self.geom_obs_th = geometric_obstacle_thresh
        self.sem_haz_th = semantic_hazard_thresh
        self.geom_flat_th = geometric_flat_thresh

    def analyze(
        self,
        semantic: SemanticResult,
        geometry: DepthGeometryResult,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Analyze conflicts and compute fused pixel traversability.

        Returns:
            fused_traversability: shape (H, W), float32 in [0.0..1.0] (1.0 = fully traversable)
            disagreement_mask: shape (H, W), bool (True where significant conflict detected)
            unknown_mask: shape (H, W), bool (True where terrain is unobserved/invalid)
            fusion_confidence: scalar in [0.0..1.0]
        """
        p_sem = semantic.traversability_mask
        p_geom_hazard = geometry.geometric_cost
        valid_depth = geometry.depth_validity_mask
        terrain_classes = semantic.terrain_class_map

        # Invert geometric hazard to get geometric traversability
        p_geom_trav = 1.0 - p_geom_hazard

        # 1. Geometry Veto: Semantics thinks it's open ground, but depth detects physical obstacle
        geom_veto = (p_sem >= self.sem_trav_th) & (p_geom_hazard >= self.geom_obs_th) & valid_depth

        # 2. Semantic Veto: Depth sees flat ground, but semantics warns of mud, puddle, or water
        is_visual_hazard = (terrain_classes == TerrainClass.WATER_PUDDLE) | (p_sem <= self.sem_haz_th)
        sem_veto = (p_geom_hazard <= self.geom_flat_th) & is_visual_hazard & valid_depth

        # Combined disagreement
        disagreement_mask = geom_veto | sem_veto

        # 3. Probabilistic Fusion
        fused = np.zeros_like(p_sem)

        # Region A: Both sensors valid and agreeing
        both_valid = valid_depth & (~disagreement_mask)
        fused[both_valid] = 0.45 * p_sem[both_valid] + 0.55 * p_geom_trav[both_valid]

        # Region B: Geometry Veto dominates -> force to 0 (impassable physical obstacle)
        fused[geom_veto] = 0.0

        # Region C: Semantic Veto dominates -> force to lower traversability
        fused[sem_veto] = np.minimum(p_sem[sem_veto], 0.20)

        # Region D: Depth is invalid (Rule 12: do not treat as free space!)
        invalid_depth_mask = ~valid_depth
        fused[invalid_depth_mask] = np.clip(p_sem[invalid_depth_mask] * 0.5, 0.0, 0.50)

        # Unknown mask: invalid depth AND non-sky
        is_not_sky = terrain_classes != TerrainClass.UNKNOWN
        unknown_mask = invalid_depth_mask & is_not_sky

        # 4. Fusion Confidence: Focus on forward ground region (lower 60% of image)
        h = p_sem.shape[0]
        ground_roi_y = int(h * 0.40)
        roi_disagree = disagreement_mask[ground_roi_y:, :]
        roi_invalid = invalid_depth_mask[ground_roi_y:, :]

        disagree_ratio = float(np.mean(roi_disagree))
        invalid_ratio = float(np.mean(roi_invalid))

        base_conf = min(semantic.confidence, geometry.confidence)
        # Moderate penalty for disagreement or invalid depth in the ground region
        penalty = 1.0 * disagree_ratio + 0.6 * invalid_ratio
        fusion_confidence = float(np.clip(base_conf - penalty, 0.15, 0.96))

        return fused, disagreement_mask, unknown_mask, fusion_confidence
