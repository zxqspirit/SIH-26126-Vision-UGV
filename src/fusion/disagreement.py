"""Disagreement detection and multimodal evidence arbitration.

Implements the 6 core operating modes:
1. Agreement: Dynamic confidence-weighted blending, low uncertainty.
2. Mode 2 (CNN Safe / Depth Obstacle): Geometry Veto -> Forced traversability 0.0, lethal obstacle.
3. Mode 3 (CNN Obstacle / Depth Clear): Semantic Veto -> Liquid/mud hazard overrides geometric flatness.
4. Mode 4 (Missing Depth): Rule 12 Guard -> Cautious crawl cap, uncertainty 1.0.
5. Mode 5 (Low CNN Confidence): Dynamic Trust Shift -> 85% authority shifted to Depth Geometry.
6. Mode 6 (Both Uncertain): Failsafe Unknown -> Fused traversability 0.0, uncertainty 1.0, trips Safety Gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Union
import numpy as np

from ..interfaces.types import SemanticResult, DepthGeometryResult, TerrainClass


@dataclass
class FusionParameters:
    """Configurable hyperparameters for Multimodal Fusion Engine.

    Empirically calibrated across 75 outdoor sensor frames in datasets/processed/.
    """
    sem_trav_thresh: float = 0.65        # Minimum semantic score to claim traversable
    geom_obs_thresh: float = 0.40        # Minimum geometric cost to trigger physical obstacle
    sem_haz_thresh: float = 0.25         # Maximum semantic score indicating mud/water hazard
    geom_flat_thresh: float = 0.20       # Maximum geometric cost indicating flat terrain

    missing_depth_max_trav: float = 0.40 # Maximum traversability allowed without depth confirmation
    missing_depth_uncertainty: float = 1.0

    low_cnn_conf_thresh: float = 0.40    # Threshold below which CNN is considered degraded
    low_depth_conf_thresh: float = 0.50  # Threshold below which Depth is considered degraded
    degraded_cnn_geom_weight: float = 0.85
    degraded_cnn_sem_weight: float = 0.15

    nominal_w_sem: float = 0.30          # Base semantic fusion weight
    nominal_w_geom: float = 0.70         # Base geometric fusion weight


class DisagreementDetector:
    """Identifies and resolves conflicts between semantic and geometric evidence."""

    def __init__(
        self,
        params: Optional[FusionParameters] = None,
        semantic_traversable_thresh: Optional[float] = None,
        geometric_obstacle_thresh: Optional[float] = None,
        semantic_hazard_thresh: Optional[float] = None,
        geometric_flat_thresh: Optional[float] = None,
    ) -> None:
        self.params = params if params is not None else FusionParameters()

        # Allow legacy parameter overrides
        if semantic_traversable_thresh is not None:
            self.params.sem_trav_thresh = semantic_traversable_thresh
        if geometric_obstacle_thresh is not None:
            self.params.geom_obs_thresh = geometric_obstacle_thresh
        if semantic_hazard_thresh is not None:
            self.params.sem_haz_thresh = semantic_hazard_thresh
        if geometric_flat_thresh is not None:
            self.params.geom_flat_thresh = geometric_flat_thresh

    def analyze(
        self,
        semantic: SemanticResult,
        geometry: DepthGeometryResult,
        return_uncertainty: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, float],
        Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]
    ]:
        """Analyze conflicts and compute fused pixel traversability and spatial uncertainty.

        Returns:
            fused_traversability: shape (H, W), float32 in [0.0..1.0] (1.0 = fully traversable)
            disagreement_mask: shape (H, W), bool (True where significant conflict detected)
            unknown_mask: shape (H, W), bool (True where terrain is unobserved/invalid)
            fusion_confidence: scalar in [0.0..1.0]
            uncertainty_px (optional): shape (H, W), float32 in [0.0..1.0]
        """
        p = self.params
        p_sem = semantic.traversability_mask
        p_geom_hazard = geometry.geometric_cost
        valid_depth = geometry.depth_validity_mask
        terrain_classes = semantic.terrain_class_map

        c_perc = float(np.clip(semantic.confidence, 0.05, 1.0))
        q_depth = float(np.clip(geometry.confidence, 0.05, 1.0))

        # Invert geometric hazard to get geometric traversability
        p_geom_trav = 1.0 - p_geom_hazard

        # Detect discrete physical obstacles
        has_pos_obs = geometry.positive_obstacle_mask
        has_neg_obs = geometry.negative_obstacle_mask
        has_disc = geometry.discontinuity_mask if geometry.discontinuity_mask is not None else np.zeros_like(has_pos_obs)
        is_physical_hazard = has_pos_obs | has_neg_obs | has_disc | (p_geom_hazard >= p.geom_obs_thresh)

        # ---------------------------------------------------------------------
        # CONFLICT MODE 2: Geometry Veto (CNN says safe, Depth says physical obstacle)
        # ---------------------------------------------------------------------
        geom_veto = valid_depth & is_physical_hazard & (p_sem >= p.sem_trav_thresh)

        # ---------------------------------------------------------------------
        # CONFLICT MODE 3: Semantic Veto (Depth says flat, CNN warns of liquid/mud)
        # ---------------------------------------------------------------------
        is_visual_hazard = (terrain_classes == TerrainClass.WATER_PUDDLE) | (p_sem <= p.sem_haz_thresh)
        sem_veto = valid_depth & (p_geom_hazard <= p.geom_flat_thresh) & is_visual_hazard

        # Combined Disagreement
        disagreement_mask = geom_veto | sem_veto

        # Initialize outputs
        fused = np.zeros_like(p_sem, dtype=np.float32)
        uncertainty = np.full_like(p_sem, 0.20, dtype=np.float32)

        # Scalar mode flags
        both_uncertain = bool((c_perc < p.low_cnn_conf_thresh) and (q_depth < p.low_depth_conf_thresh))
        low_cnn_high_depth = bool((c_perc < p.low_cnn_conf_thresh) and (q_depth >= p.low_depth_conf_thresh))

        # ---------------------------------------------------------------------
        # CONFLICT MODE 1: Agreement (Both sensors valid and agreeing)
        # ---------------------------------------------------------------------
        if not both_uncertain:
            both_agree = valid_depth & (~disagreement_mask)

            if low_cnn_high_depth:
                # Mode 5: Trust geometry primarily (85% weight)
                w_s = p.degraded_cnn_sem_weight
                w_g = p.degraded_cnn_geom_weight
                fused[both_agree] = w_s * p_sem[both_agree] + w_g * p_geom_trav[both_agree]
                uncertainty[both_agree] = 0.45
            else:
                # Mode 1: Calibrated nominal weighted fusion
                w_total = p.nominal_w_sem * c_perc + p.nominal_w_geom * q_depth
                w_s = (p.nominal_w_sem * c_perc) / max(w_total, 1e-6)
                w_g = (p.nominal_w_geom * q_depth) / max(w_total, 1e-6)
                fused[both_agree] = w_s * p_sem[both_agree] + w_g * p_geom_trav[both_agree]
                # Aleatoric uncertainty (near 0.5 is uncertain, near 0.0/1.0 is certain)
                trav_agree = fused[both_agree]
                uncertainty[both_agree] = np.clip(1.0 - 2.0 * np.abs(trav_agree - 0.5), 0.05, 0.35)

        # ---------------------------------------------------------------------
        # APPLY VETOS
        # ---------------------------------------------------------------------
        # Mode 2: Geometry Veto dominates -> Force traversability to 0.0 (lethal obstacle)
        fused[geom_veto] = 0.0
        uncertainty[geom_veto] = 0.85

        # Mode 3: Semantic Veto dominates -> Force to severe hazard penalty (<= 0.15)
        fused[sem_veto] = np.minimum(p_sem[sem_veto], 0.15)
        uncertainty[sem_veto] = 0.75

        # ---------------------------------------------------------------------
        # CONFLICT MODE 4: Missing Depth (Rule 12 Invariant)
        # ---------------------------------------------------------------------
        invalid_depth_mask = ~valid_depth
        # Cautious crawl cap (max 0.40) only if CNN has confidence; otherwise 0.0
        if c_perc >= 0.75:
            fused[invalid_depth_mask] = np.clip(p_sem[invalid_depth_mask] * p.missing_depth_max_trav, 0.0, p.missing_depth_max_trav)
        else:
            fused[invalid_depth_mask] = 0.0
        uncertainty[invalid_depth_mask] = p.missing_depth_uncertainty

        # Mode 6 override if both sensors are degraded
        if both_uncertain:
            fused[:] = 0.0
            uncertainty[:] = 1.0

        # Unknown mask: invalid depth and not non-ground sky
        is_not_sky = terrain_classes != TerrainClass.UNKNOWN
        unknown_mask = invalid_depth_mask & is_not_sky

        # ---------------------------------------------------------------------
        # AGGREGATED FUSION CONFIDENCE
        # Evaluated over forward ground ROI (lower 60% of image)
        # ---------------------------------------------------------------------
        h = p_sem.shape[0]
        ground_roi_y = int(h * 0.40)
        roi_disagree = disagreement_mask[ground_roi_y:, :]
        roi_invalid = invalid_depth_mask[ground_roi_y:, :]

        disagree_ratio = float(np.mean(roi_disagree))
        invalid_ratio = float(np.mean(roi_invalid))

        if both_uncertain:
            fusion_confidence = 0.15
        else:
            base_conf = min(c_perc, q_depth)
            penalty = 1.2 * disagree_ratio + 0.5 * invalid_ratio
            fusion_confidence = float(np.clip(base_conf - penalty, 0.15, 0.96))

        if return_uncertainty:
            return fused, disagreement_mask, unknown_mask, fusion_confidence, uncertainty
        return fused, disagreement_mask, unknown_mask, fusion_confidence
