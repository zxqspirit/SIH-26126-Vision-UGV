"""Traversability Map Engine: unified, configurable traversability representation.

Bridges perception -> depth geometry -> fusion -> navigation planning by producing:
1. Level Grid: 6-level classified traversability (PREFERRED through UNKNOWN)
2. Cost Grid: Navigation-ready uint8 cost with uncertainty + pose inflation
3. Obstacle Grid: Binary lethal obstacle map (strict superset of fusion obstacles)
4. Uncertainty Grid: Spatial confidence for safety gate
5. Visual Map: RGBA color-coded overlay for dashboard display

All cost thresholds and weights are configurable via TraversabilityCostConfig.
No hard-coded final weights — all values must be validated experimentally.

Enforces Rule 11: Unknown terrain is NEVER classified as free space.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple
import numpy as np

from ..interfaces.types import (
    SemanticResult,
    DepthGeometryResult,
    FusedTraversabilityResult,
    VisualOdometryResult,
    TraversabilityLevel,
    TraversabilityCostConfig,
    TraversabilityMapResult,
    TrackingStatus,
)


# RGBA colormap for visual traversability overlay
LEVEL_COLORS: Dict[int, Tuple[int, int, int, int]] = {
    TraversabilityLevel.PREFERRED.value:   (46, 204, 113, 192),   # Green
    TraversabilityLevel.FREE.value:        (130, 224, 170, 192),  # Lime
    TraversabilityLevel.MEDIUM_RISK.value: (244, 208, 63, 192),   # Yellow
    TraversabilityLevel.HIGH_RISK.value:   (230, 126, 34, 192),   # Orange
    TraversabilityLevel.BLOCKED.value:     (231, 76, 60, 192),    # Red
    TraversabilityLevel.UNKNOWN.value:     (149, 165, 166, 64),   # Gray (transparent)
}


class TraversabilityMapEngine:
    """Produces a unified traversability representation from multimodal sensor fusion.

    Inputs:
        - FusedTraversabilityResult (terrain semantics + obstacle geometry merged)
        - SemanticResult (terrain class information)
        - DepthGeometryResult (geometric obstacle masks)
        - VisualOdometryResult (pose and tracking state)

    Outputs:
        - TraversabilityMapResult containing all 4 output maps + visual overlay
    """

    def __init__(self, config: Optional[TraversabilityCostConfig] = None) -> None:
        self.config = config if config is not None else TraversabilityCostConfig()

    def update_config(self, config: TraversabilityCostConfig) -> None:
        """Hot-swap configuration without recreating the engine."""
        self.config = config

    def process(
        self,
        fused: FusedTraversabilityResult,
        semantic: SemanticResult,
        geometry: DepthGeometryResult,
        odometry: VisualOdometryResult,
    ) -> TraversabilityMapResult:
        """Compute full traversability map from upstream outputs.

        Args:
            fused: BEV costmap, uncertainty, disagreement from fusion engine
            semantic: Terrain class map and CNN confidence
            geometry: Obstacle masks and geometric confidence
            odometry: Pose, tracking status, and VO confidence

        Returns:
            TraversabilityMapResult with all output maps
        """
        t_start = time.perf_counter()
        cfg = self.config

        grid_h, grid_w = fused.fused_costmap.shape

        # ====================================================================
        # Step 1: Base cost computation from configurable terrain base costs
        # ====================================================================
        tc_grid = getattr(fused, 'terrain_class_grid', None)
        if tc_grid is not None:
            base_cost = np.zeros((grid_h, grid_w), dtype=np.float32)
            for tc_val, cost in cfg.terrain_base_costs.items():
                base_cost[tc_grid == tc_val] = float(cost)
            if fused.obstacle_mask is not None:
                base_cost[fused.obstacle_mask] = np.maximum(base_cost[fused.obstacle_mask], 240.0)
            observed = ~fused.unknown_mask
            base_cost[observed] = np.maximum(base_cost[observed], fused.fused_costmap[observed].astype(np.float32))
        else:
            base_cost = fused.fused_costmap.astype(np.float32)

        # ====================================================================
        # Step 2: Apply uncertainty-based cost inflation
        # ====================================================================
        if fused.uncertainty_grid is not None:
            uncertainty = fused.uncertainty_grid
        else:
            uncertainty = np.where(fused.unknown_mask, 1.0, 0.1).astype(np.float32)

        # High uncertainty regions: shift cost upward
        high_unc_mask = uncertainty > cfg.high_uncertainty_thresh
        base_cost[high_unc_mask] += cfg.uncertainty_cost_weight * uncertainty[high_unc_mask] * 254.0

        # ====================================================================
        # Step 3: Apply disagreement penalty
        # ====================================================================
        if fused.disagreement_mask is not None:
            base_cost[fused.disagreement_mask] += cfg.disagreement_cost_penalty

        # ====================================================================
        # Step 4: Enforce Rule 11 — Unknown cells are NEVER free
        # ====================================================================
        base_cost[fused.unknown_mask] = cfg.unknown_cost

        # ====================================================================
        # Step 5: Apply pose-dependent cost inflation
        # ====================================================================
        if odometry.tracking_status == TrackingStatus.TRACKING_LOST:
            # Inflate all non-preferred costs when localization is lost
            non_preferred = base_cost > cfg.preferred_max
            base_cost[non_preferred] *= cfg.localization_lost_inflation
        elif odometry.tracking_status == TrackingStatus.TRACKING_DEGRADED:
            non_preferred = base_cost > cfg.preferred_max
            base_cost[non_preferred] *= cfg.degraded_tracking_inflation

        # ====================================================================
        # Step 6: Clamp to uint8 range
        # ====================================================================
        cost_grid = np.clip(base_cost, 0, 255).astype(np.uint8)

        # ====================================================================
        # Step 7: Classify into 6 traversability levels
        # ====================================================================
        level_grid = np.full((grid_h, grid_w), TraversabilityLevel.UNKNOWN.value, dtype=np.uint8)

        # Unknown cells first (highest priority — they stay UNKNOWN)
        known_mask = ~fused.unknown_mask

        # Classify known cells by cost thresholds
        level_grid[known_mask & (cost_grid <= cfg.preferred_max)] = TraversabilityLevel.PREFERRED.value
        level_grid[known_mask & (cost_grid > cfg.preferred_max) & (cost_grid <= cfg.free_max)] = TraversabilityLevel.FREE.value
        level_grid[known_mask & (cost_grid > cfg.free_max) & (cost_grid <= cfg.medium_risk_max)] = TraversabilityLevel.MEDIUM_RISK.value
        level_grid[known_mask & (cost_grid > cfg.medium_risk_max) & (cost_grid <= cfg.high_risk_max)] = TraversabilityLevel.HIGH_RISK.value
        level_grid[known_mask & (cost_grid > cfg.high_risk_max)] = TraversabilityLevel.BLOCKED.value

        # ====================================================================
        # Step 8: Build obstacle grid (strict superset of fusion obstacles)
        # ====================================================================
        obstacle_grid = (level_grid == TraversabilityLevel.BLOCKED.value)
        # Ensure fusion obstacle cells are always included (monotonic safety)
        if fused.obstacle_mask is not None:
            obstacle_grid = obstacle_grid | fused.obstacle_mask

        # ====================================================================
        # Step 9: Generate RGBA visual overlay
        # ====================================================================
        visual_map = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
        for level_val, rgba in LEVEL_COLORS.items():
            mask = level_grid == level_val
            visual_map[mask] = rgba

        # ====================================================================
        # Step 10: Compute level counts
        # ====================================================================
        level_counts = {}
        for level in TraversabilityLevel:
            level_counts[level.name] = int(np.sum(level_grid == level.value))

        # ====================================================================
        # Step 11: Compute overall confidence
        # ====================================================================
        # Confidence = weighted average of upstream confidences, penalized by unknown ratio
        unknown_ratio = fused.unknown_mask.sum() / max(fused.unknown_mask.size, 1)
        upstream_conf = min(fused.confidence, semantic.confidence, geometry.confidence)
        confidence = float(upstream_conf * (1.0 - 0.3 * unknown_ratio))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return TraversabilityMapResult(
            level_grid=level_grid,
            cost_grid=cost_grid,
            obstacle_grid=obstacle_grid,
            uncertainty_grid=uncertainty,
            visual_map=visual_map,
            level_counts=level_counts,
            confidence=round(confidence, 4),
            latency_ms=round(latency_ms, 3),
            resolution_m=fused.resolution_m,
            grid_size_m=fused.grid_size_m,
        )

    def cost_to_level(self, cost: int) -> TraversabilityLevel:
        """Classify a single cost value into a traversability level."""
        cfg = self.config
        if cost <= cfg.preferred_max:
            return TraversabilityLevel.PREFERRED
        elif cost <= cfg.free_max:
            return TraversabilityLevel.FREE
        elif cost <= cfg.medium_risk_max:
            return TraversabilityLevel.MEDIUM_RISK
        elif cost <= cfg.high_risk_max:
            return TraversabilityLevel.HIGH_RISK
        elif cost <= cfg.blocked_max:
            return TraversabilityLevel.BLOCKED
        else:
            return TraversabilityLevel.UNKNOWN

    def get_terrain_base_cost(self, terrain_class_value: int) -> int:
        """Look up the configurable base cost for a terrain class."""
        return self.config.terrain_base_costs.get(terrain_class_value, self.config.unknown_cost)
