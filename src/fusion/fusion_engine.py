"""Semantic + geometric fusion engine.

Converts fused multimodal evidence into an ego-centric 2D Bird's-Eye-View (BEV) costmap grid.
Enforces Rule 11: Unknown terrain is never automatically free space (default_unknown_cost = 128).
Projects:
  - Fused Traversability Costmap [0..254]
  - Obstacle Mask (bool: lethal collision boundaries)
  - Spatial Uncertainty Grid [0.0..1.0]
  - Disagreement & Unknown Grids
"""

from __future__ import annotations

import time
from typing import Tuple, Optional
import numpy as np

from ..interfaces.types import (
    CameraIntrinsics,
    SemanticResult,
    DepthGeometryResult,
    FusedTraversabilityResult,
)
from .disagreement import DisagreementDetector, FusionParameters


class FusionEngine:
    """Fuses semantic segmentation and 3D depth geometry into a local BEV costmap."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        grid_size_m: Tuple[float, float] = (10.0, 10.0),  # 10m forward, 10m wide (-5 to +5)
        resolution_m: float = 0.1,  # 10 cm per cell -> 100x100 grid
        default_unknown_cost: int = 128,  # Rule 11: non-zero unknown penalty
        params: Optional[FusionParameters] = None,
    ) -> None:
        self.intrinsics = intrinsics
        self.grid_size_m = grid_size_m
        self.resolution_m = resolution_m
        self.default_unknown_cost = default_unknown_cost
        self.params = params if params is not None else FusionParameters()
        self.disagreement_detector = DisagreementDetector(params=self.params)

        # Grid dimensions
        self.grid_h = int(grid_size_m[0] / resolution_m)  # Forward cells (X: 0 to 10m)
        self.grid_w = int(grid_size_m[1] / resolution_m)  # Lateral cells (Y: -5 to +5m)
        self.origin_y_cell = self.grid_w // 2  # Y=0 is center column

    def fuse(
        self,
        semantic: SemanticResult,
        geometry: DepthGeometryResult,
        points_base_link: np.ndarray,
    ) -> FusedTraversabilityResult:
        """Fuse multimodal evidence into an ego-centric local costmap."""
        start_time = time.perf_counter()

        # 1. Pixel-level conflict resolution, veto enforcement, and uncertainty (< 3 ms)
        (
            fused_trav_px,
            disagree_px,
            unknown_px,
            fused_conf,
            uncertainty_px,
        ) = self.disagreement_detector.analyze(semantic, geometry, return_uncertainty=True)

        # 2. Initialize 2D BEV grids
        # Rule 11: Unknown cells are initialized with default_unknown_cost (128), NOT 0 (free space)!
        costmap_grid = np.full((self.grid_h, self.grid_w), self.default_unknown_cost, dtype=np.uint8)
        observed_count = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        accum_cost = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        max_cost_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        accum_uncertainty = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        disagree_grid = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        obstacle_grid = np.zeros((self.grid_h, self.grid_w), dtype=bool)

        # 3. Project 3D points from base_link to 2D BEV grid cells
        # Subsample step=2 for high-speed projection without losing resolution
        valid_depth = geometry.depth_validity_mask[::2, ::2]
        x_pts = points_base_link[::2, ::2, 0][valid_depth]
        y_pts = points_base_link[::2, ::2, 1][valid_depth]

        # Convert fused pixel traversability [0..1] to cost [0..254]
        # traversability 1.0 (open path) -> cost 0
        # traversability 0.0 (solid obstacle) -> cost 254 (lethal)
        sub_trav = fused_trav_px[::2, ::2][valid_depth]
        px_cost = ((1.0 - sub_trav) * 254.0).astype(np.float32)
        disagree_vals = disagree_px[::2, ::2][valid_depth]
        uncertainty_vals = uncertainty_px[::2, ::2][valid_depth]

        # Map to grid indices
        # Forward: row 0 is at UGV, row grid_h-1 is 10m forward
        row_idx = (x_pts / self.resolution_m).astype(np.int32)
        # Lateral: col 0 is -5m (right), center is 0m, col grid_w-1 is +5m (left)
        col_idx = (self.origin_y_cell + (y_pts / self.resolution_m)).astype(np.int32)

        # Filter points within grid bounds
        in_grid = (row_idx >= 0) & (row_idx < self.grid_h) & (col_idx >= 0) & (col_idx < self.grid_w)
        r_valid = row_idx[in_grid]
        c_valid = col_idx[in_grid]
        cost_valid = px_cost[in_grid]
        dis_valid = disagree_vals[in_grid]
        unc_valid = uncertainty_vals[in_grid]

        # Vectorized accumulation using numpy ufuncs (< 4 ms)
        np.add.at(accum_cost, (r_valid, c_valid), cost_valid)
        np.add.at(accum_uncertainty, (r_valid, c_valid), unc_valid)
        np.add.at(observed_count, (r_valid, c_valid), 1)
        np.maximum.at(max_cost_grid, (r_valid, c_valid), cost_valid)

        if np.any(dis_valid):
            disagree_grid[r_valid[dis_valid], c_valid[dis_valid]] = True

        observed_mask = observed_count > 0

        # Mean pooling for normal traversable terrain
        mean_cost = np.zeros_like(accum_cost)
        mean_cost[observed_mask] = accum_cost[observed_mask] / observed_count[observed_mask]

        # Hazard Max-Pooling: if any lethal obstacle (cost >= 220) fell in cell, cell is lethal
        final_cost = mean_cost.copy()
        lethal_in_cell = observed_mask & (max_cost_grid >= 220.0)
        final_cost[lethal_in_cell] = max_cost_grid[lethal_in_cell]

        costmap_grid[observed_mask] = np.clip(final_cost[observed_mask], 0, 254).astype(np.uint8)

        # Obstacle grid: True where cell cost >= 220 (impassable)
        obstacle_grid[observed_mask] = costmap_grid[observed_mask] >= 220

        # Uncertainty Grid: unobserved cells receive maximum uncertainty (1.0)
        uncertainty_grid = np.ones((self.grid_h, self.grid_w), dtype=np.float32)
        uncertainty_grid[observed_mask] = accum_uncertainty[observed_mask] / observed_count[observed_mask]

        # Rule 11: Unknown mask indicates cells that were never observed by the camera
        unknown_grid = ~observed_mask

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return FusedTraversabilityResult(
            fused_costmap=costmap_grid,
            disagreement_mask=disagree_grid,
            unknown_mask=unknown_grid,
            confidence=fused_conf,
            resolution_m=self.resolution_m,
            origin_x_m=0.0,
            origin_y_m=self.grid_size_m[1] / 2.0,
            grid_size_m=self.grid_size_m,
            latency_ms=latency_ms,
            fused_traversability=fused_trav_px,
            obstacle_mask=obstacle_grid,
            uncertainty_grid=uncertainty_grid,
            uncertainty_px=uncertainty_px,
        )
