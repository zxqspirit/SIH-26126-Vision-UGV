"""2D local traversability costmap with obstacle inflation.

Enforces Rule 11: Unknown terrain is never automatically free space.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import cv2

from ..interfaces.types import FusedTraversabilityResult


class Costmap2D:
    """2D grid representation of local terrain costs and obstacle clearance."""

    def __init__(
        self,
        grid_size_m: Tuple[float, float] = (10.0, 10.0),
        resolution_m: float = 0.1,
        inscribed_radius_m: float = 0.35,  # UGV footprint radius
        inflation_radius_m: float = 0.85,  # Safe obstacle clearance
        default_unknown_cost: int = 128,  # Rule 11
    ) -> None:
        self.grid_size_m = grid_size_m
        self.resolution_m = resolution_m
        self.inscribed_radius = inscribed_radius_m
        self.inflation_radius = inflation_radius_m
        self.default_unknown_cost = default_unknown_cost

        self.grid_h = int(grid_size_m[0] / resolution_m)
        self.grid_w = int(grid_size_m[1] / resolution_m)
        self.origin_y_cell = self.grid_w // 2

        # Initialize grid with default unknown cost (Rule 11)
        self.grid = np.full((self.grid_h, self.grid_w), default_unknown_cost, dtype=np.uint8)
        self.inflated_grid = np.copy(self.grid)
        self.dist_to_lethal_m = np.full((self.grid_h, self.grid_w), 10.0, dtype=np.float32)

    def world_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert robot base_link coordinates (x forward, y left) to grid (row, col)."""
        row = int(round(x_m / self.resolution_m))
        col = int(round(self.origin_y_cell + (y_m / self.resolution_m)))
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid (row, col) to robot base_link coordinates."""
        x_m = float(row * self.resolution_m)
        y_m = float((col - self.origin_y_cell) * self.resolution_m)
        return x_m, y_m

    def is_in_bounds(self, row: int, col: int) -> bool:
        """Check if grid coordinates are within the local map."""
        return (0 <= row < self.grid_h) and (0 <= col < self.grid_w)

    def update_from_fused_result(self, fused: FusedTraversabilityResult) -> None:
        """Update costmap grid from fusion engine output and inflate obstacles."""
        self.grid = np.copy(fused.fused_costmap)

        # Enforce Rule 11: unknown cells must carry a penalty
        if fused.unknown_mask is not None:
            self.grid[fused.unknown_mask] = np.maximum(self.grid[fused.unknown_mask], self.default_unknown_cost)

        self._inflate_obstacles()

    def _inflate_obstacles(self) -> None:
        """Inflate lethal obstacles using Euclidean distance transform."""
        self.inflated_grid = np.copy(self.grid)

        # Lethal obstacles: cost >= 220
        lethal_mask = (self.grid >= 220).astype(np.uint8)

        if np.any(lethal_mask):
            # Compute distance to nearest lethal obstacle in meters
            dist_map_px = cv2.distanceTransform(1 - lethal_mask, cv2.DIST_L2, 5)
            self.dist_to_lethal_m = dist_map_px * self.resolution_m

            # Within inscribed radius: lethal cost (254)
            inscribed_mask = self.dist_to_lethal_m <= self.inscribed_radius
            self.inflated_grid[inscribed_mask] = 254

            # Within inflation radius: decaying cost buffer [1..253]
            inflation_zone = (self.dist_to_lethal_m > self.inscribed_radius) & (self.dist_to_lethal_m <= self.inflation_radius)
            if np.any(inflation_zone):
                decay = np.exp(-3.0 * (self.dist_to_lethal_m[inflation_zone] - self.inscribed_radius) / (self.inflation_radius - self.inscribed_radius))
                inflated_costs = (253.0 * decay).astype(np.uint8)
                self.inflated_grid[inflation_zone] = np.maximum(self.inflated_grid[inflation_zone], inflated_costs)
        else:
            self.dist_to_lethal_m.fill(10.0)

    def get_cost(self, x_m: float, y_m: float) -> int:
        """Get cost at world position in base_link frame. Unknown/out-of-bounds is non-free."""
        row, col = self.world_to_grid(x_m, y_m)
        if not self.is_in_bounds(row, col):
            return self.default_unknown_cost
        return int(self.inflated_grid[row, col])

    def get_obstacle_distance(self, x_m: float, y_m: float) -> float:
        """Get metric distance to nearest lethal obstacle."""
        row, col = self.world_to_grid(x_m, y_m)
        if not self.is_in_bounds(row, col):
            return 0.0
        return float(self.dist_to_lethal_m[row, col])

    def check_footprint_clearance(self, x_m: float, y_m: float) -> Tuple[bool, float, int]:
        """Check if UGV footprint at (x, y) is collision-free.

        Returns:
            is_clear: bool (False if in lethal zone)
            clearance_m: exact metric distance to nearest lethal obstacle
            max_cost: maximum cost inside footprint
        """
        center_row, center_col = self.world_to_grid(x_m, y_m)
        radius_cells = max(1, int(round(self.inscribed_radius / self.resolution_m)))

        r_min = max(0, center_row - radius_cells)
        r_max = min(self.grid_h, center_row + radius_cells + 1)
        c_min = max(0, center_col - radius_cells)
        c_max = min(self.grid_w, center_col + radius_cells + 1)

        footprint_slice = self.inflated_grid[r_min:r_max, c_min:c_max]
        if footprint_slice.size == 0:
            return False, 0.0, 255

        max_cost = int(np.max(footprint_slice))
        is_clear = max_cost < 254
        clearance_m = float(self.dist_to_lethal_m[min(center_row, self.grid_h - 1), min(center_col, self.grid_w - 1)])

        return is_clear, clearance_m, max_cost
