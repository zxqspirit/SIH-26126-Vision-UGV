"""8-Connected A* Global Path Planner on 2D Traversability Costmap.

Finds optimal topological route from current UGV position to target GoalPose.
Enforces:
- Strict avoidance of lethal obstacle regions (cost >= 220)
- Risk penalty for high/medium risk terrain
- Respect of vehicle footprint via inflated costmap
- Smooth path generation
"""

from __future__ import annotations

import heapq
import math
import time
from typing import List, Tuple, Optional, Set
import numpy as np

from ..interfaces.types import GoalPose, NavigationPath
from .costmap_2d import Costmap2D


class AStarGlobalPlanner:
    """A* grid pathfinder on 2D local costmap."""

    # 8-connected motion primitives: (dx, dy, metric_distance)
    MOTIONS: List[Tuple[int, int, float]] = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
    ]

    def __init__(
        self,
        lethal_cost_thresh: int = 220,
        risk_weight: float = 2.5,
        max_iterations: int = 15000,
    ) -> None:
        self.lethal_thresh = lethal_cost_thresh
        self.risk_weight = risk_weight
        self.max_iterations = max_iterations

    def plan(
        self,
        costmap: Costmap2D,
        start_x_m: float = 0.0,
        start_y_m: float = 0.0,
        goal: Optional[GoalPose] = None,
    ) -> NavigationPath:
        """Compute an A* path from start position to goal on costmap."""
        start_time = time.perf_counter()

        # If no goal provided, default to forward lookahead point along heading (+6.0m)
        if goal is None:
            goal_x = 6.0
            goal_y = 0.0
            tolerance_cells = int(round(0.6 / costmap.resolution_m))
        else:
            goal_x = goal.x
            goal_y = goal.y
            tolerance_cells = max(1, int(round(goal.tolerance_m / costmap.resolution_m)))

        # Convert start and goal to grid coordinates
        s_row, s_col = costmap.world_to_grid(start_x_m, start_y_m)
        g_row, g_col = costmap.world_to_grid(goal_x, goal_y)

        # Clip goal to costmap bounds if outside
        g_row = max(0, min(costmap.grid_h - 1, g_row))
        g_col = max(0, min(costmap.grid_w - 1, g_col))

        # Check start bounds
        if not costmap.is_in_bounds(s_row, s_col):
            return NavigationPath(waypoints=[], is_valid=False)

        # If start is inside a lethal zone (e.g. initial pose edge overlap), find nearest non-lethal cell
        if costmap.inflated_grid[s_row, s_col] >= self.lethal_thresh:
            s_row, s_col = self._find_nearest_free_cell(costmap, s_row, s_col)

        # Priority queue for A*: entries are (f_score, g_score, (row, col))
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, 0.0, (s_row, s_col)))

        came_from_r = np.full((costmap.grid_h, costmap.grid_w), -1, dtype=np.int32)
        came_from_c = np.full((costmap.grid_h, costmap.grid_w), -1, dtype=np.int32)
        g_score = np.full((costmap.grid_h, costmap.grid_w), float('inf'), dtype=np.float32)
        g_score[s_row, s_col] = 0.0

        closed_mask = np.zeros((costmap.grid_h, costmap.grid_w), dtype=bool)

        reached_node: Optional[Tuple[int, int]] = None
        closest_node: Tuple[int, int] = (s_row, s_col)
        closest_dist = math.hypot(s_row - g_row, s_col - g_col)

        iterations = 0
        grid = costmap.inflated_grid
        res = costmap.resolution_m
        h_grid, w_grid = costmap.grid_h, costmap.grid_w

        while open_set and iterations < self.max_iterations:
            iterations += 1
            current_f, current_g, current = heapq.heappop(open_set)
            curr_r, curr_c = current

            if closed_mask[curr_r, curr_c]:
                continue
            closed_mask[curr_r, curr_c] = True

            dist_to_goal = math.hypot(curr_r - g_row, curr_c - g_col)
            if dist_to_goal < closest_dist:
                closest_dist = dist_to_goal
                closest_node = current

            # Goal check within tolerance
            if dist_to_goal <= tolerance_cells:
                reached_node = current
                break

            for dr, dc, step_cost_mult in self.MOTIONS:
                nr, nc = curr_r + dr, curr_c + dc

                if not (0 <= nr < h_grid and 0 <= nc < w_grid):
                    continue
                if closed_mask[nr, nc]:
                    continue

                cell_cost = int(grid[nr, nc])
                if cell_cost >= self.lethal_thresh:
                    continue  # Lethal obstacle avoidance

                # Edge cost: metric distance + terrain risk penalty
                step_dist_m = step_cost_mult * res
                risk_penalty = (cell_cost / 254.0) * self.risk_weight
                tentative_g = current_g + step_dist_m * (1.0 + risk_penalty)

                if tentative_g < g_score[nr, nc]:
                    came_from_r[nr, nc] = curr_r
                    came_from_c[nr, nc] = curr_c
                    g_score[nr, nc] = tentative_g
                    h_score = math.hypot(nr - g_row, nc - g_col) * res
                    f_score = tentative_g + h_score
                    heapq.heappush(open_set, (f_score, tentative_g, (nr, nc)))

        # Reconstruct path from reached_node (or closest_node if goal wasn't strictly reached)
        target_node = reached_node if reached_node is not None else closest_node
        if target_node == (s_row, s_col):
            start_x, start_y = costmap.grid_to_world(s_row, s_col)
            return NavigationPath(
                waypoints=[(start_x, start_y)],
                total_length_m=0.0,
                accumulated_cost=float(grid[s_row, s_col]),
                is_valid=(reached_node is not None),
            )

        raw_path: List[Tuple[int, int]] = [target_node]
        cr, cc = target_node
        while came_from_r[cr, cc] != -1:
            pr, pc = int(came_from_r[cr, cc]), int(came_from_c[cr, cc])
            raw_path.append((pr, pc))
            cr, cc = pr, pc
        raw_path.reverse()

        # Convert grid coordinates to world coordinates (meters)
        waypoints: List[Tuple[float, float]] = []
        total_len = 0.0
        total_cost = 0.0
        prev_pt: Optional[Tuple[float, float]] = None

        # Subsample path (every 2-3 cells) for clean waypoints
        step_stride = 2 if len(raw_path) > 10 else 1
        subsampled_path = raw_path[::step_stride]
        if raw_path[-1] not in subsampled_path:
            subsampled_path.append(raw_path[-1])

        for r, c in subsampled_path:
            x_m, y_m = costmap.grid_to_world(r, c)
            waypoints.append((round(x_m, 2), round(y_m, 2)))
            total_cost += float(grid[r, c])
            if prev_pt is not None:
                total_len += math.hypot(x_m - prev_pt[0], y_m - prev_pt[1])
            prev_pt = (x_m, y_m)

        is_valid = reached_node is not None or closest_dist <= tolerance_cells * 1.5

        return NavigationPath(
            waypoints=waypoints,
            total_length_m=round(total_len, 2),
            accumulated_cost=round(total_cost, 1),
            is_valid=is_valid,
        )

    def _find_nearest_free_cell(self, costmap: Costmap2D, r: int, c: int) -> Tuple[int, int]:
        """BFS outwards to find the nearest non-lethal cell."""
        for radius in range(1, 10):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < costmap.grid_h and 0 <= nc < costmap.grid_w:
                        if costmap.inflated_grid[nr, nc] < self.lethal_thresh:
                            return nr, nc
        return r, c
