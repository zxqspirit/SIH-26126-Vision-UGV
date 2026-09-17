"""Tests for the 2D traversability costmap and Rule 11."""

import pytest
import numpy as np
from src.planning.costmap_2d import Costmap2D
from src.interfaces.types import FusedTraversabilityResult


@pytest.fixture
def costmap():
    return Costmap2D(grid_size_m=(10.0, 10.0), resolution_m=0.1, default_unknown_cost=128)


def test_unknown_cell_is_not_free(costmap):
    """Rule 11: Unknown / unobserved terrain is NEVER automatically free space."""
    # Freshly initialized grid represents unobserved world
    cost = costmap.get_cost(x_m=5.0, y_m=0.0)
    assert cost != 0, "Rule 11 violation: unobserved cell was marked free space (0)!"
    assert cost == 128, f"Expected default unknown penalty (128), got {cost}"

    # Query out-of-bounds position
    oob_cost = costmap.get_cost(x_m=25.0, y_m=0.0)
    assert oob_cost != 0, "Rule 11 violation: out-of-bounds was marked free space!"


def test_obstacle_inflation(costmap):
    """Lethal obstacles must inflate costs within the inscribed and inflation radii."""
    # Place a lethal obstacle at (X=3.0m, Y=0.0m) -> row 30, col 50
    fused_grid = np.zeros((100, 100), dtype=np.uint8)
    fused_grid[30, 50] = 254  # Lethal obstacle

    fused_res = FusedTraversabilityResult(
        fused_costmap=fused_grid,
        disagreement_mask=np.zeros((100, 100), dtype=bool),
        unknown_mask=np.zeros((100, 100), dtype=bool),
        confidence=0.90,
    )
    costmap.update_from_fused_result(fused_res)

    # Inscribed radius (0.35m = ~3 cells around 30, 50): cost must be 254
    assert costmap.get_cost(3.0, 0.0) == 254
    assert costmap.get_cost(3.2, 0.0) == 254

    # Inflation radius (0.5m away from obstacle): cost must be in buffer [1..253]
    inflated_cost = costmap.get_cost(3.5, 0.0)
    assert 0 < inflated_cost < 254, f"Expected inflated cost in (0, 254), got {inflated_cost}"

    # Far away (7.0m): cost must be 0 (free)
    assert costmap.get_cost(7.0, 0.0) == 0


def test_costmap_bounds(costmap):
    """Bounds checking correctly identifies in-grid vs out-of-grid coordinates."""
    assert costmap.is_in_bounds(0, 0)
    assert costmap.is_in_bounds(99, 99)
    assert not costmap.is_in_bounds(-1, 50)
    assert not costmap.is_in_bounds(100, 50)
