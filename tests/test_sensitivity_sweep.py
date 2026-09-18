"""Unit tests for the Traversability Sensitivity Sweeper."""

import pytest
import numpy as np

from src.interfaces.types import TraversabilityCostConfig, TerrainClass
from src.traversability.sensitivity_sweep import (
    TraversabilitySensitivitySweeper,
    SweepPointResult,
    TerrainSensitivitySummary,
)


def test_sweeper_initialization():
    """Sweeper correctly initializes with dataset scenarios and nominal config."""
    sweeper = TraversabilitySensitivitySweeper()
    assert len(sweeper.cached_frames) > 0, "No frames loaded from datasets!"
    assert sweeper.nominal_config is not None
    assert len(sweeper.SWEEP_MULTIPLIERS) == 5


def test_evaluate_config_returns_valid_metrics():
    """evaluate_config computes all key trajectory quality metrics."""
    sweeper = TraversabilitySensitivitySweeper()
    # Evaluate with default nominal config
    metrics, trajs, levels = sweeper.evaluate_config(sweeper.nominal_config)

    assert "path_found_rate" in metrics
    assert "mean_v" in metrics
    assert "mean_cost" in metrics
    assert "mean_clearance_m" in metrics
    assert "trajectory_shift_rate" in metrics

    assert 0.0 <= metrics["path_found_rate"] <= 1.0
    assert 0.0 <= metrics["mean_v"] <= 0.85
    assert len(trajs) == len(sweeper.cached_frames)
    assert len(levels) > 0


def test_perturbation_changes_cost():
    """Perturbing a terrain base cost changes resulting mean trajectory cost."""
    sweeper = TraversabilitySensitivitySweeper()

    cfg_low = TraversabilityCostConfig()
    cfg_low.terrain_base_costs[TerrainClass.LOW_GRASS.value] = 18  # -50%

    cfg_high = TraversabilityCostConfig()
    cfg_high.terrain_base_costs[TerrainClass.LOW_GRASS.value] = 52  # +50%

    metrics_low, _, _ = sweeper.evaluate_config(cfg_low)
    metrics_high, _, _ = sweeper.evaluate_config(cfg_high)

    assert metrics_low["path_found_rate"] > 0.0
    assert metrics_high["path_found_rate"] > 0.0
    # Cost with higher terrain base cost must be greater than or equal to lower
    assert metrics_high["mean_cost"] >= metrics_low["mean_cost"]
