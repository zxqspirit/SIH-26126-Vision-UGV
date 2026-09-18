"""Unit tests for standardized visual SLAM benchmark metrics and evaluation functions."""

import pytest
import numpy as np
from scripts.benchmark_visual_slam import compute_ate, compute_rpe, run_benchmark_on_scenario


def test_compute_ate_identity():
    """Identical trajectories must have zero ATE RMSE."""
    n = 20
    t = np.linspace(0, 10, n)
    ref = np.column_stack([t, np.sin(t), np.zeros(n)])
    est = np.copy(ref)

    rmse, mean_err, max_err, errors = compute_ate(est, ref, align=True)
    assert rmse == pytest.approx(0.0, abs=1e-6)
    assert mean_err == pytest.approx(0.0, abs=1e-6)
    assert max_err == pytest.approx(0.0, abs=1e-6)


def test_compute_ate_rigid_transform():
    """Umeyama alignment must recover rigid translation and rotation with zero residual error."""
    n = 25
    t = np.linspace(0, 5, n)
    ref = np.column_stack([t, 0.5 * t, np.zeros(n)])

    # Apply 90 deg Z-rotation and [2, 3, 0] translation
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    translation = np.array([2.0, 3.0, 0.0])
    est = (R @ ref.T).T + translation

    rmse, mean_err, max_err, errors = compute_ate(est, ref, align=True)
    assert rmse == pytest.approx(0.0, abs=1e-5)
    assert max_err == pytest.approx(0.0, abs=1e-5)


def test_compute_rpe_constant_offset():
    """Relative pose error measures local displacement discrepancy."""
    n = 10
    ref = np.zeros((n, 3))
    ref[:, 0] = np.arange(n) * 1.0  # 1m per step

    est = np.zeros((n, 3))
    est[:, 0] = np.arange(n) * 0.98  # 0.98m per step (2% error)

    drift_pct, drift_rate = compute_rpe(est, ref, delta_step=1)
    assert drift_pct == pytest.approx(2.0, abs=0.1)


def test_benchmark_scenario_execution():
    """Benchmark runner successfully evaluates scenario_1_open_path on real data."""
    scenario_dir = "datasets/processed/scenario_1_open_path"
    metrics = run_benchmark_on_scenario(scenario_dir)

    assert metrics.total_frames == 15
    assert metrics.mean_latency_ms < 50.0
    assert metrics.fps > 20.0
    assert metrics.status == "PASS"
    assert metrics.mean_inlier_count > 100.0
