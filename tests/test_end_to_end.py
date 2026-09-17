"""Full end-to-end integration tests for SIH 26126 Autonomous Navigation Pipeline."""

import pytest
from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
from src.pipeline import NavigationPipeline
from src.interfaces.types import UGVMotionCommand


def test_end_to_end_pipeline_cycle():
    """Complete sensor frame passes through all 8 stages and outputs a valid UGVMotionCommand."""
    loader = OutdoorDatasetLoader("datasets/processed/scenario_1_open_path")
    assert len(loader) > 0, "Scenario 1 has no frames!"

    frame = loader.get_frame(0)
    assert frame is not None

    pipeline = NavigationPipeline()
    cmd, telemetry = pipeline.process_frame(frame)

    # 1. Output must be a UGVMotionCommand
    assert isinstance(cmd, UGVMotionCommand)
    assert 0.0 <= cmd.linear_velocity <= 1.0
    assert -1.5 <= cmd.angular_velocity <= 1.5
    assert cmd.steering_direction in ("FORWARD", "SLIGHT_LEFT", "SLIGHT_RIGHT", "HARD_LEFT", "HARD_RIGHT", "STOP")
    assert 0.0 <= cmd.confidence <= 1.0

    # 2. Telemetry must include all intermediate representations
    assert "semantic" in telemetry
    assert "geometry" in telemetry
    assert "fused" in telemetry
    assert "odometry" in telemetry
    assert "planning" in telemetry
    assert "safety" in telemetry

    # 3. Processing latency must be within reasonable limits (< 500ms on CPU)
    assert telemetry["total_latency_ms"] < 500.0
