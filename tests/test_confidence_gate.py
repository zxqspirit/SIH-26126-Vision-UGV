"""Tests for the confidence-based safety gate (Rule 9, 10, 13)."""

import pytest
from src.safety.safety_gate import SafetyGate
from src.interfaces.types import SafetyState, TrackingStatus


@pytest.fixture
def gate():
    return SafetyGate()


def test_high_confidence_passes(gate):
    """When all sensor confidences are high and tracking is OK, commands pass unmodified."""
    res = gate.arbitrate(
        nominal_v=0.80,
        nominal_w=0.00,
        c_perc=0.85,
        c_geom=0.90,
        c_vo=0.88,
        c_fusion=0.82,
        tracking_status=TrackingStatus.TRACKING_OK,
        min_obstacle_dist_m=5.0,
    )
    assert res.safety_state == SafetyState.HIGH_CONFIDENCE
    assert res.commanded_linear_velocity == 0.80
    assert res.commanded_angular_velocity == 0.00
    assert not res.is_emergency_stop
    assert res.speed_scale_factor == 1.0


def test_low_confidence_triggers_stop(gate):
    """Rule 13: Low confidence (< 0.25) must trigger an immediate emergency safety stop."""
    res = gate.arbitrate(
        nominal_v=0.80,
        nominal_w=0.10,
        c_perc=0.18,  # Below low threshold (0.25)
        c_geom=0.90,
        c_vo=0.88,
        c_fusion=0.82,
        tracking_status=TrackingStatus.TRACKING_OK,
        min_obstacle_dist_m=5.0,
    )
    assert res.safety_state == SafetyState.SAFETY_STOP
    assert res.commanded_linear_velocity == 0.0
    assert res.commanded_angular_velocity == 0.0
    assert res.is_emergency_stop
    assert any("Emergency safety stop" in r for r in res.audit_reasons)


def test_medium_confidence_scales_velocity(gate):
    """Moderate confidence (0.45 <= C < 0.75) scales speed down and increases clearance."""
    res = gate.arbitrate(
        nominal_v=0.80,
        nominal_w=0.20,
        c_perc=0.55,
        c_geom=0.60,
        c_vo=0.70,
        c_fusion=0.55,
        tracking_status=TrackingStatus.TRACKING_OK,
        min_obstacle_dist_m=3.0,
    )
    assert res.safety_state == SafetyState.CAUTIOUS_DEGRADED
    assert res.commanded_linear_velocity <= gate.v_cautious_max
    assert res.commanded_linear_velocity < 0.80
    assert res.clearance_inflation_factor == 1.3
    assert not res.is_emergency_stop


def test_localization_loss_triggers_stop(gate):
    """When VO tracking is lost, vehicle immediately halts and flags LOCALIZATION_LOST."""
    res = gate.arbitrate(
        nominal_v=0.60,
        nominal_w=0.00,
        c_perc=0.90,
        c_geom=0.95,
        c_vo=0.10,
        c_fusion=0.90,
        tracking_status=TrackingStatus.TRACKING_LOST,
        min_obstacle_dist_m=5.0,
    )
    assert res.safety_state == SafetyState.LOCALIZATION_LOST
    assert res.commanded_linear_velocity == 0.0
    assert res.commanded_angular_velocity == 0.0
    assert res.is_emergency_stop
    assert any("Visual odometry lost" in r for r in res.audit_reasons)
