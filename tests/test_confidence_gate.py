"""Tests for the confidence-based safety gate (Rule 9, 10, 13).

Validates:
- 4 deterministic safety states: HIGH, MEDIUM, LOW, CRITICAL
- 5 input signals: CNN confidence, Depth quality, Disagreement, VO quality, Temporal consistency
- Behavioral mapping:
    HIGH = normal recommendation
    MEDIUM = lower speed / larger margin
    LOW = conservative recommendation / re-observe
    CRITICAL = safe-stop recommendation
- Mandatory logging for every decision: timestamp, confidence, reason, action
"""

import pytest
from src.safety.safety_gate import SafetyGate
from src.interfaces.types import (
    SafetyAction,
    SafetyDecisionLog,
    SafetyState,
    TrackingStatus,
)


@pytest.fixture
def gate():
    return SafetyGate()


# ---------------------------------------------------------------------------
# 1. State & Behavior Tests
# ---------------------------------------------------------------------------

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
        timestamp=100.0,
    )
    assert res.safety_state == SafetyState.HIGH
    assert res.action == SafetyAction.NORMAL.value
    assert res.commanded_linear_velocity == 0.80
    assert res.commanded_angular_velocity == 0.00
    assert not res.is_emergency_stop
    assert res.speed_scale_factor == 1.0
    assert res.clearance_inflation_factor == 1.0


def test_medium_confidence_scales_velocity_and_expands_margin(gate):
    """Moderate confidence (0.45 <= C < 0.75) scales speed down and increases clearance margin."""
    res = gate.arbitrate(
        nominal_v=0.80,
        nominal_w=0.20,
        c_perc=0.55,
        c_geom=0.60,
        c_vo=0.70,
        c_fusion=0.55,
        tracking_status=TrackingStatus.TRACKING_OK,
        min_obstacle_dist_m=3.0,
        timestamp=101.0,
    )
    assert res.safety_state == SafetyState.MEDIUM
    assert res.action == SafetyAction.LOWER_SPEED_EXPAND_MARGIN.value
    assert res.commanded_linear_velocity <= gate.v_cautious_max
    assert res.commanded_linear_velocity < 0.80
    assert res.clearance_inflation_factor == 1.3
    assert not res.is_emergency_stop


def test_low_confidence_conservative_crawl_and_large_margin(gate):
    """Low confidence (0.25 <= C < 0.45) triggers conservative crawl and 1.6x clearance margin."""
    res = gate.arbitrate(
        nominal_v=0.80,
        nominal_w=0.10,
        c_perc=0.35,  # In [0.25, 0.45)
        c_geom=0.85,
        c_vo=0.85,
        c_fusion=0.85,
        tracking_status=TrackingStatus.TRACKING_OK,
        min_obstacle_dist_m=3.0,
        timestamp=102.0,
    )
    assert res.safety_state == SafetyState.LOW
    assert res.action == SafetyAction.CONSERVATIVE_REOBSERVE.value
    assert res.commanded_linear_velocity <= gate.v_crawl_max
    assert res.clearance_inflation_factor == 1.6
    assert not res.is_emergency_stop


def test_low_confidence_triggers_stop(gate):
    """Rule 13: Critical low confidence (< 0.25) must trigger an immediate emergency safe-stop."""
    res = gate.arbitrate(
        nominal_v=0.80,
        nominal_w=0.10,
        c_perc=0.18,  # Below low threshold (0.25)
        c_geom=0.90,
        c_vo=0.88,
        c_fusion=0.82,
        tracking_status=TrackingStatus.TRACKING_OK,
        min_obstacle_dist_m=5.0,
        timestamp=103.0,
    )
    assert res.safety_state == SafetyState.CRITICAL
    assert res.action == SafetyAction.SAFE_STOP.value
    assert res.commanded_linear_velocity == 0.0
    assert res.commanded_angular_velocity == 0.0
    assert res.is_emergency_stop
    assert any("Emergency safety stop" in r for r in res.audit_reasons)


def test_localization_loss_triggers_stop(gate):
    """When VO tracking is lost, vehicle immediately halts in CRITICAL state."""
    res = gate.arbitrate(
        nominal_v=0.60,
        nominal_w=0.00,
        c_perc=0.90,
        c_geom=0.95,
        c_vo=0.10,
        c_fusion=0.90,
        tracking_status=TrackingStatus.TRACKING_LOST,
        min_obstacle_dist_m=5.0,
        timestamp=104.0,
    )
    assert res.safety_state == SafetyState.CRITICAL
    assert res.action == SafetyAction.SAFE_STOP.value
    assert res.commanded_linear_velocity == 0.0
    assert res.commanded_angular_velocity == 0.0
    assert res.is_emergency_stop
    assert any("Visual odometry lost" in r for r in res.audit_reasons)


# ---------------------------------------------------------------------------
# 2. Input Signal Inspection Tests
# ---------------------------------------------------------------------------

def test_depth_quality_degradation_downshifts_state(gate):
    """Depth geometry quality drop causes safe state downshift."""
    res = gate.arbitrate(
        nominal_v=0.70,
        nominal_w=0.0,
        c_perc=0.90,
        c_geom=0.30,  # Depth degraded
        c_vo=0.90,
        c_fusion=0.85,
        timestamp=105.0,
    )
    assert res.safety_state == SafetyState.LOW
    assert res.commanded_linear_velocity <= gate.v_crawl_max


def test_semantic_geometric_disagreement_penalty(gate):
    """High disagreement ratio penalizes overall confidence and forces downshift."""
    # With 0 disagreement, confidence is 0.85 -> HIGH
    res_agree = gate.arbitrate(
        nominal_v=0.60, nominal_w=0.0,
        c_perc=0.85, c_geom=0.85, c_vo=0.85, c_fusion=0.85,
        disagreement_ratio=0.0,
        timestamp=106.0,
    )
    assert res_agree.safety_state == SafetyState.HIGH

    # With severe disagreement (>= 0.60), immediately trips CRITICAL safe stop
    res_crit_disagree = gate.arbitrate(
        nominal_v=0.60, nominal_w=0.0,
        c_perc=0.85, c_geom=0.85, c_vo=0.85, c_fusion=0.85,
        disagreement_ratio=0.65,
        timestamp=107.0,
    )
    assert res_crit_disagree.safety_state == SafetyState.CRITICAL
    assert res_crit_disagree.action == SafetyAction.SAFE_STOP.value
    assert res_crit_disagree.is_emergency_stop


def test_temporal_sudden_drop_triggers_reobserve_hold(gate):
    """Sudden drop in confidence (> 0.30) triggers temporal re-observe hold."""
    # First frame: high confidence
    gate.arbitrate(
        nominal_v=0.70, nominal_w=0.0,
        c_perc=0.90, c_geom=0.90, c_vo=0.90, c_fusion=0.90,
        timestamp=108.0,
    )
    # Second frame: sudden drop to 0.40 (delta = 0.50 > 0.30)
    res_drop = gate.arbitrate(
        nominal_v=0.70, nominal_w=0.0,
        c_perc=0.40, c_geom=0.40, c_vo=0.90, c_fusion=0.40,
        timestamp=109.0,
    )
    assert res_drop.safety_state == SafetyState.LOW
    assert res_drop.action == SafetyAction.CONSERVATIVE_REOBSERVE.value
    assert res_drop.decision_log.reobserve_active is True
    assert res_drop.commanded_linear_velocity == 0.0  # Translation paused to re-observe


# ---------------------------------------------------------------------------
# 3. Mandatory Decision Logging Invariant Tests
# ---------------------------------------------------------------------------

def test_every_decision_logs_required_fields(gate):
    """Every decision must log: timestamp, confidence, reason, action."""
    test_ts = 12345.678
    res = gate.arbitrate(
        nominal_v=0.50,
        nominal_w=0.05,
        c_perc=0.80,
        c_geom=0.82,
        c_vo=0.85,
        c_fusion=0.80,
        timestamp=test_ts,
    )
    log = res.decision_log
    assert log is not None
    assert log.timestamp == test_ts
    assert 0.0 <= log.confidence <= 1.0
    assert isinstance(log.reason, str) and len(log.reason) > 0
    assert log.action in [a.value for a in SafetyAction]

    # Check logger in-memory retrieval
    latest = gate.logger.get_latest()
    assert latest is not None
    assert latest.timestamp == test_ts
    assert latest.confidence == log.confidence
    assert latest.reason == log.reason
    assert latest.action == log.action


def test_logger_summary_statistics(gate):
    """Logger compiles summary statistics across cycles."""
    gate.logger.clear()
    for i in range(10):
        gate.arbitrate(
            nominal_v=0.5, nominal_w=0.0,
            c_perc=0.85, c_geom=0.85, c_vo=0.85, c_fusion=0.85,
            timestamp=float(i),
        )
    stats = gate.logger.get_summary_statistics()
    assert stats["total_records"] == 10
    assert stats["mean_confidence"] > 0.70
    assert "HIGH" in stats["state_distribution"]
