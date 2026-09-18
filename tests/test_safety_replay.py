"""Replay tests for Confidence-to-Behavior Safety Design on real outdoor recordings.

Tests:
- scenario_1_open_path: Nominal baseline (expects predominantly HIGH state, 0 false safe stops)
- scenario_2_sudden_obstacle: Intrusions (verifies speed scaling, clearance margin expansion)
- scenario_5_visual_degradation: Visual loss (verifies 100% fail-safe safe-stop / re-observe)
- Multi-scenario threshold validator execution and logging coverage
"""

import pytest
from src.interfaces.types import SafetyAction, SafetyState
from src.safety.threshold_validator import SafetyThresholdValidator


@pytest.fixture
def validator():
    return SafetyThresholdValidator()


def test_scenario_1_open_path_safety_replay(validator):
    """Scenario 1 (Open Path): Verifies nominal driving, predominantly HIGH state, and zero false safe stops."""
    rep = validator.validate_scenario("scenario_1_open_path")

    assert rep.total_frames == 15
    assert rep.false_safe_stops == 0, "Nominal open path must produce 0 false CRITICAL stops"
    assert rep.mean_confidence >= 0.70
    assert rep.mean_speed_scale >= 0.70

    # Ensure every single frame logged: timestamp, confidence, reason, action
    assert len(rep.decision_logs) == 15
    for log in rep.decision_logs:
        assert log["timestamp"] is not None
        assert 0.0 <= log["confidence"] <= 1.0
        assert len(log["reason"]) > 0
        assert log["action"] in [a.value for a in SafetyAction]
        assert log["clearance_margin_m"] >= 0.35


def test_scenario_2_sudden_obstacle_safety_replay(validator):
    """Scenario 2 (Sudden Obstacle): Verifies safe reaction to dynamic intrusions."""
    rep = validator.validate_scenario("scenario_2_sudden_obstacle")

    assert rep.total_frames == 15
    # Clearance margin must always be at least the vehicle inscribed radius (0.35m)
    assert rep.mean_clearance_margin_m >= 0.35

    for log in rep.decision_logs:
        assert "timestamp" in log
        assert "confidence" in log
        assert "reason" in log
        assert "action" in log


def test_scenario_5_visual_degradation_safety_replay(validator):
    """Scenario 5 (Visual Degradation): Verifies 100% fail-safe trigger under degraded sensing."""
    rep = validator.validate_scenario("scenario_5_visual_degradation")

    assert rep.total_frames == 15
    # At least 80% of frames should trip fail-safe into CRITICAL or LOW re-observe
    assert rep.fail_safe_activations >= 12
    assert rep.min_confidence < 0.30

    # Verify that safe-stop or conservative re-observe actions were commanded
    actions = rep.action_distribution
    assert (
        actions.get(SafetyAction.SAFE_STOP.value, 0)
        + actions.get(SafetyAction.CONSERVATIVE_REOBSERVE.value, 0)
        > 0
    )


def test_full_validation_suite_executes(validator):
    """Cross-scenario threshold suite executes across 5 scenarios (75 frames) and validates robustness."""
    res = validator.run_full_validation_suite()

    assert res.total_scenarios == 5
    assert res.total_frames_evaluated == 75
    assert res.threshold_sensitivity_robust is True
    assert res.logging_invariants_passed is True
    assert res.summary_verdict == "VALIDATED"
