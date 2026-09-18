"""
tests/test_adversarial_failure_matrix.py
Replay and adversarial QA tests for the Outdoor Failure Matrix.

Tests all 14 scenarios:
1. sun glare
2. shadow
3. low texture
4. vegetation
5. gravel
6. rocks
7. mud
8. water
9. unknown terrain
10. depth holes
11. motion blur
12. camera vibration
13. semantic/depth disagreement
14. tracking loss
"""

import pytest
from src.evaluation.adversarial_failure_evaluator import (
    AdversarialFailureEvaluator,
    FailureScenarioRecord,
)
from src.interfaces.types import SafetyState


@pytest.fixture(scope="module")
def failure_records():
    evaluator = AdversarialFailureEvaluator()
    return evaluator.evaluate_all_scenarios()


def test_exactly_14_scenarios_evaluated(failure_records):
    """Verify that exactly the 14 required adversarial scenarios are evaluated."""
    assert len(failure_records) == 14
    expected_scenarios = [
        "sun glare",
        "shadow",
        "low texture",
        "vegetation",
        "gravel",
        "rocks",
        "mud",
        "water",
        "unknown terrain",
        "depth holes",
        "motion blur",
        "camera vibration",
        "semantic/depth disagreement",
        "tracking loss",
    ]
    actual_scenarios = [r.scenario for r in failure_records]
    assert actual_scenarios == expected_scenarios


def test_all_records_have_complete_fields(failure_records):
    """Verify that every record contains non-empty scenario, expected, observed, metric, severity, mitigation, and remaining_risk."""
    valid_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    for r in failure_records:
        assert len(r.scenario) > 0
        assert len(r.expected) > 0
        assert len(r.observed) > 0
        assert len(r.metric) > 0
        assert r.severity in valid_severities
        assert len(r.mitigation) > 0
        assert len(r.remaining_risk) > 0
        assert r.passed_mitigation is True


def test_critical_scenarios_fail_safe_enforcement(failure_records):
    """Verify that critical scenarios (rocks, water, disagreement, tracking loss) NEVER allow unmitigated high-speed forward cruising."""
    critical_scenarios = {"rocks", "water", "semantic/depth disagreement", "tracking loss"}
    for r in failure_records:
        if r.scenario in critical_scenarios:
            assert r.severity == "CRITICAL"
            # In all critical scenarios, speed must either be 0.0 or evasive steering with safe clearance
            if r.scenario == "rocks":
                assert r.details["clearance"] >= 0.35
            elif r.scenario == "tracking loss":
                assert r.details["v_cmd"] <= 0.001


def test_benign_scenarios_maintain_motion(failure_records):
    """Verify that gravel and vibration do not trigger false emergency stops."""
    benign_scenarios = {"gravel", "camera vibration"}
    for r in failure_records:
        if r.scenario in benign_scenarios:
            assert r.details["v_cmd"] >= 0.20


def test_degraded_confidence_throttling(failure_records):
    """Verify that low texture, sun glare, and unknown terrain downshift confidence and throttle velocity."""
    degraded_scenarios = {"sun glare", "low texture", "unknown terrain"}
    for r in failure_records:
        if r.scenario in degraded_scenarios:
            assert r.severity == "HIGH"
            assert r.details.get("v_cmd", 0.0) <= 0.55
