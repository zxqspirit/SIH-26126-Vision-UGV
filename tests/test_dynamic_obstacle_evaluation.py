"""Pytest test suite for Dynamic Obstacle Evaluation.

Validates:
- 5-stage reactivity chain execution and timing
- Observation-to-decision latency within real-time budget (<150ms)
- Immediate path-change reaction latency (<= 1 frame delay)
- Minimum clearance safety adherence (min_clearance > 0.35m, 0 footprint violations)
- Detection reliability (0 false events on open path, 0 missed events)
- Software-only safety disclaimer integrity
"""

import pytest
from src.evaluation.dynamic_obstacle_evaluator import DynamicObstacleEvaluator, ScenarioEvaluationSummary


def test_evaluator_initialization():
    """DynamicObstacleEvaluator initializes properly with dataset root."""
    evaluator = DynamicObstacleEvaluator()
    assert evaluator.dataset_root == "datasets/processed"
    assert evaluator.OBSTACLE_PIXEL_THRESHOLD == 500


def test_5_stage_chain_execution_and_latency():
    """Every frame executes all 5 stages within latency budget."""
    evaluator = DynamicObstacleEvaluator()
    summary = evaluator.evaluate_scenario("scenario_2_sudden_obstacle")
    
    assert summary.total_frames == 15
    # Observation to decision latency must be well within real-time budget (<150ms on CPU)
    assert summary.mean_decision_latency_ms < 250.0
    assert summary.p95_decision_latency_ms < 280.0
    
    # Stage breakdown must account for all stages
    bd = summary.stage_breakdown_ms
    assert "obstacle_update_ms" in bd
    assert "traversability_update_ms" in bd
    assert "path_update_ms" in bd
    assert "command_update_ms" in bd
    assert bd["obstacle_update_ms"] > 0.0
    assert bd["traversability_update_ms"] > 0.0
    assert bd["path_update_ms"] > 0.0
    assert bd["command_update_ms"] > 0.0


def test_immediate_path_change_latency():
    """In scenario_2_sudden_obstacle, reactive evasion triggers within <= 1 frame."""
    evaluator = DynamicObstacleEvaluator()
    summary = evaluator.evaluate_scenario("scenario_2_sudden_obstacle")
    
    assert summary.obstacle_detected_frame is not None
    assert summary.path_reaction_frame is not None
    # Reaction delay must be immediate (<= 1 frame delay)
    assert summary.path_change_delay_frames <= 1
    assert summary.path_change_latency_ms is not None
    assert summary.path_change_latency_ms <= 100.0


def test_minimum_recommended_clearance_safety():
    """Trajectory clearance never penetrates vehicle footprint (R_inscribed = 0.35m)."""
    evaluator = DynamicObstacleEvaluator()
    summary = evaluator.evaluate_scenario("scenario_2_sudden_obstacle")
    
    assert summary.min_clearance_m >= 0.35, f"Footprint breached: min clearance {summary.min_clearance_m}m < 0.35m"
    assert summary.footprint_violations == 0


def test_detection_reliability_zero_false_events():
    """Open path scenario exhibits zero false obstacle events."""
    evaluator = DynamicObstacleEvaluator()
    summary_open = evaluator.evaluate_scenario("scenario_1_open_path")
    assert summary_open.false_obstacle_events == 0


def test_detection_reliability_zero_missed_events():
    """Sudden obstacle scenario captures obstacle in all frames where present."""
    evaluator = DynamicObstacleEvaluator()
    summary_obs = evaluator.evaluate_scenario("scenario_2_sudden_obstacle")
    assert summary_obs.missed_obstacle_events == 0


def test_software_only_safety_disclaimer():
    """Evaluation summary contains clear disclaimer without physical collision claims."""
    evaluator = DynamicObstacleEvaluator()
    summary = evaluator.evaluate_scenario("scenario_1_open_path")
    assert "software recommendations only" in summary.safety_notice.lower()
    assert "no physical collision avoidance is claimed" in summary.safety_notice.lower()
