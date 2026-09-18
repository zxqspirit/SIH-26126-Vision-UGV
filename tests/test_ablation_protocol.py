"""Test suite for research ablation protocol and runner."""

import os
import json
import pytest
from src.evaluation.ablation_runner import AblationRunner, AblationMode
from src.evaluation.ablation_reporter import AblationReporter


def test_ablation_runner_executes_all_modes():
    """Verify all 5 modes execute on real sequence frames without NaN values."""
    runner = AblationRunner()
    telemetry, summaries = runner.run_all_ablations(
        scenarios=["scenario_1_open_path"],
        max_frames_per_scenario=3,
    )

    assert len(telemetry) == 15  # 3 frames * 5 modes
    assert len(summaries) == 5

    for t in telemetry:
        assert t.fps > 0.0
        assert t.t_total_ms > 0.0
        assert not any(v != v for v in [t.perc_confidence, t.min_clearance_m, t.recommended_v])

    # Mode E should have non-zero safety state
    mode_e_records = [t for t in telemetry if t.mode == AblationMode.MODE_E_FULL_SYSTEM.value]
    assert len(mode_e_records) == 3
    assert all(r.safety_state in ["HIGH", "MEDIUM", "LOW", "CRITICAL"] for r in mode_e_records)


def test_mode_a_false_safe_on_3d_obstacle():
    """Mode A (CNN only) incurs false-safe cases on 3D obstacles due to flat planar depth."""
    runner = AblationRunner()
    telemetry, summaries = runner.run_all_ablations(
        scenarios=["scenario_2_sudden_obstacle"],
        max_frames_per_scenario=5,
    )

    mode_a_summary = summaries[AblationMode.MODE_A_CNN_ONLY.value]
    assert mode_a_summary.total_false_safe_cases > 0


def test_mode_b_false_safe_on_planar_hazard():
    """Mode B (Depth only) incurs false-safe cases on mud/vegetation without semantic CNN."""
    runner = AblationRunner()
    telemetry, summaries = runner.run_all_ablations(
        scenarios=["scenario_3_terrain_boundary"],
        max_frames_per_scenario=5,
    )

    mode_b_summary = summaries[AblationMode.MODE_B_DEPTH_ONLY.value]
    assert mode_b_summary.total_false_safe_cases > 0


def test_mode_e_mitigates_all_false_safes_and_enforces_safety():
    """Mode E maintains 0 false safe cases and activates safe speed throttling/holds."""
    runner = AblationRunner()
    telemetry, summaries = runner.run_all_ablations(
        scenarios=["scenario_2_sudden_obstacle", "scenario_5_visual_degradation"],
        max_frames_per_scenario=5,
    )

    mode_e_summary = summaries[AblationMode.MODE_E_FULL_SYSTEM.value]
    assert mode_e_summary.total_false_safe_cases == 0
    assert mode_e_summary.cautious_throttle_frames + mode_e_summary.safe_stop_frames > 0


def test_ablation_export_and_report_generation(tmp_path):
    """Verify CSV, JSON, PNG charts, and Markdown reports are successfully created."""
    runner = AblationRunner()
    telemetry, summaries = runner.run_all_ablations(
        scenarios=["scenario_1_open_path"],
        max_frames_per_scenario=2,
    )

    out_dir = str(tmp_path / "ablation_test")
    csv_p, json_p = runner.export_results(telemetry, summaries, output_dir=out_dir)

    assert os.path.exists(csv_p)
    assert os.path.exists(json_p)

    reporter = AblationReporter(ablation_json_path=json_p)
    charts = reporter.generate_all_charts()
    assert len(charts) == 3
    for cp in charts:
        assert os.path.exists(cp)
        assert os.path.getsize(cp) > 1000

    report_p = reporter.generate_markdown_report(charts)
    assert os.path.exists(report_p)
    with open(report_p, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Ablation Study" in content
    assert "Mode A" in content
    assert "Mode E" in content
