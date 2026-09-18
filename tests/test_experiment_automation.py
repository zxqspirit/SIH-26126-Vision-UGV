"""
tests/test_experiment_automation.py
Automated Testing for Experiment Automation, Non-Overwriting Storage, and Replay Reproducibility.
"""

import json
import os
import shutil
import pytest
from scripts.run_experiment import run_experiment
from scripts.evaluate_run import compare_two_runs, audit_single_run
from scripts.generate_report import generate_single_run_report, generate_comparison_report


@pytest.fixture(scope="module")
def test_exp_base_dir(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("test_experiments")
    return str(base_dir)


def test_experiment_execution_and_metadata_completeness(test_exp_base_dir):
    """Verify that run_experiment records all mandatory metadata and metrics without omission."""
    exp_id = "EXP_TEST_METADATA"
    res = run_experiment(
        scenario="scenario_1_open_path",
        experiment_id=exp_id,
        output_base_dir=test_exp_base_dir,
        frame_limit=3,
        verbose=False,
    )

    run_dir = res["run_dir"]
    assert os.path.exists(run_dir)

    # 1. Required files must exist
    required_files = [
        "experiment_meta.json",
        "configuration.json",
        "metrics.json",
        "failure_summary.json",
        "telemetry.jsonl",
        "summary.md",
    ]
    for rf in required_files:
        p = os.path.join(run_dir, rf)
        assert os.path.exists(p), f"Missing required file: {rf}"

    # 2. Metadata completeness
    meta = res["metadata"]
    assert meta["experiment_id"] == exp_id
    assert meta["scenario"] == "scenario_1_open_path"
    assert "git_commit" in meta
    assert "commit_hash" in meta["git_commit"]
    assert "model_version" in meta
    assert "model_name" in meta["model_version"]
    assert "timestamp_start" in meta
    assert "timestamp_end" in meta

    # 3. Configuration completeness
    cfg = res["configuration"]
    assert "traversability" in cfg
    assert "safety" in cfg
    assert "navigation" in cfg

    # 4. Metrics & Failures
    metrics = res["metrics"]
    assert metrics["total_frames"] == 3
    assert "mean_fps" in metrics
    assert "mean_latency_ms" in metrics
    assert "safety_state_distribution" in metrics
    assert "failure_summary" in res


def test_never_overwrite_existing_results(test_exp_base_dir):
    """Invariant: Re-running with an existing experiment ID must NEVER overwrite existing results."""
    fixed_id = "EXP_TEST_NO_OVERWRITE"

    # First run
    res1 = run_experiment(
        scenario="scenario_1_open_path",
        experiment_id=fixed_id,
        output_base_dir=test_exp_base_dir,
        frame_limit=2,
        verbose=False,
    )
    dir1 = res1["run_dir"]
    assert os.path.basename(dir1) == fixed_id

    # Place a unique marker file in the first directory
    marker_path = os.path.join(dir1, "unique_marker.txt")
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("ORIGINAL_RUN_MARKER")

    # Second run with IDENTICAL ID
    res2 = run_experiment(
        scenario="scenario_1_open_path",
        experiment_id=fixed_id,
        output_base_dir=test_exp_base_dir,
        frame_limit=2,
        verbose=False,
    )
    dir2 = res2["run_dir"]

    # Invariant checks:
    # 1. New directory was created with disambiguated name
    assert dir1 != dir2
    assert os.path.exists(dir1)
    assert os.path.exists(dir2)

    # 2. Original directory contents were NOT overwritten or deleted
    assert os.path.exists(marker_path)
    with open(marker_path, "r", encoding="utf-8") as f:
        assert f.read() == "ORIGINAL_RUN_MARKER"

    # 3. Disambiguated directory has its own metadata
    with open(os.path.join(dir2, "experiment_meta.json"), "r", encoding="utf-8") as f:
        meta2 = json.load(f)
    assert meta2["experiment_id"] == os.path.basename(dir2)


def test_reproducibility_two_runs_identical_within_tolerances(test_exp_base_dir):
    """Execute the same sequence twice and verify outputs match within strict numerical tolerances."""
    res_a = run_experiment(
        scenario="scenario_1_open_path",
        experiment_id="EXP_TEST_REPRO_A",
        output_base_dir=test_exp_base_dir,
        frame_limit=5,
        verbose=False,
    )
    res_b = run_experiment(
        scenario="scenario_1_open_path",
        experiment_id="EXP_TEST_REPRO_B",
        output_base_dir=test_exp_base_dir,
        frame_limit=5,
        verbose=False,
    )

    eval_result = compare_two_runs(
        res_a["run_dir"],
        res_b["run_dir"],
        tol_v=1e-4,
        tol_w=1e-4,
        tol_conf=1e-4,
        tol_clearance=1e-4,
        verbose=False,
    )

    assert eval_result["reproducibility_passed"] is True
    assert eval_result["total_frames_compared"] == 5
    assert eval_result["mismatches"]["tolerance_violations_count"] == 0
    assert eval_result["mismatches"]["safety_state_mismatches_count"] == 0
    assert eval_result["mismatches"]["steering_direction_mismatches_count"] == 0
    assert eval_result["max_observed_differences"]["max_diff_linear_velocity"] <= 1e-4
    assert eval_result["max_observed_differences"]["max_diff_angular_velocity"] <= 1e-4


def test_generate_report_utilities(test_exp_base_dir):
    """Verify single-run and comparative report generators produce valid non-empty markdown."""
    run_dir = os.path.join(test_exp_base_dir, "EXP_TEST_METADATA")

    # Single run report
    md_single = generate_single_run_report(run_dir)
    assert "# Experiment Technical Report: EXP_TEST_METADATA" in md_single
    assert "## 2. Executive Performance Metrics" in md_single
    assert "## 3. Safety State Distribution" in md_single

    # Comparison report
    eval_res = compare_two_runs(
        os.path.join(test_exp_base_dir, "EXP_TEST_REPRO_A"),
        os.path.join(test_exp_base_dir, "EXP_TEST_REPRO_B"),
        verbose=False,
    )
    eval_json_path = os.path.join(test_exp_base_dir, "eval_temp.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_res, f, indent=2)

    md_comp = generate_comparison_report(eval_json_path)
    assert "# Replay Reproducibility Verification Report" in md_comp
    assert "REPRODUCIBILITY VERIFIED (PASS)" in md_comp
    assert "PASS (Zero Drift)" in md_comp
