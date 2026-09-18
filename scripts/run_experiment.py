#!/usr/bin/env python3
"""
scripts/run_experiment.py
Experiment Automation Orchestrator for SIH 26126.

Workflow:
recorded data -> replay -> perception -> depth -> fusion -> localization -> navigation -> safety -> metrics

Every run records:
- experiment ID
- Git commit
- model version
- scenario
- configuration
- metrics
- failure summary

Enforces: NEVER overwrite existing results.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import uuid
import numpy as np
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_replay import run_replay
from src.pipeline import NavigationPipeline


def get_git_commit() -> Dict[str, Any]:
    """Retrieve Git commit hash and repository status."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status_output) > 0
        return {"commit_hash": commit_hash, "is_dirty": is_dirty, "short_hash": commit_hash[:8]}
    except Exception as e:
        return {"commit_hash": "UNKNOWN_GIT_HASH", "is_dirty": True, "error": str(e)}


def get_model_version(pipeline: Optional[NavigationPipeline] = None) -> Dict[str, Any]:
    """Identify active perception model architecture and version."""
    pipe = pipeline if pipeline is not None else NavigationPipeline()
    model_path = getattr(pipe.perception.net, "model_path", None)
    is_onnx = getattr(pipe.perception.net, "session", None) is not None
    return {
        "model_name": "TraversabilityNet",
        "model_version": "v1.0.0-SIH26126",
        "runtime": "ONNXRuntime" if is_onnx else "ClassicalColorTextureFallback",
        "model_path": str(model_path) if model_path else "default_internal",
    }


def get_pipeline_configuration(pipeline: Optional[NavigationPipeline] = None) -> Dict[str, Any]:
    """Extract complete active algorithmic parameters from pipeline instances."""
    pipe = pipeline if pipeline is not None else NavigationPipeline()
    trav_cfg = pipe.traversability.config
    safety = pipe.safety_gate
    decision = pipe.decision_engine

    return {
        "traversability": {
            "preferred_max": trav_cfg.preferred_max,
            "free_max": trav_cfg.free_max,
            "medium_risk_max": trav_cfg.medium_risk_max,
            "high_risk_max": trav_cfg.high_risk_max,
            "blocked_max": trav_cfg.blocked_max,
            "terrain_base_costs": {str(k): v for k, v in trav_cfg.terrain_base_costs.items()},
            "uncertainty_cost_weight": trav_cfg.uncertainty_cost_weight,
            "disagreement_cost_penalty": trav_cfg.disagreement_cost_penalty,
            "localization_lost_inflation": trav_cfg.localization_lost_inflation,
        },
        "safety": {
            "th_high": getattr(safety, "th_high", 0.75),
            "th_med": getattr(safety, "th_med", 0.45),
            "th_low": getattr(safety, "th_low", 0.25),
            "emergency_dist": getattr(safety, "emergency_dist", 0.35),
            "v_cautious_max": getattr(safety, "v_cautious_max", 0.35),
            "v_crawl_max": getattr(safety, "v_crawl_max", 0.15),
            "inscribed_radius": getattr(safety, "inscribed_radius", 0.30),
        },
        "navigation": {
            "max_v": getattr(decision, "max_v", 0.80),
            "max_w": getattr(decision, "max_w", 0.60),
            "min_v": getattr(decision, "min_v", 0.0),
            "goal_tolerance": getattr(decision, "goal_tolerance", 0.50),
        },
        "intrinsics": {
            "width": pipe.intrinsics.width,
            "height": pipe.intrinsics.height,
            "fx": pipe.intrinsics.fx,
            "fy": pipe.intrinsics.fy,
            "cx": pipe.intrinsics.cx,
            "cy": pipe.intrinsics.cy,
        }
    }


def compute_experiment_metrics(replay_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Aggregate per-frame replay data into comprehensive benchmark metrics and failure summary."""
    frames = replay_data["frames"]
    total = len(frames)
    if total == 0:
        return {}, []

    latencies = [f["latency_ms"] for f in frames]
    confidences = [f["safety"]["confidence"] for f in frames]
    velocities = [f["command"]["linear_velocity"] for f in frames]
    clearances = [f["planning"]["min_clearance_m"] for f in frames if f["planning"]["min_clearance_m"] is not None]

    # State distributions
    safety_states: Dict[str, int] = {}
    nav_states: Dict[str, int] = {}
    tracking_statuses: Dict[str, int] = {}
    failure_summary: List[Dict[str, Any]] = []

    emergency_stops = 0
    tracking_losses = 0
    clearance_breaches = 0

    for f in frames:
        s_state = f["safety"]["state"]
        n_state = f["command"]["navigation_state"]
        t_stat = f["odometry"]["tracking_status"]

        safety_states[s_state] = safety_states.get(s_state, 0) + 1
        nav_states[n_state] = nav_states.get(n_state, 0) + 1
        tracking_statuses[t_stat] = tracking_statuses.get(t_stat, 0) + 1

        if f["safety"]["is_emergency_stop"]:
            emergency_stops += 1
            failure_summary.append({
                "frame_id": f["frame_id"],
                "type": "EMERGENCY_STOP",
                "reason": f["safety"]["primary_reason"],
                "confidence": f["safety"]["confidence"],
            })

        if t_stat == "TRACKING_LOST":
            tracking_losses += 1
            failure_summary.append({
                "frame_id": f["frame_id"],
                "type": "LOCALIZATION_TRACKING_LOST",
                "inliers": f["odometry"]["inliers"],
                "vo_confidence": f["odometry"]["confidence"],
            })

        clear_m = f["planning"]["min_clearance_m"]
        if clear_m is not None and clear_m < 0.35 and not f["safety"]["is_emergency_stop"]:
            clearance_breaches += 1
            failure_summary.append({
                "frame_id": f["frame_id"],
                "type": "CLEARANCE_MARGIN_WARNING",
                "clearance_m": clear_m,
                "threshold_m": 0.35,
            })

    metrics = {
        "total_frames": total,
        "mean_fps": replay_data["mean_fps"],
        "mean_latency_ms": round(float(sum(latencies) / total), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "max_latency_ms": round(float(max(latencies)), 2),
        "mean_confidence": round(float(sum(confidences) / total), 4),
        "min_confidence": round(float(min(confidences)), 4),
        "mean_linear_velocity_mps": round(float(sum(velocities) / total), 4),
        "max_linear_velocity_mps": round(float(max(velocities)), 4),
        "min_clearance_observed_m": round(float(min(clearances)), 4) if clearances else None,
        "safety_state_distribution": {
            k: {"count": v, "percentage": round((v / total) * 100.0, 1)} for k, v in safety_states.items()
        },
        "navigation_state_distribution": {
            k: {"count": v, "percentage": round((v / total) * 100.0, 1)} for k, v in nav_states.items()
        },
        "tracking_status_distribution": {
            k: {"count": v, "percentage": round((v / total) * 100.0, 1)} for k, v in tracking_statuses.items()
        },
        "anomalies": {
            "emergency_stops_count": emergency_stops,
            "tracking_loss_frames": tracking_losses,
            "clearance_breaches_count": clearance_breaches,
            "total_failure_events": len(failure_summary),
        }
    }
    return metrics, failure_summary


def run_experiment(
    scenario: str,
    experiment_id: Optional[str] = None,
    output_base_dir: str = "experiments/runs",
    config_override: Optional[Dict[str, Any]] = None,
    frame_limit: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute a fully tracked, non-overwriting autonomous experiment."""
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not experiment_id:
        uid_short = str(uuid.uuid4())[:8]
        experiment_id = f"EXP_{now_str}_{scenario}_{uid_short}"

    run_dir = os.path.join(output_base_dir, experiment_id)

    # STRICT INVARIANT: Never overwrite existing results
    if os.path.exists(run_dir):
        # Disallow silent overwrite; create a disambiguated non-conflicting path
        counter = 1
        while os.path.exists(f"{run_dir}_{counter}"):
            counter += 1
        run_dir = f"{run_dir}_{counter}"
        experiment_id = f"{experiment_id}_{counter}"
        if verbose:
            print(f"[Notice] Disambiguated experiment ID to prevent overwrite: {experiment_id}")

    os.makedirs(run_dir, exist_ok=False)

    if verbose:
        print(f"================================================================")
        print(f"Starting Experiment: {experiment_id}")
        print(f"Scenario:            {scenario}")
        print(f"Output Directory:    {run_dir}")
        print(f"================================================================")

    # 1. Metadata Collection
    git_info = get_git_commit()
    model_info = get_model_version()
    config_info = get_pipeline_configuration()
    if config_override:
        # Merge overrides
        for section, params in config_override.items():
            if section in config_info and isinstance(params, dict):
                config_info[section].update(params)
            else:
                config_info[section] = params

    # 2. Run Replay
    t_start = datetime.datetime.now().isoformat()
    replay_data = run_replay(
        scenario=scenario,
        config=config_info,
        frame_limit=frame_limit,
        verbose=verbose,
    )
    t_end = datetime.datetime.now().isoformat()

    # 3. Compute Metrics and Failure Summary
    metrics, failure_summary = compute_experiment_metrics(replay_data)

    experiment_metadata = {
        "experiment_id": experiment_id,
        "timestamp_start": t_start,
        "timestamp_end": t_end,
        "scenario": scenario,
        "git_commit": git_info,
        "model_version": model_info,
        "environment": {
            "python_version": sys.version.split()[0],
            "os": sys.platform,
        },
    }

    # 4. Save Structured Non-Overwriting Outputs
    meta_path = os.path.join(run_dir, "experiment_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(experiment_metadata, f, indent=2)

    config_path = os.path.join(run_dir, "configuration.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    failures_path = os.path.join(run_dir, "failure_summary.json")
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump({"total_events": len(failure_summary), "events": failure_summary}, f, indent=2)

    # Line-delimited telemetry log
    telemetry_path = os.path.join(run_dir, "telemetry.jsonl")
    with open(telemetry_path, "w", encoding="utf-8") as f:
        for frame in replay_data["frames"]:
            f.write(json.dumps(frame) + "\n")

    # Generate Markdown Summary
    md_summary = generate_experiment_markdown_summary(experiment_metadata, config_info, metrics, failure_summary)
    summary_path = os.path.join(run_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(md_summary)

    if verbose:
        print(f"\nExperiment completed successfully.")
        print(f"Results archived at: {run_dir}")
        print(f" - Metadata:        {meta_path}")
        print(f" - Configuration:   {config_path}")
        print(f" - Metrics:         {metrics_path}")
        print(f" - Failure Summary: {failures_path}")
        print(f" - Telemetry Log:   {telemetry_path}")
        print(f" - Report:          {summary_path}")

    return {
        "experiment_id": experiment_id,
        "run_dir": run_dir,
        "metadata": experiment_metadata,
        "configuration": config_info,
        "metrics": metrics,
        "failure_summary": failure_summary,
    }


def generate_experiment_markdown_summary(
    meta: Dict[str, Any],
    cfg: Dict[str, Any],
    met: Dict[str, Any],
    fails: List[Dict[str, Any]]
) -> str:
    """Produce executive Markdown report for an experiment run."""
    lines = [
        f"# Experiment Run Report: {meta['experiment_id']}",
        "",
        f"- **Scenario:** `{meta['scenario']}`",
        f"- **Git Commit:** `{meta['git_commit']['commit_hash']}` (Dirty: `{meta['git_commit']['is_dirty']}`)",
        f"- **Model Version:** `{meta['model_version']['model_name']} {meta['model_version']['model_version']}` ({meta['model_version']['runtime']})",
        f"- **Execution Window:** {meta['timestamp_start']} -> {meta['timestamp_end']}",
        "",
        "## Performance Metrics",
        "",
        f"| Metric | Value |",
        f"|:---|:---|",
        f"| Total Frames | `{met.get('total_frames', 0)}` |",
        f"| Mean FPS | `{met.get('mean_fps', 0.0):.1f} FPS` |",
        f"| Mean Latency | `{met.get('mean_latency_ms', 0.0):.1f} ms` |",
        f"| 95th Percentile Latency | `{met.get('p95_latency_ms', 0.0):.1f} ms` |",
        f"| Mean Confidence | `{met.get('mean_confidence', 0.0):.3f}` |",
        f"| Min Confidence | `{met.get('min_confidence', 0.0):.3f}` |",
        f"| Mean Commanded Speed | `{met.get('mean_linear_velocity_mps', 0.0):.2f} m/s` |",
        f"| Min Clearance Observed | `{met.get('min_clearance_observed_m', 'N/A')} m` |",
        f"| Emergency Stops Count | `{met.get('anomalies', {}).get('emergency_stops_count', 0)}` |",
        "",
        "## Safety State Distribution",
        "",
        "| State | Frame Count | Percentage |",
        "|:---|:---|:---|",
    ]
    for st, v in met.get("safety_state_distribution", {}).items():
        lines.append(f"| `{st}` | {v['count']} | {v['percentage']:.1f}% |")

    lines.extend([
        "",
        "## Failure and Anomaly Summary",
        "",
        f"- **Total Anomaly Events:** `{len(fails)}`",
    ])
    if fails:
        lines.append("")
        lines.append("| Frame ID | Anomaly Type | Reason / Inliers |")
        lines.append("|:---|:---|:---|")
        for ev in fails[:15]:
            desc = ev.get("reason") or ev.get("inliers") or ev.get("clearance_m") or "N/A"
            lines.append(f"| `{ev.get('frame_id')}` | `{ev.get('type')}` | {desc} |")
        if len(fails) > 15:
            lines.append(f"| ... | ... ({len(fails)-15} more events) | ... |")
    else:
        lines.append("- Zero safety breaches, emergency stops, or localization dropouts detected.")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run tracked, non-overwriting autonomous navigation experiment.")
    parser.add_argument("--scenario", default="scenario_1_open_path", help="Scenario name")
    parser.add_argument("--id", default=None, help="Custom experiment ID")
    parser.add_argument("--output-dir", default="experiments/runs", help="Base output directory")
    parser.add_argument("--limit", type=int, default=None, help="Frame limit")
    parser.add_argument("--config", default=None, help="Optional JSON config file to override defaults")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose frame logging")
    args = parser.parse_args()

    cfg_override = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_override = json.load(f)

    res = run_experiment(
        scenario=args.scenario,
        experiment_id=args.id,
        output_base_dir=args.output_dir,
        config_override=cfg_override,
        frame_limit=args.limit,
        verbose=not args.quiet,
    )
    print(f"\nExperiment completed: {res['experiment_id']}")


if __name__ == "__main__":
    main()
