#!/usr/bin/env python3
"""
scripts/generate_report.py
Report Generation Utility for Autonomous Experiments in SIH 26126.

Supports:
1. Single Run Executive Markdown Report
2. Comparative Replay Verification Report (Run A vs Run B)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def generate_single_run_report(run_dir: str) -> str:
    """Build a comprehensive markdown report from run directory artifacts."""
    meta_path = os.path.join(run_dir, "experiment_meta.json")
    cfg_path = os.path.join(run_dir, "configuration.json")
    met_path = os.path.join(run_dir, "metrics.json")
    fail_path = os.path.join(run_dir, "failure_summary.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(met_path, "r", encoding="utf-8") as f:
        met = json.load(f)
    with open(fail_path, "r", encoding="utf-8") as f:
        fails_data = json.load(f)
        fails = fails_data.get("events", [])

    lines = [
        f"# Experiment Technical Report: {meta['experiment_id']}",
        "",
        "## 1. Provenance & Environment",
        "",
        f"- **Experiment ID:** `{meta['experiment_id']}`",
        f"- **Scenario:** `{meta['scenario']}`",
        f"- **Git Commit:** `{meta['git_commit']['commit_hash']}` (Dirty: `{meta['git_commit']['is_dirty']}`)",
        f"- **Model Version:** `{meta['model_version']['model_name']}` ({meta['model_version']['model_version']}) - Runtime: `{meta['model_version']['runtime']}`",
        f"- **Start Time:** `{meta['timestamp_start']}`",
        f"- **End Time:** `{meta['timestamp_end']}`",
        f"- **Environment:** Python `{meta['environment']['python_version']}` on `{meta['environment']['os']}`",
        "",
        "## 2. Executive Performance Metrics",
        "",
        "| Metric | Value | Reference Standard |",
        "|:---|:---|:---|",
        f"| Total Processed Frames | `{met.get('total_frames', 0)}` | Complete Scenario |",
        f"| Mean Frame Rate | `{met.get('mean_fps', 0.0):.1f} FPS` | $\\ge 6.0\\text{{ FPS}}$ (Real-time Target) |",
        f"| Mean Processing Latency | `{met.get('mean_latency_ms', 0.0):.1f} ms` | $\\le 200\\text{{ ms}}$ Bound |",
        f"| 95th Percentile Latency | `{met.get('p95_latency_ms', 0.0):.1f} ms` | Real-time Upper Margin |",
        f"| Max Latency Spike | `{met.get('max_latency_ms', 0.0):.1f} ms` | Peak Workload |",
        f"| Mean Master Confidence ($C_{{total}}$) | `{met.get('mean_confidence', 0.0):.3f}` | $[0.0, 1.0]$ Multi-sensor Gate |",
        f"| Minimum Confidence | `{met.get('min_confidence', 0.0):.3f}` | Fail-Safe Trip Threshold ($0.25$) |",
        f"| Mean Commanded Velocity | `{met.get('mean_linear_velocity_mps', 0.0):.2f} m/s` | Velocity Recommendation |",
        f"| Max Commanded Velocity | `{met.get('max_linear_velocity_mps', 0.0):.2f} m/s` | Speed Ceiling ($0.80\\text{{ m/s}}$) |",
        f"| Minimum Clearance Observed | `{met.get('min_clearance_observed_m', 'N/A')} m` | Buffer Floor ($0.35\\text{{ m}}$) |",
        f"| Total Emergency Stops | `{met.get('anomalies', {}).get('emergency_stops_count', 0)}` | Fail-safe Invocations |",
        f"| Tracking Loss Frames | `{met.get('anomalies', {}).get('tracking_loss_frames', 0)}` | Rule 13 Trigger |",
        "",
        "## 3. Safety State Distribution",
        "",
        "| Safety State | Frames | Distribution % | Operational Mode |",
        "|:---|:---|:---|:---|",
    ]

    for st, d in met.get("safety_state_distribution", {}).items():
        mode_desc = {
            "HIGH": "Nominal progression (Full speed)",
            "MEDIUM": "Caution mode (Scaled speed, wider margin)",
            "LOW": "Conservative crawl / Re-observe",
            "CRITICAL": "Emergency Safe-Stop recommendation",
        }.get(st, "Unknown state")
        lines.append(f"| `{st}` | {d['count']} | {d['percentage']:.1f}% | {mode_desc} |")

    lines.extend([
        "",
        "## 4. Active Pipeline Configuration",
        "",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
        "",
        "## 5. Failure and Anomaly Incident Log",
        "",
        f"- **Total Recorded Anomaly Events:** `{len(fails)}`",
    ])

    if fails:
        lines.append("")
        lines.append("| Frame | Type | Detail / Audit Record |")
        lines.append("|:---|:---|:---|")
        for ev in fails:
            desc = ev.get("reason") or ev.get("inliers") or f"Clearance {ev.get('clearance_m')}m" or "N/A"
            lines.append(f"| `{ev.get('frame_id')}` | `{ev.get('type')}` | {desc} |")
    else:
        lines.append("- Zero safety breaches, emergency stops, or tracking dropouts encountered during this run.")

    lines.append("")
    return "\n".join(lines)


def generate_comparison_report(comparison_json_path: str) -> str:
    """Build a comparative reproducibility report between Run A and Run B."""
    with open(comparison_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    passed = data.get("reproducibility_passed", False)
    status_badge = "REPRODUCIBILITY VERIFIED (PASS)" if passed else "REPRODUCIBILITY FAILED (DIVERGENCE)"

    lines = [
        f"# Replay Reproducibility Verification Report",
        "",
        f"**Verdict:** `{status_badge}`",
        "",
        f"- **Scenario:** `{data.get('scenario')}`",
        f"- **Run A Experiment:** `{data.get('run_a_id')}`",
        f"- **Run B Experiment:** `{data.get('run_b_id')}`",
        f"- **Total Frames Compared:** `{data.get('total_frames_compared')}`",
        "",
        "## 1. Numerical Tolerances & Observed Differences",
        "",
        "| Controlled Parameter | Enforced Tolerance | Max Observed Difference | Within Tolerance? |",
        "|:---|:---|:---|:---:|",
        f"| Commanded Linear Velocity ($v$) | $\\le {data['tolerances']['tol_linear_velocity_mps']:.6f}\\text{{ m/s}}$ | `{data['max_observed_differences']['max_diff_linear_velocity']:.8f} m/s` | {'PASS' if data['max_observed_differences']['max_diff_linear_velocity'] <= data['tolerances']['tol_linear_velocity_mps'] else 'FAIL'} |",
        f"| Commanded Angular Velocity ($\\omega$) | $\\le {data['tolerances']['tol_angular_velocity_radps']:.6f}\\text{{ rad/s}}$ | `{data['max_observed_differences']['max_diff_angular_velocity']:.8f} rad/s` | {'PASS' if data['max_observed_differences']['max_diff_angular_velocity'] <= data['tolerances']['tol_angular_velocity_radps'] else 'FAIL'} |",
        f"| Master Multi-Sensor Confidence ($C$) | $\\le {data['tolerances']['tol_confidence']:.6f}$ | `{data['max_observed_differences']['max_diff_confidence']:.8f}` | {'PASS' if data['max_observed_differences']['max_diff_confidence'] <= data['tolerances']['tol_confidence'] else 'FAIL'} |",
        f"| Obstacle Clearance Margin ($d$) | $\\le {data['tolerances']['tol_clearance_m']:.6f}\\text{{ m}}$ | `{data['max_observed_differences']['max_diff_clearance_m']:.8f} m` | {'PASS' if data['max_observed_differences']['max_diff_clearance_m'] <= data['tolerances']['tol_clearance_m'] else 'FAIL'} |",
        "",
        "## 2. Categorical Invariants",
        "",
        "| Check | Mismatches | Status |",
        "|:---|:---|:---:|",
        f"| Safety State Identity | `{data['mismatches']['safety_state_mismatches_count']} / {data['total_frames_compared']}` | {'PASS (100% Identical)' if data['mismatches']['safety_state_mismatches_count'] == 0 else 'FAIL'} |",
        f"| Commanded Steering Direction | `{data['mismatches']['steering_direction_mismatches_count']} / {data['total_frames_compared']}` | {'PASS (100% Identical)' if data['mismatches']['steering_direction_mismatches_count'] == 0 else 'FAIL'} |",
        f"| Total Tolerance Violations | `{data['mismatches']['tolerance_violations_count']}` | {'PASS (Zero Drift)' if data['mismatches']['tolerance_violations_count'] == 0 else 'FAIL'} |",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate executive reports for autonomous navigation experiments.")
    parser.add_argument("--run", default=None, help="Path to single run directory")
    parser.add_argument("--comparison", default=None, help="Path to comparison JSON output")
    parser.add_argument("--output", default=None, help="Path to write Markdown report")
    args = parser.parse_args()

    if args.run:
        report_text = generate_single_run_report(args.run)
    elif args.comparison:
        report_text = generate_comparison_report(args.comparison)
    else:
        parser.print_help()
        sys.exit(1)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report written to: {args.output}")
    else:
        print(report_text)


if __name__ == "__main__":
    main()
