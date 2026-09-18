#!/usr/bin/env python3
"""
scripts/evaluate_run.py
Evaluation and Comparative Tolerances Checker for SIH 26126 Experiments.

Modes:
1. Single Run Audit:
   python scripts/evaluate_run.py --run experiments/runs/EXP_XXX
2. Comparative Replay Verification (Run A vs Run B):
   python scripts/evaluate_run.py --run-a experiments/runs/EXP_A --run-b experiments/runs/EXP_B [--tol-v 0.0001] [--tol-w 0.0001]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_run_telemetry(run_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load metadata and line-delimited telemetry log from an experiment directory."""
    meta_path = os.path.join(run_dir, "experiment_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing experiment metadata in: {run_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tele_path = os.path.join(run_dir, "telemetry.jsonl")
    if not os.path.exists(tele_path):
        raise FileNotFoundError(f"Missing telemetry log in: {run_dir}")
    frames: List[Dict[str, Any]] = []
    with open(tele_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                frames.append(json.loads(line))

    return meta, frames


def compare_two_runs(
    run_a_dir: str,
    run_b_dir: str,
    tol_v: float = 1e-4,
    tol_w: float = 1e-4,
    tol_conf: float = 1e-4,
    tol_clearance: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compare two experiment runs frame-by-frame against defined numerical tolerances."""
    meta_a, frames_a = load_run_telemetry(run_a_dir)
    meta_b, frames_b = load_run_telemetry(run_b_dir)

    if len(frames_a) != len(frames_b):
        return {
            "reproducibility_passed": False,
            "reason": f"Frame count mismatch: Run A has {len(frames_a)} frames, Run B has {len(frames_b)} frames.",
            "run_a_id": meta_a["experiment_id"],
            "run_b_id": meta_b["experiment_id"],
        }

    total_frames = len(frames_a)
    max_diff_v = 0.0
    max_diff_w = 0.0
    max_diff_conf = 0.0
    max_diff_clear = 0.0

    state_mismatches = 0
    steering_mismatches = 0
    status_mismatches = 0
    tolerance_violations = 0
    frame_diffs: List[Dict[str, Any]] = []

    for i in range(total_frames):
        fa = frames_a[i]
        fb = frames_b[i]

        va = fa["command"]["linear_velocity"]
        vb = fb["command"]["linear_velocity"]
        diff_v = abs(va - vb)
        max_diff_v = max(max_diff_v, diff_v)

        wa = fa["command"]["angular_velocity"]
        wb = fb["command"]["angular_velocity"]
        diff_w = abs(wa - wb)
        max_diff_w = max(max_diff_w, diff_w)

        ca = fa["safety"]["confidence"]
        cb = fb["safety"]["confidence"]
        diff_c = abs(ca - cb)
        max_diff_conf = max(max_diff_conf, diff_c)

        cla = fa["planning"]["min_clearance_m"] or 0.0
        clb = fb["planning"]["min_clearance_m"] or 0.0
        diff_cl = abs(cla - clb)
        max_diff_clear = max(max_diff_clear, diff_cl)

        sa_state = fa["safety"]["state"]
        sb_state = fb["safety"]["state"]
        if sa_state != sb_state:
            state_mismatches += 1

        steer_a = fa["command"]["steering_direction"]
        steer_b = fb["command"]["steering_direction"]
        if steer_a != steer_b:
            steering_mismatches += 1

        is_viol = (diff_v > tol_v) or (diff_w > tol_w) or (diff_c > tol_conf) or (diff_cl > tol_clearance)
        if is_viol:
            tolerance_violations += 1
            frame_diffs.append({
                "frame_id": i,
                "diff_v": diff_v,
                "diff_w": diff_w,
                "diff_confidence": diff_c,
                "diff_clearance": diff_cl,
                "state_a": sa_state,
                "state_b": sb_state,
            })

    passed = (
        tolerance_violations == 0
        and state_mismatches == 0
        and steering_mismatches == 0
    )

    result = {
        "reproducibility_passed": passed,
        "run_a_id": meta_a["experiment_id"],
        "run_b_id": meta_b["experiment_id"],
        "scenario": meta_a["scenario"],
        "total_frames_compared": total_frames,
        "tolerances": {
            "tol_linear_velocity_mps": tol_v,
            "tol_angular_velocity_radps": tol_w,
            "tol_confidence": tol_conf,
            "tol_clearance_m": tol_clearance,
        },
        "max_observed_differences": {
            "max_diff_linear_velocity": round(max_diff_v, 8),
            "max_diff_angular_velocity": round(max_diff_w, 8),
            "max_diff_confidence": round(max_diff_conf, 8),
            "max_diff_clearance_m": round(max_diff_clear, 8),
        },
        "mismatches": {
            "tolerance_violations_count": tolerance_violations,
            "safety_state_mismatches_count": state_mismatches,
            "steering_direction_mismatches_count": steering_mismatches,
        },
        "frame_diff_samples": frame_diffs[:10],
    }

    if verbose:
        print("================================================================")
        print("COMPARATIVE REPRODUCIBILITY AUDIT")
        print(f"Run A: {meta_a['experiment_id']}")
        print(f"Run B: {meta_b['experiment_id']}")
        print(f"Scenario: {meta_a['scenario']} ({total_frames} frames)")
        print("----------------------------------------------------------------")
        print(f"Max Delta Linear Velocity:  {max_diff_v:.6f} m/s (Tol: {tol_v:.6f})")
        print(f"Max Delta Angular Velocity: {max_diff_w:.6f} rad/s (Tol: {tol_w:.6f})")
        print(f"Max Delta Confidence:       {max_diff_conf:.6f} (Tol: {tol_conf:.6f})")
        print(f"Max Delta Clearance:        {max_diff_clear:.6f} m (Tol: {tol_clearance:.6f})")
        print(f"Safety State Mismatches:    {state_mismatches} / {total_frames}")
        print(f"Steering Mismatches:        {steering_mismatches} / {total_frames}")
        print("----------------------------------------------------------------")
        if passed:
            print("[PASS] ZERO DRIFT: Outputs are strictly deterministic and reproducible within tolerance.")
        else:
            print(f"[FAIL] Divergence detected: {tolerance_violations} frames violated numerical tolerances.")
        print("================================================================")

    return result


def audit_single_run(run_dir: str, verbose: bool = True) -> Dict[str, Any]:
    """Audit quality and safety compliance of a single experiment run."""
    meta, frames = load_run_telemetry(run_dir)
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Invariants
    total_frames = metrics["total_frames"]
    mean_lat = metrics["mean_latency_ms"]
    min_clear = metrics["min_clearance_observed_m"]

    # Criteria:
    # 1. Pipeline operates within real-time bounds (mean latency < 250ms)
    # 2. Clearance never breached without emergency stop
    lat_ok = mean_lat < 250.0
    clear_ok = (min_clear is None) or (min_clear >= 0.30)

    audit = {
        "experiment_id": meta["experiment_id"],
        "scenario": meta["scenario"],
        "audit_passed": lat_ok and clear_ok,
        "checks": {
            "real_time_latency_pass": lat_ok,
            "safe_clearance_pass": clear_ok,
            "mean_latency_ms": mean_lat,
            "min_clearance_m": min_clear,
        }
    }

    if verbose:
        print("=== SINGLE RUN AUDIT ===")
        print(f"Experiment:    {meta['experiment_id']}")
        print(f"Mean Latency:  {mean_lat:.1f} ms -> {'PASS' if lat_ok else 'FAIL'}")
        print(f"Min Clearance: {min_clear or 0.0:.2f} m -> {'PASS' if clear_ok else 'FAIL'}")
        print(f"Verdict:       {'AUDIT PASSED' if audit['audit_passed'] else 'AUDIT FAILED'}")

    return audit


def main():
    parser = argparse.ArgumentParser(description="Evaluate or compare experiment runs within defined tolerances.")
    parser.add_argument("--run", default=None, help="Path to single run directory for audit")
    parser.add_argument("--run-a", default=None, help="Path to Run A directory for comparison")
    parser.add_argument("--run-b", default=None, help="Path to Run B directory for comparison")
    parser.add_argument("--tol-v", type=float, default=1e-4, help="Linear velocity tolerance in m/s")
    parser.add_argument("--tol-w", type=float, default=1e-4, help="Angular velocity tolerance in rad/s")
    parser.add_argument("--tol-conf", type=float, default=1e-4, help="Confidence tolerance")
    parser.add_argument("--tol-clear", type=float, default=1e-4, help="Clearance tolerance in meters")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging")
    args = parser.parse_args()

    if args.run_a and args.run_b:
        res = compare_two_runs(
            args.run_a,
            args.run_b,
            tol_v=args.tol_v,
            tol_w=args.tol_w,
            tol_conf=args.tol_conf,
            tol_clearance=args.tol_clear,
            verbose=not args.quiet,
        )
    elif args.run:
        res = audit_single_run(args.run, verbose=not args.quiet)
    else:
        parser.print_help()
        sys.exit(1)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"Saved evaluation result to: {args.output}")


if __name__ == "__main__":
    main()
