#!/usr/bin/env python3
"""Run Research Ablation Evaluation across Modes A, B, C, D, E.

Compares 5 progressive configurations across identical real outdoor test sequences:
A = CNN only
B = depth only
C = CNN + depth
D = CNN + depth + localization
E = full system + confidence safety

Generates:
- experiments/ablation/ablation_results.csv
- experiments/ablation/ablation_results.json
- experiments/ablation/latency_by_mode.png
- experiments/ablation/clearance_and_safety.png
- experiments/ablation/radar_module_contributions.png
- experiments/ablation/ablation_report.md
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.evaluation.ablation_reporter import AblationReporter
from src.evaluation.ablation_runner import AblationRunner


def main():
    parser = argparse.ArgumentParser(description="Run research ablation study on real recorded datasets.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "scenario_1_open_path",
            "scenario_2_sudden_obstacle",
            "scenario_3_terrain_boundary",
            "scenario_4_depth_degradation",
            "scenario_5_visual_degradation",
        ],
        help="Scenarios to evaluate.",
    )
    parser.add_argument("--max-frames", type=int, default=15, help="Max frames per scenario.")
    parser.add_argument("--output-dir", type=str, default="experiments/ablation", help="Directory for artifacts.")
    args = parser.parse_args()

    print("=" * 70)
    print("        RESEARCH ABLATION STUDY: UGV VISION & SAFETY STACK")
    print("=" * 70)
    print(f"Scenarios: {args.scenarios}")
    print(f"Max frames per scenario: {args.max_frames}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 70)

    runner = AblationRunner()
    telemetry, summaries = runner.run_all_ablations(
        scenarios=args.scenarios,
        max_frames_per_scenario=args.max_frames,
    )

    csv_path, json_path = runner.export_results(telemetry, summaries, output_dir=args.output_dir)

    print("\nGenerating visualization charts and executive Markdown report...")
    reporter = AblationReporter(ablation_json_path=json_path)
    chart_paths = reporter.generate_all_charts()
    report_path = reporter.generate_markdown_report(chart_paths)

    print("\n" + "=" * 70)
    print("               ABLATION SUMMARY MATRIX (EMPIRICAL)")
    print("=" * 70)
    print(f"{'Mode':<18} | {'Lat (ms)':<8} | {'FPS':<5} | {'MinClr (m)':<10} | {'FSafe':<5} | {'Viol':<5} | {'SafeStop':<8}")
    print("-" * 70)
    for mode_name, s in summaries.items():
        print(f"{mode_name:<18} | {s.mean_decision_latency_ms:<8.1f} | {s.fps:<5.1f} | {s.min_clearance_observed_m:<10.3f} | {s.total_false_safe_cases:<5} | {s.total_footprint_violations:<5} | {s.safe_stop_frames:<8}")
    print("=" * 70)
    print(f"\nComplete Report: {report_path}")
    print(f"Raw CSV Data:    {csv_path}")
    print(f"Charts:")
    for cp in chart_paths:
        print(f"  - {cp}")


if __name__ == "__main__":
    main()
