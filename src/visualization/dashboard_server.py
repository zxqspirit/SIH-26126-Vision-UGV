"""Interactive Laptop Mission Control Dashboard Server for SIH 26126.

Serves a rich, glassmorphic real-time visual telemetry dashboard
allowing judges, mentors, and engineers to inspect all internal pipeline states:
RGB, Semantic Mask, Metric Depth, 2D BEV Costmap, DWA Trajectories,
Visual Odometry XY Path, Multi-Source Confidence Bars, and the Deterministic Safety Arbiter.
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import socketserver
import sys
import urllib.parse
from typing import Dict, Any, Optional
import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
from src.pipeline import NavigationPipeline

PORT = 5000
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Cache pipelines and datasets per scenario
LOADERS: Dict[str, OutdoorDatasetLoader] = {}
PIPELINES: Dict[str, NavigationPipeline] = {}
FRAME_CACHE: Dict[str, Dict[int, Dict[str, Any]]] = {}


def get_or_create_pipeline(scenario_name: str) -> tuple[OutdoorDatasetLoader, NavigationPipeline]:
    """Retrieve or instantiate pipeline and dataset loader for scenario."""
    if scenario_name not in LOADERS:
        scenario_dir = os.path.join(REPO_ROOT, "datasets", "processed", scenario_name)
        if not os.path.exists(scenario_dir):
            raise FileNotFoundError(f"Scenario not found: {scenario_dir}")
        LOADERS[scenario_name] = OutdoorDatasetLoader(scenario_dir)
        PIPELINES[scenario_name] = NavigationPipeline()
        FRAME_CACHE[scenario_name] = {}
    return LOADERS[scenario_name], PIPELINES[scenario_name]


def process_and_cache_frame(scenario_name: str, frame_idx: int) -> Dict[str, Any]:
    """Process a single frame and cache visualizations and telemetry."""
    loader, pipeline = get_or_create_pipeline(scenario_name)
    if frame_idx in FRAME_CACHE[scenario_name]:
        return FRAME_CACHE[scenario_name][frame_idx]

    # Need to run sequentially if previous frames aren't processed (for odometry)
    cur_idx = 0
    while cur_idx <= frame_idx:
        if cur_idx not in FRAME_CACHE[scenario_name]:
            frame = loader.get_frame(cur_idx)
            if frame is None:
                break
            cmd, tele = pipeline.process_frame(frame)

            # Generate encoded JPEG images for frontend
            # 1. RGB with optional semantic overlay
            rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            sem_mask = tele["semantic"].traversability_mask
            # Green overlay for traversable regions
            overlay = rgb_bgr.copy()
            green_tint = np.zeros_like(rgb_bgr)
            green_tint[:, :, 1] = 255  # Green channel
            alpha_mask = np.clip(sem_mask, 0.0, 1.0)[:, :, np.newaxis]
            blended = (rgb_bgr * (1.0 - 0.4 * alpha_mask) + green_tint * (0.4 * alpha_mask)).astype(np.uint8)

            _, rgb_jpg = cv2.imencode(".jpg", rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _, sem_jpg = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # 2. Depth colormap (Turbo / Inferno)
            depth = frame.depth_m
            valid = np.isfinite(depth) & (depth > 0.1) & (depth < 15.0)
            norm_depth = np.zeros_like(depth, dtype=np.uint8)
            if np.any(valid):
                norm_depth[valid] = np.clip((depth[valid] / 10.0) * 255.0, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(norm_depth, cv2.COLORMAP_TURBO)
            depth_color[~valid] = [30, 30, 30]  # Dark gray for invalid depth
            _, depth_jpg = cv2.imencode(".jpg", depth_color, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # 3. 2D Costmap Image (100x100)
            cost_grid = tele["fused"].fused_costmap
            cost_bgr = cv2.cvtColor(cost_grid, cv2.COLOR_GRAY2BGR)
            # Mark lethal obstacles in bright red
            cost_bgr[cost_grid >= 220] = [0, 0, 255]
            # Mark unknown in dark blue
            cost_bgr[cost_grid == 128] = [60, 30, 10]
            # Resize for visual clarity (300x300)
            cost_large = cv2.resize(cost_bgr, (300, 300), interpolation=cv2.INTER_NEAREST)
            # Flip vertically so forward is up
            cost_large = cv2.flip(cost_large, 0)
            _, cost_jpg = cv2.imencode(".jpg", cost_large)

            # Candidate trajectories rollout points for SVG/canvas rendering
            candidate_paths = []
            for traj in tele["planning"].candidate_trajectories[:18]:
                pts_2d = [{"x": round(float(p[0]), 2), "y": round(float(p[1]), 2)} for p in traj.points]
                candidate_paths.append({
                    "points": pts_2d,
                    "is_valid": traj.is_valid,
                    "v": round(float(traj.linear_velocity), 2),
                    "w": round(float(traj.angular_velocity), 2),
                })

            selected_path = []
            if tele["planning"].selected_trajectory is not None:
                selected_path = [{"x": round(float(p[0]), 2), "y": round(float(p[1]), 2)} for p in tele["planning"].selected_trajectory.points]

            cached_item = {
                "frame_id": cur_idx,
                "timestamp": round(float(frame.timestamp), 2),
                "fps": tele["fps"],
                "latency_ms": tele["total_latency_ms"],
                "images": {
                    "rgb": base64.b64encode(rgb_jpg).decode("utf-8"),
                    "semantic": base64.b64encode(sem_jpg).decode("utf-8"),
                    "depth": base64.b64encode(depth_jpg).decode("utf-8"),
                    "costmap": base64.b64encode(cost_jpg).decode("utf-8"),
                },
                "motion": {
                    "linear_velocity": cmd.linear_velocity,
                    "angular_velocity": cmd.angular_velocity,
                    "steering_direction": cmd.steering_direction,
                    "navigation_state": cmd.navigation_state,
                    "safety_state": cmd.safety_state,
                    "confidence": cmd.confidence,
                },
                "confidence_breakdown": {
                    "c_perc": round(float(tele["semantic"].confidence), 3),
                    "c_geom": round(float(tele["geometry"].confidence), 3),
                    "c_vo": round(float(tele["odometry"].confidence), 3),
                    "c_fusion": round(float(tele["fused"].confidence), 3),
                    "c_total": round(float(tele["safety"].overall_confidence), 3),
                },
                "odometry": {
                    "x": round(float(tele["odometry"].x), 3),
                    "y": round(float(tele["odometry"].y), 3),
                    "yaw": round(float(tele["odometry"].yaw), 3),
                    "inliers": tele["odometry"].inlier_count,
                    "status": tele["odometry"].tracking_status.value,
                    "history": [{"x": round(float(h[0]), 2), "y": round(float(h[1]), 2)} for h in pipeline.odometry.trajectory_history],
                },
                "planning": {
                    "status": tele["planning"].status,
                    "selected_path": selected_path,
                    "candidate_paths": candidate_paths,
                },
                "safety": {
                    "audit_reasons": tele["safety"].audit_reasons,
                    "speed_scale": tele["safety"].speed_scale_factor,
                    "is_emergency_stop": tele["safety"].is_emergency_stop,
                }
            }
            FRAME_CACHE[scenario_name][cur_idx] = cached_item
        cur_idx += 1

    return FRAME_CACHE[scenario_name].get(frame_idx, {})


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the REST API and static dashboard files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/scenarios":
            scenarios = [
                {"id": "scenario_1_open_path", "name": "Scenario 1: Open Traversable Trail (High Confidence)"},
                {"id": "scenario_2_sudden_obstacle", "name": "Scenario 2: Sudden Positive Obstacle on Path"},
                {"id": "scenario_3_terrain_boundary", "name": "Scenario 3: Non-Traversable Vegetation Boundary"},
                {"id": "scenario_4_depth_degradation", "name": "Scenario 4: Glare & Invalid Depth (Rule 12)"},
                {"id": "scenario_5_visual_degradation", "name": "Scenario 5: Visual Feature Loss (Rule 13 E-Stop)"},
            ]
            self._send_json({"scenarios": scenarios})
            return

        elif path == "/api/telemetry":
            query = urllib.parse.parse_qs(parsed.query)
            scenario = query.get("scenario", ["scenario_1_open_path"])[0]
            frame_idx = int(query.get("frame", [0])[0])

            try:
                loader, _ = get_or_create_pipeline(scenario)
                total_frames = len(loader)
                frame_idx = max(0, min(frame_idx, total_frames - 1))
                data = process_and_cache_frame(scenario, frame_idx)
                data["total_frames"] = total_frames
                self._send_json(data)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        elif path == "/api/reset":
            query = urllib.parse.parse_qs(parsed.query)
            scenario = query.get("scenario", ["scenario_1_open_path"])[0]
            if scenario in PIPELINES:
                PIPELINES[scenario].reset()
                FRAME_CACHE[scenario].clear()
            self._send_json({"status": "reset", "scenario": scenario})
            return

        # Serve index.html by default
        if path == "/" or not os.path.exists(os.path.join(STATIC_DIR, path.lstrip("/"))):
            self.path = "/index.html"

        return super().do_GET()

    def _send_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


def run_server(port: int = PORT) -> None:
    """Start local dashboard HTTP server."""
    os.makedirs(STATIC_DIR, exist_ok=True)
    with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
        print(f"=================================================================")
        print(f"SIH 26126: TerrainSight Laptop Dashboard running at http://localhost:{port}")
        print(f"Press Ctrl+C to stop.")
        print(f"=================================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")


if __name__ == "__main__":
    run_server()
