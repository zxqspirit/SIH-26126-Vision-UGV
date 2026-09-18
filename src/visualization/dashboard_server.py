"""Interactive SIH Judge Mission Control Dashboard Server for SIH 26126.

Serves a rich, glassmorphic real-time visual telemetry dashboard
allowing judges, mentors, and engineers to inspect all internal pipeline states:
RGB, Semantic Mask, Metric Depth, 2D BEV Costmap, 6-Level Traversability,
3D Obstacles, Global A* Path, DWA Trajectories, Visual Odometry History,
Multi-Source Confidence Bars, and the Deterministic Safety Arbiter.

Explains:
- WHY PATH CHANGED
- WHY SPEED REDUCED
- WHY STOPPED

Displays 6 Standardized Judge States:
- AUTONOMOUS
- CAUTION
- UNCERTAIN
- SAFE STOP
- LOCALIZATION LOST
- RECOVERING
"""

from __future__ import annotations

import base64
import http.server
import json
import os
import re
import socketserver
import sys
import urllib.parse
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import email
from email.message import Message

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.outdoor_dataset_loader import OutdoorDatasetLoader
from src.interfaces.types import TrackingStatus
from src.pipeline import NavigationPipeline
from src.sensors.live_pipeline import LiveCameraStreamer, MockLiveCamera, ImageSequenceCamera

PORT = 5000
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
UPLOAD_BASE = os.path.join(REPO_ROOT, "datasets", "uploads")
os.makedirs(UPLOAD_BASE, exist_ok=True)

# Global caches
LOADERS: Dict[str, OutdoorDatasetLoader] = {}
PIPELINES: Dict[str, NavigationPipeline] = {}
FRAME_CACHE: Dict[str, Dict[int, Dict[str, Any]]] = {}

# Live streaming state
LIVE_STREAMER: Optional[LiveCameraStreamer] = None
LIVE_PIPELINE: Optional[NavigationPipeline] = None
CURRENT_LIVE_SCENARIO: Optional[str] = None




def parse_upload_payload(headers: Dict[str, str], body_bytes: bytes) -> List[Tuple[str, bytes]]:
    """Extract all files (filename, file_bytes) from multipart/form-data or raw octet stream."""
    ct = headers.get("content-type", headers.get("Content-Type", ""))
    files: List[Tuple[str, bytes]] = []
    if "multipart/form-data" in ct:
        msg = email.message_from_bytes(f"Content-Type: {ct}\r\n\r\n".encode("utf-8") + body_bytes)
        for part in msg.walk():
            filename = part.get_filename()
            if filename:
                payload = part.get_payload(decode=True)
                if payload:
                    files.append((filename, payload))
        if files:
            return files

    # Raw binary upload fallback
    return [("uploaded_media.jpg", body_bytes)]


def process_uploaded_media(files: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    """Process uploaded single image, video, or multi-image batch (up to 60 frames) into a scenario."""
    if not files:
        raise ValueError("No files provided for processing")

    # Case 1: Multiple images uploaded simultaneously (Batch of frames, e.g., 60 frames)
    if len(files) > 1:
        # Natural sort key so frame_1, frame_2, ..., frame_10, frame_60 sort chronologically
        def natural_sort_key(item: Tuple[str, bytes]):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', item[0])]
        
        files.sort(key=natural_sort_key)
        # Cap at 60 frames max for model evaluation
        files = files[:60]
        
        scenario_id = "uploaded_sequence"
        scenario_dir = os.path.join(UPLOAD_BASE, scenario_id)
        rgb_dir = os.path.join(scenario_dir, "rgb")
        depth_dir = os.path.join(scenario_dir, "depth")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        for folder in [rgb_dir, depth_dir]:
            for f in os.listdir(folder):
                try:
                    os.remove(os.path.join(folder, f))
                except Exception:
                    pass

        h, w = 480, 640
        processed_count = 0
        for idx, (fname, data_bytes) in enumerate(files):
            bgr = cv2.imdecode(np.frombuffer(data_bytes, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            if (bgr.shape[1], bgr.shape[0]) != (w, h):
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

            frame_name = f"frame_{idx:04d}"
            cv2.imwrite(os.path.join(rgb_dir, f"{frame_name}.jpg"), bgr)

            # Generate perspective ground-plane depth gradient (1.0m to 10.0m)
            depth_m = np.zeros((h, w), dtype=np.float32)
            horizon = int(h * 0.45)
            ground_rows = h - horizon
            grad = np.linspace(10.0, 1.2, ground_rows, dtype=np.float32)[:, np.newaxis]
            depth_m[horizon:, :] = np.tile(grad, (1, w))
            np.save(os.path.join(depth_dir, f"{frame_name}.npy"), depth_m)
            processed_count += 1

        if processed_count == 0:
            raise ValueError("None of the uploaded images could be decoded")

        metadata = {
            "scenario_name": scenario_id,
            "media_type": "sequence",
            "total_frames": processed_count,
            "resolution": [w, h],
        }
        with open(os.path.join(scenario_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "status": "success",
            "scenario_id": scenario_id,
            "scenario": scenario_id,
            "media_type": "sequence",
            "total_frames": processed_count,
            "filename": f"batch_{processed_count}_frames",
        }

    # Case 2: Single file uploaded (either video or single image)
    filename, data_bytes = files[0]
    ext = os.path.splitext(filename)[1].lower()
    is_video = ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    scenario_id = "uploaded_video" if is_video else "uploaded_image"
    scenario_dir = os.path.join(UPLOAD_BASE, scenario_id)
    rgb_dir = os.path.join(scenario_dir, "rgb")
    depth_dir = os.path.join(scenario_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    for folder in [rgb_dir, depth_dir]:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except Exception:
                pass

    raw_temp_path = os.path.join(scenario_dir, f"raw_upload{ext}")
    with open(raw_temp_path, "wb") as f:
        f.write(data_bytes)

    extracted_frames = 0
    if not is_video:
        bgr = cv2.imread(raw_temp_path)
        if bgr is None:
            raise ValueError(f"Failed to decode uploaded image: {filename}")
        if (bgr.shape[1], bgr.shape[0]) != (640, 480):
            bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(rgb_dir, "frame_0000.jpg"), bgr)

        h, w = 480, 640
        depth_m = np.zeros((h, w), dtype=np.float32)
        horizon = int(h * 0.45)
        ground_rows = h - horizon
        grad = np.linspace(10.0, 1.2, ground_rows, dtype=np.float32)[:, np.newaxis]
        depth_m[horizon:, :] = np.tile(grad, (1, w))
        np.save(os.path.join(depth_dir, "frame_0000.npy"), depth_m)
        extracted_frames = 1
    else:
        cap = cv2.VideoCapture(raw_temp_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to decode uploaded video: {filename}")

        idx = 0
        max_frames = 60
        while cap.isOpened() and idx < max_frames:
            ret, bgr = cap.read()
            if not ret or bgr is None:
                break
            if (bgr.shape[1], bgr.shape[0]) != (640, 480):
                bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)

            frame_name = f"frame_{idx:04d}"
            cv2.imwrite(os.path.join(rgb_dir, f"{frame_name}.jpg"), bgr)

            h, w = 480, 640
            depth_m = np.zeros((h, w), dtype=np.float32)
            horizon = int(h * 0.45)
            ground_rows = h - horizon
            grad = np.linspace(10.0, 1.2, ground_rows, dtype=np.float32)[:, np.newaxis]
            depth_m[horizon:, :] = np.tile(grad, (1, w))
            np.save(os.path.join(depth_dir, f"{frame_name}.npy"), depth_m)
            idx += 1

        cap.release()
        extracted_frames = idx
        if extracted_frames == 0:
            raise ValueError("No valid video frames could be extracted from uploaded video")

    metadata = {
        "scenario_name": scenario_id,
        "source_filename": filename,
        "media_type": "video" if is_video else "image",
        "total_frames": extracted_frames,
        "resolution": [640, 480],
    }
    with open(os.path.join(scenario_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "status": "success",
        "scenario_id": scenario_id,
        "scenario": scenario_id,
        "media_type": "video" if is_video else "image",
        "total_frames": extracted_frames,
        "filename": filename,
    }

def get_or_create_pipeline(scenario_name: str) -> Tuple[OutdoorDatasetLoader, NavigationPipeline]:
    """Retrieve or instantiate pipeline and dataset loader for scenario."""
    if scenario_name not in LOADERS:
        scenario_dir = os.path.join(REPO_ROOT, "datasets", "processed", scenario_name)
        if not os.path.exists(scenario_dir):
            upload_dir = os.path.join(UPLOAD_BASE, scenario_name)
            if os.path.exists(upload_dir):
                scenario_dir = upload_dir
            else:
                raise FileNotFoundError(f"Scenario not found: {scenario_name}")
        LOADERS[scenario_name] = OutdoorDatasetLoader(scenario_dir)
        PIPELINES[scenario_name] = NavigationPipeline()
        FRAME_CACHE[scenario_name] = {}
    return LOADERS[scenario_name], PIPELINES[scenario_name]


def derive_judge_state(tele: Dict[str, Any]) -> str:
    """Derive the 6 standardized judge states deterministically from backend data."""
    odometry = tele["odometry"]
    safety = tele["safety"]
    decision = tele["decision"]

    tracking_status_str = odometry.tracking_status.value if hasattr(odometry.tracking_status, "value") else str(odometry.tracking_status)
    safety_state_str = safety.safety_state.value if hasattr(safety.safety_state, "value") else str(safety.safety_state)
    nav_state_str = decision.navigation_state.value if hasattr(decision.navigation_state, "value") else str(decision.navigation_state)

    # 1. LOCALIZATION LOST
    if tracking_status_str == "TRACKING_LOST" or odometry.confidence < 0.20:
        return "LOCALIZATION LOST"

    # 2. RECOVERING
    reloc_val = getattr(odometry, "relocalization_state", None)
    reloc_str = reloc_val.value if hasattr(reloc_val, "value") else str(reloc_val) if reloc_val else ""
    if reloc_str in ("RELOCALIZING", "SEARCHING", "RECOVERED") or nav_state_str in ("RECOVERY_HOLD", "RECOVERING"):
        return "RECOVERING"

    # 3. SAFE STOP
    if safety.is_emergency_stop or safety_state_str == "CRITICAL" or nav_state_str == "ESTOP":
        return "SAFE STOP"
    if decision.status in ("OBSTACLE_BLOCKED", "NO_TRAVERSABLE_PATH") and safety.commanded_linear_velocity <= 0.001:
        return "SAFE STOP"

    # 4. UNCERTAIN
    reobserve_on = bool(safety.decision_log and safety.decision_log.reobserve_active)
    if safety_state_str == "LOW" or reobserve_on or nav_state_str in ("CAUTIOUS_CRAWL", "RE_OBSERVE"):
        return "UNCERTAIN"

    # 5. CAUTION
    if safety_state_str == "MEDIUM" or tracking_status_str == "TRACKING_DEGRADED" or nav_state_str == "CAUTIOUS_EXPLORATION":
        return "CAUTION"

    # 6. AUTONOMOUS
    return "AUTONOMOUS"


def compute_explainability(tele: Dict[str, Any], cmd: Any) -> Dict[str, str]:
    """Extract causal explanations directly from backend decision and safety logs."""
    decision = tele["decision"]
    safety = tele["safety"]
    cost_exp = decision.cost_explanation
    dominant_terrain = cost_exp.primary_terrain if cost_exp else "TRAVERSABLE"
    explanation_txt = cost_exp.explanation_text if cost_exp else ""

    # 1. WHY PATH CHANGED
    steer_val = decision.recommended_steering
    steering_str = steer_val.value if hasattr(steer_val, 'value') else str(steer_val)

    if decision.status == "OBSTACLE_BLOCKED":
        why_path = (
            f"DIRECT PATH BLOCKED: Positive obstacle detected in forward corridor. "
            f"Evasive steering commanded to preserve {decision.min_clearance_m:.2f}m boundary clearance."
        )
    else:
        why_path = (
            f"Steering {steering_str} (w = {decision.recommended_angular_velocity:+.2f} rad/s). "
            f"Trajectory selected over {dominant_terrain} terrain maintaining {decision.min_clearance_m:.2f}m clearance. "
            f"{explanation_txt}"
        )

    # 2. WHY SPEED REDUCED
    speed_scale = safety.speed_scale_factor
    if speed_scale < 0.99:
        primary_reason = safety.audit_reasons[0] if safety.audit_reasons else "Speed reduced under uncertainty"
        why_speed = (
            f"Speed scaled to {speed_scale * 100:.0f}% (v = {safety.commanded_linear_velocity:.2f} m/s). "
            f"Cause: {primary_reason}"
        )
    else:
        why_speed = (
            f"Full nominal speed (100%, v = {safety.commanded_linear_velocity:.2f} m/s). "
            f"Perception and localization confidences nominal across clear corridor."
        )

    # 3. WHY STOPPED
    is_stopped = safety.commanded_linear_velocity <= 0.001 or safety.is_emergency_stop
    if is_stopped:
        if safety.decision_log and safety.decision_log.reason:
            stop_reason = safety.decision_log.reason
        elif safety.audit_reasons:
            stop_reason = safety.audit_reasons[0]
        else:
            stop_reason = "Vehicle stopped by safety gate"
        why_stopped = f"STOPPED / SAFE HOLD: {stop_reason}"
    else:
        why_stopped = f"NOT STOPPED: Forward progression active at {safety.commanded_linear_velocity:.2f} m/s."

    return {
        "why_path_changed": why_path,
        "why_speed_reduced": why_speed,
        "why_stopped": why_stopped,
    }


def serialize_frame_telemetry(
    frame: Any,
    cmd: Any,
    tele: Dict[str, Any],
    pipeline: NavigationPipeline,
    cur_idx: int,
    scenario_name: str = "",
) -> Dict[str, Any]:
    """Encode images and format JSON telemetry from a processed frame."""
    # Optimization 1: Fast Vectorized Visualization Serialization
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 70, cv2.IMWRITE_JPEG_OPTIMIZE, 0]

    # 1. RGB with optional semantic overlay (downscaled to 320x240 for web UI previews)
    rgb_small = cv2.resize(frame.rgb, (320, 240), interpolation=cv2.INTER_AREA)
    rgb_bgr = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)

    sem_mask = cv2.resize(tele["semantic"].traversability_mask, (320, 240), interpolation=cv2.INTER_NEAREST)
    alpha_mask = np.clip(sem_mask, 0.0, 1.0)[:, :, np.newaxis]
    blended = np.clip(rgb_bgr * (1.0 - 0.35 * alpha_mask) + np.array([0, 89, 0], dtype=np.float32) * alpha_mask, 0, 255).astype(np.uint8)

    _, rgb_jpg = cv2.imencode(".jpg", rgb_bgr, jpeg_params)
    _, sem_jpg = cv2.imencode(".jpg", blended, jpeg_params)

    # 2. Depth colormap (Turbo) - downscale to 320x240
    depth_small = cv2.resize(frame.depth_m, (320, 240), interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(depth_small) & (depth_small > 0.1) & (depth_small < 15.0)
    norm_depth = np.zeros_like(depth_small, dtype=np.uint8)
    if np.any(valid):
        norm_depth[valid] = np.clip((depth_small[valid] / 10.0) * 255.0, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(norm_depth, cv2.COLORMAP_TURBO)
    depth_color[~valid] = [30, 30, 30]  # Dark gray for invalid depth
    _, depth_jpg = cv2.imencode(".jpg", depth_color, jpeg_params)

    # 3. 2D Costmap Image (100x100 -> 240x240, flipped so forward is UP)
    cost_grid = tele["fused"].fused_costmap
    cost_bgr = cv2.cvtColor(cost_grid, cv2.COLOR_GRAY2BGR)
    cost_bgr[cost_grid >= 220] = [0, 0, 255]  # Red lethal
    cost_bgr[cost_grid == 128] = [60, 30, 10]  # Dark blue unknown
    cost_large = cv2.resize(cv2.flip(cost_bgr, 0), (240, 240), interpolation=cv2.INTER_NEAREST)
    _, cost_jpg = cv2.imencode(".jpg", cost_large, jpeg_params)

    # 4. 6-Level Visual Traversability Map (100x100 -> 240x240)
    trav_rgba = tele["traversability"].visual_map
    trav_bgr = cv2.cvtColor(trav_rgba, cv2.COLOR_RGBA2BGR)
    trav_large = cv2.resize(cv2.flip(trav_bgr, 0), (240, 240), interpolation=cv2.INTER_NEAREST)
    _, trav_jpg = cv2.imencode(".jpg", trav_large, jpeg_params)

    # 5. Candidate rollouts for canvas overlay
    candidate_paths = []
    for traj in tele["planning"].candidate_trajectories[:24]:
        pts_2d = [{"x": round(float(p[0]), 2), "y": round(float(p[1]), 2)} for p in traj.points]
        candidate_paths.append({
            "points": pts_2d,
            "is_valid": traj.is_valid,
            "v": round(float(traj.linear_velocity), 2),
            "w": round(float(traj.angular_velocity), 2),
            "cost": round(float(traj.cost), 3),
        })

    # Selected rollout
    selected_path = []
    if tele["planning"].selected_trajectory is not None:
        selected_path = [
            {"x": round(float(p[0]), 2), "y": round(float(p[1]), 2)}
            for p in tele["planning"].selected_trajectory.points
        ]

    # Global A* recommended path
    global_path = []
    if tele["decision"].recommended_path is not None:
        global_path = [
            {"x": round(float(wp[0]), 2), "y": round(float(wp[1]), 2)}
            for wp in tele["decision"].recommended_path.waypoints
        ]

    # Derive state & explanations
    judge_state = derive_judge_state(tele)
    explain = compute_explainability(tele, cmd)

    return {
        "scenario": scenario_name,
        "frame_id": cur_idx,
        "timestamp": round(float(frame.timestamp), 3),
        "fps": tele["fps"],
        "latency_ms": tele["total_latency_ms"],
        "judge_state": judge_state,
        "explainability": explain,
        "images": {
            "rgb": base64.b64encode(rgb_jpg).decode("utf-8"),
            "semantic": base64.b64encode(sem_jpg).decode("utf-8"),
            "depth": base64.b64encode(depth_jpg).decode("utf-8"),
            "costmap": base64.b64encode(cost_jpg).decode("utf-8"),
            "traversability": base64.b64encode(trav_jpg).decode("utf-8"),
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
            "disagreement": round(float(tele["safety_decision_log"]["signals"].get("disagreement_ratio", 0.0)), 3)
            if tele.get("safety_decision_log") else 0.0,
            "temporal": round(float(tele["safety_decision_log"]["signals"].get("temporal_consistency", 1.0)), 3)
            if tele.get("safety_decision_log") else 1.0,
        },
        "odometry": {
            "x": round(float(tele["odometry"].x), 3),
            "y": round(float(tele["odometry"].y), 3),
            "yaw": round(float(tele["odometry"].yaw), 3),
            "inliers": tele["odometry"].inlier_count,
            "status": tele["odometry"].tracking_status.value,
            "history": [
                {"x": round(float(h[0]), 2), "y": round(float(h[1]), 2)}
                for h in pipeline.odometry.trajectory_history
            ],
        },
        "planning": {
            "status": tele["decision"].status,
            "min_clearance_m": round(float(tele["decision"].min_clearance_m), 3),
            "global_path": global_path,
            "selected_path": selected_path,
            "candidate_paths": candidate_paths,
            "cost_breakdown": {
                "progress": tele["decision"].cost_explanation.progress_score if tele["decision"].cost_explanation else 0.0,
                "clearance": tele["decision"].cost_explanation.clearance_score if tele["decision"].cost_explanation else 0.0,
                "traversability": tele["decision"].cost_explanation.traversability_score if tele["decision"].cost_explanation else 0.0,
                "heading": tele["decision"].cost_explanation.heading_score if tele["decision"].cost_explanation else 0.0,
                "dominant_terrain": tele["decision"].cost_explanation.primary_terrain if tele["decision"].cost_explanation else "UNKNOWN",
                "explanation": tele["decision"].cost_explanation.explanation_text if tele["decision"].cost_explanation else "",
            },
        },
        "safety": {
            "action": tele["safety"].action,
            "audit_reasons": tele["safety"].audit_reasons,
            "speed_scale": tele["safety"].speed_scale_factor,
            "clearance_inflation": tele["safety"].clearance_inflation_factor,
            "is_emergency_stop": tele["safety"].is_emergency_stop,
            "decision_log": tele.get("safety_decision_log"),
        },
    }


def process_and_cache_frame(scenario_name: str, frame_idx: int) -> Dict[str, Any]:
    """Process a single frame and cache visualizations and telemetry."""
    loader, pipeline = get_or_create_pipeline(scenario_name)
    if frame_idx in FRAME_CACHE[scenario_name]:
        return FRAME_CACHE[scenario_name][frame_idx]

    cur_idx = 0
    while cur_idx <= frame_idx:
        if cur_idx not in FRAME_CACHE[scenario_name]:
            frame = loader.get_frame(cur_idx)
            if frame is None:
                break
            cmd, tele = pipeline.process_frame(frame)
            cached_item = serialize_frame_telemetry(frame, cmd, tele, pipeline, cur_idx, scenario_name)
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
                {"id": "scenario_4_depth_degradation", "name": "Scenario 4: Depth Degradation & Sensor Dropout"},
                {"id": "scenario_5_visual_degradation", "name": "Scenario 5: Visual Feature Loss (Rule 13 E-Stop)"},
                {"id": "live_kaggle_offroad", "name": "LIVE SENSOR: Kaggle Off-Road Trail Stream (515 4K Frames)"},
                {"id": "live_webcam", "name": "LIVE SENSOR: Real USB / DirectShow Camera (Device 0)"},
                {"id": "live_camera", "name": "LIVE SENSOR: Virtual Simulated Stream (Single-Slot Ring Buffer)"},
            ]
            seq_up = os.path.join(UPLOAD_BASE, "uploaded_sequence", "rgb", "frame_0000.jpg")
            if os.path.exists(seq_up):
                meta_file = os.path.join(UPLOAD_BASE, "uploaded_sequence", "metadata.json")
                n_seq = 60
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            n_seq = json.load(f).get("total_frames", 60)
                    except Exception:
                        pass
                scenarios.insert(0, {"id": "uploaded_sequence", "name": f"UPLOADED SEQUENCE: Custom Batch ({n_seq} Frames)"})
            img_up = os.path.join(UPLOAD_BASE, "uploaded_image", "rgb", "frame_0000.jpg")
            if os.path.exists(img_up):
                scenarios.insert(0, {"id": "uploaded_image", "name": "UPLOADED IMAGE: Custom Uploaded Frame"})
            vid_up = os.path.join(UPLOAD_BASE, "uploaded_video", "rgb", "frame_0000.jpg")
            if os.path.exists(vid_up):
                scenarios.insert(0, {"id": "uploaded_video", "name": "UPLOADED VIDEO: Custom Uploaded Sequence"})
            self._send_json({"scenarios": scenarios})
            return

        elif path == "/api/telemetry":
            query = urllib.parse.parse_qs(parsed.query)
            scenario = query.get("scenario", ["scenario_1_open_path"])[0]

            if scenario.startswith("live_"):
                self._handle_live_telemetry(scenario)
                return

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

        elif path == "/api/live_telemetry":
            query = urllib.parse.parse_qs(parsed.query)
            scenario = query.get("scenario", ["live_camera"])[0]
            self._handle_live_telemetry(scenario)
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

    def _handle_live_telemetry(self, scenario: str = "live_camera") -> None:
        """Process live frame from LiveCameraStreamer with dynamic sensor source switching."""
        global LIVE_STREAMER, LIVE_PIPELINE, CURRENT_LIVE_SCENARIO
        try:
            # Recreate streamer if scenario source changed
            if LIVE_STREAMER is None or CURRENT_LIVE_SCENARIO != scenario:
                if LIVE_STREAMER is not None:
                    try:
                        LIVE_STREAMER.stop(timeout_s=0.5)
                    except Exception:
                        pass

                source_label = "Virtual Simulated Stream"
                if scenario == "live_kaggle_offroad":
                    kaggle_path = r"C:\SIH\tests\TrainingImages\TrainingImages\OriginalImages"
                    if not os.path.exists(kaggle_path):
                        # Fallback to local sample data
                        kaggle_path = os.path.join(REPO_ROOT, "demo", "sample_data", "scenario_1_open_path", "rgb")
                    cam = ImageSequenceCamera(kaggle_path, target_fps=15.0, loop=True)
                    LIVE_STREAMER = LiveCameraStreamer(camera_source=cam, target_fps=15.0, sequence_name="kaggle_offroad")
                    source_label = "Kaggle Off-Road Trail Stream (515 Frames)"
                elif scenario == "live_webcam":
                    # Attempt physical webcam 0, fallback to mock if unavailable
                    test_cap = cv2.VideoCapture(0)
                    if test_cap.isOpened():
                        test_cap.release()
                        LIVE_STREAMER = LiveCameraStreamer(camera_source=0, target_fps=20.0, sequence_name="webcam_device_0")
                        source_label = "Real USB / DirectShow Camera (Device 0)"
                    else:
                        cam = MockLiveCamera(target_fps=20.0, frame_width=640, frame_height=480, simulate_depth=True)
                        LIVE_STREAMER = LiveCameraStreamer(camera_source=cam, target_fps=20.0, sequence_name="webcam_fallback_mock")
                        source_label = "Webcam Unavailable (Fallback to Virtual Sensor)"
                else:
                    cam = MockLiveCamera(target_fps=20.0, frame_width=640, frame_height=480, simulate_depth=True)
                    LIVE_STREAMER = LiveCameraStreamer(camera_source=cam, target_fps=20.0, sequence_name="virtual_sensor")
                    source_label = "Virtual Simulated Stream (Single-Slot Ring Buffer)"

                LIVE_STREAMER.start()
                LIVE_PIPELINE = NavigationPipeline()
                CURRENT_LIVE_SCENARIO = scenario

            frame = LIVE_STREAMER.read_next_frame(timeout_s=0.6)
            metrics = LIVE_STREAMER.get_health_metrics()

            source_label_map = {
                "live_kaggle_offroad": "Kaggle Off-Road Trail Stream",
                "live_webcam": "USB / DirectShow Webcam",
                "live_camera": "Virtual Simulated Sensor",
            }
            active_label = source_label_map.get(scenario, "Live Sensor")

            if frame is None:
                self._send_json({
                    "is_live": True,
                    "health": metrics.to_dict(),
                    "live_health": metrics.to_dict(),
                    "live_sensor": {
                        "source_id": scenario,
                        "source_label": active_label,
                        "is_live": True,
                        "health_state": metrics.health_state.value,
                        "capture_fps": metrics.capture_fps,
                        "processing_fps": metrics.processing_fps,
                        "dropped_frames": metrics.dropped_frames,
                    },
                    "judge_state": metrics.health_state.value,
                })
                return

            cmd, tele = LIVE_PIPELINE.process_frame(frame)
            data = serialize_frame_telemetry(frame, cmd, tele, LIVE_PIPELINE, frame.frame_id)
            data["is_live"] = True
            data["live_health"] = metrics.to_dict()
            data["live_sensor"] = {
                "source_id": scenario,
                "source_label": active_label,
                "is_live": True,
                "health_state": metrics.health_state.value,
                "capture_fps": metrics.capture_fps,
                "processing_fps": metrics.processing_fps,
                "dropped_frames": metrics.dropped_frames,
                "frame_latency_ms": metrics.frame_latency_ms,
                "total_captured": metrics.total_captured,
            }
            data["total_frames"] = 1000
            self._send_json(data)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)

    
    def do_POST(self) -> None:
        """Handle file uploads for custom user images and videos to run model."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/upload":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length <= 0:
                    self._send_json({"error": "Empty upload payload"}, status=400)
                    return

                raw_body = self.rfile.read(content_length)
                files = parse_upload_payload(dict(self.headers), raw_body)

                if not files:
                    self._send_json({"error": "No valid file data received"}, status=400)
                    return

                res = process_uploaded_media(files)
                scenario_id = res["scenario_id"]
                res["scenario"] = scenario_id

                # Reset loaders and caches for this scenario
                LOADERS.pop(scenario_id, None)
                PIPELINES.pop(scenario_id, None)
                FRAME_CACHE.pop(scenario_id, None)

                # Pre-cache frame 0 so initial telemetry is ready for immediate display
                telemetry = process_and_cache_frame(scenario_id, 0)
                res["telemetry"] = telemetry

                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self._send_json({"error": f"Unknown POST endpoint: {path}"}, status=404)

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
        print(f"SIH 26126: Technical Mission Control Dashboard at http://localhost:{port}")
        print(f"Press Ctrl+C to stop.")
        print(f"=================================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")


if __name__ == "__main__":
    run_server()
