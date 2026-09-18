"""
tests/test_dashboard.py
Unit and Integration Tests for SIH Judge Dashboard Server, Telemetry, and Explainability Engine.
"""

import threading
import time
import urllib.request
import json
import pytest
import numpy as np
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from src.visualization.dashboard_server import (
    derive_judge_state,
    compute_explainability,
    process_and_cache_frame,
    LOADERS,
    FRAME_CACHE,
    DashboardHandler,
)
from src.planning.navigation_decision_engine import NavigationDecisionResult, PathCostBreakdown
from src.safety.safety_gate import SafetyResult, SafetyDecisionLog, SafetyState, SafetyAction


def make_mock_tele(
    tracking_status="TRACKING_OK",
    vo_conf=0.9,
    reloc_state=None,
    safety_state=SafetyState.HIGH,
    is_e_stop=False,
    cmd_v=0.8,
    cmd_w=0.0,
    nav_state="TRACKING",
    decision_status="PATH_FOUND",
    reobserve=False
):
    """Helper to mock tele dict structure for testing derive_judge_state."""
    odometry = SimpleNamespace(
        tracking_status=tracking_status,
        confidence=vo_conf,
        relocalization_state=reloc_state
    )
    decision_log = SimpleNamespace(
        reobserve_active=reobserve,
        reason="Mock test reason"
    )
    safety = SimpleNamespace(
        safety_state=safety_state,
        is_emergency_stop=is_e_stop,
        commanded_linear_velocity=cmd_v,
        commanded_angular_velocity=cmd_w,
        decision_log=decision_log
    )
    decision = SimpleNamespace(
        navigation_state=nav_state,
        status=decision_status
    )
    return {
        "odometry": odometry,
        "safety": safety,
        "decision": decision
    }


def test_derive_judge_state_canonical_states():
    """Verify that derive_judge_state accurately maps backend signals into all 6 canonical judge states."""
    # 1. AUTONOMOUS
    tele_auto = make_mock_tele(safety_state=SafetyState.HIGH, vo_conf=0.9, tracking_status="TRACKING_OK")
    assert derive_judge_state(tele_auto) == "AUTONOMOUS"

    # 2. CAUTION
    tele_caution = make_mock_tele(safety_state=SafetyState.MEDIUM, vo_conf=0.8)
    assert derive_judge_state(tele_caution) == "CAUTION"

    # 3. UNCERTAIN
    tele_uncert = make_mock_tele(safety_state=SafetyState.LOW, vo_conf=0.7)
    assert derive_judge_state(tele_uncert) == "UNCERTAIN"

    # 4. LOCALIZATION LOST
    tele_lost = make_mock_tele(tracking_status="TRACKING_LOST", vo_conf=0.05)
    assert derive_judge_state(tele_lost) == "LOCALIZATION LOST"

    # 5. SAFE STOP
    tele_stop = make_mock_tele(safety_state=SafetyState.CRITICAL, is_e_stop=True, cmd_v=0.0)
    assert derive_judge_state(tele_stop) == "SAFE STOP"

    # 6. RECOVERING
    tele_recov = make_mock_tele(reloc_state="RELOCALIZING", nav_state="RECOVERY_HOLD")
    assert derive_judge_state(tele_recov) == "RECOVERING"


def test_compute_explainability_all_fields():
    """Verify that compute_explainability accurately formats why_path_changed, why_speed_reduced, and why_stopped."""
    cost_breakdown = PathCostBreakdown(
        total_score=0.45,
        progress_score=0.8,
        clearance_score=0.2,
        traversability_score=0.6,
        heading_score=0.9,
        average_cost=18.5,
        min_clearance_m=0.72,
        primary_terrain="FREE",
        explanation_text="Trajectory selected over FREE terrain maintaining 0.72m clearance."
    )
    decision = SimpleNamespace(
        recommended_linear_velocity=0.5,
        recommended_angular_velocity=0.15,
        recommended_steering="SLIGHT_LEFT",
        status="PATH_FOUND",
        cost_explanation=cost_breakdown,
        min_clearance_m=0.72,
        navigation_state="TRACKING"
    )
    safety_log = SafetyDecisionLog(
        timestamp=100.0,
        confidence=0.7,
        reason="Caution mode: Obstacle proximity scaled speed to 60%",
        action="SCALE_SPEED_MEDIUM",
        state="MEDIUM",
        signals={},
        nominal_v=0.75,
        recommended_v=0.45,
        recommended_w=0.15,
        clearance_margin_m=0.72,
        reobserve_active=False
    )
    safety = SafetyResult(
        overall_confidence=0.7,
        safety_state=SafetyState.MEDIUM,
        commanded_linear_velocity=0.45,
        commanded_angular_velocity=0.15,
        speed_scale_factor=0.6,
        clearance_inflation_factor=1.2,
        is_emergency_stop=False,
        audit_reasons=["Caution mode: Obstacle proximity scaled speed to 60%"],
        action=SafetyAction.LOWER_SPEED_EXPAND_MARGIN,
        decision_log=safety_log
    )

    tele = {
        "decision": decision,
        "safety": safety
    }

    exp = compute_explainability(tele, safety)

    assert "why_path_changed" in exp
    assert "why_speed_reduced" in exp
    assert "why_stopped" in exp

    # Check path change reasoning
    assert "SLIGHT_LEFT" in exp["why_path_changed"]
    assert "0.72m" in exp["why_path_changed"]

    # Check speed reduction reasoning
    assert "60%" in exp["why_speed_reduced"]
    assert "Obstacle proximity" in exp["why_speed_reduced"]

    # Check stop explanation when in motion
    assert "NOT STOPPED" in exp["why_stopped"]
    assert "0.45 m/s" in exp["why_stopped"]


def test_compute_explainability_emergency_stop():
    """Verify explainability generates appropriate stop reason when emergency stop is active."""
    cost_breakdown = PathCostBreakdown(
        total_score=10.0,
        progress_score=0.0,
        clearance_score=0.0,
        traversability_score=1.0,
        heading_score=0.0,
        average_cost=255.0,
        min_clearance_m=0.1,
        primary_terrain="BLOCKED",
        explanation_text="Trajectory blocked."
    )
    decision = SimpleNamespace(
        recommended_linear_velocity=0.0,
        recommended_angular_velocity=0.0,
        recommended_steering="STOP",
        status="OBSTACLE_BLOCKED",
        cost_explanation=cost_breakdown,
        min_clearance_m=0.1,
        navigation_state="ESTOP"
    )
    safety_log = SafetyDecisionLog(
        timestamp=102.0,
        confidence=0.1,
        reason="Visual odometry lost (status=TRACKING_LOST, C_vo=0.05). Holding position.",
        action="SAFE_STOP_CRITICAL",
        state="CRITICAL",
        signals={},
        nominal_v=0.75,
        recommended_v=0.0,
        recommended_w=0.0,
        clearance_margin_m=0.1,
        reobserve_active=False
    )
    safety = SafetyResult(
        overall_confidence=0.1,
        safety_state=SafetyState.CRITICAL,
        commanded_linear_velocity=0.0,
        commanded_angular_velocity=0.0,
        speed_scale_factor=0.0,
        clearance_inflation_factor=2.0,
        is_emergency_stop=True,
        audit_reasons=["Visual odometry lost (status=TRACKING_LOST, C_vo=0.05). Holding position."],
        action=SafetyAction.SAFE_STOP,
        decision_log=safety_log
    )

    tele = {
        "decision": decision,
        "safety": safety
    }

    exp = compute_explainability(tele, safety)
    assert "STOPPED / SAFE HOLD" in exp["why_stopped"]
    assert "TRACKING_LOST" in exp["why_stopped"]


def test_process_and_cache_frame_structure():
    """Verify that process_and_cache_frame generates all backend topic payloads required by the judge dashboard."""
    data = process_and_cache_frame("scenario_1_open_path", 0)

    # 1. Root telemetry
    assert data["frame_id"] == 0
    assert "fps" in data
    assert "latency_ms" in data
    assert data["judge_state"] in ["AUTONOMOUS", "CAUTION", "UNCERTAIN", "SAFE STOP", "LOCALIZATION LOST", "RECOVERING"]

    # 2. Explainability
    exp = data["explainability"]
    assert "why_path_changed" in exp and len(exp["why_path_changed"]) > 0
    assert "why_speed_reduced" in exp and len(exp["why_speed_reduced"]) > 0
    assert "why_stopped" in exp and len(exp["why_stopped"]) > 0

    # 3. Images (JPEG base64 encoded)
    imgs = data["images"]
    for img_key in ["rgb", "depth", "semantic", "costmap", "traversability"]:
        assert img_key in imgs
        assert isinstance(imgs[img_key], str)
        assert len(imgs[img_key]) > 100

    # 4. Planning (A* and DWA)
    planning = data["planning"]
    assert "global_path" in planning
    assert len(planning["global_path"]) > 0
    assert "selected_path" in planning
    assert len(planning["selected_path"]) > 0
    assert "candidate_paths" in planning

    # 5. Multi-Source Confidence
    conf = data["confidence_breakdown"]
    for c_key in ["c_perc", "c_geom", "c_vo", "c_fusion", "c_total", "disagreement", "temporal"]:
        assert c_key in conf
        assert 0.0 <= conf[c_key] <= 1.0

    # 6. Motion
    motion = data["motion"]
    assert "linear_velocity" in motion
    assert "angular_velocity" in motion
    assert "steering_direction" in motion
    assert "navigation_state" in motion


def test_dashboard_http_server_endpoints():
    """Test actual HTTP GET endpoints against an ephemeral ThreadingHTTPServer instance."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    port = server.server_port

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Test /api/scenarios
        req = urllib.request.urlopen(f"{base_url}/api/scenarios", timeout=5)
        assert req.status == 200
        resp_data = json.loads(req.read().decode("utf-8"))
        scenarios = resp_data["scenarios"]
        scenario_ids = [s["id"] for s in scenarios]
        assert "scenario_1_open_path" in scenario_ids
        assert "scenario_2_sudden_obstacle" in scenario_ids
        assert "live_camera" in scenario_ids
        assert "live_kaggle_offroad" in scenario_ids
        assert "live_webcam" in scenario_ids

        # 2. Test /api/telemetry
        req_t = urllib.request.urlopen(f"{base_url}/api/telemetry?scenario=scenario_1_open_path&frame=0", timeout=10)
        assert req_t.status == 200
        telem = json.loads(req_t.read().decode("utf-8"))
        assert telem["frame_id"] == 0
        assert "judge_state" in telem
        assert "explainability" in telem
        assert "images" in telem
        assert "traversability" in telem["images"]

        # 3. Test /api/reset
        req_r = urllib.request.urlopen(f"{base_url}/api/reset?scenario=scenario_1_open_path", timeout=5)
        assert req_r.status == 200
        res_reset = json.loads(req_r.read().decode("utf-8"))
        assert res_reset["status"] == "reset"

        # 4. Test /api/live_telemetry
        req_l = urllib.request.urlopen(f"{base_url}/api/live_telemetry", timeout=10)
        assert req_l.status == 200
        live_telem = json.loads(req_l.read().decode("utf-8"))
        assert "judge_state" in live_telem
        assert "explainability" in live_telem
        assert live_telem.get("is_live") is True

        # 5. Test /api/live_telemetry with live_kaggle_offroad
        req_k = urllib.request.urlopen(f"{base_url}/api/live_telemetry?scenario=live_kaggle_offroad", timeout=10)
        assert req_k.status == 200
        kaggle_telem = json.loads(req_k.read().decode("utf-8"))
        assert kaggle_telem.get("is_live") is True
        assert "live_sensor" in kaggle_telem

    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_upload_image_and_video():
    """Test POST /api/upload with image and video to execute the full UGV pipeline."""
    import cv2
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    port = server.server_port

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Create a synthetic test image (640x480 RGB)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_img[240:, :] = [34, 139, 34]  # Green ground
        dummy_img[:240, :] = [235, 206, 135] # Sky
        success, img_encoded = cv2.imencode(".jpg", dummy_img)
        assert success
        img_bytes = img_encoded.tobytes()

        # Build multipart/form-data body
        boundary = "----WebKitFormBoundarySIH26126Test"
        c_type = f"multipart/form-data; boundary={boundary}"
        
        parts = [
            f"--{boundary}".encode("utf-8"),
            b'Content-Disposition: form-data; name="file"; filename="field_test.jpg"',
            b'Content-Type: image/jpeg',
            b'',
            img_bytes,
            f"--{boundary}--".encode("utf-8"),
            b''
        ]
        body = b"\r\n".join(parts)

        # POST /api/upload
        req = urllib.request.Request(
            f"{base_url}/api/upload",
            data=body,
            headers={
                "Content-Type": c_type,
                "Content-Length": str(len(body)),
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "success"
            assert data["scenario_id"] == "uploaded_image"
            assert data["media_type"] == "image"
            assert "telemetry" in data
            telem = data["telemetry"]
            assert "judge_state" in telem
            assert "motion" in telem
            assert "linear_velocity" in telem["motion"]
            assert "angular_velocity" in telem["motion"]
            assert "confidence_breakdown" in telem
            assert "explainability" in telem
            assert "images" in telem

        # Verify scenario is registered in /api/scenarios
        req_scen = urllib.request.urlopen(f"{base_url}/api/scenarios", timeout=5)
        scen_data = json.loads(req_scen.read().decode("utf-8"))
        scen_ids = [s["id"] for s in scen_data["scenarios"]]
        assert "uploaded_image" in scen_ids

        # Verify telemetry can be queried via /api/telemetry
        req_get = urllib.request.urlopen(f"{base_url}/api/telemetry?scenario=uploaded_image&frame=0", timeout=10)
        assert req_get.status == 200
        get_telem = json.loads(req_get.read().decode("utf-8"))
        assert get_telem["frame_id"] == 0
        assert get_telem["scenario"] == "uploaded_image"
        assert "judge_state" in get_telem

        # Test empty payload failure
        empty_req = urllib.request.Request(
            f"{base_url}/api/upload",
            data=b"",
            headers={"Content-Length": "0"},
            method="POST"
        )
        try:
            urllib.request.urlopen(empty_req, timeout=5)
            assert False, "Should have raised HTTPError for empty payload"
        except urllib.error.HTTPError as e:
            assert e.code == 400

    finally:
        server.shutdown()
        server.server_close()
