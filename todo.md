# SIH 26126 — Project Tracker

> This file is the operational source of truth for unfinished work.
> Keep tasks small enough to verify.

---

## STATUS LEGEND

- `[ ]` not started
- `[-]` in progress
- `[x]` completed
- `[!]` blocked
- `[?]` decision required

---

# PHASE 0 — FOUNDATION

- [x] Confirm Problem Statement 26126 requirements
- [x] Select hybrid architecture: stereo/RGB-D + lightweight CNN + depth geometry + VO/SLAM + planner + confidence fallback
- [x] Finalize project name: TerrainSight UGV
- [x] Create Git repository and branch strategy
- [x] Create reproducible development environment
- [x] Record compute hardware: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 13.1), Ubuntu 24.04 WSL2, ROS 2 Jazzy
- [x] Select ROS 2 version: ROS 2 Jazzy Jalisco
- [x] Reference camera mount model: Forward-facing RGB-D / Stereo (height: 0.45m, downward pitch: 12 deg)
- [x] Define software output contract: `linear_velocity`, `angular_velocity`, `steering_direction`, `navigation_state`, `confidence`, `safety_state` (No motor actuation)

---

# PHASE 1 — SYSTEM DEFINITION

- [x] Draw final system architecture (Sensor -> Perception -> Depth Geometry -> Fusion -> VO -> Costmap & Planner -> Safety Gate -> Motion Command -> Laptop Dashboard)
- [x] Define strongly-typed data interfaces between modules (`src/interfaces/types.py`)
- [x] Define camera coordinate frames (Optical: X right, Y down, Z forward)
- [x] Define robot coordinate frames (Base_link: X forward, Y left, Z up)
- [x] Define target operating speed: nominal 0.80 m/s, cautious 0.35 m/s, crawl 0.15 m/s
- [x] Define minimum obstacle clearance: 0.45 m emergency stopping distance, 0.85 m inflation
- [x] Define emergency-stop behavior: deterministic zero command on obstacle proximity or confidence < 0.25
- [x] Define confidence states: `HIGH_CONFIDENCE`, `CAUTIOUS_DEGRADED`, `LOW_CONFIDENCE_SLOW`, `SAFETY_STOP`, `LOCALIZATION_LOST`
- [x] Define planner input/output contract: DWA trajectory rollouts with forward progress scoring
- [x] Define logging format: JSON frame telemetry with audit reason logs

---

# PHASE 2 — CAMERA & REAL OUTDOOR DATA

- [x] Camera intrinsics and extrinsics configuration (`config/camera/camera_params.yaml`)
- [x] Real outdoor visual dataset loader (`src/datasets/outdoor_dataset_loader.py`)
- [x] Outdoor scenario generation suite (`scripts/generate_sample_scenarios.py`)
- [x] Curate 5 realistic outdoor scenarios (open path, sudden obstacle, terrain boundary, glare/invalid depth, feature loss)
- [x] Enforce Rule 12: Invalid depth is never treated as free space

---

# PHASE 3 — PERCEPTION BASELINE

- [x] Define terrain segmentation classes (paved, dirt path, low grass, gravel, high vegetation, solid obstacles, puddle)
- [x] Implement lightweight CNN model runner with ONNX support (`src/perception/traversability_net.py`)
- [x] Implement deterministic classical perception fallback (`src/perception/color_texture_fallback.py`)
- [x] Excess Green (ExG) and texture energy extraction
- [x] Horizon prior and ground zone boundary calculation
- [x] Compute perception confidence $C_{perc}$ based on lighting adequacy and class margin
- [x] Measure inference latency: <30 ms on CPU

---

# PHASE 4 — DEPTH GEOMETRY

- [x] 3D metric point cloud back-projection (`src/depth_geometry/point_cloud.py`)
- [x] Camera optical frame to base_link transformation
- [x] Ground plane estimation in base_link (`src/depth_geometry/ground_estimator.py`)
- [x] Positive obstacle detection (step threshold > 15 cm) (`src/depth_geometry/obstacle_detector.py`)
- [x] Negative obstacle / drop-off detection
- [x] Depth validity masking and uncertainty penalization (Rule 12)
- [x] Compute depth geometry confidence $C_{geom}$

---

# PHASE 5 — FUSION

- [x] Multimodal evidence fusion (`src/fusion/disagreement.py`, `src/fusion/fusion_engine.py`)
- [x] Geometry Veto: Physical obstacle overrides semantic traversability claims
- [x] Semantic Veto: Visual hazards (puddle, mud) override geometric flatness
- [x] Enforce Rule 11: Unknown / unobserved terrain is not automatically free space
- [x] 2D Bird's-Eye-View (BEV) local costmap grid generation (10m x 10m forward grid)
- [x] Compute fusion confidence $C_{fusion}$

---

# PHASE 6 — VISUAL ODOMETRY / LOCALIZATION

- [x] Feature-based Visual Odometry engine (`src/localization/visual_odometry.py`)
- [x] Lucas-Kanade pyramidal optical flow and PnP-RANSAC 6-DOF motion estimation
- [x] Cumulative dead-reckoning pose tracking $(x, y, \theta)$
- [x] Tracking health diagnostics: inlier count, feature density, flow variance (`src/localization/tracking_diagnostics.py`)
- [x] Tracking status states: `TRACKING_OK`, `TRACKING_DEGRADED`, `TRACKING_LOST`
- [x] Compute VO confidence $C_{vo}$

---

# PHASE 7 — PLANNING & CONTROL

- [x] 2D local costmap with Euclidean obstacle distance transform and inflation (`src/planning/costmap_2d.py`)
- [x] Dynamic Window Approach (DWA) local trajectory planner (`src/planning/dwa_planner.py`)
- [x] Kinematically feasible circular arc rollouts
- [x] Multi-objective trajectory scoring: forward progress, clearance, terrain cost, heading alignment
- [x] Standardized `UGVMotionCommand` formatting (`src/control/motion_command_generator.py`)

---

# PHASE 8 — CONFIDENCE SAFETY LAYER

- [x] Deterministic multi-source confidence arbiter (`src/safety/safety_gate.py`)
- [x] Multi-source aggregation (weakest-link: $C_{total} = \min(C_{perc}, C_{geom}, C_{vo}, C_{fusion})$)
- [x] Enforce Rule 9: Deterministic and inspectable state machine
- [x] Enforce Rule 10: No LLM control of physical systems
- [x] Enforce Rule 13: Confidence produces actual behavior change (speed scaling, clearance expansion, safety stop)
- [x] Inspectable audit reason logs for every decision

---

# PHASE 9 — INTEGRATION & VERIFICATION

- [x] End-to-end navigation pipeline orchestrator (`src/pipeline.py`)
- [x] Comprehensive pytest test suite: 17/17 tests passing (`tests/`)
- [x] Automated dataset replay and assertion verification runner (`scripts/replay_dataset.py`)
- [x] Verified all 5 outdoor test scenarios
- [x] ROS 2 Jazzy launch file (`launch/terrain_sight_launch.py`)

---

# PHASE 10 — LAPTOP DASHBOARD DEMO

- [x] Real-time Mission Control Dashboard Server (`src/visualization/dashboard_server.py`)
- [x] Interactive glassmorphic browser UI (`src/visualization/static/index.html`)
- [x] 4 synchronized monitors: RGB/Semantics, Metric Depth Colormap, BEV Costmap + DWA Trajectories, VO Path
- [x] Motion dials: linear velocity, angular velocity, steering needle
- [x] Multi-source confidence breakdown meters
- [x] Live Deterministic Safety Arbiter Inspector
- [x] Interactive playback controls: Play, Pause, Step, Scrub, Scenario Selector

---

# DECISION LOG

### Decision 001
**Topic:** Project Constraints & Execution Paradigm  
**Status:** accepted  
**Decision:** Software-only autonomous navigation stack validated strictly on real outdoor visual and depth data. No physical UGV motor actuation. No Gazebo, no simulation.  
**Reason:** Strict adherence to SIH 26126 guidelines and user project constraints.  
**Evidence:** `README.md`, `pipeline.py`, `scripts/replay_dataset.py`.

### Decision 002
**Topic:** Perception Architecture  
**Status:** accepted  
**Decision:** Lightweight CNN semantic segmentation (Fast-SCNN / MobileNetV3 / ONNX) with deterministic color/texture/ExG classical fallback. Do NOT train a VLM.  
**Reason:** Low mobile latency (<30 ms), high determinism, zero reliance on external APIs or heavy models.  
**Evidence:** `src/perception/traversability_net.py`, `src/perception/color_texture_fallback.py`.

### Decision 003
**Topic:** Rule 11 & Rule 12 Invariants  
**Status:** accepted  
**Decision:** Unknown terrain is penalized by default (cost 128), and invalid depth (0.0, NaN) carries an uncertainty hazard penalty (>= 0.50). Neither is ever marked as free space.  
**Reason:** Safety invariant: sensor blindness or unobserved areas must not induce aggressive forward motion.  
**Evidence:** `src/depth_geometry/obstacle_detector.py`, `src/fusion/fusion_engine.py`, `tests/test_depth_geometry.py`, `tests/test_costmap.py`.

### Decision 004
**Topic:** Rule 13 Confidence Behavior Scaling  
**Status:** accepted  
**Decision:** Multi-source confidence strictly regulates velocity limits and safety stops: $C \ge 0.75 \implies 0.8\text{ m/s}$, $0.45 \le C < 0.75 \implies 0.35\text{ m/s}$, $0.25 \le C < 0.45 \implies 0.15\text{ m/s}$, $C < 0.25 \implies 0.0\text{ m/s}$ (Emergency Stop).  
**Reason:** Confidence must produce an actual behavior change, not merely decorative metrics.  
**Evidence:** `src/safety/safety_gate.py`, `tests/test_confidence_gate.py`.
