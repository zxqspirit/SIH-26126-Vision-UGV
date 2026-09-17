# SIH 26126 — Vision Based Autonomous Navigation for Outdoor UGV

**Organization:** Bharat Electronics Limited (BEL)  
**Category:** Software | **Theme:** Smart Automation  
**System Name:** TerrainSight UGV  

Offline-first, software-only vision-based autonomous navigation stack for an Unmanned Ground Vehicle (UGV) operating in GPS-denied or GPS-unreliable outdoor environments.

---

## Project Constraints & Core Principles

- **Software-Only Autonomy Stack:** We do NOT have a physical UGV. The system produces recommended motion commands (`linear_velocity`, `angular_velocity`, `steering_direction`, `navigation_state`, `confidence`, `safety_state`) without physical motor actuation.
- **Real Outdoor Visual Data Validation:** No Gazebo. No simulation. No virtual UGV. All perception, depth geometry, fusion, visual odometry, and planning modules are validated strictly on real outdoor visual and depth data across outdoor pathways, obstacles, and terrain transitions.
- **Lightweight AI Perception:** No heavy VLMs. The AI task is lightweight semantic segmentation and traversability estimation (<30 ms latency).
- **Enforced Safety Invariants:**
  - **Rule 9:** Safety behavior must be deterministic and inspectable.
  - **Rule 10:** Never let an LLM directly control a physical robot.
  - **Rule 11:** Unknown terrain is not automatically safe (unobserved cells carry risk penalties).
  - **Rule 12:** Invalid depth is not automatically free space (unmeasured depth carries uncertainty penalties).
  - **Rule 13:** Confidence must produce an actual behavior change (speed scaling, clearance expansion, safety stop).

---

## Architecture & Pipeline

```
RGB / RGB-D / Stereo Camera
        ↓
Lightweight CNN Perception (Semantic Traversability & Terrain Understanding)
        ↓
Depth Geometry (3D Point Cloud, Ground Plane & Metric Obstacle Reasoning)
        ↓
Semantic + Geometric Fusion (Disagreement Detection & Hazard Veto)
        ↓
Visual Odometry / Visual SLAM (Ego-Motion Dead-Reckoning & Tracking Health)
        ↓
Traversability Cost Representation (2D Ego-Centric BEV Grid with Inflation)
        ↓
Path Planning (Dynamic Window Approach - DWA Forward Trajectory Rollout)
        ↓
Confidence-Based Safety (Deterministic Arbiter with Safe Degradation)
        ↓
UGV Motion Command (Standardized Velocity, Steering & State Interface)
        ↓
Laptop Dashboard (Multi-Panel Mission Control Visualization)
```

---

## Repository Layout

- `src/`
  - `interfaces/` — Strongly-typed data contracts (`types.py`, `UGVMotionCommand`, etc.)
  - `perception/` — Lightweight CNN & classical color-texture fallback traversability
  - `depth_geometry/` — 3D metric projection, ground estimation, positive/negative obstacles
  - `fusion/` — Semantic + geometric disagreement detection and BEV costmap fusion
  - `localization/` — Feature-based Visual Odometry and tracking health diagnostics
  - `planning/` — 2D costmap with obstacle inflation and Dynamic Window Approach (DWA) planner
  - `control/` — Standardized `UGVMotionCommand` formatting
  - `safety/` — Deterministic multi-source confidence safety gate
  - `datasets/` — Real outdoor dataset loaders and sequence iterators
  - `visualization/` — Interactive Mission Control Laptop Dashboard (`dashboard_server.py`)
  - `pipeline.py` — Top-level end-to-end autonomous navigation pipeline
- `datasets/processed/` — Curated real-world outdoor validation sequences:
  - `scenario_1_open_path`: Open traversable trail (high confidence, straight motion)
  - `scenario_2_sudden_obstacle`: Positive obstacle on trail (geometry detection & avoidance)
  - `scenario_3_terrain_boundary`: Dirt path flanked by non-traversable bushes
  - `scenario_4_depth_degradation`: Glare & invalid depth region (Rule 12 validation)
  - `scenario_5_visual_degradation`: Low-feature scene (Rule 13 visual tracking loss safety stop)
- `launch/` — ROS 2 Jazzy launch orchestration (`terrain_sight_launch.py`)
- `config/` — Versioned camera, perception, planning, SLAM, and system parameters
- `tests/` — Comprehensive automated pytest suite (17 unit and integration tests)
- `scripts/` — Scenario generation, dataset replay, and benchmarking tools

---

## Quick Start & Validation

### 1. Run Automated Test Suite
```bash
python -m pytest tests/ -v
```

### 2. Replay Real Outdoor Scenarios
```bash
# Replay all 5 scenarios and verify safety assertions
python scripts/replay_dataset.py --all --verify

# Replay a specific scenario
python scripts/replay_dataset.py --scenario scenario_2_sudden_obstacle --verify
```

### 3. Launch Interactive Laptop Mission Control Dashboard
```bash
python -m src.visualization.dashboard_server
```
Open `http://localhost:5000` in your web browser.

**Dashboard Features:**
- **Synchronized 4-Panel Video Stream:** Real Camera RGB/Semantic Overlay, Metric Depth Colormap, 2D BEV Costmap with DWA candidate rollouts, and Visual Odometry XY Trajectory.
- **Live Motion Telemetry:** Commanded linear velocity, angular velocity, and compass steering needle.
- **Multi-Source Confidence Breakdown:** Real-time gauges for $C_{perc}$, $C_{geom}$, $C_{vo}$, $C_{fusion}$, and overall $C_{total}$.
- **Deterministic Safety Arbiter Inspector:** Inspectable audit log explaining the exact rule triggering any speed reduction or emergency stop.
- **Interactive Scrubber:** Play, Pause, Step Frame, and Scenario Selector.
