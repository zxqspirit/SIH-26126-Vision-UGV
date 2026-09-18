# Final Architecture Specification

**Project:** SIH 26126 — Vision-Based Autonomous Navigation for Outdoor UGV in GPS-Denied Environments  
**Organization:** Bharat Electronics Limited (BEL)  
**System Name:** TerrainSight UGV  
**System Classification:** Offline-First, Software-Only Vision Autonomy Stack  
**Document Release:** v1.0-RC1 (Final Technical Release)  

---

> [!IMPORTANT]
> ### Critical Deployment Boundary & Terminology Enforcement
> - **PROVEN NOW (Current Software Prototype):** A fully integrated, real-data-validated 10-stage vision navigation software stack executing on authentic outdoor camera and depth recordings. All velocity and steering outputs are **software recommendations only**. Validated on 75 multi-modal frames (375 pipeline cycles) with zero false-safe failures, $83.16\text{ ms}$ latency ($12.0\text{ FPS}$ on CPU), and deterministic fail-safe throttles.
> - **FUTURE PHYSICAL UGV DEPLOYMENT (Roadmap):** Integrating physical motor controller drivers (CAN/PWM), RTK-GPS/wheel-encoder state estimation, physical emergency braking actuators, compute-box ruggedization (IP67), and field trials on rolling vehicle chassis.
> - **Explicit Negative Claims:** We do **NOT** claim physical testing, simulation, production-ready, military-grade, 100% safe, or fully autonomous physical UGV.

---

## 1. System Overview & Architectural Topology

TerrainSight UGV solves the problem of outdoor ground navigation when GNSS/GPS signals are jammed, degraded, or unavailable. Rather than relying on heavy vision-language models (VLMs) or simulation environments, the system implements a high-efficiency, multi-modal vision pipeline that couples **lightweight CNN semantic segmentation** with **metric 3D geometry**, **feature-based visual odometry**, **multi-source confidence filtering**, and **deterministic safety arbiters**.

### 10-Stage Pipeline Topology

```
+------------------------------------------------------------------------------------------------------------------+
|                                    TERRAINSIGHT UGV 10-STAGE PIPELINE TOPOLOGY                                   |
+------------------------------------------------------------------------------------------------------------------+
                                                        │
┌─────────────────────────┐                             ▼
│  Stage 1: Real Outdoor  │ ──> RGB Stream (640x480x3 uint8) + Metric Depth Stream (640x480 float32)
│      Data Ingestion     │     Replay / Live Camera Interface with Monotonic Nanosecond Timestamps
└───────────┬─────────────┘
            │
            ├────────────────────────────────────────┬────────────────────────────────────────┐
            ▼                                        ▼                                        ▼
┌─────────────────────────┐              ┌─────────────────────────┐              ┌─────────────────────────┐
│ Stage 2: Preprocessing  │              │  Stage 3: AI Perception │              │ Stage 4: Depth/Geometry │
│ Frame Scaling & Normal. │              │ Lightweight Semantic    │              │ 3D Pointcloud Backproj  │
│ ROI Dynamic Extraction  │              │ Segmentation (Fast-SCNN/│              │ Ground RANSAC Estimation│
└───────────┬─────────────┘              │ MobileNetV3 / Fallback) │              │ Positive/Negative Veto  │
            │                            └───────────┬─────────────┘              └───────────┬─────────────┘
            │                                        │                                        │
            └────────────────────────────────────────┼────────────────────────────────────────┘
                                                     ▼
                                         ┌─────────────────────────┐
                                         │     Stage 5: Fusion     │
                                         │ Continuous Log-Odds     │
                                         │ Geometric Obstacle Veto │
                                         │ Semantic Hazard Veto    │
                                         └───────────┬─────────────┘
                                                     │
                                                     ├────────────────────────────────────────┐
                                                     ▼                                        ▼
                                         ┌─────────────────────────┐              ┌─────────────────────────┐
                                         │ Stage 6: Visual Odometry│              │  Stage 7: Traversability│
                                         │ FAST / ORB Keypoints    │              │ 6-Level 2.5D BEV Costmap│
                                         │ RANSAC Ego-Motion Pose  │              │ Dynamic Cost Inflation  │
                                         │ Feature Inlier Health   │              │ Uncertainty Hazard Veto │
                                         └───────────┬─────────────┘              └───────────┬─────────────┘
                                                     │                                        │
                                                     └────────────────────────────────────────┘
                                                                 │
                                                                 ▼
                                                     ┌─────────────────────────┐
                                                     │    Stage 8: Planning    │
                                                     │ Global A* Waypoint Path │
                                                     │ DWA Trajectory Rollouts │
                                                     │ Footprint Clearance Ck  │
                                                     └───────────┬─────────────┘
                                                                 │
                                                                 ▼
                                                     ┌─────────────────────────┐
                                                     │ Stage 9: Safety Gate &  │
                                                     │     Motion Command      │
                                                     │ Multi-Source Confidence │
                                                     │ Deterministic FSM Arbiter│
                                                     │ Recommended [v, w] Cmd  │
                                                     └───────────┬─────────────┘
                                                                 │
                                                                 ▼
                                                     ┌─────────────────────────┐
                                                     │  Stage 10: SIH Mission  │
                                                     │     Control Dashboard   │
                                                     │ Real-time Stream & Dials│
                                                     │ Explainability Engine   │
                                                     │ "WHY" Decision Audit Log│
                                                     └─────────────────────────┘
```

---

## 2. Hardware Specification & Compute Baseline

| Parameter | Laboratory Target / Benchmark System | Target UGV Platform (Future Deployment) |
|:---|:---|:---|
| **Primary Compute** | Intel Core CPU (Multi-core x86_64, Windows) | NVIDIA Jetson Orin Nano / Xavier NX |
| **GPU Acceleration** | NVIDIA GeForce RTX 4050 Laptop (CUDA 13.1) | 1024-core Ampere GPU (TensorRT / FP16) |
| **System Memory** | 16 GB DDR5 Host RAM | 8 GB / 16 GB Unified LPDDR5 |
| **Primary Sensing** | Forward-facing Stereo / RGB-D Camera | Intel RealSense D435i / D455 / OAK-D Pro |
| **Lens Extrinsics** | Mount Height: $0.45\text{ m}$, Downward Pitch: $12.0^\circ$ | Rigid shock-isolated forward mast ($0.45\text{ m}$) |
| **Power Envelope** | Line / AC Adapter ($90\text{ W}$) | DC-DC $12\text{V}$ Regulated ($15\text{ W} - 25\text{ W}$) |
| **Throughput Target** | $\ge 10\text{ FPS}$ on pure CPU ($12.0\text{ FPS}$ measured) | $\ge 20\text{ FPS}$ with TensorRT FP16 |

---

## 3. Coordinate Systems & Reference Frames

All spatial reasoning adheres to standard robotics coordinate conventions:

```
Camera Optical Frame:                      Robot Base Frame (base_link):
        +Z (Forward / Optical Axis)               +X (Forward Velocity)
       /                                                 ^
      /                                                  |
     /                                                   |
    +-----> +X (Image Right)                +Y <---------+ (Robot Center)
    |                                       (Left)       |
    |                                                    |
    v                                                   +Z (Upward Height)
    +Y (Image Down)
```

- **Optical Frame:** Origin at primary camera left optical center. $+X$ points right across image columns; $+Y$ points downwards across image rows; $+Z$ points forward along optical line of sight.
- **Base Link Frame (`base_link`):** Origin at ground-projection of vehicle footprint center. $+X$ points forward along vehicle heading; $+Y$ points leftwards; $+Z$ points upwards perpendicular to ground plane.
- **Extrinsic Transformation:** Rigid $4\times 4$ Euclidean transform matrix incorporating sensor mounting height ($0.45\text{ m}$) and downward pitch angle ($-12.0^\circ$).

---

## 4. Component Subsystems Breakdown

### 1. Perception Engine (`src/perception/`)
- **Semantic Segmentation:** Fast-SCNN / MobileNetV3 backbone optimized for outdoor terrain classification.
- **Semantic Classes:** Trail/Road, Grass, Gravel/Rock, Obstacle/Tree, Water/Mud, Sky/Void.
- **Classical Fallback:** Deterministic HSV color slicing, ExG (Excess Green index), and Gabor texture filtering activated if CNN confidence drops below threshold or model execution fails.

### 2. Depth Geometry Engine (`src/depth_geometry/`)
- **Back-Projection:** De-projects metric depth pixels $(u, v, Z)$ into 3D camera pointclouds using calibrated focal length ($f_x, f_y$) and principal points ($c_x, c_y$).
- **Ground Plane Estimation:** RANSAC planar estimator extracts ground normal and ground elevation within the near-field ground ROI ($1.0\text{ m} \le Z \le 3.5\text{ m}$).
- **Elevation Slicing:** Points rising $> 0.15\text{ m}$ above the estimated ground surface are flagged as positive obstacles; points sinking $< -0.12\text{ m}$ below the ground plane are flagged as negative obstacles (ditches/holes).

### 3. Fusion Engine (`src/fusion/`)
- **Mode 1 (Continuous Fusion):** Log-odds Bayesian accumulation merging semantic obstacle probability with geometric depth obstacle probability.
- **Mode 2 (Geometric Elevation Veto):** If depth geometry detects a physical 3D elevation barrier $> 0.15\text{ m}$, the cell is unconditionally vetoed as **BLOCKED (cost 255)**, even if the CNN visually misclassified it as traversable terrain.
- **Mode 3 (Semantic Hazard Veto):** If CNN semantics detect water or slick mud (which appear geometrically flat to stereo/monocular depth), the cell is unconditionally vetoed as **HIGH RISK (cost 200)** or **BLOCKED**.

### 4. Visual Odometry Engine (`src/localization/`)
- **Feature Extraction:** FAST corner detection coupled with ORB rotation-invariant descriptors.
- **Frame-to-Frame Tracking:** 2D-to-2D feature matching with RANSAC 5-point essential matrix estimation and metric scale recovery from synchronized depth.
- **Health Monitoring:** Continuous measurement of tracked inliers. If inlier count drops below 30, a `TRACKING_DEGRADED` warning is triggered; below 15 triggers `TRACKING_LOST`.

### 5. Traversability & Costmap Engine (`src/traversability/`, `src/planning/costmap_2d.py`)
- **Grid Representation:** $100\times 100$ 2.5D Bird’s-Eye-View (BEV) grid covering $5.0\text{ m} \times 5.0\text{ m}$ ($0.05\text{ m/cell}$ resolution).
- **6-Level Costmap Classes:**
  1. `PREFERRED_PATH` (Cost: 0) — Smooth trail/road.
  2. `FREE_TERRAIN` (Cost: 30) — Flat grass/firm dirt.
  3. `MEDIUM_RISK` (Cost: 80) — Uneven terrain, loose gravel.
  4. `HIGH_RISK` (Cost: 160) — Dense brush, boundary proximity.
  5. `BLOCKED` (Cost: 255) — Physical barriers, water bodies, negative dropoffs.
  6. `UNKNOWN` (Cost: 128) — Unobserved cells (Rule 11: unknown is never free space).
- **Inflation Layer:** Gaussian-decay obstacle inflation ensuring vehicle footprint ($0.60\text{ m} \times 0.45\text{ m}$) maintains minimum clearance $\ge 0.35\text{ m}$.

### 6. Planning & Decision Engine (`src/planning/`)
- **Global Path Planning:** Grid-based A\* algorithm evaluating lowest-cost traversal paths from robot origin to designated sub-goals.
- **Local Dynamic Window Approach (DWA):** Rollout of kinematically feasible velocity candidate arcs $(v, \omega)$ evaluating heading alignment, clearance to obstacles, terrain cost, and velocity magnitude.

### 7. Safety Gate & Decision Arbiter (`src/safety/`)
- **Multi-Source Confidence:** Integrated metric combining CNN prediction entropy, depth validity ratio, semantic/depth agreement score, and VO tracking inliers:
  $$C_{\text{total}} = 0.30 C_{\text{perc}} + 0.30 C_{\text{depth}} + 0.20 C_{\text{agree}} + 0.20 C_{\text{vo}}$$
- **Deterministic 4-State Arbiter FSM:**
  - `HIGH` ($C \ge 0.75$): Nominal software recommendation ($v_{\text{rec}} = 0.80\text{ m/s}$).
  - `MEDIUM` ($0.45 \le C < 0.75$): Throttled recommendation ($v_{\text{rec}} = 0.35\text{ m/s}$, expanded clearance margin).
  - `LOW` ($0.25 \le C < 0.45$): Crawl recommendation ($v_{\text{rec}} = 0.15\text{ m/s}$, conservative routing).
  - `CRITICAL` ($C < 0.25$ or tracking lost): Deterministic safe-stop recommendation ($v_{\text{rec}} = 0.00\text{ m/s}$, active position hold).

### 8. Explainable Dashboard Server (`src/visualization/dashboard_server.py`)
- Real-time Flask & WebSocket mission control UI operating at `http://localhost:5000`.
- 4 synchronized video/grid streams: RGB/Semantics overlay, Metric Depth colormap, BEV Costmap with DWA trajectory rollouts, and VO Odometry path.
- **Explainability "WHY" Engine:** Automatically details exactly why paths altered, why speeds throttled, or why safe-stop recommendations triggered.

---

## 5. Architectural Invariants & Non-Negotiable Rules

1. **Rule 9 (Deterministic Safety):** Safety arbitration logic is strictly deterministic code with transparent conditional rules; never delegated to probabilistic heuristics.
2. **Rule 10 (Zero Direct Control):** No LLM or generative foundation model is ever in the motor control loop.
3. **Rule 11 (Unknown Terrain Penalty):** Unobserved grid cells carry a base penalty ($128$) and are never assumed to be free space.
4. **Rule 12 (Invalid Depth Uncertainty):** Missing or corrupted depth (glare/absorption) carries an uncertainty hazard penalty and is never treated as traversable ground.
5. **Rule 13 (Actionable Confidence Scaling):** Confidence must actively alter vehicle speed recommendations and safety envelopes; it is never a passive telemetry metric.

---

## 6. Proven Now vs. Future Physical UGV Deployment

| Subsystem Dimension | Proven Now (Current Software Release) | Future Physical UGV Deployment (Field Integration) |
|:---|:---|:---|
| **Platform Actuation** | Pure software velocity and yaw recommendations | CAN Bus / Roboteq motor controller closed-loop PWM |
| **Sensor Data Source** | Real outdoor synchronized RGB + Depth datasets | Live RealSense D435i / D455 USB 3.2 gen 1 stream |
| **State Estimation** | Monocular/Stereo Visual Odometry (Ego-motion) | Multi-sensor EKF (VO + 9-DoF IMU + Wheel Encoders) |
| **Global Waypoints** | Local coordinate frame sub-goals | RTK-GPS global coordinate waypoint mission planner |
| **Emergency Braking** | Deterministic software zero-velocity command | Hardwired physical mechanical disc brakes & e-stop |
| **Environment Coverage** | 5 outdoor benchmark sequences (75 frames) | All-weather, day/night long-duration trail endurance |
