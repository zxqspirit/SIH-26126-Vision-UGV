# SIH Technical Demonstration Runbook

**Project:** SIH 26126 — Vision-Based Autonomous Navigation for Outdoor UGV  
**System Name:** TerrainSight UGV  
**Document Release:** v1.0-RC1 (Demonstration Master Guide)  
**Target Audience:** SIH Evaluation Panel, BEL Industry Judges, Demonstration Operators  

---

> [!IMPORTANT]
> ### Golden Rule of SIH Demonstration
> 1. **Data Source:** Strictly **REAL OUTDOOR DATA** (`datasets/processed/` scenarios 1 through 5). No synthetic data. No simulation. No Gazebo.
> 2. **Autonomy Scope:** Demonstrating software-only vision autonomy. All velocity and yaw outputs are **software recommendations** for an onboard motion controller.
> 3. **Explainability:** When the robot steers, slows, or stops, use the dashboard's **Explainability Engine** to show the judges the mathematical causality (Observation -> Interpretation -> Action).

---

## 1. Quick-Start Terminal Commands

Follow these three commands to launch the complete evaluation and demonstration environment:

### Step 1: Verify Regression Suite (Pre-Flight Check)
Open a terminal in the repository root and confirm all 121 tests pass:
```powershell
python -m pytest tests/ -q
```
*Expected Output:* `121 passed in ~90s (100% PASS)`

### Step 2: Launch Real-Time Mission Control Dashboard
In the primary terminal, run the dashboard server:
```powershell
python src/visualization/dashboard_server.py
```
*Expected Output:*
```
 * Running on http://127.0.0.1:5000
 * Mission Control WebSocket broadcast active
 * Explainability "WHY" Engine initialized
```

### Step 3: Open Browser Interface
Open Google Chrome or Microsoft Edge and navigate to:
```
http://localhost:5000
```
You will see the 4 synchronized visual streams, navigation dials, multi-source confidence meters, and the live Explainability Decision Console.

---

## 2. The 10-Stage Demonstration Sequence

During the demo, guide the judges through the exact sequential data transformation pipeline:

```
[ 1. Real Outdoor Data ]  ──> Authentic RGB + Metric Depth stream from camera
           │
           ▼
[ 2. AI Perception ]      ──> Lightweight CNN semantic segmentation (Trail, Grass, Rock, Water)
           │
           ▼
[ 3. Depth Geometry ]     ──> 3D Pointcloud, ground plane RANSAC, elevation barriers (dz > 0.15m)
           │
           ▼
[ 4. Multi-Modal Fusion ] ──> Mode 1 Log-Odds + Mode 2 Geometric Elevation Veto + Mode 3 Semantic Veto
           │
           ▼
[ 5. Visual Localization] ──> FAST/ORB feature tracking, ego-motion pose, inlier health check
           │
           ▼
[ 6. Traversability ]     ──> 6-level 2.5D Bird's-Eye-View Costmap with Gaussian obstacle inflation
           │
           ▼
[ 7. Planning & Path ]    ──> Global A* optimal path + 15 DWA candidate forward rollouts
           │
           ▼
[ 8. Multi-Source Conf ]  ──> Composite metric combining CNN entropy, depth ratio, agreement, VO
           │
           ▼
[ 9. Motion Command ]     ──> Software recommended [v, w] command with deterministic FSM safety
           │
           ▼
[ 10. Dashboard & WHY ]   ──> Real-time operator UI with instant causality explanation
```

---

## 3. Five-Scenario Demonstration Walkthrough Script

Use the Scenario Dropdown on the top-right of the dashboard (`http://localhost:5000`) to step through the 5 real outdoor scenarios:

### Act 1: Scenario 1 — Open Trail Autonomous Traversal
- **Action:** Select `scenario_1_open_path` and click **Play** (or step through frames).
- **What to show the judges:**
  - **RGB Stream:** Open outdoor dirt trail surrounded by low grass.
  - **Depth Stream:** Uniform, smooth ground depth gradient from $1.0\text{m}$ to $5.0\text{m}$.
  - **Costmap:** Clean green `PREFERRED_PATH` channel along the path center.
  - **Gauges:** Speed dial indicates **$0.80\text{ m/s}$**, steering needle points centered ($0.0^\circ$), Confidence shows **$0.85$ (`HIGH`)**.
- **Judge Narration:**  
  *"In nominal conditions, the fusion engine merges high CNN confidence ($0.85$) with clean geometric depth ($92\%$ valid). A\* plans along the preferred path, and DWA recommends our maximum cruising speed of $0.80\text{ m/s}$."*

### Act 2: Scenario 2 — Sudden Dynamic Obstacle Pop-Up
- **Action:** Select `scenario_2_sudden_obstacle`, scrub to frame 6.
- **What to show the judges:**
  - **Perception & Depth:** At frame 6, a physical barrier enters the near-field ($Z = 2.1\text{ m}$).
  - **Costmap:** Cell turns instantly red (`BLOCKED`, Cost 255) with a $0.35\text{ m}$ yellow inflation buffer.
  - **Reaction Latency:** The path shifts immediately in the same cycle ($0$ frame lag, $< 100\text{ ms}$).
  - **Explainability Console:**  
    `[WHY PATH CHANGED]: Obstacle detected at (X=2.1m, Y=-0.2m); replanned global A* trajectory to right flank (Clearance: 0.41m).`
- **Judge Narration:**  
  *"Notice the zero-frame reaction latency. The moment geometric elevation exceeds $0.15\text{ m}$, Mode 2 Geometric Veto blocks the cell. The planner recalculates an evasive maneuver maintaining a safe $0.41\text{ m}$ clearance, exceeding our $0.35\text{ m}$ requirement."*

### Act 3: Scenario 3 — Complex Terrain & Soft Boundary Transition
- **Action:** Select `scenario_3_terrain_boundary`.
- **What to show the judges:**
  - **Perception:** Path curves from asphalt onto rough grass with loose gravel on the side.
  - **Costmap:** 6 distinct cost levels visualized: Path (0, green), Grass (30, cyan), Gravel (80, yellow).
  - **DWA Rollouts:** DWA candidate arcs bend naturally to favor low-cost ground even though grass is physically flat.
- **Judge Narration:**  
  *"This proves why depth alone is insufficient. Geometrically, the grass and gravel are flat. But our semantic costmap assigns higher traversal cost to rough terrain, keeping the recommended trajectory firmly on the firm trail."*

### Act 4: Scenario 4 — Solar Glare & Depth Degradation (Rule 12 Invariant)
- **Action:** Select `scenario_4_depth_degradation`.
- **What to show the judges:**
  - **Depth Stream:** Solar glare blanks out $70\%$ of the depth map (black void).
  - **Costmap:** Black areas are marked with striped hazard shading (Rule 12: invalid depth is uncertainty hazard, never free space).
  - **Confidence & Speed:** Confidence drops from $0.85$ to $0.30$ (`LOW`). Speed dial automatically scales down from $0.80\text{ m/s}$ to **$0.15\text{ m/s}$ (Crawl)**.
  - **Explainability Console:**  
    `[WHY SPEED REDUCED]: Depth validity fell to 28.4% due to solar blooming; throttled to conservative crawl (0.15 m/s) under Rule 12.`
- **Judge Narration:**  
  *"Many systems crash here by assuming missing depth is empty space. In TerrainSight, Rule 12 enforces an uncertainty penalty. Furthermore, Rule 13 automatically throttles recommended speed to a crawl."*

### Act 5: Scenario 5 — Severe Visual Degradation & Tracking Loss (Rule 13 Fail-Safe)
- **Action:** Select `scenario_5_visual_degradation`, advance to frame 10.
- **What to show the judges:**
  - **Visual Stream:** Severe optical motion blur and low illumination.
  - **VO Health:** Feature inliers drop to $11 < 15$. VO State turns red: `TRACKING_LOST`.
  - **Safety Arbiter:** State changes to **`CRITICAL`**. Linear velocity drops immediately to **$0.00\text{ m/s}$**.
  - **Explainability Console:**  
    `[WHY STOPPED]: Visual odometry inliers dropped to 11 (< 15 threshold); asserted deterministic safe-stop position hold under Rule 13.`
- **Judge Narration:**  
  *"When optical tracking collapses, we do not guess or blunder forward. The system immediately recommends an active safe stop, logging the root cause and holding position until valid features return."*

---

## 4. Live Judge Q&A Anchors

- If asked: *"Is this running in simulation?"*  
  **Answer:** *"No. Every single frame you see is authentic multi-modal outdoor data recorded on physical pathways with real stereo/depth sensors. We have zero Gazebo code in this project."*
- If asked: *"Why isn't the physical robot driving right now?"*  
  **Answer:** *"Per SIH problem statement rules and disciplined safety engineering, this prototype is an offline-first software navigation stack. We provide verified, deterministic motion recommendations without risking physical hardware collision before formal validation."*
- If asked: *"What is your latency?"*  
  **Answer:** *"Our end-to-end cycle is $83.16\text{ ms}$ ($12.0\text{ FPS}$) on standard CPU without needing GPU acceleration, with $< 100\text{ ms}$ obstacle reaction time."*
