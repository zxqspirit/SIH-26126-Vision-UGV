# SIH Judge Demonstration Guide & Choreography

**Project:** SIH 26126 — Vision-Based Autonomous Navigation for Outdoor UGV  
**System Name:** TerrainSight UGV  
**Document Classification:** Official Live Demonstration Operator Guide  
**Evaluation Setting:** Live Technical Jury Presentation (Smart India Hackathon)  

---

> [!IMPORTANT]
> ### Demonstration Principles
> - **Source Integrity:** Strictly REAL OUTDOOR DATA (`datasets/processed/` scenario 1 to 5). No synthetic imagery, no Gazebo simulation.
> - **Operational Boundary:** All motion outputs are **software recommendations**. Emphasize disciplined verification prior to physical motor actuation.
> - **Explainability:** Always anchor visual actions to the Explainability Console ("Observation -> Interpretation -> Action").

---

## 1. Pre-Demonstration System Setup

Before calling the judging panel to the terminal, execute this 60-second verification routine:

### Step 1: Execute Regression Pre-Check
In your primary terminal:
```powershell
python -m pytest tests/ -q
```
*Verify:* All 121 tests pass with code 0 (`121 passed in ~95s`).

### Step 2: Start Mission Control Dashboard
In the primary terminal:
```powershell
python src/visualization/dashboard_server.py
```
*Verify:* Terminal prints `Running on http://127.0.0.1:5000` and `Mission Control WebSocket broadcast active`.

### Step 3: Open Mission Control UI
Open Google Chrome or Microsoft Edge in full-screen mode (F11) to:
```
http://localhost:5000
```
Confirm the 4 visual monitor panels, motion dials, confidence meters, and Explainability Console are initialized and ready.

---

## 2. Walkthrough Sequence: 10-Stage Pipeline Demonstration

Guide the evaluators through the exact data transformation sequence:

```
1. Real Outdoor Data   ──> 640x480 RGB + Metric Depth stream from camera
2. Perception          ──> Fast-SCNN / MobileNetV3 semantic segmentation
3. Depth / Geometry    ──> 3D Pointcloud, RANSAC ground fitting, elevation barriers
4. Multi-Modal Fusion  ──> Mode 1 Continuous Log-Odds + Dual Vetoes (Mode 2 & 3)
5. Visual Localization ──> FAST/ORB feature tracking & metric ego-motion
6. Traversability Map  ──> 6-level 2.5D Bird's-Eye-View Costmap with inflation
7. Planning & Path     ──> Global A* pathfinder + 15 DWA candidate rollouts
8. Confidence Arbiter  ──> Multi-source confidence equation (0.30 C_p + 0.30 C_d + 0.20 C_a + 0.20 C_v)
9. Motion Command      ──> Software recommended [v, w] command with FSM safety
10. Dashboard & WHY    ──> Real-time operator display and mathematical causality audit
```

---

## 3. Detailed Scenario Demonstration Scripts

### Scenario 1: Open Path Nominal Cruising
- **Operator Action:** Select `scenario_1_open_path` from the top-right dropdown, press **Play**.
- **Panel Focus:**
  - *Monitor 1 (RGB/Semantics):* Path classified in green; surrounding grass classified in cyan.
  - *Monitor 2 (Depth):* Smooth, contiguous ground gradient extending out to 5.0m.
  - *Monitor 3 (Costmap & DWA):* Clear green low-cost corridor along path center.
  - *Motion Dials:* Speed indicates **$0.80\text{ m/s}$**, Steering needle centered ($0.0^\circ$).
  - *Confidence:* Meters read $0.85$ (`HIGH`).
- **Narrative Script:**  
  *"Judges, this demonstrates our nominal autonomous baseline on real outdoor trail data. The fusion engine merges high CNN confidence with clean metric depth. Because confidence exceeds our 0.75 threshold, the safety arbiter recommends our top cruising speed of 0.80 m/s with centered steering along the global A* path."*

---

### Scenario 2: Dynamic Obstacle Pop-Up (Zero-Frame Reaction)
- **Operator Action:** Switch to `scenario_2_sudden_obstacle`, scrub to frame 6.
- **Panel Focus:**
  - *Perception & Depth:* Physical obstacle suddenly occupies the center-right pathway ($Z = 2.1\text{ m}$).
  - *Costmap:* The corresponding cell instantly turns red (`BLOCKED`, Cost 255) with a 0.35m inflation safety zone.
  - *DWA Rollout:* Candidate trajectories roll out to the left flank, selecting an evasive arc.
  - *Clearance:* Minimum clearance reads $0.41\text{ m}$ (exceeding our $0.35\text{ m}$ safety envelope).
  - *Explainability Console:*  
    `[WHY PATH CHANGED]: Mode 2 Geometric Elevation Veto triggered by +0.22m physical obstacle at X=2.1m; replanned global path to left flank with 0.41m clearance.`
- **Narrative Script:**  
  *"At frame 6, a physical obstacle appears. Notice the zero-frame latency (< 100 ms). Mode 2 Geometric Veto unconditionally blocks the cell, and DWA immediately routes the vehicle around the hazard with 0.41m clearance. The explainability console provides full mathematical transparency into why the path altered."*

---

### Scenario 3: Complex Terrain & Soft Boundary Traversal
- **Operator Action:** Select `scenario_3_terrain_boundary`.
- **Panel Focus:**
  - *Perception:* Asphalt path curves into rough grass and gravel shoulders.
  - *Costmap:* Notice the 6 distinct cost color levels: Trail (0, dark green), Flat Grass (30, light green), Gravel (80, yellow).
  - *Trajectory:* Trajectory hugs the path curve even though grass is physically flat.
- **Narrative Script:**  
  *"Here we see why depth alone is inadequate. To a 3D LiDAR or depth sensor, the grass and asphalt look coplanar. But our semantic perception assigns a higher cost to rough grass, keeping the vehicle on the firm trail and preventing wheel slippage."*

---

### Scenario 4: Solar Glare & Missing Depth (Rule 12 Invariant)
- **Operator Action:** Select `scenario_4_depth_degradation`.
- **Panel Focus:**
  - *Depth Monitor:* Over 70% of the depth image is blanked out by intense solar blooming.
  - *Costmap:* Blank sectors are hatched with uncertainty penalties (cost 128).
  - *Speed Dial:* Speed automatically drops from $0.80\text{ m/s}$ to **$0.15\text{ m/s}$ (Conservative Crawl)**.
  - *Explainability Console:*  
    `[WHY SPEED REDUCED]: Depth validity fell to 28.4% due to optical blooming; Rule 12 uncertainty penalty applied, throttling speed to 0.15 m/s crawl.`
- **Narrative Script:**  
  *"Most robotics algorithms fail catastrophically in solar glare by assuming missing depth is empty air. Under our Rule 12 invariant, invalid depth is treated as an uncertainty hazard. Total confidence drops to LOW, and the safety gate automatically throttles the vehicle to a safe crawl."*

---

### Scenario 5: Visual Degradation & Tracking Loss (Rule 13 Fail-Safe)
- **Operator Action:** Select `scenario_5_visual_degradation`, advance to frame 10.
- **Panel Focus:**
  - *RGB Monitor:* Severe optical motion blur and low illumination.
  - *VO Status Meter:* FAST/ORB inliers drop to $11 < 15$. State turns red: `TRACKING_LOST`.
  - *Safety Arbiter:* State flips to **`CRITICAL`**.
  - *Motion Dials:* Linear velocity immediately drops to **$0.00\text{ m/s}$**.
  - *Explainability Console:*  
    `[WHY STOPPED]: Feature tracking lost (11 inliers < 15 threshold); asserted deterministic safe-stop position hold under Rule 13.`
- **Narrative Script:**  
  *"When visual features collapse due to motion blur or low light, TerrainSight does not guess. The system immediately executes Rule 13: a deterministic safe-stop recommendation, logging the root cause and holding position until valid features return."*

---

## 4. Closing Judge Demonstration Takeaways

1. **Empirically Proven:** 121 automated regression tests passing, 0 false-safe decisions across 375 multi-modal evaluations, 12 FPS on pure CPU.
2. **Transparent Explainability:** Every speed reduction, evasive turn, and stop is accompanied by a mathematical reason in the Explainability Console.
3. **Strict Engineering Discipline:** We demonstrate proven software capabilities on real outdoor datasets without making ungrounded claims of physical testing or military readiness.
