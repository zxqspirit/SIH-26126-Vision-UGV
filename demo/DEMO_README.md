# TerrainSight UGV — Live Demonstration Package

**SIH Problem Statement:** 26126 — Vision Based Autonomous Navigation for Outdoor UGV in GPS-Denied Environments  
**Organization / Problem Beneficiary:** Bharat Electronics Limited (BEL)  
**System Name:** TerrainSight UGV  
**Release Version:** v1.0-RC1  
**Package Purpose:** Self-contained, portable live evaluation package for technical juries, judges, and robotics engineers.  

---

> [!IMPORTANT]
> ### Rigorous Engineering Boundaries (Please Read First)
> 1. **Data Source:** Strictly **REAL OUTDOOR SENSOR DATA** recorded across real dirt pathways, sudden obstacles, and terrain transitions. No Gazebo. No synthetic data.
> 2. **Autonomy Scope:** Demonstrating a software-only autonomy stack. All linear velocities and steering angles are **software recommendations** for an onboard motor controller. No physical motor actuation is claimed.
> 3. **Explainability First:** Every vehicle action (speed reduction, path deviation, or safe stop) is explained mathematically by the Explainability "WHY" Engine on the dashboard.

---

## 1. Directory Structure

```
demo/
├── demo_config/                # Centralized demonstration configuration parameters
│   ├── pipeline_demo.yaml      # Master end-to-end pipeline settings
│   ├── camera_params.yaml      # Forward-facing camera intrinsics and mount extrinsics
│   ├── perception_params.yaml  # CNN input resolution and confidence thresholds
│   ├── planning_params.yaml    # Costmap resolution, vehicle footprint, DWA limits
│   └── safety_params.yaml      # Multi-source confidence thresholds & speed gates
├── sample_data/                # Real outdoor synchronized multi-modal scenarios
│   ├── scenario_1_open_path/   # 15 frames: Clear trail path (Nominal cruising baseline)
│   │   ├── rgb/                # 640x480 RGB JPG images
│   │   ├── depth/              # 640x480 Metric float32 NPY depth arrays (meters)
│   │   └── metadata.json       # Scenario ground truth & timestamp metadata
│   └── scenario_2_sudden_obstacle/ # 15 frames: Dynamic barrier pop-up (Zero-frame evasive reaction)
│       ├── rgb/
│       ├── depth/
│       └── metadata.json
├── model/                      # Lightweight segmentation network & class mappings
│   ├── mobilenetv3_lraspp_best.onnx # Optimized ONNX semantic segmentation model (<11ms)
│   ├── training_metadata.json  # Training convergence metrics & hyperparameter log
│   └── class_labels.json       # 6 terrain class IDs, palette colors, and cost weights
├── run_demo.sh                 # POSIX Bash one-click demo launcher (Linux / WSL / macOS)
├── run_demo.bat                # Windows batch one-click demo launcher
└── DEMO_README.md              # This technical evaluation guide
```

---

## 2. Quick Start: Launching the Demonstration

### Option A: One-Click Shell Script (Linux / WSL / macOS)
Open a terminal in the repository root and execute:
```bash
./demo/run_demo.sh
```
*Options:*
- `./demo/run_demo.sh --port 5000` — Specify custom dashboard port.
- `./demo/run_demo.sh --cli --scenario scenario_2_sudden_obstacle` — Run headless terminal replay.

### Option B: Windows Native (Command Prompt / PowerShell)
Double-click `demo/run_demo.bat` or run:
```cmd
demo\run_demo.bat
```

### Option C: Direct Python Command
```powershell
python src/visualization/dashboard_server.py
```

Once started, open your web browser (Chrome, Edge, or Firefox) to:
```
http://localhost:5000
```

---

## 3. What the Demonstration Shows

The web dashboard displays 4 synchronized monitor feeds, navigation instruments, and an Explainability Console updating at $\approx 12\text{ FPS}$:

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│  Monitor 1: RGB & Semantic Overlay   │   Monitor 2: Metric Depth Colormap   │
│  - Real outdoor camera view          │  - Calibrated range from 1.0m to 5.0m│
│  - 6-class segmentation overlay      │  - Ground plane RANSAC elevation fit │
├──────────────────────────────────────┼──────────────────────────────────────┤
│  Monitor 3: BEV Traversability Map   │   Monitor 4: Visual Odometry Path    │
│  - 100x100 2.5D Bird's-Eye-View Grid │  - FAST/ORB feature tracking points  │
│  - A* Global Path & 15 DWA rollouts  │  - 6-DoF ego-motion dead-reckoning   │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### Instruments & Telemetry:
- **Speedometer:** Displays recommended forward velocity ($0.0\text{ m/s}$ to $0.80\text{ m/s}$).
- **Steering Needle:** Indicates recommended angular yaw rate and direction (`STRAIGHT`, `LEFT`, `RIGHT`, `STOP`).
- **Confidence Meters:** Real-time breakdown of $C_{\text{perc}}$, $C_{\text{depth}}$, $C_{\text{agree}}$, and $C_{\text{vo}}$.
- **Safety State Indicator:** `HIGH` (Green), `MEDIUM` (Yellow), `LOW` (Orange), `CRITICAL` (Red).

---

## 4. Guided Demonstration Walkthrough for Evaluators

### Step 1: Nominal Cruising Baseline (`scenario_1_open_path`)
1. In the top-right scenario dropdown, select `scenario_1_open_path`.
2. Click **Play** (or use the step button to step frame-by-frame).
3. **Observations:**
   - CNN segments the dirt trail as `PREFERRED_PATH` (green).
   - Depth geometry confirms flat ground ($dz < 0.15\text{m}$).
   - Total confidence exceeds $0.75$ (`HIGH`).
   - Planner recommends maximum speed ($0.80\text{ m/s}$) with zero steering deviation.

### Step 2: Dynamic Obstacle & Zero-Frame Evasive Replanning (`scenario_2_sudden_obstacle`)
1. Select `scenario_2_sudden_obstacle` and advance to **Frame 6**.
2. **Observations:**
   - A physical barrier abruptly occupies the right side of the trail at $Z = 2.1\text{ m}$.
   - Depth Geometry detects elevation $+0.22\text{ m} > 0.15\text{ m}$.
   - **Mode 2 Geometric Veto** instantly turns the cell `BLOCKED` (Cost 255) with a $0.35\text{ m}$ inflation buffer.
   - Reaction latency is $< 100\text{ ms}$ ($0$ frame delay).
   - DWA evaluates 15 candidate trajectories and immediately steers left around the hazard.
   - Observed clearance is $\mathbf{0.41\text{ m}}$ (exceeding our $0.35\text{ m}$ safety envelope).
   - **Explainability Console Output:**
     ```
     [WHY PATH CHANGED]: Mode 2 Geometric Elevation Veto triggered by +0.22m physical obstacle at X=2.1m; replanned global path to left flank with 0.41m clearance.
     ```

---

## 5. Summary of Verified Benchmark Metrics

- **End-to-End Cycle Latency:** $83.16\text{ ms}$ ($12.0\text{ FPS}$ on host CPU).
- **Obstacle Reaction Latency:** $< 100.0\text{ ms}$ ($0$ frame delay).
- **Minimum Obstacle Clearance:** $0.36\text{ m}$ (Safety threshold: $\ge 0.35\text{ m}$).
- **Footprint Violations:** $0$ across all test runs.
- **False-Safe Catastrophic Errors:** $0 / 75\text{ frames}$ in the full stack (compared to $15$ in CNN-only and $60$ in Depth-only).
- **Unit & Integration Regression Suite:** $121 / 121\text{ tests}$ passing.

---

## 6. Technical Release & Evaluation Documents

For full algorithmic, architectural, and adversarial evaluation details, refer to:
- [**RELEASE_REPORT.md**](../RELEASE_REPORT.md) — Master technical release report.
- [**SIH Presentation Deck**](../docs/presentation/sih_presentation_deck.md) — 10-slide technical jury presentation.
- [**Judge Demonstration Guide**](../docs/demo/judge_demonstration_guide.md) — Complete operator script.
- [**SIH Red-Team Review**](../docs/benchmarks/outdoor_failure_matrix.md) — Adversarial failure matrix and known limitations.
