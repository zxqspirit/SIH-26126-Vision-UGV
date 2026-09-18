# Artifact: Ablation Protocol

**Role:** Research-Evaluation Engineer  
**Scope:** Quantitative and causal ablation comparison of 5 progressive UGV perception, localization, planning, and safety configurations on identical real outdoor recorded sequences.  
**Governing Standard:** Empirical truth only. Never fabricate values. All metrics computed frame-by-frame directly from recorded camera feeds.

---

## 1. Executive Summary & Objective

In complex off-road and outdoor unstructured navigation, modular autonomous stacks combine diverse perception, geometric depth, visual odometry, and confidence-gating components. A critical engineering question for deployment and certification is:
> *What specific failure mode does each module address, and what are the quantitative tradeoffs in safety, clearance, latency, and path efficiency across configurations?*

This protocol defines the exact test methodology, mathematical metrics, configuration flags, and causal failure attribution for five progressive configurations ($A \to E$):

```
Configuration Hierarchy:
[A: CNN Only] ──────────────────────────► Semantic classes only; planar assumption
     ▲
[B: Depth Only] ────────────────────────► 3D Pointcloud geometry only; no semantic semantics
     ▲
[C: CNN + Depth] ───────────────────────► Fused 2.5D elevation + semantics + dual vetoes
     ▲
[D: CNN + Depth + Localization] ────────► Fused 2.5D + Visual Odometry tracking & cost scaling
     ▲
[E: Full System + Confidence Safety] ───► Fused 2.5D + VO + Multi-sensor confidence FSM & speed governor
```

---

## 2. Progressive Ablation Configurations ($A - E$)

Every configuration is executed on the **identical sequence of recorded outdoor sensor frames**:

### Mode A: CNN Only
- **Perception:** Semantic segmentation (ResNet-FCN-50) active.
- **Depth / Geometry:** **DISABLED**. Constant nominal planar depth assumption ($Z = 3.0\text{ m}$ uniform plane); pointcloud slope, roughness, and geometric vetoes disabled.
- **Localization:** **DISABLED**. Stationary pose identity matrix ($x=0, y=0, \theta=0$); pose-dependent cost inflation disabled.
- **Safety Gate:** **DISABLED**. Fixed nominal forward throttle ($v = 0.50\text{ m/s}$); confidence-based speed scaling bypassed; emergency stop overrides disabled.
- **Primary Failure Modes Exposed:** Negative obstacles (potholes, dropoffs, ditches), thin geometry (posts, wires, low wires), and semantic color misclassifications without geometric checks.

### Mode B: Depth Only
- **Perception:** **DISABLED**. Semantic segmentation bypassed; all pixels treated as uniform generic terrain ($C_{\text{base}} = 50$). Semantic vetoes disabled.
- **Depth / Geometry:** **ACTIVE**. MiDaS metric depth + pointcloud projection, ground-plane RANSAC, elevation analysis, roughness calculation, and geometric vetoes active.
- **Localization:** **DISABLED**. Stationary pose identity matrix.
- **Safety Gate:** **DISABLED**. Fixed nominal forward throttle ($v = 0.50\text{ m/s}$).
- **Primary Failure Modes Exposed:** Viscous terrain (mud, sand, puddles) appearing geometrically flat but non-traversable, low-contrast terrain transitions, and depth absorption/dropout holes.

### Mode C: CNN + Depth (Perception Fusion)
- **Perception:** **ACTIVE**.
- **Depth / Geometry:** **ACTIVE**.
- **Fusion:** **ACTIVE**. Mode 1 continuous fused cost, Mode 2 geometric elevation veto, Mode 3 semantic obstacle veto.
- **Localization:** **DISABLED**. Stationary pose identity matrix; no path integration or tracking degradation awareness.
- **Safety Gate:** **DISABLED**. Fixed nominal forward throttle ($v = 0.50\text{ m/s}$).
- **Primary Failure Modes Exposed:** Visual odometry drift accumulation, inability to detect tracking loss during optical degradation (sun glare, optical blackout), leading to blind forward progression.

### Mode D: CNN + Depth + Localization
- **Perception:** **ACTIVE**.
- **Depth / Geometry:** **ACTIVE**.
- **Fusion:** **ACTIVE**.
- **Localization:** **ACTIVE**. Visual Odometry feature tracking (FAST/ORB), motion estimation, pose history logging, and pose-dependent cost scaling active.
- **Safety Gate:** **DISABLED**. Bypasses the 4-state Confidence Safety Gate; continues commanding nominal speeds ($v = 0.50\text{ m/s}$) even when VO reports severe tracking loss or degraded confidence.
- **Primary Failure Modes Exposed:** Full-speed traversal during severe sensor degradation (motion blur, glare, sensor disagreement) without throttle moderation or safe-stop.

### Mode E: Full System + Confidence Safety (Production Baseline)
- **Perception:** **ACTIVE**.
- **Depth / Geometry:** **ACTIVE**.
- **Fusion:** **ACTIVE**.
- **Localization:** **ACTIVE**.
- **Temporal Consistency:** **ACTIVE** (Exponential moving average + transient spike rejection).
- **Safety Gate:** **ACTIVE**. Multi-sensor confidence evaluation ($C_{\text{total}} = 0.35 C_{\text{perc}} + 0.30 C_{\text{geom}} + 0.20 C_{\text{vo}} + 0.15 C_{\text{temp}}$), 4-state FSM (`HIGH`, `MEDIUM`, `LOW`, `CRITICAL`), throttle scaling ($1.0 \to 0.6 \to 0.2 \to 0.0$), and deterministic safe stop (Rule 12 & Rule 13).
- **Mitigation Guarantee:** Complete coverage of all 14 outdoor adversarial failure modes.

---

## 3. Predefined Metric Dimensions

Before executing a single frame, the evaluation metrics are formally specified:

### Dimension 1: Perception Quality
- **Mean Perception Confidence ($\overline{C_{\text{perc}}}$):** Average softmax confidence of semantic terrain predictions ($[0.0, 1.0]$).
- **Valid Depth Ratio ($R_{\text{depth}}$):** Fraction of image pixels possessing valid metric depth values ($Z \in [0.2, 15.0]\text{ m}$).
- **Semantic-Geometric Disagreement Rate ($R_{\text{disagree}}$):** Percentage of BEV cells where 3D elevation and 2D semantic classification conflict ($> 50$ cost delta).

### Dimension 2: Obstacle Handling
- **Positive Obstacles Detected ($N_{\text{obs}}$):** Count of identified physical obstacle regions in the field of view.
- **Obstacle BEV Footprint ($A_{\text{obs}}$):** Number of occupied obstacle cells in the $100 \times 100$ local costmap ($0.1\text{ m/cell}$).
- **Minimum Vehicle Clearance ($d_{\text{min}}$):** Minimum distance from assumed vehicle bounding box ($1.2\text{ m} \times 0.8\text{ m}$) to any occupied obstacle cell along the recommended rollout:
  $$d_{\text{min}} = \min_{t, i} \text{dist}(\text{Footprint}(x_t, y_t, \theta_t), \text{Obstacle}_i)$$
- **Footprint Violations ($N_{\text{viol}}$):** Count of frames where recommended vehicle trajectory breaches the safety margin ($d_{\text{min}} < 0.35\text{m}$).

### Dimension 3: False-Safe Cases (Critical Safety Hazard)
- **False-Safe Count ($N_{\text{false\_safe}}$):** Number of frames where a lethal obstacle, ditch, mud pit, or barrier was incorrectly classified as traversable ($C \le 127$) due to a disabled perception/depth module.
  - *Mode A False-Safe:* Geometric drop or barrier classified as clear because CNN color matched ground.
  - *Mode B False-Safe:* Planar mud puddle or water body classified as clear because surface was flat ($Z \approx \text{constant}$).

### Dimension 4: Path Quality
- **Plan Validity Rate ($R_{\text{valid}}$):** Percentage of frames producing a geometrically feasible, collision-free global A* path to the goal.
- **Path Smoothness ($\sigma_{\kappa}$):** Standard deviation of waypoint curvature along the recommended trajectory:
  $$\kappa_i = \frac{|\Delta \theta_i|}{\Delta s_i}, \quad \sigma_{\kappa} = \sqrt{\frac{1}{N}\sum (\kappa_i - \bar{\kappa})^2}$$
- **Evasive Maneuver Rate ($R_{\text{evasive}}$):** Fraction of frames requiring sharp steering adjustments ($|\omega| \ge 0.40\text{ rad/s}$) to clear obstacles.

### Dimension 5: Path Cost
- **Mean Global Path Cost ($\overline{J_{\text{global}}}$):** Normalized cumulative cost per meter along the planned A* path:
  $$J_{\text{global}} = \frac{1}{L} \sum_{k=1}^K C(x_k, y_k) \Delta s_k$$
- **Mean Trajectory Score ($\overline{S_{\text{dwa}}}$):** DWA trajectory optimization objective score balancing goal progress, clearance, and terrain cost.

### Dimension 6: System Latency
- **Per-Stage Latency ($t_{\text{stage}}$ in ms):** Compute time for Perception, Depth, Fusion, Localization, Planning, and Safety.
- **Mean Decision Latency ($\overline{t_{\text{decision}}}$ in ms):** Total time from raw observation arrival to software command emission.
- **P95 Latency ($t_{\text{p95}}$ in ms):** 95th-percentile observation-to-decision latency.
- **Effective Processing Rate (FPS):** Throughput in frames per second ($1000 / \overline{t_{\text{decision}}}$).

### Dimension 7: Localization Quality
- **Mean Inlier Feature Count ($\overline{N_{\text{inliers}}}$):** Average tracked FAST/ORB keypoints per frame.
- **Tracking Loss Rate ($R_{\text{lost}}$):** Percentage of frames triggering `TRACKING_LOST` ($N_{\text{inliers}} < 15$ or pose covariance explosion).
- **Trajectory Length ($L_{\text{odom}}$ in meters):** Total integrated metric path length traversed by VO estimate.

### Dimension 8: Recovery & Safety Assurance
- **Cautious Throttle Activations ($N_{\text{cautious}}$):** Frames where speed was proactively moderated ($v \in (0.0, 0.40]\text{ m/s}$) due to degraded confidence.
- **Deterministic Safe-Stops ($N_{\text{stop}}$):** Frames where speed was commanded to $0.00\text{ m/s}$ due to `CRITICAL` risk, tracking loss (Rule 13), or unresolvable sensor disagreement.

---

## 4. Test Matrix & Datasets

All five configurations are benchmarked across five authentic recorded outdoor sequences ($15$ frames each, $75$ frames per mode, $375$ total evaluations):
1. `scenario_1_open_path`: High-texture gravel trail with clear forward path.
2. `scenario_2_sudden_obstacle`: Central rock/barrier requiring lateral avoidance.
3. `scenario_3_terrain_boundary`: Complex transitions across grass, mud, and gravel.
4. `scenario_4_depth_degradation`: Restricted clearance corridor between bushes and rocky margins.
5. `scenario_5_visual_degradation`: Optical blur, sun glare, and low-texture tracking loss.

---

## 5. Failure Mode Causality Mapping

| Failure Mode | Manifests in Mode(s) | Root Cause | Mitigating Module | Solved in Mode |
|:---|:---:|:---|:---|:---:|
| **Planar Mud / Water Trap** | B (Depth only) | Planar mud reflects laser/stereo depth as flat ground; geometric checks see zero elevation step. | Semantic CNN Segmentation & Semantic Veto | **C, D, E** |
| **Negative Obstacle (Ditch/Drop)** | A (CNN only) | Optical textures in ditch blend with background; 2D CNN predicts continuous grass. | 3D Pointcloud Elevation & Geometric Veto | **B, C, D, E** |
| **Untextured Monolithic Barrier** | A (CNN only) | Featureless concrete/rock barrier has no contrast; CNN classifies as uniform gravel. | Metric Depth Pointcloud Elevation | **B, C, D, E** |
| **Drift-Induced Map Corruption** | A, B, C | Open-loop costmaps overlay old observations onto wrong spatial coordinates. | Visual Odometry & Pose Integration | **D, E** |
| **Blind Progression in Glare** | A, B, C, D | Extreme overexposure washes out perception, but open-loop planner drives full speed. | Confidence Safety Gate (`CRITICAL` hold) | **E** |
| **Tracking Loss Crash** | A, B, C, D | VO loses lock in featureless sand; planner commands full throttle with stale pose. | Multi-Sensor Confidence FSM (Rule 13 Safe Stop) | **E** |
| **Sensor Disagreement Hazard** | A, B, C, D | Green spray-painted rock looks like grass to CNN but is a 1m rock to Depth. | Multi-Sensor Confidence Gate & Disagreement Penalty | **E** |

---

## 6. Output Deliverables
1. `experiments/ablation/ablation_results.csv`: Complete raw tabular records for all 375 frame executions.
2. `experiments/ablation/ablation_results.json`: Hierarchical aggregated metrics per mode and per scenario.
3. `experiments/ablation/latency_by_mode.png`: Stacked bar chart of per-stage compute latency across Modes A-E.
4. `experiments/ablation/clearance_and_safety.png`: Grouped comparative bar chart of minimum clearance and false-safe events.
5. `experiments/ablation/radar_module_contributions.png`: 8-axis radar chart showing trade-offs between speed, latency, clearance, and safety.
6. `experiments/ablation/ablation_report.md`: Comprehensive executive report detailing findings, Pareto trade-offs, and certification conclusions.
