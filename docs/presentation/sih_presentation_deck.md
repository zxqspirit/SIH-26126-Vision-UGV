# SIH 26126 — Technical Presentation Deck
## TerrainSight UGV: Vision-Based Autonomous Navigation for Outdoor UGV in GPS-Denied Environments

**Organization:** Bharat Electronics Limited (BEL)  
**Category:** Software | **Theme:** Smart Automation  
**Team / System:** TerrainSight UGV  
**Document Classification:** Official SIH Technical Deck & Presentation Script  

---

### Slide 1: Title & Executive Summary
- **Headline:** TerrainSight UGV — GPS-Denied Vision Autonomy for Outdoor UGVs
- **Problem Statement (SIH 26126):** Develop vision-based autonomous navigation for an Unmanned Ground Vehicle (UGV) operating in outdoor environments where GNSS/GPS signals are denied, jammed, or unreliable.
- **Core Value Proposition:** A high-throughput (12 FPS on CPU), multi-modal software navigation stack that fuses lightweight CNN semantic segmentation with 3D depth geometry, visual odometry, and deterministic confidence-aware safety arbiters to produce safe motion recommendations.
- **Strict Boundary Disclaimer:** Software-only autonomy prototype evaluated on real outdoor datasets. Outputs are software motion recommendations; does not claim physical collision avoidance or military-grade field readiness.

---

### Slide 2: The Tactical Problem & Operating Environment
- **Operational Reality:**
  - Modern defense, patrol, and search-and-rescue operations face intense GNSS electronic jamming, dense foliage canopy attenuation, and urban canyon signal loss.
  - Outdoor cross-country terrain features treacherous hazards: steep ditches, hidden rocks, slippery mud slicks, specular water puddles, and dynamic moving obstacles.
- **Why Existing Approaches Fail:**
  - *GPS-Dependent Waypoint Navigation:* Completely immobilized when satellites are jammed.
  - *Pure Active LiDAR:* Emits detectable laser signatures, consumes high power (15W-30W), and is blind to flat surface hazards (mud, water puddles, oil slicks).
  - *Heavy Vision-Language Models (VLMs):* Severe latency (>1000ms), unpredictable hallucinations, non-deterministic safety.
- **Our Approach:** Passive, low-SWaP multi-modal vision: Semantic AI + Metric 3D Depth + Visual Odometry + Deterministic Safety FSM.

---

### Slide 3: 10-Stage Decoupled System Architecture
- **Stage 1: Real Outdoor Data Ingestion:** Synchronized 640x480 RGB + Metric Float Depth stream with nanosecond timestamps.
- **Stage 2: High-Speed Preprocessing:** Dynamic scaling, ROI selection, and intrinsic calibration.
- **Stage 3: Lightweight AI Perception:** Fast-SCNN / MobileNetV3 semantic segmentation with deterministic color/texture/ExG fallback (10.85 ms).
- **Stage 4: Metric Depth Geometry:** 3D Pointcloud back-projection, RANSAC ground surface estimation, elevation slicing (18.51 ms).
- **Stage 5: Multi-Modal Fusion Engine:** Mode 1 Continuous Log-Odds + Mode 2 Geometric Elevation Veto + Mode 3 Semantic Hazard Veto (12.37 ms).
- **Stage 6: Visual Odometry & Localization:** FAST corners + ORB descriptors + RANSAC ego-motion tracking (3.99 ms).
- **Stage 7: 2.5D Bird's-Eye-View Traversability Map:** 6 distinct cost levels with Gaussian footprint inflation.
- **Stage 8: Global & Local Path Planning:** Global A* pathfinder + Dynamic Window Approach (DWA) 15-arc candidate rollouts (16.32 ms).
- **Stage 9: Confidence Arbiter & Motion Generator:** Multi-source confidence gate with deterministic Rule 13 fail-safe FSM (0.16 ms).
- **Stage 10: Real-Time Mission Control Dashboard:** 4 synchronized display monitors + Explainability \'WHY\' Engine (9.02 ms).

---

### Slide 4: Multi-Modal Fusion: The Dual-Veto Paradigm
- **Continuous Mode 1:** Bayesian log-odds accumulation merging semantic traversability probability with depth obstacle probability.
- **Mode 2 (Geometric Elevation Veto):**
  - Problem: A camouflaged rock or low wall matches the color of the dirt path. The CNN misclassifies it as safe trail.
  - Action: Metric depth detects a +0.20m physical height barrier above the RANSAC ground plane.
  - Resolution: Depth geometry unconditionally overrides the CNN, marking the cell **BLOCKED (Cost 255)**.
- **Mode 3 (Semantic Hazard Veto):**
  - Problem: Standing water puddles or slick mud slicks appear perfectly flat (dz = 0.00m) to depth cameras and LiDAR.
  - Action: AI perception detects water specular reflection and mud texture with high probability (>0.85).
  - Resolution: Perception unconditionally overrides depth, marking the cell **HIGH RISK (Cost 200)** or **BLOCKED (Cost 255)**.

---

### Slide 5: Confidence-to-Behavior Safety Design (Rule 13 Invariant)
- **Multi-Source Confidence Equation:**
  \\mathbf{C_{\\text{total}} = 0.30 C_{\\text{perc}} + 0.30 C_{\\text{depth}} + 0.20 C_{\\text{agree}} + 0.20 C_{\\text{vo}}}
- **Deterministic 4-State Safety Arbiter:**
  - **HIGH ( \\ge 0.75$):** Nominal traversal up to .80\\text{ m/s}$.
  - **MEDIUM (.45 \\le C < 0.75$):** Throttled speed (.35\\text{ m/s}$), expanded safety clearance buffer.
  - **LOW (.25 \\le C < 0.45$):** Conservative crawl (.15\\text{ m/s}$), route around blind spots.
  - **CRITICAL ( < 0.25$ or Tracking Lost):** **Deterministic Safe Stop (.00\\text{ m/s}$)** with active position hold.
- **Causality Logging:** Every decision logs 	imestamp, confidence, 
eason, and ction.

---

### Slide 6: Five-Mode Ablation Study: Why Every Module Is Essential
- Evaluated across 375 multi-modal cycles on 5 real outdoor scenarios:
  - **Mode A (CNN Only):** 15 false-safe frames (20%). Blind to elevation obstacles and negative ditches.
  - **Mode B (Depth Only):** 60 false-safe frames (80%). Blind to water puddles, mud traps, and trail boundaries.
  - **Mode C (CNN + Depth):** 15 false-safe frames (20%). Catches physical barriers and water, but lacks ego-motion.
  - **Mode D (CNN + Depth + Loc):** 10 false-safe frames (13.3%). Accurate tracking, but unthrottled in tracking loss.
  - **Mode E (Full Stack):** **0 FALSE-SAFE FRAMES (0.0%)!** Eliminates all catastrophic navigation hazards.

---

### Slide 7: Verified Empirical Metrics & System Performance
- **End-to-End Decision Latency:** .16\\text{ ms}$ (.0\\text{ FPS}$ on host CPU, optimized from .19\\text{ ms}$).
- **Obstacle Reaction Latency:** $< 100.0\\text{ ms}$ ($ frame delay).
- **Minimum Observed Clearance:** .36\\text{ m}$ (comfortably exceeding .35\\text{ m}$ footprint requirement).
- **Footprint Collisions:** $ violations across all test runs.
- **Visual Odometry Inliers:** Mean .4$ tracked features.
- **Regression Suite:** 121/121 automated unit and integration tests passing in \\text{ s}$.

---

### Slide 8: Adversarial Outdoor Robustness (14 Evaluated Failure Modes)
- All 14 failure modes mitigated via deterministic architectural safeguards:
  - *Sun Glare & Depth Holes:* Rule 12 uncertainty mapping + speed throttled to .15\\text{ m/s}$.
  - *Shadows & High Contrast:* Geometric RANSAC ground surface invariance.
  - *Mud & Water Puddles:* Mode 3 Semantic Hazard Veto forces evasive routing.
  - *Rock & Camouflaged Barriers:* Mode 2 Geometric Elevation Veto marks cost 255.
  - *Motion Blur & Severe Vibration:* VO health monitor detects inlier collapse ($< 15$) and executes instant safe-stop.

---

### Slide 9: Live Mission Control Dashboard & Explainability Engine
- Operating at http://localhost:5000 for live judge interaction:
  - 4 Synchronized Streams: RGB/Semantics, Metric Depth Colormap, BEV Costmap + DWA, VO Odometry Path.
  - Real-time Gauges: Speedometer, Steering Needle, Multi-Source Confidence Breakdown.
  - **Explainability Console:** Live mathematical audit explaining:
    - *WHY PATH CHANGED* (e.g., dynamic obstacle detected at 2.1m, rerouted to right flank).
    - *WHY SPEED REDUCED* (e.g., depth validity dropped to 28.4%, scaled to 0.15 m/s crawl).
    - *WHY STOPPED* (e.g., VO tracking lost, asserted safe-stop position hold).

---

### Slide 10: Clear Operational Boundary & Physical Deployment Roadmap
- **PROVEN NOW (Software Prototype):**
  - Verified on authentic outdoor multi-modal datasets.
  - Complete 10-stage perception, geometry, fusion, planning, and safety stack.
  - 0 false-safe decisions, 12 FPS on CPU, 121 automated tests.
  - Software motion recommendations only.
- **FUTURE PHYSICAL UGV DEPLOYMENT (Roadmap):**
  - *Compute:* Container deployment on NVIDIA Jetson Orin Nano / AGX Orin (TensorRT FP16).
  - *Sensors:* Live Intel RealSense D435i/D455 integration via live_pipeline.py.
  - *Actuation:* Micro-ROS CAN bus bridge to Roboteq / Curtis motor controllers.
  - *Safety:* Hardware electromechanical disc brakes and physical radio e-stop backing up software safe-stop logic.
