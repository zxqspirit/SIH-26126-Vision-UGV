# SIH Judge & Industry Expert FAQ

**Project:** SIH 26126 — Vision-Based Autonomous Navigation for Outdoor UGV  
**Organization:** Bharat Electronics Limited (BEL)  
**System Name:** TerrainSight UGV  
**Document Release:** v1.0-RC1 (Judge Examination Reference)  
**Tone Standard:** Engineering-first, calm, grounded, transparent, zero marketing buzzwords.  

---

## 1. Core Architectural Questions

### Q1: Why focus strictly on vision rather than relying on GPS or RTK-GNSS?
**Answer:**  
In defense and tactical outdoor scenarios specified by Bharat Electronics Limited (BEL), GNSS signals are easily jammed, spoofed, or physically attenuated by dense forest canopies, urban canyons, and mountain valleys. TerrainSight UGV is explicitly designed for **GPS-denied environments**, using onboard passive vision (RGB + stereo/metric depth) to provide both local traversability perception and metric ego-motion estimation without radiating active RF signals or depending on external satellite constellations.

### Q2: Why not use modern Vision-Language Models (VLMs) or Generative AI for navigation?
**Answer:**  
Foundation VLMs (such as CLIP, GPT-4V, or large multi-modal transformers) suffer from three fatal flaws in real-time outdoor edge robotics:
1. **Excessive Latency:** Inference times range from $800\text{ ms}$ to $3000\text{ ms}$ per query, making dynamic obstacle avoidance at $1\text{ m/s}$ impossible.
2. **Non-Deterministic Hallucinations:** Generative models lack formal mathematical bounds and can output contradictory actions for near-identical scenes.
3. **Violates Rule 10 Safety Invariant:** *"Never let a probabilistic foundation model directly command vehicle motion."*  
TerrainSight UGV uses a lightweight, deterministic CNN ($10.85\text{ ms}$ inference) coupled with classical 3D projective geometry and deterministic finite-state arbiters ($0.16\text{ ms}$).

### Q3: Why not use 3D LiDAR as the primary sensor?
**Answer:**  
While active LiDAR provides precise range data, it has major tactical and operational trade-offs:
1. **Active Emission:** In contested environments, active laser pulses can be detected by opposing optical sensors. Vision is completely passive.
2. **Inability to Discern Surface Hazards:** LiDAR pointclouds reflect off the surface of standing water, mud slicks, or oil patches, treating them as flat, drivable ground. Our dual-stream fusion uses semantic perception to detect surface texture hazards that LiDAR misses.
3. **SWaP-C Constraints:** Multi-beam 3D LiDAR significantly increases payload weight, power consumption ($15\text{W}-30\text{W}$), and hardware cost compared to passive stereo/RGB-D camera units.

---

## 2. Safety, Failure Modes & Edge Cases

### Q4: How does your system detect negative obstacles like ditches, potholes, and drop-offs?
**Answer:**  
Negative obstacles are the most dangerous hazards in outdoor ground robotics. In our **Depth Geometry Engine** (`src/depth_geometry/`), 3D pointclouds are back-projected from metric depth and fitted to a local ground surface via RANSAC. Cells where measured depth drops more than $0.12\text{ m}$ below the expected ground plane ($dz < -0.12\text{ m}$) or where ground plane support abruptly vanishes are classified as negative hazards and marked **BLOCKED (cost 255)** in the traversability costmap.

### Q5: What happens when the camera is blinded by intense direct sunlight or lens glare?
**Answer:**  
Solar blooming causes optical saturation and depth dropout ($71.6\%$ missing depth in Scenario 4). Most robotics stacks commit a catastrophic error by treating unmeasured zero-depth pixels as open free space. In TerrainSight UGV:
1. **Rule 12 Invariant:** Missing depth is classified as an **uncertainty hazard** and assigned a base risk cost ($128$).
2. **Confidence Throttling:** The depth health score collapses ($C_{\text{depth}} \approx 0.28$), lowering total confidence to `LOW`.
3. **Action:** The system automatically throttles recommended forward velocity from $0.80\text{ m/s}$ to a conservative crawl ($0.15\text{ m/s}$), steering only through the remaining valid visual sector.

### Q6: How does the system handle moving or pop-up dynamic obstacles?
**Answer:**  
Obstacles entering the field of view are detected across both modalities within a single frame cycle:
1. Geometric elevation slicers flag positive points ($dz > 0.15\text{ m}$).
2. Mode 2 Geometric Veto writes a cost of $255$ directly into the local costmap.
3. Our DWA trajectory evaluator checks candidate rollouts against the updated costmap with an obstacle reaction latency of **$< 100\text{ ms}$ ($0$ frame delay)**.
4. If an evasive path with $\ge 0.35\text{ m}$ clearance exists, DWA swerves smoothly; if blocked, the safety gate issues an immediate safe-stop command.

### Q7: What occurs if visual feature tracking fails (e.g., in heavy dust, smoke, or pitch darkness)?
**Answer:**  
The Visual Odometry Engine monitors FAST/ORB inlier counts every cycle. If inliers drop below $15$ features:
1. The VO health monitor asserts `TRACKING_LOST`.
2. Total confidence drops to `CRITICAL`.
3. The Confidence Arbiter executes **Rule 13**: a deterministic **Safe-Stop recommendation ($v_{\text{rec}} = 0.00\text{ m/s}$)** with an active position hold. The robot does not guess or drive blindly.

---

## 3. Deployment, Integration & Physical Roadmap

### Q8: Why are your motion outputs framed as "Software Recommendations" rather than driving physical motors?
**Answer:**  
Per SIH 26126 guidelines and disciplined aerospace/robotics software engineering standards, autonomy stacks must be thoroughly validated on real sensor datasets before hardware deployment. By outputting strongly-typed software recommendations (`UGVMotionCommand`: linear velocity, angular velocity, steering mode, confidence, safety state), we verify algorithmic correctness, timing bounds, and fail-safe logic with zero risk of physical vehicle crashes or hardware damage.

### Q9: What exact steps are required to deploy this software onto a physical UGV chassis?
**Answer:**  
The architecture was engineered for modular zero-code-change deployment:
1. **Hardware Compute:** Flash our Docker container onto an NVIDIA Jetson Orin Nano / AGX Orin module.
2. **Sensor Interface:** Connect an Intel RealSense D435i / D455 via USB 3.2. Our `src/sensors/live_pipeline.py` implements the exact identical downstream interface as the dataset loader.
3. **Motor Driver Interface:** Bridge `UGVMotionCommand` to the vehicle's CAN Bus / Roboteq motor controller via a micro-ROS node subscribing to `/cmd_vel`.
4. **State Estimation:** Feed VO outputs into a robot_localization EKF node fused with wheel encoders and a 9-DoF IMU.

### Q10: How do you justify claiming zero false-safe failures in the full system?
**Answer:**  
In our 5-configuration ablation study across 375 multi-modal evaluations on real outdoor sequences:
- Mode A (CNN only) had **15 false-safe frames** (blind to negative obstacles and elevation).
- Mode B (Depth only) had **60 false-safe frames** (blind to water and mud hazards).
- Mode C & D had **15 and 10 false-safe frames** (lacked tracking fail-safes).
- **Mode E (Full System)** achieved **0 false-safe frames** because whenever sensor data degraded, Rule 11, Rule 12, and Rule 13 deterministic arbiters throttled speed or stopped the vehicle rather than executing a high-risk traversal.
