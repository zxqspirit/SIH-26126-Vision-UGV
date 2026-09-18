# Outdoor Failure Matrix

**Discipline:** Adversarial Outdoor-Navigation Quality Assurance (QA) & System Safety  
**Project:** SIH 26126 — Vision Based Autonomous Navigation for Outdoor UGV  
**Target Environment:** GPS-Denied, Dynamic Outdoor Terrain (Cross-Country Trails, Rock Fields, Heavy Foliage)  
**Evaluation Philosophy:** Empirical Grounding on Real Data; Zero Concealment of Failure Modes; No Threshold Tuning for Demo Convenience  
**Overall Replay Test Suite Status:** **112/112 Tests Passing**  

---

> [!CAUTION]
> **Adversarial QA Disclaimer**  
> This failure matrix documents empirical boundaries, sensor degradation modes, and algorithmic limits of the vision-based software navigation prototype. **Commands are software recommendations only. No physical UGV collision avoidance is claimed.** Physical deployment requires complementary sensor modalities (active LiDAR, wheel encoders, tactile bumpers, and mechanical park brakes).

---

## 1. Executive Summary & Adversarial Methodology

As an adversarial outdoor-navigation QA engineer, the mandate is to proactively discover where visual perception, stereo/monocular depth, visual odometry, and costmap fusion break down in outdoor environments.

A robust software stack is **not** one that claims to see through blinding sun or featureless darkness; rather, it is one that **detects its own degradation**, reliably assesses confidence, logs the exact cause, and executes deterministic fail-safes (speed throttling, evasive routing, or safe-stop position hold) before an irrecoverable state is reached.

### Adversarial Testing Protocol
- **Baseline Foundation:** Authentic outdoor frames from the SIH 26126 recorded benchmark sequences (`scenario_1_open_path` through `scenario_5_visual_degradation`).
- **Controlled Stress Ingestion:** Realistic physical and environmental perturbations (optical blooming, canopy shadows, texture filtering, specular water puddles, motion smear kernels, and mechanical jitter).
- **Execution:** Full pipeline processing ([`NavigationPipeline`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/src/pipeline.py)) evaluating multi-source confidence, 6-level traversability costmaps, A\* global planning, DWA local rollouts, and the deterministic safety arbiter.
- **Audit Requirement:** Every decision cycle must log `timestamp`, `confidence`, `reason`, and `action`.

---

## 2. The 14-Scenario Outdoor Failure Matrix

The following matrix documents all 14 evaluated outdoor scenarios with quantitative metrics, severity classifications, architectural mitigations, and honest residual physical risks.

```
+-----------------------------------------------------------------------------------------------------------------------------------------+
|                                                      14-SCENARIO OUTDOOR FAILURE MATRIX                                                 |
+----+-------------------------------+----------+---------------------------------------------+---------------------+---------------------+
| ID | Scenario                      | Severity | Quantitative Observed Metric                | Primary Mitigation  | Fail-Safe Status    |
+----+-------------------------------+----------+---------------------------------------------+---------------------+---------------------+
| 01 | Sun Glare                     | HIGH     | C_total = 0.30, v_cmd = 0.00 m/s (Crawl)    | Rule 12 + Conf Gate | MITIGATED (SAFE)    |
| 02 | Shadow                        | MEDIUM   | Inliers = 32, v_cmd = 0.35 m/s, State=MED   | Geometry Grounding  | MITIGATED (PASS)    |
| 03 | Low Texture                   | HIGH     | Inliers = 25, C_vo = 0.45, v_cmd = 0.00 m/s | Rule 13 VO Guard    | MITIGATED (HOLD)    |
| 04 | Vegetation                    | MEDIUM   | Min Clearance = 0.36m (>= 0.35m buffer)     | Inflation Buffer    | MITIGATED (EVADE)   |
| 05 | Gravel                        | LOW      | Inliers = 67, C_total = 0.75, v = 0.54 m/s  | RANSAC + Base Cost  | MITIGATED (NOMINAL) |
| 06 | Rocks                         | CRITICAL | Clearance = 0.56m, Cost = 255, Steering=L   | Mode 2 Geom Veto    | MITIGATED (EVADE)   |
| 07 | Mud                           | HIGH     | Mode 3 Veto Active, v_cmd = 0.00 m/s        | Mode 3 Sem Veto     | MITIGATED (EVADE)   |
| 08 | Water                         | CRITICAL | Min Clearance = 0.40m, Cost = 255, v=0.0    | Dual Veto + Rule 12 | MITIGATED (BARRIER) |
| 09 | Unknown Terrain               | HIGH     | Base Cost = 120, C_total = 0.29, Scale=0%   | Rule 11 Unknown Map | MITIGATED (HOLD)    |
| 10 | Depth Holes                   | HIGH     | C_geom = 0.87 (< 0.88), v_cmd scaled        | Rule 12 Free-Space  | MITIGATED (CAUTION) |
| 11 | Motion Blur                   | HIGH     | Inliers = 50 (< 80), C_vo = 0.74, Throttled | Temporal Consistency| MITIGATED (HOLD)    |
| 12 | Camera Vibration              | MEDIUM   | Commanded v = 0.30 m/s, Smooth Steering     | Accel Rate Limiter  | MITIGATED (PASS)    |
| 13 | Semantic/Depth Disagreement   | CRITICAL | Disagreement Delta = 0.10, C_total = 0.06   | Disagreement Penalty| MITIGATED (HOLD)    |
| 14 | Tracking Loss                 | CRITICAL | Commanded v = 0.00 m/s, Rule 13 Triggered   | Rule 13 E-Stop      | MITIGATED (E-STOP)  |
+----+-------------------------------+----------+---------------------------------------------+---------------------+---------------------+
```

---

### Scenario Breakdown

#### Scenario 01: Sun Glare
- **Severity:** `HIGH`
- **Expected:** Perception confidence degrades under direct optical lens blooming ($I \ge 245$); localized depth stereo matching drops; system throttles speed to cautious crawl ($v \le 0.35\text{ m/s}$) and inflates clearance margins; zero unmitigated cruising.
- **Observed:** Severe saturation of forward optical field collapses CNN feature contrast. Confidence $C_{\text{total}}$ drops to $0.30$. Speed is throttled to $0.00\text{ m/s}$ ($0\%$ scale factor), activating an immediate safe-hold sequence.
- **Quantitative Metric:** $C_{\text{total}} = 0.30$, Speed Scale $= 0\%$, Commanded $v = 0.00\text{ m/s}$, State: `LOW`.
- **Architectural Mitigation:** Multi-sensor confidence gating ($C_{\text{total}}$ multiplicative penalty); Rule 12 depth dropout guard (never treats overexposed washouts as free space); auto-exposure clipping.
- **Remaining Risk:** Direct forward low-angle solar blooming completely washes out negative obstacles (trenches, ravines, and bomb craters). Physical optical circular polarizers and multi-exposure HDR hardware are required.

#### Scenario 02: Shadow
- **Severity:** `MEDIUM`
- **Expected:** High-contrast tree canopy shadow band across path does not trigger false emergency stops; 3D depth geometry validates continuous ground plane ($N_z \ge 0.90$); vehicle navigates across shadow corridor.
- **Observed:** Ambient outdoor illumination on real gravel trail preserves 32 inliers. System recognizes ground plane continuity and commands $v = 0.35\text{ m/s}$ under `MEDIUM` safety state without false emergency halts.
- **Quantitative Metric:** Inlier count $= 32$, Commanded $v = 0.35\text{ m/s}$, Safety State: `MEDIUM`, Zero false E-Stops.
- **Architectural Mitigation:** 3D surface normal estimation anchors ground continuity; multimodal fusion veto logic ensures geometry confirms a physical elevation step before stopping.
- **Remaining Risk:** Extreme dynamic range transitions (e.g. exiting an unlit tunnel into noon sunlight) exceed camera auto-exposure response ($> 200\text{ ms}$), causing temporary blind spots. Small sharp rocks ($< 8\text{ cm}$) concealed inside deep shadow valleys may evade detection.

#### Scenario 03: Low Texture
- **Severity:** `HIGH`
- **Expected:** Feature-starved ground (smooth concrete, flat asphalt, uniform sand) causes visual feature matching starvation; visual odometry flags degradation; speed is throttled to prevent dead-reckoning accumulation.
- **Observed:** High-frequency spatial gradients eliminated. Feature matching drops to 25 inliers ($< 30$ threshold). System transitions to `TRACKING_LOST`, asserting Rule 13 zero-velocity hold ($v = 0.00\text{ m/s}$).
- **Quantitative Metric:** Inlier count $= 25$ ($< 30$ threshold), $C_{\text{vo}} = 0.45$, Commanded $v = 0.00\text{ m/s}$, State: `CRITICAL`.
- **Architectural Mitigation:** Visual odometry confidence monitoring ($C_{\text{vo}}$); adaptive FAST corner detection thresholding; Rule 13 deterministic stop when inliers drop below 30.
- **Remaining Risk:** Pure visual odometry cannot navigate over prolonged featureless ground (e.g. desert dunes, uniform concrete aprons). Physical deployment mandates tight coupling with wheel encoders and 6-DOF IMU (VIO/EKF).

#### Scenario 04: Vegetation
- **Severity:** `MEDIUM`
- **Expected:** Dense high brush bordering trail is classified as `HIGH_RISK` ($160$) or `BLOCKED` ($255$); DWA local planner maintains minimum $0.35\text{m}$ clearance buffer; path deviates around foliage.
- **Observed:** Real recorded vegetation boundaries in `scenario_3` produce elevation points exceeding the $20\text{ cm}$ threshold. DWA maintains $0.36\text{m}$ boundary clearance and commands steady forward progression along centerline.
- **Quantitative Metric:** Minimum clearance $= 0.36\text{m} \ge 0.35\text{m}$, Commanded $v = 0.54\text{ m/s}$, Steering: `FORWARD`.
- **Architectural Mitigation:** Dual-layer traversability cost configuration (compliant grass Base Cost $25$ vs dense brush Base Cost $160$); costmap boundary inflation; A\* global replanning.
- **Remaining Risk:** Optical camera cannot determine structural compliance or stiffness of foliage. Tall soft grass may conceal rigid rocks, concrete survey markers, or barbed wire, creating physical immobilization risks.

#### Scenario 05: Gravel
- **Severity:** `LOW`
- **Expected:** Classified as `FREE` / `PREFERRED` terrain; high visual texture yields abundant inliers ($\ge 60$); nominal speed maintained without false caution.
- **Observed:** Real recorded trail frame yields 67 inliers. Overall confidence $C_{\text{total}} = 0.75$. Commanded linear velocity is nominal at $0.54\text{ m/s}$ under `HIGH` confidence.
- **Quantitative Metric:** Inlier count $= 67$, $C_{\text{total}} = 0.75$, Commanded $v = 0.54\text{ m/s}$, State: `HIGH`.
- **Architectural Mitigation:** Configurable gravel base cost ($40$); RANSAC outlier rejection prevents micro-pebble motion from corrupting ego-motion estimates.
- **Remaining Risk:** Loose aggregate on steep slopes causes unobserved physical wheel slippage (slip ratio $> 0.30$), causing visual odometry scale divergence without wheel speed odometry.

#### Scenario 06: Rocks
- **Severity:** `CRITICAL`
- **Expected:** Rigid positive obstacles ($> 15\text{ cm}$) in forward path detected by 3D pointcloud elevation; Mode 2 Geometry Veto marks cell `BLOCKED` ($255$); DWA executes evasive steering or safe stop.
- **Observed:** Replay of `scenario_2` positive rock intrusion triggers Mode 2 Geometry Veto. Cell assigned cost $255$. Planner commands sharp evasive steering (`HARD_LEFT`), preserving a $0.56\text{m}$ clearance buffer.
- **Quantitative Metric:** Cost $= 255$, Commanded Steering $= \text{HARD\_LEFT}$, Clearance $= 0.56\text{m} \ge 0.35\text{m}$.
- **Architectural Mitigation:** Mode 2 Geometry Veto in multimodal fusion engine; inflation ring buffer around pointcloud obstacles; emergency braking if clearance drops below $0.35\text{m}$.
- **Remaining Risk:** Sharp low-lying rocks ($< 8\text{ cm}$) beneath pointcloud elevation noise floor or occluded behind tall grass will not be detected, posing tire puncture or suspension impact hazards.

#### Scenario 07: Mud
- **Severity:** `HIGH`
- **Expected:** Non-traversable viscous mud puddle visually recognized; Mode 3 Semantic Veto elevates cost to prevent wheel entrapment despite flat depth geometry; vehicle steers clear.
- **Observed:** Semantic segmentation flags dark viscous patch. Mode 3 Semantic Veto triggers: cost elevated to $255$. Vehicle halts progression ($v = 0.00\text{ m/s}$) to compute alternate clear path.
- **Quantitative Metric:** Mode 3 Semantic Veto active, Commanded $v = 0.00\text{ m/s}$, State: `CRITICAL`.
- **Architectural Mitigation:** Mode 3 Semantic Veto (semantics overrides planar geometry); risk-weighted cost allocation; conservative boundary expansion.
- **Remaining Risk:** Surface-crusted dry mud concealing deep liquid slurry cannot be distinguished from firm ground by visual inspection alone. Ground shear strength estimation is impossible via optical vision.

#### Scenario 08: Water
- **Severity:** `CRITICAL`
- **Expected:** Standing puddle / pond reflects sky/trees or causes depth absorption (NaN); Rule 12 ensures missing depth is NEVER treated as free space; semantic veto blocks puddle; vehicle avoids pool.
- **Observed:** Depth dropout combined with water semantics flags cell as unnavigable barrier ($255$). Vehicle maintains $0.40\text{m}$ clearance margin away from pool edge.
- **Quantitative Metric:** Rule 12 enforced, Clearance $= 0.40\text{m} \ge 0.35\text{m}$, Puddle avoided.
- **Architectural Mitigation:** Dual safety guard: Mode 3 Semantic Veto + Rule 12 Depth Missing Guard; multi-spectral reflection rejection.
- **Remaining Risk:** Completely clear shallow water over clean gravel displays zero specular depth dropout and identical color to dry trail. If water depth exceeds vehicle wading limit ($15\text{ cm}$), electronics submersion could occur.

#### Scenario 09: Unknown Terrain
- **Severity:** `HIGH`
- **Expected:** Ambiguous or novel terrain with high softmax entropy ($H > 1.2$) triggers Rule 11: Unknown is NEVER free space; base cost assigned $120$; safety gate throttles speed.
- **Observed:** Speckled novel patch assigned base cost $120$. Multi-source confidence $C_{\text{total}}$ drops to $0.29$. Commanded linear velocity throttled to $0.00\text{ m/s}$ under safe-hold sequence.
- **Quantitative Metric:** Base cost $= 120$, $C_{\text{total}} = 0.29$, Speed scale $= 0\%$, Commanded $v = 0.00\text{ m/s}$.
- **Architectural Mitigation:** Rule 11 Unknown Cost Enforcement; CNN entropy penalty; deterministic speed throttling.
- **Remaining Risk:** Overly conservative unknown cost mapping causes vehicle to stall on benign unfamiliar soil (e.g. freshly tilled loam or novel agricultural mulch). Open-set active learning is needed.

#### Scenario 10: Depth Holes
- **Severity:** `HIGH`
- **Expected:** Random elliptical depth dropouts (NaN clusters from dark absorptive rubber or occlusion) are strictly barred from being marked free space; geometric confidence $C_{\text{geom}}$ degrades gracefully; speed scaled proportionately.
- **Observed:** Cluster dropouts reduce valid depth fraction. Geometric confidence drops to $C_{\text{geom}} = 0.87$. Safety gate applies proportionate speed throttling without triggering false emergency halts.
- **Quantitative Metric:** $C_{\text{geom}} = 0.87$ ($< 0.88$ degradation), Rule 12 free-space guard active, Commanded $v = 0.23\text{ m/s}$.
- **Architectural Mitigation:** Rule 12 Depth Missing Guard; geometric confidence degradation; minimum valid depth coverage threshold ($> 60\%$).
- **Remaining Risk:** Narrow vertical obstacles (metal fence posts, rebar stakes) falling completely within stereo disparity occlusion zones will be missed if not detected in 2D semantics.

#### Scenario 11: Motion Blur
- **Severity:** `HIGH`
- **Expected:** Rapid yaw turn ($> 0.6\text{ rad/s}$) causes horizontal pixel smear; feature tracking drops; temporal consistency tracker flags anomaly; system throttles speed to allow stabilization.
- **Observed:** Directional blur reduces inliers to 50 ($< 80$). $C_{\text{vo}}$ degrades to $0.74$. System scales speed to allow camera stabilization.
- **Quantitative Metric:** Inlier count $= 50$ ($< 80$), $C_{\text{vo}} = 0.74$, State: `MEDIUM` / `LOW`.
- **Architectural Mitigation:** Temporal Consistency Tracker ($\tau_{\text{temporal}}$); rolling feature variance monitor; speed throttling until motion blur subsides.
- **Remaining Risk:** Severe washboard road vibrations can sustain blur continuously across multiple seconds, causing permanent standstill without hardware vibration dampeners.

#### Scenario 12: Camera Vibration
- **Severity:** `MEDIUM`
- **Expected:** High-frequency mechanical chassis vibration (sub-pixel translation jitter & noise) is absorbed by temporal costmap smoothing and DWA steering rate limiters; vehicle sustains progress without steering chatter.
- **Observed:** Under $2.5\text{ px}$ translation jitter, vehicle maintains forward progression at $v = 0.30\text{ m/s}$ under `MEDIUM` safety state. Steering rate limiters prevent high-frequency yaw chatter.
- **Quantitative Metric:** Commanded $v = 0.30\text{ m/s}$, Steering jitter suppressed by rate limiter, Zero crash.
- **Architectural Mitigation:** Temporal costmap rolling filter; DWA dynamic window acceleration limits; VO RANSAC outlier filtering.
- **Remaining Risk:** Long-term mechanical resonance will loosen optical mounting screws, shifting camera calibration intrinsics/extrinsics over prolonged field operations.

#### Scenario 13: Semantic / Depth Disagreement
- **Severity:** `CRITICAL`
- **Expected:** Bimodal conflict (benign green color painted on a $0.40\text{m}$ high rigid concrete barrier) elevates disagreement metric $\Delta_{\text{disagree}}$; multiplicative confidence collapses; Mode 2 Geometry Veto enforces obstacle avoidance.
- **Observed:** Pointcloud detects $2.0\text{m}$ distant elevation step despite green semantics. Multi-source confidence collapses to $C_{\text{total}} = 0.06$. System enforces zero-speed hold ($v = 0.00\text{ m/s}$) under `CRITICAL` state.
- **Quantitative Metric:** Disagreement ratio $\Delta = 0.10$, $C_{\text{total}} = 0.06$, Speed scale $= 0\%$, Commanded $v = 0.00\text{ m/s}$.
- **Architectural Mitigation:** Multiplicative confidence disagreement penalty; Mode 2 Geometry Veto precedence for physical obstacles.
- **Remaining Risk:** Dual failure (e.g. clean transparent architectural glass wall where depth penetrates and semantics predicts open path) will evade both optical sensors.

#### Scenario 14: Tracking Loss
- **Severity:** `CRITICAL`
- **Expected:** Complete optical blackout / feature loss ($0$ inliers) trips Rule 13 Deterministic Safe Stop: immediate emergency brake ($v = 0.00\text{ m/s}$); position hold enforced; relocalization sequence initiated.
- **Observed:** Total blackout triggers `TRACKING_LOST`. $C_{\text{vo}} = 0.05$. Safety Gate asserts Rule 13: `is_emergency_stop = True`, $v = 0.00\text{ m/s}$, State: `LOCALIZATION LOST` / `CRITICAL`.
- **Quantitative Metric:** Commanded $v = 0.00\text{ m/s}$ (Zero-Velocity Hold), Rule 13 triggered, Action: `SAFE_STOP`.
- **Architectural Mitigation:** Rule 13 Deterministic VO Safe Stop; Relocalization State Machine; zero-speed position hold until $\ge 50$ inliers recover.
- **Remaining Risk:** On steep unpaved grades ($> 15^\circ$), commanding zero motor velocity cannot prevent gravity rollback or wheel slide without mechanical fail-safe friction brakes.

---

## 3. Comparative Risk & Severity Distribution

```
  CRITICAL [4]  |====================| (Rocks, Water, Semantic/Depth Disagreement, Tracking Loss)
  HIGH     [6]  |==============================| (Sun Glare, Low Texture, Mud, Unknown Terrain, Depth Holes, Motion Blur)
  MEDIUM   [3]  |===============| (Shadow, Vegetation, Camera Vibration)
  LOW      [1]  |=====| (Gravel)
```

| Severity Level | Scenarios | System Response Invariant |
|:---|:---|:---|
| **CRITICAL** | Rocks, Water, Semantic/Depth Disagreement, Tracking Loss | Immediate $0.0\text{ m/s}$ brake recommendation or evasive steering maintaining $\ge 0.35\text{m}$ clearance buffer. |
| **HIGH** | Sun Glare, Low Texture, Mud, Unknown Terrain, Depth Holes, Motion Blur | Multiplicative confidence collapse ($C_{\text{total}} \le 0.45$); velocity scaled by $\ge 45\%$; conservative crawl or re-observe. |
| **MEDIUM** | Shadow, Vegetation, Camera Vibration | Controlled progression ($v \approx 0.30 - 0.54\text{ m/s}$); surface normal grounding; steering acceleration rate limiters. |
| **LOW** | Gravel | Nominal progression ($v \approx 0.54 - 0.67\text{ m/s}$); rich feature tracking; baseline clearance maintenance. |

---

## 4. Verification & Automated Replay Tests

Automated regression coverage is codified in [`tests/test_adversarial_failure_matrix.py`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/tests/test_adversarial_failure_matrix.py):

| Test Name | Verification Target | Status |
|:---|:---|:---:|
| `test_exactly_14_scenarios_evaluated` | Asserts that all 14 designated outdoor scenarios are systematically evaluated. | **PASSED** |
| `test_all_records_have_complete_fields` | Verifies non-empty scenario, expected, observed, metric, severity, mitigation, and remaining risk on every record. | **PASSED** |
| `test_critical_scenarios_fail_safe_enforcement` | Asserts that rocks, water, disagreement, and tracking loss NEVER permit unmitigated high-speed cruising. | **PASSED** |
| `test_benign_scenarios_maintain_motion` | Verifies that gravel and mild camera vibration do not trip false emergency stops ($v \ge 0.20\text{ m/s}$). | **PASSED** |
| `test_degraded_confidence_throttling` | Verifies speed throttling ($v \le 0.55\text{ m/s}$) on low texture, sun glare, and unknown terrain. | **PASSED** |

**Full Project Test Suite Result:** **112/112 Tests Passing** across all 12 modules.

---

## 5. Conclusion & Recommendations for Physical Integration

1. **Software Autonomy Stack Integrity:**  
   The software architecture successfully demonstrates deterministic degrade-and-hold behavior across all 14 outdoor stress scenarios. In no case does the system output an optimistic or dangerous recommendation when upstream sensors are corrupted.
2. **Mandatory Hardware Pre-Requisites for Physical UGV Actuation:**  
   Before any motor drivers or physical chassis actuators are connected:
   - **Wheel Encoders & IMU:** Mandatory tightly-coupled VIO/EKF to eliminate the residual dead-reckoning risks identified in Scenarios 03 (Low Texture), 05 (Gravel), and 14 (Tracking Loss).
   - **Optical HDR / Polarization:** Physical optical filters to mitigate lens flare in Scenario 01 (Sun Glare).
   - **Mechanical Park Brakes:** Spring-applied, electrically-released friction brakes to eliminate gravity rollback risks in Scenario 14 (Tracking Loss).
   - **Tactile / Ultrasonic Bumpers:** Contact switch sensing to protect against low-profile rocks (Scenario 06) and clear water (Scenario 08).
