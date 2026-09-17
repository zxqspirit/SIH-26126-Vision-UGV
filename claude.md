# SIH 26126 — Vision-Based Autonomous UGV
## Claude Code Project Foundation

> **Project status:** Internal SIH prototype / pre-document phase  
> **Problem Statement:** SIH 26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor environment  
> **Organization:** Bharat Electronics Limited (BEL)  
> **Category:** Software  
> **Theme:** Smart Automation  
> **Current selected direction:** Stereo / RGB-D Visual Odometry + Lightweight CNN Path/Terrain Segmentation + Confidence-Based Fallback

---

## 1. Project Mission

Build a credible, demonstrable and modular autonomous-navigation software stack for an outdoor UGV operating without GPS, using onboard vision as the primary sensing modality.

The prototype must demonstrate:

1. Safe/traversable-path perception.
2. Visual localization / odometry without GPS.
3. Dynamic obstacle avoidance.
4. A planner/controller that converts perception into vehicle motion commands.
5. Safe degradation when perception becomes uncertain.

The project is judged as an SIH engineering prototype. Therefore, **working integration, measurable performance, robustness, explainability and demo reliability are more important than adding fashionable algorithms.**

---

## 2. Current Winning Architecture

Treat the following as the current baseline architecture unless the team deliberately changes it after evidence-based testing.

### Primary sensor
- Stereo camera OR RGB-D camera.
- Candidate hardware: Intel RealSense family, OAK-D family, or equivalent.
- Camera is primary; optional secondary modality is only a fallback.

### Perception
- Lightweight CNN / segmentation network.
- Outputs:
  - traversable path / ground mask
  - basic terrain classes or terrain cost
  - optional uncertainty/confidence
- Prefer a small model over a heavy detector if the task can be solved with segmentation + geometry.

### Geometry
- Depth-based obstacle masking.
- Use height/discontinuity/depth consistency to detect positive obstacles and likely negative obstacles.
- Fuse geometry with learned segmentation.

### Localization
- Stereo/RGB-D visual odometry / SLAM.
- Candidate stack:
  - RTAB-Map
  - ORB-SLAM3 where camera support and integration are appropriate
  - another well-supported visual/depth SLAM stack if tests justify it
- Do not implement SLAM from scratch.

### Planning
- Local costmap generated from fused perception + depth.
- Classical, interpretable planner/controller first.
- Candidate algorithms:
  - DWA
  - TEB
  - MPPI
  - A* / D* Lite for graph/grid planning where appropriate
- Planner must respect vehicle kinematics and safety margins.

### Confidence-based fallback
Use system confidence to change behavior.

Example states:
- **HIGH confidence:** normal autonomous navigation.
- **MEDIUM confidence:** lower speed + larger clearance + conservative planner.
- **LOW confidence:** stop / slow crawl / re-observe / attempt recovery.
- **LOCALIZATION LOST:** stop safely, rotate/reobserve if safe, attempt relocalization.
- **SENSOR QUALITY FAILURE:** degrade to the safest available modality; never pretend certainty.

---

## 3. Non-Negotiable Engineering Principles

### Safety > Completion
A UGV that safely stops is better than one that reaches the goal by colliding.

### Geometry + semantics
Neither depth alone nor segmentation alone should be the sole authority in difficult scenes. Fuse them.

### Confidence must affect behavior
Confidence is not a dashboard decoration. It must modify speed, clearance, planner selection or trigger recovery.

### Build around measurable interfaces
Every module must have:
- inputs
- outputs
- timestamps
- coordinate frame assumptions
- confidence/health information
- logging

### Offline-first
The core navigation stack must operate without cloud connectivity.

### Modular, replaceable components
A model, SLAM backend or planner should be replaceable without rewriting the entire project.

### Demo reliability
Prefer deterministic and debuggable systems over end-to-end black-box control.

### Evidence before complexity
Do not add RL, thermal, radar, event cameras, multi-robot coordination, foundation models, etc. merely to make the architecture sound advanced. Add them only when an identified failure mode justifies the complexity.

---

## 4. Expected Repository Structure

Keep the repository organized approximately like this:

```text
.
├── claude.md
├── voice.md
├── business.md
├── todo.md
├── agent.md
├── README.md
├── docs/
│   ├── architecture/
│   ├── research/
│   ├── experiments/
│   └── demo/
├── config/
│   ├── camera/
│   ├── perception/
│   ├── slam/
│   ├── planning/
│   └── system/
├── src/
│   ├── perception/
│   ├── depth_geometry/
│   ├── localization/
│   ├── fusion/
│   ├── planning/
│   ├── control/
│   ├── safety/
│   ├── interfaces/
│   └── visualization/
├── models/
│   ├── checkpoints/
│   └── metadata/
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── annotations/
├── simulation/
├── tests/
├── scripts/
├── logs/
└── results/
```

Do not create directories just because they look architecturally impressive. Add them when they are actually needed.

---

## 5. Source-of-Truth Rules

Before changing an implementation:
1. Read relevant existing files.
2. Understand the current data flow.
3. Reuse working code where possible.
4. Make the smallest change that solves the problem.
5. Run the relevant test/verification.
6. Update documentation only when the implementation really changed.

Never assume a library API from memory when version differences may matter. Check the installed version and current documentation available to the project.

---

## 6. Coordinate Frames

The project should explicitly document these frames when ROS/robotics middleware is used:

- `camera_link`
- `camera_optical_frame`
- `base_link`
- `odom`
- `map`

Do not silently mix image coordinates, camera coordinates and robot coordinates.

Every transformation must state:
- source frame
- target frame
- timestamp
- units
- handedness

---

## 7. Core Runtime Data Flow

```text
Stereo / RGB-D Camera
        │
        ├── RGB ────────────────┐
        │                       │
        └── Depth ───────┐      │
                         ▼      ▼
                  Depth Quality  Lightweight CNN
                         │      │
                         └──┬───┘
                            ▼
                  Semantic + Geometric Fusion
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       Traversability / Cost         Obstacle Map
              │                           │
              └─────────────┬─────────────┘
                            ▼
                   Local Costmap / World Model
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
          Visual VO/SLAM              Goal / Planner
              │                           │
              └─────────────┬─────────────┘
                            ▼
                 Confidence / Safety Layer
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          Motion command          Slow / Stop /
                                  Recovery
```

---

## 8. Perception Requirements

The perception system should prefer a compact task formulation.

At minimum:
- traversable
- non-traversable
- uncertain

Preferred extension:
- grass
- soil
- gravel/rock
- mud
- water
- vegetation
- man-made obstacle
- unknown

Do not create dozens of terrain classes unless the data justifies them.

The perception module should output:
- mask
- class probabilities / cost
- confidence
- inference latency
- frame timestamp

---

## 9. Depth Requirements

Depth is valuable because it supplies metric geometry, but it is not automatically reliable outdoors.

Track:
- valid-depth percentage
- median depth noise in relevant range
- holes/missing depth
- overexposure/IR degradation where applicable
- temporal consistency

Never treat invalid depth as free space.

The geometry layer should be able to say:
- obstacle detected
- possible drop / negative obstacle
- uncertain geometry
- no reliable depth

---

## 10. Localization Requirements

A localization module must expose:
- pose
- velocity estimate if available
- tracking state
- covariance / quality metric if available
- relocalization events
- map/odom consistency

Test the system in:
- textured terrain
- low-texture terrain
- repeated visual patterns
- sudden illumination changes
- camera vibration
- partial occlusion

Do not claim “GPS-denied autonomy” just because GPS is unplugged in one easy indoor test. Demonstrate representative outdoor cases.

---

## 11. Planner / Control Requirements

The planner must operate on a representation created from perception.

Minimum requirements:
- collision checking
- safety margin
- goal direction
- vehicle footprint
- maximum speed
- stopping distance
- recovery behavior

The controller must be smooth enough to avoid oscillation.

Prefer a staged implementation:
1. straight safe path
2. simple local avoidance
3. dynamic obstacle handling
4. recovery
5. optimization

---

## 12. Confidence & Safety Layer

Confidence may depend on:
- segmentation confidence
- depth validity
- depth/semantic agreement
- visual tracking quality
- localization covariance
- temporal consistency
- scene-change indicators

Example conceptual score:

```text
C_total =
  w1 * C_segmentation +
  w2 * C_depth +
  w3 * C_localization +
  w4 * C_temporal_consistency
```

The actual weighting must be experimentally determined and documented.

The safety layer should map confidence to behavior:

```text
C >= high_threshold:
    normal speed

medium <= C < high:
    reduced speed
    increased clearance

low <= C < medium:
    slow / re-observe / conservative mode

C < low:
    stop safely
    recovery or operator intervention
```

Never invent “confidence” values without defining how they are computed.

---

## 13. Performance Targets

These are **proposed engineering targets**, not official SIH requirements.

Initial targets to validate:
- End-to-end perception + planning loop: preferably >= 10 Hz, target 15–20 Hz where hardware permits.
- Perception inference: preferably <100 ms per frame; target <60 ms.
- Local obstacle response: ideally <200 ms from perception observation to updated control decision.
- Navigation success in controlled demo scenarios: >=90% across repeated trials.
- Collision rate: 0 in the final curated demo scenario; measure broader test-set collision rate honestly.
- Recoverable localization-loss event: automatic recovery in a meaningful fraction of tested cases.
- System should log enough information to reproduce failures.

Never manipulate metrics to make the project look better.

---

## 14. Hardware Strategy

Start with the minimum viable hardware:
- UGV chassis / simulated robot
- stereo or RGB-D camera
- compute unit
- motor controller
- wheel encoders where available
- IMU where available

Do not add LiDAR, radar, thermal, drone or extra sensors until a documented failure mode requires them.

A laptop is acceptable for early algorithm development.
Embedded deployment is the stronger final demonstration.

---

## 15. Software Strategy

Preferred general stack:
- Ubuntu
- ROS 2 when integration value outweighs setup cost
- Python for experimentation/training
- C++ for latency-critical robotics nodes where necessary
- OpenCV
- PyTorch
- ONNX / TensorRT where supported
- RViz / Foxglove or another strong visualization tool
- Git

When selecting a package, verify:
- active maintenance
- license
- hardware support
- ROS version compatibility
- CPU/GPU footprint
- camera support
- reproducibility

---

## 16. Dataset Rules

Dataset decisions are first-class engineering decisions.

A training dataset should represent:
- daylight variation
- shadows
- glare
- dust if available
- vegetation
- soil
- gravel/rocks
- puddles/water
- narrow paths
- slopes
- obstacles
- humans/animals where relevant
- camera motion

When local Indian data is scarce:
- combine public datasets + team-collected field data
- label a focused custom validation set
- do not claim “Indian terrain robustness” based only on generic internet datasets

The validation set must be kept separate from training.

---

## 17. Experiment Discipline

Every experiment should record:
- hypothesis
- configuration
- dataset/split
- hardware
- model version
- software commit
- metric
- failure examples
- conclusion
- next action

Use repeatable experiment names, e.g.:

```text
EXP-001_baseline_segmentation
EXP-002_depth_obstacle_mask
EXP-003_fusion_v1
EXP-004_confidence_speed_control
EXP-005_vo_stability_outdoor
```

---

## 18. What Claude Must Not Do

Do not:
- rewrite the whole repository unnecessarily
- introduce huge dependencies without need
- replace stable algorithms just for novelty
- train enormous models when a small model will work
- fabricate benchmarks
- fabricate citations
- fabricate hardware capabilities
- claim real-world safety certification
- claim military readiness
- claim production robustness from simulation alone
- make undocumented changes to interfaces
- silently disable safety checks
- silently bypass errors

---

## 19. Decision Rule for New Ideas

Before implementing an idea, answer:

1. What concrete failure mode does it solve?
2. Why does the current stack fail?
3. What measurable metric should improve?
4. What is the extra compute/hardware/complexity cost?
5. Can we test it within the project timeline?
6. Does it improve SIH demo strength?
7. Can we explain it clearly to a professor/judge?

If the answer is weak, do not implement it.

---

## 20. SIH Demo Philosophy

The final demo should make the audience understand the system in seconds.

Prefer visible demonstrations:
- RGB image
- segmentation overlay
- depth view
- traversability/cost view
- obstacle mask
- estimated trajectory
- local path
- current confidence
- robot command
- safety state

The judge should be able to see:

**what the robot sees → what it thinks is safe → where it localizes → what path it chooses → why it slows/stops → that it recovers.**

---

## 21. Current Scope Boundaries

### Must have
- RGB-D or stereo input
- traversability segmentation
- depth obstacle reasoning
- visual odometry/localization
- local planning
- collision avoidance
- confidence-based safety behavior
- logs and evaluation

### Good to have
- terrain cost map
- relocalization
- dynamic obstacle tracking
- embedded deployment
- offline operator dashboard
- automatic recovery

### Only after core system works
- thermal/radar fallback
- learned planner
- RL
- multi-robot
- event camera
- advanced semantic SLAM

---

## 22. Definition of Done

A feature is not “done” when the code runs once.

It is done when:
- interface is documented
- failure behavior is defined
- test exists
- measured result is logged
- visualization/debugging exists where useful
- reproducible command exists
- docs are updated if behavior changed

---

## 23. Research & Citation Rule

When writing technical documents for the team:
- cite primary sources where possible
- distinguish published evidence from our own proposed design
- include year/version for volatile software/hardware claims
- never turn a hypothesis into a fact
- never copy vendor marketing language as engineering evidence

