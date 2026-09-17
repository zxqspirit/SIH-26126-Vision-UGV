# SIH 26126 — Project Tracker

> This file is the operational source of truth for unfinished work.
> Keep tasks small enough to verify.

---

## STATUS LEGEND

- `[ ]` not started
- `[-]` in progress
- `[x]` completed
- `[!]` blocked
- `[?]` decision required

---

# PHASE 0 — FOUNDATION

- [x] Confirm Problem Statement 26126 requirements
- [x] Select hybrid architecture: stereo/RGB-D + lightweight CNN + depth geometry + VO/SLAM + planner + confidence fallback
- [x] Finalize project name
- [x] Create Git repository and branch strategy
- [x] Create reproducible development environment
- [!] Record exact hardware availability — blocked pending physical inventory
- [!] Record compute hardware and GPU — blocked pending physical inventory
- [x] Select ROS 2 version if ROS 2 is used
- [x] Select camera — D435i reference model for simulation; physical camera still unverified
- [x] Define UGV drivetrain and footprint — provisional simulation model; physical measurements required before drive tests

---

# PHASE 1 — SYSTEM DEFINITION

- [ ] Draw final system architecture
- [ ] Define data interfaces between modules
- [ ] Define camera coordinate frames
- [ ] Define robot coordinate frames
- [ ] Define target operating speed
- [ ] Define minimum obstacle clearance
- [ ] Define emergency-stop behavior
- [ ] Define confidence states
- [ ] Define planner input/output contract
- [ ] Define logging format

---

# PHASE 2 — CAMERA & DEPTH

- [ ] Camera calibration
- [ ] RGB stream verification
- [ ] Depth stream verification
- [ ] Stereo alignment / RGB-depth registration
- [ ] Depth validity statistics outdoors
- [ ] Depth quality under direct sunlight
- [ ] Depth quality on grass
- [ ] Depth quality on soil
- [ ] Depth quality on gravel
- [ ] Depth quality near reflective/wet surfaces
- [ ] Record camera failure cases

---

# PHASE 3 — PERCEPTION BASELINE

- [ ] Define segmentation classes
- [ ] Assemble public training data
- [ ] Collect local outdoor samples
- [ ] Label focused validation set
- [ ] Train lightweight baseline
- [ ] Measure segmentation quality
- [ ] Measure inference latency
- [ ] Export to ONNX if beneficial
- [ ] Test accelerated inference
- [ ] Visualize mask + confidence
- [ ] Build failure-case gallery

---

# PHASE 4 — DEPTH GEOMETRY

- [ ] Implement depth filtering
- [ ] Implement ground/plane reasoning where appropriate
- [ ] Implement positive obstacle detection
- [ ] Investigate negative obstacle / drop detection
- [ ] Compute obstacle clearance
- [ ] Generate local obstacle representation
- [ ] Measure depth-to-obstacle latency
- [ ] Validate against manually labeled scenes

---

# PHASE 5 — FUSION

- [ ] Define semantic + geometric fusion rules
- [ ] Implement disagreement detection
- [ ] Implement traversability cost map
- [ ] Add uncertainty representation
- [ ] Test semantic-only vs depth-only vs fused
- [ ] Record ablation results
- [ ] Tune fusion thresholds
- [ ] Confirm failure behavior

---

# PHASE 6 — VISUAL ODOMETRY / SLAM

- [ ] Benchmark 2 candidate SLAM/VO stacks
- [ ] Verify camera compatibility
- [ ] Integrate pose output
- [ ] Test textured outdoor scene
- [ ] Test low-texture terrain
- [ ] Test rapid illumination change
- [ ] Test camera vibration
- [ ] Measure tracking loss frequency
- [ ] Test relocalization
- [ ] Record drift

---

# PHASE 7 — PLANNING & CONTROL

- [ ] Create local costmap
- [ ] Add UGV footprint
- [ ] Implement baseline planner
- [ ] Implement collision checking
- [ ] Tune safety clearance
- [ ] Tune velocity limits
- [ ] Test static obstacle avoidance
- [ ] Test sudden obstacle appearance
- [ ] Test dead-end recovery
- [ ] Test oscillation / corner cases
- [ ] Measure planning latency

---

# PHASE 8 — CONFIDENCE SAFETY LAYER

- [ ] Define confidence inputs
- [ ] Define confidence aggregation
- [ ] Calibrate thresholds
- [ ] Implement HIGH state
- [ ] Implement MEDIUM state
- [ ] Implement LOW state
- [ ] Implement STOP state
- [ ] Implement relocalization recovery
- [ ] Validate confidence vs actual failures
- [ ] Add operator-facing status

---

# PHASE 9 — INTEGRATION

- [ ] Connect camera to full stack
- [ ] Connect perception to fusion
- [ ] Connect fusion to costmap
- [ ] Connect SLAM to planner
- [ ] Connect planner to controller
- [ ] Connect safety layer to final command gate
- [ ] End-to-end latency measurement
- [ ] End-to-end logging
- [ ] Replay test from recorded data
- [ ] Full offline test

---

# PHASE 10 — SIMULATION

- [ ] Choose simulator
- [ ] Create or import UGV model
- [ ] Simulate camera
- [ ] Simulate outdoor terrain
- [ ] Simulate static obstacles
- [ ] Simulate dynamic obstacles
- [ ] Simulate sensor degradation
- [ ] Compare simulation behavior vs real camera data

---

# PHASE 11 — REAL-WORLD VALIDATION

Minimum scenario set:

- [ ] Open grass
- [ ] Soil path
- [ ] Gravel / rocks
- [ ] Narrow path
- [ ] Uneven ground
- [ ] Shadows
- [ ] Glare
- [ ] Vegetation
- [ ] Puddle / wet patch
- [ ] Sudden static obstacle
- [ ] Moving obstacle
- [ ] Low-texture scene
- [ ] Visual tracking degradation
- [ ] Recovery scenario

---

# PHASE 12 — BENCHMARKING

For every major experiment record:
- [ ] FPS
- [ ] latency
- [ ] segmentation metric
- [ ] obstacle precision/recall where meaningful
- [ ] localization drift
- [ ] navigation success rate
- [ ] collision rate
- [ ] path length / efficiency
- [ ] recovery time
- [ ] intervention count
- [ ] CPU/GPU use
- [ ] memory use

---

# PHASE 13 — SIH PRESENTATION

- [ ] One-page problem framing
- [ ] Architecture diagram
- [ ] Why depth?
- [ ] Why lightweight CNN?
- [ ] Why fusion?
- [ ] Why confidence-based fallback?
- [ ] Competitive comparison
- [ ] USP slide
- [ ] Metrics slide
- [ ] Demo flow
- [ ] Failure-case slide
- [ ] Business/use-case slide
- [ ] Future roadmap
- [ ] Final video
- [ ] Offline backup demo recording

---

# CRITICAL BLOCKERS

- [ ] Camera unavailable
- [ ] No suitable compute
- [ ] No UGV / simulator path
- [ ] No usable outdoor dataset
- [ ] ROS/device compatibility issue
- [ ] SLAM instability
- [ ] End-to-end latency too high
- [ ] Depth unusable under intended conditions
- [ ] Planner unsafe
- [ ] Training data too narrow

---

# DECISION LOG

Record major decisions here.

### Decision 001
**Topic:** Primary depth sensor  
**Status:** pending  
**Decision:**  
**Reason:**  
**Evidence:**

### Decision 002
**Topic:** Segmentation architecture  
**Status:** pending  
**Decision:**  
**Reason:**  
**Evidence:**

### Decision 003
**Topic:** SLAM/VO backend  
**Status:** pending  
**Decision:**  
**Reason:**  
**Evidence:**

### Decision 004
**Topic:** Local planner  
**Status:** pending  
**Decision:**  
**Reason:**  
**Evidence:**

### Decision 005
**Topic:** ROS 2 runtime and simulator  
**Status:** accepted  
**Decision:** ROS 2 Jazzy Jalisco with Gazebo Harmonic.  
**Reason:** Supported modular baseline for simulation and eventual UGV integration.  
**Evidence:** `execution-roadmap.md`; `docs/architecture/phase-0-decisions.md`.

### Decision 006
**Topic:** Project identity and workflow  
**Status:** accepted  
**Decision:** TerrainSight UGV; `main` plus short-lived feature branches and pull requests.  
**Reason:** Stable integration with reviewable module ownership.  
**Evidence:** `README.md`; `docs/architecture/phase-0-decisions.md`.

### Decision 007
**Topic:** Simulation reference platform  
**Status:** provisional  
**Decision:** D435i-style RGB-D camera and differential-drive UGV model.  
**Reason:** Allows ROS topics, frames, planning and safety work before real hardware is verified.  
**Evidence:** `execution-roadmap.md`; `docs/architecture/hardware-inventory.md`.

---

# WEEKLY RULE

At the end of each working session:
1. Mark completed tasks.
2. Add blockers.
3. Record important measurements.
4. Choose the next 3 highest-value tasks.
5. Never leave “research” as a vague task; turn it into a decision or experiment.
