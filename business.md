# SIH 26126 — Business Profile

## 1. Venture / Product Concept

### Working product
A modular **vision-based autonomy software stack for outdoor UGVs** operating in GPS-denied or GPS-unreliable environments.

### Working product description
The software combines:
- stereo / RGB-D sensing
- lightweight traversability and terrain segmentation
- metric depth reasoning
- visual odometry / SLAM
- local path planning
- confidence-aware safety fallback

The intended value is not simply “object detection.” It is a navigation layer that can turn perception into safe motion.

---

## 2. Primary Customer Segments

### A. Defence & security
Potential buyers/users:
- defence integrators
- government robotics programs
- security/patrol operators
- UGV OEMs

Pain:
- unreliable GPS
- terrain variability
- need for local autonomy
- constrained connectivity

Fit:
**Very high strategic fit**, but procurement and certification barriers are substantial.

### B. Search & rescue / disaster response
Potential buyers:
- emergency response organizations
- disaster management agencies
- robotics integrators

Pain:
- hazardous terrain
- poor infrastructure
- rapidly changing obstacles
- need to keep humans away from dangerous areas

Fit:
**High.**

### C. Agriculture
Potential buyers:
- agricultural robotics firms
- large farms
- equipment OEMs
- agritech companies

Pain:
- uneven terrain
- mud
- crop rows
- variable soil and vegetation

Fit:
**High commercial potential.**

### D. Mining / construction / inspection
Potential buyers:
- mining operators
- infrastructure inspection companies
- robotics OEMs

Pain:
- rough terrain
- repeated routes
- hazardous work zones
- poor positioning conditions

Fit:
**High B2B potential.**

---

## 3. Core Value Proposition

> **Provide a deployable autonomy module that lets an outdoor UGV perceive traversable terrain, localize visually and react to obstacles without requiring continuous GPS.**

Secondary value:
- modular integration
- offline operation
- lower sensor complexity than LiDAR-heavy designs
- interpretable safety behavior
- ability to retrofit an existing UGV platform

---

## 4. What We Actually Sell

The most plausible long-term commercial product is not the student UGV itself.

It is the **autonomy software module**.

Possible packaging:

### Autonomy SDK
- perception API
- depth fusion
- localization interface
- planner interface
- diagnostics

### Edge deployment package
- optimized models
- hardware integration
- runtime configuration
- logging

### Fleet / operator software
- mission planning
- health monitoring
- route history
- event logs

### Integration services
- camera calibration
- vehicle parameter tuning
- site-specific training
- OEM integration

---

## 5. Business Model Options

### B2B / OEM licensing — preferred long-term model
Sell or license the navigation stack to UGV manufacturers and system integrators.

Possible commercial structure:
- development / integration fee
- per-platform license
- annual support contract

### B2G
Government / defence deployment through qualified integrators.

Likely model:
- pilot
- technical evaluation
- integration
- procurement / program contract
- maintenance and updates

### B2B deployment
For agriculture, mining, inspection and security.

Possible model:
- per-robot annual software license
- integration fee
- optional site-specific model adaptation

### R&D / SDK model
Provide a software development kit to research labs and robotics companies.

---

## 6. Pricing Philosophy

Do not invent a retail price during the SIH stage.

Instead, position the economics around:
- sensor BOM savings
- engineering-hour reduction
- reduced operator dependency
- fewer collisions / interventions
- ability to retrofit existing UGVs

For any future pricing study, create a real cost model using:
- hardware
- compute
- integration hours
- support
- model training
- certification
- deployment scale

---

## 7. Target Buyer

The buyer is not necessarily the person driving the UGV.

Potential buyer:
- robotics OEM
- defence integrator
- government project team
- industrial automation company

Potential user:
- field operator
- robotics engineer
- mission commander
- maintenance engineer

Always separate **buyer, user and technical decision-maker**.

---

## 8. Competitive Positioning

We should not claim “no competitors.”

The practical competitive landscape includes:
- GPS-based outdoor autonomy
- LiDAR-heavy stacks
- research-grade visual SLAM systems
- robotics middleware + custom perception
- proprietary UGV autonomy stacks

Our intended differentiation:
1. depth + semantics rather than one modality alone
2. confidence-aware behavior
3. lightweight edge deployment
4. modular integration
5. GPS-denied focus
6. terrain-aware cost rather than simple obstacle boxes

---

## 9. Defensibility

Software defensibility can come from:
- custom Indian outdoor dataset
- fusion logic
- confidence calibration
- failure recovery
- deployment optimization
- domain-specific terrain cost models
- field-tested vehicle interfaces
- integration tooling
- performance data across difficult conditions

A model checkpoint alone is not a durable moat.

---

## 10. Commercial Validation Questions

Before a commercial claim, investigate:
- target robot platform
- camera constraints
- compute budget
- operating speed
- terrain types
- acceptable failure rate
- certification requirements
- procurement model
- integration effort
- data ownership
- cybersecurity / offline requirements

---

## 11. Product Principles

### Offline-first
Core autonomy should not depend on cloud inference.

### Human override
A practical system needs a clear manual takeover path.

### Auditability
The system should record why it slowed, stopped or changed path.

### Platform independence
The software should be adaptable to multiple UGV drivetrains.

### Safety by degradation
When confidence falls, performance should degrade conservatively rather than unpredictably.

