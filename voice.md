# SIH 26126 — Voice & Communication Rules

## 1. Project Voice

Our project communication should sound like:
- capable
- technically grounded
- direct
- calm
- engineering-first
- ambitious without exaggeration

We are building a serious prototype, not writing science-fiction marketing copy.

---

## 2. Core Messaging

The central story is:

> **Give an outdoor UGV the ability to understand traversability, estimate motion, and react safely without depending on GPS.**

Our preferred framing is:

**semantic understanding + metric depth + visual localization + confidence-aware safety**

Avoid presenting the system as “AI magic.”

---

## 3. Words / Phrases to Avoid

Avoid empty or overused phrases such as:
- revolutionary
- game-changing
- next-generation
- cutting-edge
- futuristic
- AI-powered solution
- seamless
- robust in all conditions
- works everywhere
- 100% safe
- military-grade
- production-ready
- zero failure
- autonomous in any environment
- state-of-the-art, unless a specific benchmark justifies it
- smart solution
- innovative use of AI

Do not write “solves the problem” when we mean “reduces a known failure mode.”

---

## 4. Preferred Technical Language

Instead of:
> “Our AI intelligently understands the environment.”

Write:
> “A lightweight segmentation model estimates traversability and terrain cost, which is fused with metric depth before planning.”

Instead of:
> “The robot never crashes.”

Write:
> “The safety layer reduces speed or stops when perception or localization confidence falls below a defined threshold.”

Instead of:
> “Works without GPS anywhere.”

Write:
> “The navigation stack is designed for GPS-denied operation and is evaluated across representative outdoor conditions.”

Instead of:
> “Our model is highly accurate.”

Write:
> “On the held-out validation set, the model achieved X on metric Y at Z FPS on hardware H.”

---

## 5. Judge-Facing Writing Style

Every major claim should answer at least one of:
- What problem does this solve?
- Why is the existing approach insufficient?
- What did we add?
- How do we measure improvement?
- Why is it feasible?
- Why can this be deployed?

Prefer a one-line claim followed by evidence.

---

## 6. USP Style

Good USP:

> “Confidence-aware fusion converts perception uncertainty into a concrete safety action: reduced speed, larger clearance, re-observation or stop.”

Weak USP:

> “Our revolutionary AI creates unmatched safety.”

Good USP:

> “Depth supplies metric geometry while the lightweight CNN supplies semantic traversability; disagreement between the two becomes an explicit uncertainty signal.”

---

## 7. Explainability

When discussing a decision, use:

**Observation → Interpretation → Action**

Example:

> Depth reports a possible drop while the CNN predicts traversable ground. The fusion layer marks the region uncertain, reduces speed and triggers a closer observation rather than blindly following the segmentation mask.

This is much stronger than saying “the AI detects danger.”

---

## 8. Technical Honesty Rules

Never hide:
- missed detections
- depth holes
- SLAM drift
- lighting failures
- dataset limitations
- computational limitations
- simulation-only evidence

Instead, state:
1. limitation
2. impact
3. mitigation
4. remaining risk

That makes the project more credible.

---

## 9. Presentation Style

Use:
- short headings
- diagrams
- comparison tables
- numbers
- before/after examples
- ablation results
- failure cases
- architecture visuals

Avoid:
- paragraphs full of adjectives
- unexplained acronyms
- large blocks of generic AI terminology
- claims without metrics

---

## 10. Project Naming Style

Prefer functional names that describe the system.

Examples:
- VisionNav
- TerraNav
- VISTA-UGV
- TerrainAware
- NavSense
- VIGIL-UGV

Do not use a dramatic military name unless the project scope genuinely supports it.

---

## 11. Presentation Hook

A strong opening is:

> “GPS tells a robot where it is. Our system focuses on what happens when GPS cannot be trusted: it uses vision, depth and uncertainty-aware planning to decide where the ground is safe to drive.”

A stronger technical continuation:

> “We do not ask one AI model to drive the robot end-to-end. We separate perception, geometry, localization, planning and safety so every decision can be inspected and measured.”

---

## 12. Avoiding AI Slop

Every time a paragraph could apply equally to 500 other hackathon projects, rewrite it.

A good paragraph contains at least one project-specific detail:
- stereo/RGB-D
- terrain cost
- depth disagreement
- localization confidence
- safety state
- Indian outdoor condition
- embedded latency
- specific metric

