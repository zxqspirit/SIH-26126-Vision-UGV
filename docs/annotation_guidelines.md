# UGV Terrain Semantic Annotation Guidelines & SOP

**Project:** SIH 26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle (Outdoor)  
**Organization:** Bharat Electronics Limited (BEL)  
**Role:** Annotation Architect  
**Reference Artifact:** [`annotation_schema_decision.md`](file:///C:/Users/harsh/.gemini/antigravity-ide/brain/1120e7f0-317f-4594-9ac8-c567f398da47/annotation_schema_decision.md)  
**Validation Utility:** [`scripts/validate_dataset.py`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/scripts/validate_dataset.py)  

---

## 1. Executive Summary & Core Rules

Accurate semantic segmentation is the foundation of vision-based off-road autonomy. In the SIH 26126 pipeline, the perception model does not make subjective binary guesses; it identifies **physical terrain materials**. The downstream costmap generator then applies deterministic traversability penalties.

### The Three Golden Rules of Annotation:
1. **Annotate Physical Material, Not Speculative Traversability:** Never guess whether an imaginary robot can climb an obstacle. Label *what you see* (e.g., grass is grass; a rock is a rock).
2. **Label the Substrate Under Shadows:** Tree shadows and cloud shadows do **not** change the ground material. A shadowed dirt path is `TRAVERSABLE_DIRT`, not an obstacle.
3. **Pixel-Accurate Mask Output:** Masks must be saved as single-channel, 8-bit grayscale PNG images where each pixel's numeric integer value strictly equals its Class ID ($0$ to $8$).

---

## 2. The 9-Class Ontology (`TerrainClass`)

| ID | Class Name | Semantic Scope & Description | Cost $[0, 100]$ | Binary State | RGB Display Color | Hex Code |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **0** | `UNKNOWN` | Sky, image margins, totally overexposed glare ($> 250$), pitch black void ($< 5$). | 50 | Unknown | `[128, 128, 128]` Grey | `#808080` |
| **1** | `PAVED_ROAD` | Paved asphalt, smooth concrete, paved pedestrian paths, tiled ground. | 5 | Traversable | `[80, 80, 80]` Dark Grey | `#505050` |
| **2** | `TRAVERSABLE_DIRT` | Compact dirt, flat red/brown soil, dry clay paths, sand flats. | 15 | Traversable | `[160, 100, 50]` Brown | `#A06432` |
| **3** | `LOW_GRASS` | Mown grass, turf, low clover, blade height strictly $\le 10\text{ cm}$. | 20 | Traversable | `[0, 200, 0]` Bright Green | `#00C800` |
| **4** | `GRAVEL` | Crushed gravel, loose pebbles, ballast, stones $\le 5\text{ cm}$ diameter. | 25 | Traversable | `[180, 180, 120]` Beige | `#B4B478` |
| **5** | `HIGH_VEGETATION` | Tall weeds, dense brush, scrub, wild thickets, foliage $> 30\text{ cm}$. | 80 | Non-Traversable | `[0, 90, 0]` Forest Green | `#005A00` |
| **6** | `OBSTACLE_SOLID` | Protruding boulders ($> 15\text{ cm}$), brick walls, fences, poles, tree trunks. | 100 | Non-Traversable | `[220, 20, 20]` Red | `#DC1414` |
| **7** | `WATER_PUDDLE` | Standing water puddles, mud ruts, saturated clay bogs, deep ditches. | 90 | Non-Traversable | `[0, 150, 255]` Cyan | `#0096FF` |
| **8** | `DYNAMIC_OBSTACLE` | Pedestrians, moving animals, vehicles, active machinery. | 100 | Non-Traversable | `[255, 100, 0]` Orange | `#FF6400` |

---

## 3. Ambiguous-Case Boundary Decision Trees

Annotators frequently encounter ambiguous boundary scenarios in outdoor field footage. Follow these standardized rulings:

### Case A: Low Grass vs. High Vegetation
```
Is the vegetation height <= 10 cm (e.g. lawn / short ground cover)?
    ├── YES ──> Label as LOW_GRASS (3) [Traversable]
    └── NO
         └── Is it tall wild weeds (> 30 cm), thick brush, or a tree canopy?
                 ├── YES ──> Label as HIGH_VEGETATION (5) [Non-Traversable]
                 └── Intermediate (10 cm to 30 cm):
                      └── Does it obscure the ground surface?
                           ├── YES ──> HIGH_VEGETATION (5)
                           └── NO  ──> LOW_GRASS (3)
```

### Case B: Dirt vs. Mud vs. Water Puddle
- **Scenario 1 (Dry or Damp Soil):** Compact, firm earth without liquid standing on top $\rightarrow$ `TRAVERSABLE_DIRT` (2).
- **Scenario 2 (Wet Saturated Mud):** Viscous, sticky mud with visible tire ruts or wet surface sheen indicating vehicle sinkage hazard $\rightarrow$ `WATER_PUDDLE` (7).
- **Scenario 3 (Standing Puddle):** Any surface pool of liquid displaying sky reflections or rippling $\rightarrow$ `WATER_PUDDLE` (7).

### Case C: Gravel vs. Protruding Rocks
- **Loose small pebbles ($\le 5\text{ cm}$):** Walking path composed of small gravel or crushed stone $\rightarrow$ `GRAVEL` (4).
- **Protruding boulders ($> 15\text{ cm}$):** Distinct, sharp rocks jutting out of the ground that would hit a low vehicle undercarriage $\rightarrow$ `OBSTACLE_SOLID` (6). Trace the individual rock boundary tightly.
- **Medium stones ($5\text{ cm} - 15\text{ cm}$):**
  - If isolated on an otherwise clean dirt trail: Outline as `OBSTACLE_SOLID` (6).
  - If densely packed into a uniform stone path: Label as `GRAVEL` (4).

### Case D: Direct Sunlight, Shadows, and Glare
- **Tree and Building Shadows:** Always label the **underlying ground material**. If a shadow falls across a dirt trail, label the entire continuous path as `TRAVERSABLE_DIRT` (2). Do not split paths into alternating "shadow" and "sun" classes.
- **Extreme Glare / Sun Flare:** If sunlight creates total white clipping ($R=G=B=255$ with zero ground detail visible), label strictly the blinded patch as `UNKNOWN` (0).
- **Complete Black Shadow:** If underexposure renders an area pure black ($R=G=B=0$ with no texture discernable), label as `UNKNOWN` (0).

### Case E: Narrow Paths & Trail Corridors
- When a footpath narrows between overgrown bushes or rock walls, trace the exact walkable path as `TRAVERSABLE_DIRT` (2) or `GRAVEL` (4).
- Do **not** artificially shrink the trail boundary. The robot's inflation radius and safety buffer will be handled automatically by the costmap generator.

### Case F: Tree Trunks vs Overhanging Canopies
- **Tree Trunks:** The vertical wood trunk rooted in the ground is an immovable solid obstacle $\rightarrow$ `OBSTACLE_SOLID` (6).
- **Overhanging Foliage:** Leaves and branches in the air above the vehicle path $\rightarrow$ `HIGH_VEGETATION` (5).
- **Fallen Logs:** Any deadfall or horizontal log across the trail $> 10\text{ cm}$ diameter $\rightarrow$ `OBSTACLE_SOLID` (6).

---

## 4. Annotation Tool Setup (CVAT / LabelMe / AnyLabeling)

### LabelMe / CVAT Configuration
When setting up class labels in labeling software, configure the labels in exact numerical order:

```json
[
  {"id": 0, "name": "unknown", "color": "#808080"},
  {"id": 1, "name": "paved_road", "color": "#505050"},
  {"id": 2, "name": "traversable_dirt", "color": "#A06432"},
  {"id": 3, "name": "low_grass", "color": "#00C800"},
  {"id": 4, "name": "gravel", "color": "#B4B478"},
  {"id": 5, "name": "high_vegetation", "color": "#005A00"},
  {"id": 6, "name": "obstacle_solid", "color": "#DC1414"},
  {"id": 7, "name": "water_puddle", "color": "#0096FF"},
  {"id": 8, "name": "dynamic_obstacle", "color": "#FF6400"}
]
```

---

## 5. Storage Directory Structure

Annotations must follow the standard paired format:

```
datasets/annotations/
├── rgb/
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
├── masks/
│   ├── frame_0000.png    <-- 8-bit single channel grayscale PNG
│   ├── frame_0001.png
│   └── ...
└── classes.json
```

---

## 6. Pre-Commit Quality Checklist for Annotators

Before submitting an annotated batch, run the automated validator:
```bash
python scripts/validate_dataset.py --dataset-dir datasets/annotations
```

Verify that:
- [ ] No missing mask files exist for any RGB image.
- [ ] Width and height match RGB dimensions exactly.
- [ ] No invalid pixel indices outside $[0, 8]$ exist.
- [ ] Water puddles and mud ruts are strictly labeled as `WATER_PUDDLE` ($7$).
- [ ] Shadows were not labeled as obstacles.
