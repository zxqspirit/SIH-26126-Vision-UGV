# Public Off-Road Terrain & Semantic Traversability Datasets

**Project:** SIH 26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle (Outdoor)  
**Organization:** Bharat Electronics Limited (BEL)  
**Role:** Dataset Research & Collection-Planning Agent  
**Reference Artifact:** [`dataset_decision_matrix.md`](file:///C:/Users/harsh/.gemini/antigravity-ide/brain/1120e7f0-317f-4594-9ac8-c567f398da47/dataset_decision_matrix.md)  
**Field Protocol Companion:** [`docs/data_collection_protocol.md`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/docs/data_collection_protocol.md)

---

## 1. Context & Dataset Requirements

Standard autonomous driving datasets (such as Cityscapes, nuScenes, or Waymo Open) focus heavily on structured urban roads with clear lane markings, curbs, painted asphalt, and pedestrian crosswalks. 

For the **SIH 26126 outdoor UGV**, structured driving assumptions break down completely. The vehicle must negotiate:
- Unpaved dirt paths, packed soil, and loose gravel trails.
- Low grass vs. dense, non-traversable overgrown scrub and brush.
- Loose rocks, protruding boulders, and rubble piles.
- Muddy ruts, shallow puddles, and deep water hazards.
- Dappled canopy shadows, direct sun glare, and high dynamic range lighting.
- Negative obstacles such as drop-offs, erosion ditches, and trenches.

This research analyzes premier public datasets purpose-built for off-road unstructured robotics.

---

## 2. Comprehensive Candidate Profiles

### 2.1 RELLIS-3D (Texas A&M / US Army Research Lab)
- **Year:** 2021
- **Focus:** Multi-modal off-road autonomous navigation and 3D semantic segmentation.
- **Sensor Suite:** RGB cameras, Ouster OS1-64 LiDAR, Stereo pairs, RTK-GPS, and IMU mounted on a Clearpath Warthog robot.
- **Scale:** 6,235 dense pixel-annotated RGB frames; 13,556 annotated 3D LiDAR scans.
- **Classes (20):** Grass, tree, bush, mud, puddle, water (deep), concrete, dirt, rock, asphalt, rubble, barrier, sky, vehicle, object, etc.
- **Licensing:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0).
- **Strengths:**
  - Explicit distinction between **puddle** (potentially shallow/traversable with caution) and **deep water** (lethal hazard).
  - Explicit annotations for **mud** and **rubble/rock piles**.
  - Provides full ROS bag recordings enabling direct playback into ROS 2 test pipelines.
- **Weaknesses:**
  - Severe class imbalance (puddle and mud pixels represent $< 2\%$ of total annotations).
  - Geographic bias toward dry Texas brush and scrubland.
- **Project Role:** **Primary training benchmark** for our CNN traversability segmenter and multi-modal fusion layer.

---

### 2.2 RUGD (Robot Unstructured Ground Driving Dataset)
- **Year:** 2019
- **Focus:** Semantic segmentation for unstructured mobile robot navigation.
- **Sensor Suite:** Point Grey Bumblebee2 stereo camera mounted on an all-terrain mobile platform.
- **Scale:** Over 8,000 dense pixel-annotated frames captured across 12 unique outdoor trails (parks, creeks, woods, gravel roads).
- **Classes (24):** Dirt, sand, grass, gravel, bush, tree, mulch, water, asphalt, rock, log, fence, container, etc.
- **Licensing:** CC BY-NC-SA 3.0.
- **Strengths:**
  - Exceptional variety of natural trail boundaries (e.g., dirt paths meandering through tall grass).
  - Highly consistent low-vantage camera height ($\approx 0.6\text{ m}$) directly matching our UGV design envelope.
- **Weaknesses:**
  - Does not supply synchronized dense metric depth maps.
  - Less temporal variation across seasons.
- **Project Role:** Pre-training for feature extractors and boundary segmentation evaluation.

---

### 2.3 TAS500 (Technical University of Munich)
- **Year:** 2020
- **Focus:** High-resolution semantic segmentation for outdoor driving in rural and agricultural terrain.
- **Sensor Suite:** High-resolution industrial camera ($2048 \times 1024$).
- **Scale:** 500 finely annotated high-resolution images.
- **Classes (23):** Soil, sand, gravel, asphalt, low grass, high grass, tree trunk, tree crown, bush, rock, wall, building, fence, pole, sky.
- **Licensing:** CC BY-NC-SA 3.0 IGO.
- **Strengths:**
  - Crucial separation between **low grass** (traversable) and **high grass** (non-traversable/concealed obstacles).
  - Very clean annotations of fine soil and gravel transitions.
- **Weaknesses:**
  - Small total frame count (500 frames).
  - Vehicle perspective is slightly higher than small UGV chassis height.
- **Project Role:** Fine-grained boundary validation and texture transfer test.

---

### 2.4 Yamaha-CMU Off-Road Dataset (YCOR)
- **Year:** 2021
- **Focus:** Off-road traversability estimation across multi-seasonal environments.
- **Sensor Suite:** Multi-camera RGB, RTK-GPS, and IMU on Yamaha Viking all-terrain vehicle.
- **Scale:** 1,076 annotated images collected across Western Pennsylvania and Ohio across summer, fall, and winter.
- **Classes (8):** Sky, rough trail, smooth trail, traversable grass, high vegetation, low non-traversable vegetation, obstacle, void.
- **Licensing:** Academic / Research Use.
- **Strengths:**
  - Classes are already directly mapped to **traversability states** rather than generic semantic categories.
  - Captures wet leaves, muddy autumn ruts, and winter freeze.
- **Weaknesses:**
  - Coarser class granularity.
- **Project Role:** Ground-truth validation for traversability cost scoring ($C \in [0, 100]$).

---

### 2.5 GOOSE (German Outdoor and Offroad Dataset)
- **Year:** 2023
- **Focus:** Multimodal 2D and 3D perception in off-road and adverse outdoor environments.
- **Sensor Suite:** Multi-camera RGB, Near-Infrared (NIR), 3D LiDAR, IMU on mobile excavator and quadruped platforms.
- **Scale:** 10,000 multimodal pairs; extension GOOSE-Ex adds 5,000 more.
- **Classes:** Complex ontology spanning structured and natural terrain, dense vegetation, water, and ditches.
- **Licensing:** Open Research License.
- **Strengths:**
  - Modern, state-of-the-art annotations.
  - Near-Infrared imagery provides strong water/vegetation separation under harsh glare.
- **Weaknesses:**
  - Very large raw download footprints.
- **Project Role:** Robustness stress-testing under adverse lighting and high dynamic range conditions.

---

## 3. Ontological Harmonization to SIH 26126 Pipeline

The disparate labeling conventions of external datasets are unified into the standard `TerrainClass` enum defined in [`src/interfaces/types.py`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/src/interfaces/types.py):

| Project Class | ID | Traversability Cost | Normalized Color (RGB) | Composite External Classes Mapped |
| :--- | :--- | :--- | :--- | :--- |
| **`UNKNOWN`** | 0 | 50 (Cautious) | `[128, 128, 128]` (Grey) | Void, unlabeled, sky, glare-blinded zones |
| **`PAVED_ROAD`** | 1 | 5 (Optimal) | `[80, 80, 80]` (Dark Grey) | Asphalt, concrete, paved road, sidewalk |
| **`TRAVERSABLE_DIRT`** | 2 | 15 (Safe) | `[160, 100, 50]` (Brown) | Packed dirt, soil, sand, smooth trail |
| **`LOW_GRASS`** | 3 | 20 (Safe) | `[0, 200, 0]` (Green) | Low grass, mown lawn, traversable grass |
| **`GRAVEL`** | 4 | 25 (Safe) | `[180, 180, 120]` (Beige) | Gravel, pebble paths, rough trail, rubble |
| **`HIGH_VEGETATION`** | 5 | 80 (Hazard) | `[0, 90, 0]` (Dark Green) | High grass, tall weeds, bush, dense scrub, tree trunk |
| **`OBSTACLE_SOLID`** | 6 | 100 (Lethal) | `[220, 20, 20]` (Red) | Boulders, large rocks, barriers, fences, walls |
| **`WATER_PUDDLE`** | 7 | 90 (Lethal/Mud) | `[0, 150, 255]` (Blue) | Standing water, puddles, mud, bogs, marsh |

---

## 4. Dataset Acquisition & Storage Protocol

To uphold the repository's strict large-file and bandwidth constraints:
1. **Never commit multi-gigabyte raw datasets directly into Git.**
2. **Download Only Curated Slices:**
   - Instead of downloading 100+ GB full archives, extract curated 50-frame evaluation splits (`eval_50_rellis`, `eval_50_rugd`, `eval_50_tas500`).
3. **Local Storage Target:**
   - Raw downloads reside exclusively in `datasets/raw/` (ignored by Git).
   - Standardized downscaled test pairs ($640 \times 480$) live in `datasets/processed/<dataset_name>/` with JSON metadata index files.
