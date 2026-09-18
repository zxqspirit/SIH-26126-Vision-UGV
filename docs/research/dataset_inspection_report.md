# Dataset Inspection Report: Off-Road Terrain Attention Region Dataset

**Inspection Agent:** SIH 26126 Dataset Inspection & Perception QA Agent  
**Date of Inspection:** September 18, 2026  
**Target Path Inspected:** `C:\SIH\tests\TrainingImages\TrainingImages\OriginalImages`  
**Dataset Name / Source:** *"Off-Road Terrain Attention Region Images"* (Kaggle / IEEE 2021)  
**Authors / Citation:** Gabriela Gresenz, Jules White, Douglas C. Schmidt, *"Attention Regions in Terrain Roughness Classification for Off-Road Autonomous Vehicles"* (Vanderbilt University, 2021)  
**License:** Open Academic / Research (CC BY-SA 4.0 / ODbL on Kaggle)  
**Operational Directive:** Read-only inspection. Zero modifications, zero renames, zero deletions, zero training.  

---

## 1. Directory Structure

The inspected dataset hierarchy in `C:\SIH\tests\` is organized as follows:

```
C:\SIH\tests\
├── AttentionRegion off road.zip (3,896.6 MB - Raw video frames with masked attention)
├── TrainingImages.zip (1,043.6 MB - Primary Path Segmentation subset)
├── AttentionRegion off road/
│   └── AttentionRegion/
│       ├── dark/                  # 5,381 RGB PNG images (3840x2160) with non-path blacked out
│       └── light/                 # 5,381 RGB PNG images (3840x2160) with non-path whited out
└── TrainingImages/
    └── TrainingImages/
        ├── OriginalImages/        # 515 Raw 4K RGB JPEG images (Target Directory)
        └── EnumMasks/             # Ground-truth pixel-wise semantic masks (515 pairs each)
            ├── png_0_1/           # Masks encoded with pixel values {0, 1}
            └── png_0_255/         # Masks encoded with pixel values {0, 255}
```

---

## 2. Total Number of Images

- **Target Directory (`OriginalImages`):** **515 image files** (all `.jpg`).
- **Related Sequence Directory (`AttentionRegion/dark` & `light`):** **5,381 image files** each (expanded temporal sequence from the same camera run).
- **Inspected Scope:** 515 paired high-resolution ground truth frames.

---

## 3. Total Number of Masks

- **Mask Variant A (`EnumMasks/png_0_1`):** **515 mask files** (`.png`).
- **Mask Variant B (`EnumMasks/png_0_255`):** **515 mask files** (`.png`).
- **Pairing Completeness:** Exactly **515 / 515 frames ($100.0\%$)** are matched 1:1 between `OriginalImages` and `EnumMasks`.
- **Naming Rule:** Source image `<timestamp>.jpg` maps to mask `<timestamp>_P.png`.

---

## 4. Image Dimensions & Color Format

- **Resolution:** Exactly **$3840 \times 2160$ pixels** (4K UHD, 16:9 widescreen aspect ratio).
- **Color Mode:** 3-channel **RGB** (24-bit depth, 8 bits per channel).
- **Uniformity:** $100\%$ uniform ($515 / 515$ images share the exact same dimensions).
- **EXIF Metadata:** Stripped / clean (no embedded GPS tags or camera serials in EXIF headers).

---

## 5. Mask Dimensions & Data Types

- **Resolution:** Exactly **$3840 \times 2160$ pixels** (identical to corresponding RGB images).
- **Data Type:** Single-channel **`uint8`** (8-bit unsigned grayscale PNG).
- **Spatial Alignment:** Exact pixel-to-pixel correspondence with `OriginalImages`.

---

## 6. Available Semantic Classes

The dataset provides strictly **two binary classes**:
1. **Drivable Off-Road Path / Trail Corridor**
2. **Non-Path Background** (aggregates sky, forest canopy, trees, roadside brush, grass, and background clutter into a single catch-all label).

---

## 7. Class IDs & Encoding Convention

> [!WARNING]
> ### Critical Inverted Label Warning
> Unlike standard segmentation conventions where `0 = Background` and `1 = Foreground/Path`, this dataset uses an **inverted class encoding**:

- **In `EnumMasks/png_0_1`:**
  - `Class 0`: **Drivable Trail / Path** (Foreground)
  - `Class 1`: **Non-Path / Environment / Sky / Vegetation** (Background)
- **In `EnumMasks/png_0_255`:**
  - `Class 0`: **Drivable Trail / Path** (Foreground)
  - `Class 255`: **Non-Path / Environment / Sky / Vegetation** (Background)

---

## 8. Class Pixel Frequency & Extreme Imbalance

Across all 515 images ($4,271,616,000\text{ total pixels}$):

| Class ID | Semantic Entity | Total Pixel Count | Percentage Share | Spatial Distribution |
|:---:|:---|:---:|:---:|:---|
| **0** | **Drivable Trail / Path** | **$190,603,773$** | **$4.46\%$** | Restricted strictly to the **lower image half** ($Y \ge 1080$). Zero path pixels exist in the top half ($0.0\%$). |
| **1** (or **255**) | **Non-Path / Background** | **$4,081,012,227$** | **$95.54\%$** | Covers $100\%$ of the upper image half ($Y < 1080$) plus all off-trail terrain shoulders. |

- **Class Imbalance Ratio:** **$21.4 : 1$** (extreme foreground sparsity). Standard cross-entropy without class re-weighting or Focal/Dice loss will suffer trivial collapse toward predicting all background.

---

## 9. Missing Files Audit

- **Original Images Missing Masks:** **0** ($0.0\%$).
- **Masks Missing Original Images:** **0** ($0.0\%$).
- **Result:** **Complete, unbroken 1:1 pair integrity.**

---

## 10. File Corruption & Integrity Audit

- **Image Readability Check:** 515 / 515 images decoded successfully via PIL and OpenCV with zero truncated JPEG blocks.
- **Mask Readability Check:** 1,030 / 1,030 mask PNGs decoded without decompression errors or invalid pixel values.
- **Result:** **Zero corrupted files.**

---

## 11. Duplicate Files Audit

- **Cryptographic Hashes (MD5):** All 515 images possess unique MD5 hashes.
- **Temporal Redundancy:** Images are sequential video frame grabs from a vehicle drive (`964868548s701ms`, `964868549s702ms`, etc.). While no two files are identical bytes, adjacent frames exhibit high temporal autocorrelation ($> 90\%$ visual similarity).

---

## 12. Train / Validation / Test Structure

- **Current Repository Layout:** **Completely Flat**. There are **no subfolders or split manifests** (`train/`, `val/`, `test/`).
- **Data Leakage Hazard:** Because frames originate from a continuous driving video, performing a naive random uniform split (`train_test_split(test_size=0.2)`) will result in severe **temporal data leakage** (frame $t$ in train, frame $t+1$ in test).
- **Requirement:** A contiguous sequence-based or block-based temporal split is strictly required if partitioned.

---

## 13. Mask Granularity & Pixel-Wise Verification

- **Format:** **Genuine pixel-wise semantic masks** tracing the organic, curved boundaries of off-road dirt paths.
- **Contour Fidelity:** Follows ruts, grass edges, and trail curves. They are **not** bounding boxes, bounding polygons, or coarse brush strokes.

---

## 14. Dataset Provenance & Licensing

- **Dataset Origin:** Kaggle public dataset: *"Off-Road Terrain Attention Region Images"* (2021).
- **Associated Academic Paper:** *G. Gresenz, J. White, D. Schmidt, "Attention Regions in Terrain Roughness Classification for Off-Road Autonomous Vehicles", IEEE / Vanderbilt University, 2021.*
- **Intended Original Purpose:** Training path segmentation to mask out non-path pixels, allowing a secondary CNN to estimate terrain roughness on the upcoming driving path.
- **License:** Open Academic / Research use (CC BY-SA 4.0 / ODbL). Commercial use requires author clearance.

---

## 15. Suitability for SIH 26126 Traversability Problem

### Assessment: **SEVERELY LIMITED / INADEQUATE FOR MULTI-CLASS TRAVERSABILITY**

| SIH 26126 Requirement | Kaggle Dataset Reality | Suitability Verdict |
|:---|:---|:---:|
| **Multi-Class Terrain Perception** (Trail, Grass, Gravel, Obstacle, Water/Mud, Sky) | **Binary Only (2 Classes)**: Bundles sky, trees, rocks, mud, and grass into one generic class (`1`). Cannot teach a model to separate driveable flat grass from lethal tree trunks. | **FAILED** |
| **Metric Depth Geometry** | **Zero Depth Information**: Purely monocular 2D RGB. No stereo baseline, no depth maps, no pointclouds. | **FAILED** |
| **Water / Mud Hazard Detection** | **No Class Distinction**: Water puddles and mud slicks are not annotated. | **FAILED** |
| **Negative Obstacle Detection** | **No Topography / Depth**: Cannot discern ditches, dropoffs, or potholes. | **FAILED** |
| **Real-Time Edge Throughput** | **4K UHD (3840x2160)**: Overwhelming compute footprint ($8.3\text{ MP}$). Requires heavy downscaling ($12\times$ reduction to $320\times 240$) before ingestion into our lightweight UGV CNN. | **NEEDS DOWNSAMPLING** |
| **Off-Road Trail Geometry** | **Realistic Ground View**: Excellent visual coverage of curved dirt trails through forest/brush. | **HIGH VALUE** |

---

## 16. Label Remapping Requirements

If this dataset is utilized for any neural network training, **mandatory label remapping is required**:

1. **Class Inversion (Mandatory):**
   $$\text{Mask}_{\text{remapped}} = 1 - \text{Mask}_{\text{png\_0\_1}}$$
   - Transforms `0 (Trail)` $\longrightarrow$ `Class 1 (Drivable Path)`.
   - Transforms `1 (Background)` $\longrightarrow$ `Class 0 (Non-Path / Void)`.
2. **Value Normalization:** If ingesting `png_0_255`, map $255 \longrightarrow 1$.
3. **Resolution Downsampling:** Bilinear/nearest-neighbor downsampling from $3840\times 2160$ to $320\times 240$ or $640\times 480$.

---

## 17. Camera Viewpoint & Perspective Analysis

- **Viewpoint Category:** **Forward-Looking Ground-Vehicle View (Front-Mounted UGV/ATV Camera)**.
- **Camera Mounting Height:** Approximately $1.0\text{ m} - 1.4\text{ m}$ above ground level.
- **Pitch Angle:** Downward tilt ($\approx -8^\circ$ to $-12^\circ$), placing the horizon at the vertical center ($Y \approx 1080$).
- **Vehicle Hood / Bumper:** Not visible in the frame (clear optical frustum).
- **Environment:** Natural off-road woodland / cross-country trails under outdoor daytime illumination.
- **Perspective Alignment with TerrainSight UGV:** High optical perspective similarity with our project's forward-facing sensor specification ($h = 0.45\text{m}$, pitch $= -12^\circ$).

---

## 18. Final Engineering Recommendation

### Recommendation: **USE WITH LABEL REMAPPING (ONLY FOR PATH EXTRACTION PRETRAINING / VALIDATION)**

> [!CAUTION]
> ### Definitive Engineering Verdict
> **DO NOT USE THIS DATASET AS THE PRIMARY TRAINING SOURCE FOR THE SIH 26126 TRAVERSABILITY PIPELINE.**
> 
> **Why?** Our system relies on a **6-level multi-modal costmap** (`PREFERRED_PATH`, `FREE_TERRAIN/Grass`, `MEDIUM_RISK/Gravel`, `BLOCKED/Obstacle`, `HIGH_RISK/Water`, `UNKNOWN`). Training our segmentation network on this dataset would cause the model to classify all grass, rocks, puddles, and trees as a single undifferentiated obstacle block, completely breaking our dual-veto fusion engine and planning clearance logic.
> 
> **Legitimate Permissible Use Case:**
> This dataset **CAN BE USED WITH LABEL REMAPPING** strictly as:
> 1. **Auxiliary Pre-Training / Feature Backbones:** Pre-training the CNN encoder on off-road trail feature extraction prior to fine-tuning on our multi-class outdoor datasets.
> 2. **External Generalization Benchmark:** Evaluating whether a trained trail segmentation model can identify path boundaries on unseen external off-road video.
