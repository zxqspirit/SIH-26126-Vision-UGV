# Outdoor Field Data Collection Protocol (Without Physical UGV)

**Project:** SIH 26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle (Outdoor)  
**Organization:** Bharat Electronics Limited (BEL)  
**Role:** Dataset Research & Collection-Planning Agent  
**Companion Artifact:** [`dataset_decision_matrix.md`](file:///C:/Users/harsh/.gemini/antigravity-ide/brain/1120e7f0-317f-4594-9ac8-c567f398da47/dataset_decision_matrix.md)  
**Research Reference:** [`docs/research/datasets.md`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/docs/research/datasets.md)  
**Hardware Baseline:** Handheld rig, monopod, or tripod mount (no physical vehicle assumed).

---

## 1. Protocol Purpose & Operational Context

Because the project does **not possess physical UGV hardware** and **forbids simulation (no Gazebo)**, real sensor recordings captured by team members represent the primary lifeline for training CNN perception, evaluating depth geometry, testing visual odometry, and tuning safety recovery behaviors.

To ensure captured field data seamlessly transfers to a robotic ground vehicle, handheld and tripod-mounted recordings must **strictly emulate UGV kinematics, optical mounting heights, and look-ahead angles**.

---

## 2. Sensor Mounting & Optical Geometry

```
                    +-------------------------------------------------------------+
                    |                      OPERATOR (Walking)                     |
                    +-------------------------------------------------------------+
                                       |
                                       | Holds stabilized monopod / inverted gimbal
                                       v
                             [ Camera Mount / Rig ]
                                 |
                                 |  Camera Height: 0.45 m - 0.60 m
                                 |  Downward Pitch: 10° - 15°
                                 v
        ═══════════════════════[ CAMERA ]══════════════════════════
                                    \   ) ~12° pitch
                                     \
                                      \ Optical Axis (Forward + Depth)
                                       \
        ────────────────────────────────▼───────────────────────── GROUND TERRAIN
```

### 2.1 Rig Specifications
- **Camera Height ($h_{\text{cam}}$):** Exactly **$0.45\text{ m} - 0.60\text{ m}$** above the ground surface (matching the UGV chassis baseline defined in `docs/architecture/hardware-inventory.md`).
- **Look-Ahead Pitch Angle ($\theta_{\text{pitch}}$):** **$10^{\circ} - 15^{\circ}$ downward tilt** ($\approx 0.21\text{ rad}$) relative to horizontal. This ensures the ground plane enters the image frame $\approx 1.0\text{ m}$ ahead of the chassis while maintaining the horizon in the upper 15% of the frame.
- **Physical Rigs:**
  - *Option A (Recommended):* Handheld inverted 3-axis motorized gimbal mounted on a short monopod or selfie stick held low near the ground.
  - *Option B:* Fixed monopod with rubber foot walked smoothly at knee height.
  - *Option C:* Low-clearance pushcart or rolling utility dolly with camera hard-mounted at 0.50 m.
  - *Option D (Static):* Heavy tripod positioned at 0.50 m for multi-angle obstacle inspection.

---

## 3. Camera Sensor & Optical Settings

To prevent motion blur, optical artifacts, and exposure instability:

| Parameter | Recommended Setting | Rationale |
| :--- | :--- | :--- |
| **Resolution** | $1920 \times 1080$ (1080p) or $1280 \times 720$ (720p) | Standard 16:9 aspect ratio downscalable to $640 \times 480$. |
| **Framerate** | 30 FPS native (or 10 FPS if recording raw uncompressed) | Enables smooth visual odometry tracking and 10 Hz downsampling. |
| **Shutter Speed** | Fixed **$1/250\text{ s}$ to $1/500\text{ s}$** (Manual Mode) | **Critical:** Completely eliminates footstep motion blur and micro-jitter. |
| **ISO** | Fixed $100 - 400$ (Avoid Auto-ISO) | Prevents high-frequency sensor grain from corrupting texture analysis. |
| **Aperture** | $f/2.8 - f/5.6$ | Deep depth-of-field ensuring ground from $1\text{ m}$ to $20\text{ m}$ remains in sharp focus. |
| **White Balance** | Fixed (5500K for Sunlight, 6500K for Overcast) | **Never use Auto White Balance (AWB)**: color shifts cause false terrain label flips. |
| **Optical Filter** | Circular Polarizer (CPL) on standby | Suppresses extreme water puddle and soil specular glare during noon runs. |

---

## 4. Environmental Lighting Regimes

Data collection must encompass four distinct outdoor lighting conditions to build perception robustness:

```
+---------------------------------------------------------------------------------------------------+
|                                     OUTDOOR LIGHTING REGIMES                                      |
|                                                                                                   |
|  [Regime 1: Direct High-Noon]       [Regime 2: Tree Canopy Shadows]                               |
|  - Harsh vertical contrast          - Alternating bright sun and dark shade                       |
|  - Minimal horizontal shadow        - Tests camera dynamic range (HDR)                            |
|                                                                                                   |
|  [Regime 3: Golden Hour / Low Sun]  [Regime 4: Overcast / Diffuse]                                |
|  - Extreme forward glare into lens  - Low-contrast, uniform illumination                          |
|  - Long confusing ground shadows    - Ideal for subtle soil/grass color boundaries                |
+---------------------------------------------------------------------------------------------------+
```

### 4.1 Protocol per Lighting Regime:
1. **Direct Sunlight (11:00 AM – 2:00 PM):**
   - High ambient lux ($> 80,000\text{ lux}$). Set shutter speed to $1/500\text{ s}$ to avoid clipping highlights on sandy soil or dry concrete.
2. **Dappled Canopy Shadows:**
   - Walk under tree lines and forest edges where shadows create false "step" boundaries. Capture slow passages transitioning from bright open field into dark forest floor.
3. **Low-Angle Sun & Glare (Early Morning / Late Afternoon):**
   - Capture runs walking **directly toward the sun** ($\pm 30^{\circ}$) to test lens flare and sensor blinding, followed by walking **directly away from the sun** to evaluate long vehicle/chassis shadows.
4. **Diffuse Overcast / Cloudy:**
   - Flat light conditions where shadows vanish. Best condition for recording baseline ground-truth soil, rock, and grass textures without illumination bias.

---

## 5. Terrain Substrate Capture Matrix

Every session must capture isolated and transitional samples of all primary terrain classes:

| Terrain Class | Target Field Locations | Visual & Geometric Features to Record | Traversability Label |
| :--- | :--- | :--- | :--- |
| **Paved Road** | Campus perimeter, asphalt service paths | Flat, high traction, crack lines, edge transitions | `PAVED_ROAD` (Cost: 5) |
| **Traversable Dirt** | Beaten footpaths, packed red soil, dry clay | Compact surface, minor dust, small pebbles ($< 2\text{ cm}$) | `TRAVERSABLE_DIRT` (Cost: 15) |
| **Low Grass** | Mown lawns, athletic fields, dry park grass | Blade height $< 10\text{ cm}$, firm ground underneath | `LOW_GRASS` (Cost: 20) |
| **Gravel & Scree** | Driveways, railway margins, loose stone paths | Crushed stone ($2 - 5\text{ cm}$), loose rolling stones | `GRAVEL` (Cost: 25) |
| **High Vegetation** | Unmown field edges, tall weeds, thorny scrub | Foliage height $> 30\text{ cm}$ concealing ground hazards | `HIGH_VEGETATION` (Cost: 80) |
| **Rocks & Boulders** | Quarry borders, landscaped rockeries, rocky trails | Protruding rocks ($> 15\text{ cm}$ height), rock piles | `OBSTACLE_SOLID` (Cost: 100) |
| **Mud & Wet Clay** | Saturated soil near runoff ditches, water taps | Viscous wet mud, tire ruts, surface slippage zones | `WATER_PUDDLE` (Cost: 90) |
| **Water / Puddles** | Rain pools, drainage channels, irrigation runoff | Specular sky reflections, surface ripples, submerged ground | `WATER_PUDDLE` (Cost: 90) |

---

## 6. Challenging Topologies & Geometry Scenarios

To test the depth geometry node and visual SLAM without a physical robot, collect specific geometric failure modes:

1. **Narrow Trail Passages:**
   - Record narrow corridors ($0.8\text{ m} - 1.2\text{ m}$ width) bounded by thick brush on both sides or large rocks.
   - *Test Objective:* Verifies local planner does not halt prematurely when clearance is tight.
2. **Negative Obstacles (Drop-offs & Ditches):**
   - Approach an erosion ditch, drainage trench, or downward concrete curb from $5\text{ m}$ away until reaching the brink.
   - *Test Objective:* Validates depth geometry node detects ground drop ($> 15\text{ cm}$) before the vehicle moves forward.
3. **Uneven Hummocks & Slopes:**
   - Record walking up and down an incline ramp ($10^{\circ} - 20^{\circ}$ slope) and traversing undulating terrain mounds.
   - *Test Objective:* Verifies normal estimation in `sih_depth` correctly distinguishes steep non-traversable banks from flat paths.
4. **Sudden Obstacle Intrusion:**
   - Walk toward an obstacle (person walking across path, box placed on trail, fallen log) appearing suddenly from behind foliage.
   - *Test Objective:* Benchmarks safety watchdog emergency stop (E-STOP) deceleration.

---

## 7. Operator Movement & Emulation Guidelines

1. **Pace:** Walk at a slow, measured speed of **$0.5\text{ m/s} - 0.8\text{ m/s}$** ($\approx 2 - 3\text{ km/h}$). Avoid fast walking ($> 1.5\text{ m/s}$).
2. **Gait:** Use a "ninja walk" or heel-to-toe roll with bent knees to minimize vertical camera bobbing.
3. **Trajectory Shapes:**
   - **Type 1 (Straight Line):** $20\text{ m} - 30\text{ m}$ continuous straight path.
   - **Type 2 (Gentle S-Curve):** Smooth meandering motion along trail contours.
   - **Type 3 (Choke Point & Turnaround):** Enter a narrow area, perform a smooth $90^{\circ}$ or $180^{\circ}$ yaw rotation, and exit.
4. **Recording Duration:** Limit individual recording runs to **$60 - 120\text{ seconds}$** ($600 - 1200$ frames at 10 Hz). Short, focused runs prevent massive unindexed video files and ease labeling.

---

## 8. Calibration & Pre-Flight Checklist

Before capturing data on location, perform the following sequence:

- [ ] **Lens Cleaning:** Inspect and clean camera lens using microfiber cloth (fingerprint smudges cause severe halo glare under sunlight).
- [ ] **Checkerboard Recording:** Record a 15-second sequence of an $8 \times 6$ checkerboard calibration target held at distances from $0.5\text{ m}$ to $3.0\text{ m}$ across the field of view. This computes the exact $K$ matrix (`CameraInfo`).
- [ ] **Rig Geometry Check:** Measure distance from lens center to ground ($0.50\text{ m} \pm 0.05\text{ m}$) using a tape measure; verify $12^{\circ}$ tilt using a smartphone digital inclinometer app.
- [ ] **Settings Lock:** Confirm Shutter Speed ($1/250\text{ s}$ or faster), ISO ($100-400$), and White Balance are set to Manual Lock.

---

## 9. Session Directory Structure & Metadata Logging

Save each recorded session following the standardized directory schema:

```
datasets/raw/session_YYYYMMDD_HHMMSS_<site>_<terrain>_<lighting>/
├── metadata.json
├── calibration/
│   └── checkerboard.mp4
├── video/
│   └── outdoor_run.mp4
└── notes.txt
```

### Example `metadata.json`:
```json
{
  "session_id": "session_20260920_143000_campus_gravel_sun",
  "date": "2026-09-20",
  "time": "14:30:00 IST",
  "location": "North Campus Trail, BEL Facility",
  "gps_anchor": {"lat": 13.0254, "lon": 77.5482},
  "rig": {
    "type": "handheld_monopod_gimbal",
    "camera_height_m": 0.50,
    "pitch_deg": -12.0
  },
  "camera": {
    "sensor": "Wide RGB (Sony IMX)",
    "resolution": [1920, 1080],
    "fps": 30.0,
    "shutter_speed": "1/500",
    "white_balance": "5500K"
  },
  "environment": {
    "weather": "Sunny, clear sky",
    "lighting_regime": "direct_noon_sunlight",
    "primary_terrain": ["TRAVERSABLE_DIRT", "GRAVEL"],
    "hazards_present": ["loose_scree", "deep_puddle_edge", "high_weeds"]
  }
}
```
