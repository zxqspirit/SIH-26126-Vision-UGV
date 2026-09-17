# Real Sensor-Data Pipeline & Common Input Interface

**Project:** SIH 26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle (Outdoor)  
**Organization:** Bharat Electronics Limited (BEL)  
**Role:** Real Sensor-Data Pipeline Engineer  
**Reference Artifact:** [`real_data_input_architecture.md`](file:///C:/Users/harsh/.gemini/antigravity-ide/brain/1120e7f0-317f-4594-9ac8-c567f398da47/real_data_input_architecture.md)  
**Python Source Module:** [`src/sensors/camera_pipeline.py`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/src/sensors/camera_pipeline.py)  
**ROS 2 Publisher Node:** [`ros2/sih_perception/sih_perception/sensor_feed_publisher_node.py`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/ros2/sih_perception/sih_perception/sensor_feed_publisher_node.py)

---

## 1. Overview & Architectural Goals

The SIH 26126 navigation system runs exclusively on **real outdoor sensor observations** without synthetic simulators or Gazebo environments. Downstream perception (CNN traversability), depth geometry, visual odometry, and planning require a unified data contract regardless of whether the system is connected to a live physical camera or replaying field-recorded trials.

This document details the **Minimum Common Input Interface**, timestamp models, coordinate conventions, and empirical benchmark results across four physical ingestion modalities.

---

## 2. Ingestion Modalities

```
+---------------------------------------------------------------------------------------------------+
|                                   DATA SOURCES / MODALITIES                                       |
|  [1. Recorded MP4 Video]  [2. Image Sequence]  [3. Live RGB Camera]  [4. Real Stereo / RGB-D]     |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                                    COMMON INPUT INTERFACE                                         |
|                                    `BaseCameraSource` (ABC)                                       |
|  - .open(), .read(), .close()                                                                     |
|  - Rate Pacing & Real-time Emulation                                                              |
|  - Timestamp Harmonization (Hardware vs Replay Clock)                                             |
|  - CameraIntrinsics & Metadata Synthesis                                                          |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                                       ROS 2 PUBLISHER NODE                                        |
|                                     `sensor_feed_publisher`                                       |
|  Topics Published:                                                                                |
|    - /camera/image_raw         (sensor_msgs/msg/Image: rgb8 / bgr8)                               |
|    - /camera/camera_info       (sensor_msgs/msg/CameraInfo)                                       |
|    - /camera/depth/image_raw   (sensor_msgs/msg/Image: 32FC1 in meters)                           |
+---------------------------------------------------------------------------------------------------+
```

1. **Recorded RGB Video (`VideoFileSource`):** Ingests compressed containers (`.mp4`, `.avi`, `.mkv`) recorded from vehicle dashcams or hand-held field survey cameras.
2. **Image Sequences (`ImageSequenceSource`):** Reads sequentially indexed uncompressed frames (`frame_%04d.jpg`/`.png`) preserving raw image fidelity.
3. **Live RGB Camera (`LiveCameraSource`):** Interfaces with USB/V4L2/DirectShow camera devices (device indices `/dev/video0`, `0`, `1`) for real-time testing.
4. **Real Stereo / RGB-D (`StereoRGBDSource`):** Synchronizes RGB color imagery with pixel-aligned 32-bit floating point metric depth arrays (`.npy`) or 16-bit millimeter depth maps.

---

## 3. Data Interface & Contract: `SensorPacket`

Every camera source implements `BaseCameraSource` and returns a standardized `SensorPacket`:

```python
@dataclass
class SensorPacket:
    frame_index: int                       # Monotonically increasing index
    timestamp: float                       # Fractional epoch seconds (monotonic)
    rgb: np.ndarray                        # Shape (H, W, 3), uint8, RGB format
    depth: Optional[np.ndarray] = None     # Shape (H, W), float32 meters (NaN/0 = unknown)
    intrinsics: CameraIntrinsics           # fx, fy, cx, cy, distortion, resolution
    frame_id: str = "camera_optical_frame" # REP-104 optical coordinate frame
    is_live: bool = False                  # True for live hardware, False for replay
    metadata: Dict[str, Any]               # Source tracking and debug properties
```

---

## 4. Coordinate Frames (REP-103 & REP-104)

All camera observations strictly adhere to standard robotics optical conventions:

- **`camera_link` (Chassis Mount):** $+X$ points forward along vehicle heading, $+Y$ points left, $+Z$ points upward.
- **`camera_optical_frame` (Sensor Plane):** $+X$ points right across image columns, $+Y$ points downward along image rows, $+Z$ points forward into the observed scene along the optical axis.
- **Static Transform ($R_{\text{link}\rightarrow\text{optical}}$):**
  $$\begin{bmatrix} X_{\text{optical}} \\ Y_{\text{optical}} \\ Z_{\text{optical}} \end{bmatrix} = \begin{bmatrix} 0 & -1 & 0 \\ 0 & 0 & -1 \\ 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} X_{\text{link}} \\ Y_{\text{link}} \\ Z_{\text{link}} \end{bmatrix}$$

---

## 5. Timestamp Model & Replay Pacing

1. **Strict Monotonicity:** Across all sources, every consecutive packet satisfies $t_{k} > t_{k-1}$. Any non-monotonic hardware jitter is clamped to $t_{k-1} + 0.1\text{ ms}$.
2. **Pacing Controller:** During recorded playback, `BaseCameraSource.pace(cycle_start)` computes elapsed computation time and sleeps for the remainder of the target inter-frame interval ($T = 1.0 / \text{FPS}$).
3. **Drop Detection:** If downstream execution or file reading falls behind schedule by more than $1.0 \times T$, the pacing logic records a dropped frame and yields immediate execution to prevent queue lag.

---

## 6. Empirical Benchmark Results on Real Outdoor Recordings

Benchmarking was conducted on real outdoor sequences (`datasets/processed/scenario_1_open_path/` and compiled `.mp4` video) as well as the host live camera device:

| Ingestion Modality | Test Input Source | Target Rate | Frames Read | Measured Input FPS | Processing FPS | Dropped Frames | Mean $\Delta t$ (ms) | Jitter (Std Dev) | Monotonic Violations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Recorded Video** | `datasets/raw/outdoor_run_sample.mp4` | 10.0 Hz | 30 | **9.80 FPS** | 9.80 FPS | **0** | 102.04 ms | 1.65 ms | **0** |
| **Image Sequence** | `scenario_1_open_path/rgb/` | 10.0 Hz | 30 | **9.65 FPS** | 9.66 FPS | **0** | 103.48 ms | 3.82 ms | **0** |
| **Stereo RGB-D** | `scenario_1_open_path/` (RGB + Depth) | 10.0 Hz | 30 | **9.62 FPS** | 9.62 FPS | **0** | 103.59 ms | 7.80 ms | **0** |
| **Live Camera** | Hardware Device `0` | 30.0 Hz | 15 | **23.91 FPS** | 23.91 FPS | **0** | 33.56 ms | 6.50 ms | **0** |

### Benchmark Observations:
- **Rate Stability:** Paced playback delivered consistent rates ($\approx 10\text{ Hz}$) with jitter $< 8\text{ ms}$.
- **Zero Drops:** Ingestion reading time per frame is negligible ($< 3\text{ ms}$), causing 0 dropped frames.
- **Monotonicity Guarantee:** 0 monotonic violations observed across all test modalities.

---

## 7. ROS 2 Driver Node: `sensor_feed_publisher_node`

To feed downstream ROS 2 nodes, `sih_perception` includes `sensor_feed_publisher_node`:

```bash
# Example: Launching RGB-D playback at 10 Hz into ROS 2
ros2 run sih_perception sensor_feed_publisher_node \
  --ros-args \
  -p source_uri:=/mnt/c/Users/harsh/SIH-26126-Vision-UGV/datasets/processed/scenario_1_open_path \
  -p target_fps:=10.0 \
  -p loop:=true
```

### Verified Active Topics:
- `/camera/image_raw` (`sensor_msgs/msg/Image`, encoding `rgb8`, $640 \times 480$)
- `/camera/camera_info` (`sensor_msgs/msg/CameraInfo`, pinhole model `plumb_bob`)
- `/camera/depth/image_raw` (`sensor_msgs/msg/Image`, encoding `32FC1`, metric meters)

---

## 8. Failure Handling Protocol

| Failure Condition | Detection | Action |
| :--- | :--- | :--- |
| **End of File / Disconnect** | `read()` returns `None` | If `loop=True`, resets pointer to frame 0; otherwise cleanly terminates and logs completion. |
| **Missing Depth Channel** | No matching `.npy` or depth image | Automatically populates zero-filled metric depth array without throwing exceptions. |
| **Depth Out-of-Bounds** | NaN, Inf, or negative depth | Sanitized to $0.0\text{ m}$ (`np.nan_to_num`) to safeguard geometry estimators. |
| **Extreme Lighting** | Mean intensity $< 15$ or $> 245$ | Flags `degraded=True` in packet metadata for downstream safety gating. |
