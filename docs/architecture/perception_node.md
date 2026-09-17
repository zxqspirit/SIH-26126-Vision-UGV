# ROS 2 Perception Node Architecture & Implementation Specification

**Package:** `sih_perception`  
**Node Name:** `traversability_classifier_node`  
**Source Location:** [`ros2/sih_perception/sih_perception/traversability_classifier_node.py`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/ros2/sih_perception/sih_perception/traversability_classifier_node.py)  
**Inference Engine:** [`src/perception/ros_perception_node.py`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/src/perception/ros_perception_node.py)  
**Trained Model:** `MobileNetV3-Large + LR-ASPP` (`models/checkpoints/production_v1/mobilenetv3_lraspp_best.onnx`)  

---

## 1. Executive Summary & Architectural Role

The `traversability_classifier_node` serves as the primary visual perception gateway in the SIH 26126 autonomous UGV stack. It continuously consumes raw optical RGB sensor streams from the forward-facing camera and performs:
1. **Dense Semantic Segmentation:** Pixel-level classification across the 9-class `TerrainClass` ontology.
2. **Deterministic Traversability Cost Mapping:** Real-time lookup of terrain costs on a $[0, 100]$ scale (where $0 = \text{free road}$ and $100 = \text{lethal obstacle}$).
3. **Statistical Perception Confidence:** Dynamic score reflecting prediction certainty and unobserved/degraded pixel ratios.
4. **Frame-Accurate Synchronization:** Strict preservation of input header timestamps and optical frame IDs for downstream 3D fusion.

```
+---------------------------------------------------------------------------------------------------------+
|                                    PERCEPTION NODE DATAFLOW PIPELINE                                    |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   [/camera/image_raw] (sensor_msgs/Image)                                                               |
|            |                                                                                            |
|            v                                                                                            |
|   +-------------------------------------------------------------------------------------------------+   |
|   | TraversabilityClassifierNode                                                                    |   |
|   |                                                                                                 |   |
|   |  1. Input Validation Gate:                                                                      |   |
|   |     - Verifies non-null data, supported encoding (rgb8/bgr8), and valid dimensions             |   |
|   |     - Graceful degradation on corrupted frames (returns safe cost 50 with confidence 0.0)      |   |
|   |                                                                                                 |   |
|   |  2. Preprocessing:                                                                              |   |
|   |     - Bilinear resize to configurable model input (512x512)                                     |   |
|   |     - Channel standardization: ImageNet mean/std normalization                                  |   |
|   |                                                                                                 |   |
|   |  3. Model Execution:                                                                            |   |
|   |     - MobileNetV3-Large + LR-ASPP (3.22M parameters)                                            |   |
|   |     - Dual backend support: ONNX Runtime (default) or PyTorch Native                            |   |
|   |                                                                                                 |   |
|   |  4. Post-processing & Metric Telemetry:                                                         |   |
|   |     - Nearest-neighbor upsampling back to camera native resolution (640x480)                    |   |
|   |     - Array lookup into TERRAIN_COST_TABLE [0..100]                                             |   |
|   |     - Confidence calculation penalized by UNKNOWN pixel ratio                                   |   |
|   |     - Runtime latency profiling and continuous rolling FPS computation                          |   |
|   +-------------------------------------------------------------------------------------------------+   |
|            |                              |                             |                               |
|            v                              v                             v                               v
|  [/perception/              [/perception/                 [/perception/                 [/perception/   |
|   segmentation_mask]         terrain_cost]                 confidence]                   colored_mask]  |
|  (sensor_msgs/Image)        (sensor_msgs/Image)           (std_msgs/Float32)            (sensor_msgs/Image)
|  Encoding: mono8            Encoding: mono8               Scalar: [0.0, 1.0]            Encoding: rgb8  |
|  Values: 0..8               Values: 0..100                Safety gate input             Visual monitor  |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Topic & Interface Specification

### 2.1 Subscriptions
| Topic Name | Message Type | Purpose | Quality of Service (QoS) |
| :--- | :--- | :--- | :--- |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | Ingests optical RGB camera frames (`rgb8` or `bgr8`) | `SensorData` (Best Effort, Queue=5) |

### 2.2 Publications
| Topic Name | Message Type | Encoding | Semantics & Format |
| :--- | :--- | :--- | :--- |
| `/perception/segmentation_mask` | `sensor_msgs/msg/Image` | `mono8` | Single-channel 8-bit image with integer class IDs ($0$ to $8$). |
| `/perception/terrain_cost` | `sensor_msgs/msg/Image` | `mono8` | 2D traversability cost ($0 = \text{free}$, $100 = \text{lethal}$). |
| `/perception/confidence` | `std_msgs/msg/Float32` | N/A | Perception certainty scalar in $[0.0, 1.0]$. |
| `/perception/colored_mask` | `sensor_msgs/msg/Image` | `rgb8` | Colorized terrain map for operator display / telemetry dashboard. |

### 2.3 Strict Timestamp Propagation Rule
All output messages copy the exact `header.stamp` and `header.frame_id` from the triggering `/camera/image_raw` frame:
```python
mask_msg.header = in_header
cost_msg.header = in_header
```
This guarantees zero temporal jitter and enables synchronized projection in `sih_depth` and `sih_fusion`.

---

## 3. Node Configuration Parameters

Declared via standard ROS 2 parameters:

| Parameter Name | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `string` | `models/checkpoints/production_v1/mobilenetv3_lraspp_best.onnx` | Path to ONNX engine or PyTorch checkpoint |
| `model_backend` | `string` | `onnx` | Inference engine: `onnx` (recommended) or `pytorch` |
| `input_width` | `int` | `512` | Model input spatial width |
| `input_height` | `int` | `512` | Model input spatial height |
| `device` | `string` | `auto` | Execution device: `auto`, `cuda`, or `cpu` |
| `confidence_threshold`| `double` | `0.60` | Minimum nominal confidence before emitting warnings |
| `publish_colored_mask`| `bool` | `true` | Toggles publication of `/perception/colored_mask` |

---

## 4. Traversability Cost & Confidence Formulation

### 4.1 Cost Lookup Table
Each pixel class ID $c \in [0, 8]$ maps directly to a deterministic cost:

```python
TERRAIN_COST_TABLE = np.array([
    50,   # 0: UNKNOWN (Cautious neutral cost)
    5,    # 1: PAVED_ROAD (Optimal asphalt/concrete corridor)
    15,   # 2: TRAVERSABLE_DIRT (Primary UGV dirt track)
    30,   # 3: LOW_GRASS (Safe low ground cover)
    25,   # 4: GRAVEL (Loose pebble terrain)
    80,   # 5: HIGH_VEGETATION (Non-traversable tall brush)
    100,  # 6: OBSTACLE_SOLID (Lethal collision hazard)
    90,   # 7: WATER_PUDDLE (Near-lethal submergence risk)
    100,  # 8: DYNAMIC_OBSTACLE (Lethal moving obstacle)
], dtype=np.uint8)
```

### 4.2 Perception Confidence Formulation
Perception confidence balances raw softmax class probability with spatial unobserved ratio:

$$C_{\text{raw}} = \frac{1}{H \cdot W} \sum_{x, y} \max_{c} P(c \mid x, y)$$

$$C_{\text{final}} = C_{\text{raw}} \cdot \left(1.0 - 0.7 \cdot \frac{N_{\text{unknown}}}{H \cdot W}\right)$$

When $C_{\text{final}} < \text{confidence\_threshold}$, the node logs a rate-limited warning, alerting downstream `sih_safety` to initiate speed reduction or cautious maneuvers.

---

## 5. Invalid-Frame Fault Tolerance & Safety Degradation

The node implements bulletproof input guards:

1. **Null / Zero-Sized Buffer:**
   - Detects empty message payloads or zero dimensions without crashing.
   - Emits: `[WARN] Received empty camera frame; publishing fallback safe costmap`.
   - Publishes safe fallback costmap (all pixels cost $50$) with confidence $= 0.0$.
2. **Corrupt / Unsupported Encodings:**
   - Safely parses `rgb8`, `bgr8`, `mono8`, and `rgba8`.
   - If buffer unpack fails, catches exceptions cleanly and issues fallback output.
3. **Graceful Lifecycle Shutdown:**
   - Unbinds publishers/subscriptions and flushes telemetry statistics on `destroy_node()`.

---

## 6. Empirical Benchmarks on Real Outdoor Video Data

The node was benchmarked on real outdoor camera video (`datasets/raw/outdoor_run_sample.mp4`, 640×480 @ 10 FPS, real dirt trail, grass, trees, and sunlight):

```
+-------------------------------------------------------------------------------------------------------+
|                                    PERCEPTION NODE RUNTIME BENCHMARK                                  |
+--------------------------+------------------------------------+---------------------------------------+
| Benchmark Metric         | PyTorch Native Backend             | ONNX Runtime Backend (Champion)       |
+--------------------------+------------------------------------+---------------------------------------+
| Mean Total Latency       | 92.16 ms                           | 47.71 ms                              |
| Median Latency           | 90.13 ms                           | 46.14 ms                              |
| p95 Latency              | 106.43 ms                          | 55.04 ms                              |
| Min / Max Latency        | 84.2 ms / 116.5 ms                 | 42.5 ms / 66.9 ms                     |
| Latency Breakdown        | Pre: 7.6ms | Infer: 76.3ms | Post: 8.2ms | Pre: 5.3ms | Infer: 34.5ms | Post: 7.9ms |
| Continuous Throughput    | 10.9 FPS                           | 21.0 FPS                              |
| Real-Time Target (>15FPS)| Acceptable                         | EXCEEDED (> 20 FPS on CPU alone)      |
| Dropped / Corrupt Frames | 0 crashes (100% handled)           | 0 crashes (100% handled)              |
| Mean Confidence Score    | 0.608                              | 0.608                                 |
+--------------------------+------------------------------------+---------------------------------------+
```

### Key Architectural Finding:
ONNX Runtime cuts inference latency by **54.8%** ($76.3\text{ ms} \rightarrow 34.5\text{ ms}$), boosting overall node throughput to **21.0 FPS** on CPU alone, exceeding the real-time threshold required for high-speed off-road autonomous navigation.
