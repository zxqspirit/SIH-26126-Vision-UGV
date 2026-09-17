# ROS 2 Package Architecture & Subsystem Interface Contracts

**Project:** SIH 26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle (Outdoor)  
**Organization:** Bharat Electronics Limited (BEL)  
**Middleware:** ROS 2 Jazzy Jalisco (Ubuntu 24.04 / WSL2)  
**Architectural Constraint:** Strict Software-Only outdoor navigation on real sensor data. **Zero simulation / No Gazebo**.  
**Reference Artifact:** [`sih_ros_package_architecture.md`](file:///C:/Users/harsh/.gemini/antigravity-ide/brain/1120e7f0-317f-4594-9ac8-c567f398da47/sih_ros_package_architecture.md)

---

## 1. System Overview & Package Topology

The vision-based autonomous navigation stack for SIH 26126 decomposes the end-to-end autonomy flow into **8 minimum, tightly scoped ROS 2 packages**. 

Simulation packages (`sih_simulation`), Gazebo models, and synthetic plugins are **explicitly excluded** per project mandate; validation is performed exclusively against real outdoor datasets, recorded rosbags, and live camera feeds.

```
+-------------------------------------------------------------------------------------------------------+
|                                    REAL SENSOR INPUT (Cameras)                                        |
|                     Topics: /camera/image_raw, /camera/depth/image_raw, /camera/camera_info            |
+-------------------------------------------------------------------------------------------------------+
                │                                                       │
                ▼                                                       ▼
+-------------------------------+                       +-------------------------------+
|        sih_perception         |                       |           sih_depth           |
|  traversability_classifier    |                       |         depth_geometry        |
|  - Lightweight CNN Inference  |                       |  - PointCloud & Normal Vector |
|  - Semantic Segmentation      |                       |  - Slope & Step Hazards       |
+-------------------------------+                       +-------------------------------+
                │                                                       │
                │ /perception/traversability_mask                       │ /depth/geometry_hazards
                │ /perception/confidence                                │ /depth/slope_grid
                └───────────────────────────┬───────────────────────────┘
                                            ▼
                        +---------------------------------------+
                        |              sih_fusion               |
                        |            terrain_fusion             |
                        |  - Spatio-temporal sync               |
                        |  - Cost blending: C = w_s*C_s + w_g*C_g|
                        +---------------------------------------+
                                            │
                                            │ /costmap/traversability_grid (OccupancyGrid)
                                            │ /costmap/fusion_confidence
                                            ▼
+-------------------------------+   +-----------------------------------+
|       sih_localization        |   |          sih_navigation           |
|        visual_odometry        |   |           local_planner           |
|  - Feature tracking (ORB)     |   |  - Lattice / DWA trajectory search|
|  - TF2: odom -> base_link     |   |  - Evaluates costmap paths        |
+-------------------------------+   +-----------------------------------+
                │                                     │
                │ /localization/odometry              │ /navigation/cmd_vel_raw
                └───────────────────┬─────────────────┘
                                    ▼
                        +---------------------------------------+
                        |              sih_safety               |
                        |           safety_supervisor           |
                        |  - Deadman timer & Heartbeat check    |
                        |  - Confidence-gated speed scaling     |
                        |  - Emergency Stop (E-STOP) override   |
                        +---------------------------------------+
                                    │
                                    │ /cmd_vel (Final Safe Motion Command)
                                    ▼
+-------------------------------------------------------------------------------------------------------+
|       sih_visualization        |                               sih_bringup                            |
|        telemetry_bridge        |                        ugv_system.launch.py                          |
|  - RViz2 Markers & Displays    |  - Coordinates lifecycle & parameters of all 7 packages              |
|  - Dashboard WebSocket Bridge  |  - Zero simulation components                                        |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Package Specifications

### 1. `sih_perception`
- **Purpose:** Ingests live or replayed camera RGB video frames, executes lightweight CNN terrain segmentation, categorizes pixels into traversability states, and outputs probabilistic confidence ratings.
- **Node(s):**
  - `traversability_classifier_node`: Executes inference loop and publishes terrain class masks.
- **Inputs:**
  - `/camera/image_raw` (`sensor_msgs/msg/Image`)
  - `/camera/camera_info` (`sensor_msgs/msg/CameraInfo`)
- **Outputs:**
  - `/perception/traversability_mask` (`sensor_msgs/msg/Image`)
  - `/perception/confidence` (`std_msgs/msg/Float32`)
- **Dependencies:** `rclpy`, `sensor_msgs`, `std_msgs`, `cv_bridge`
- **Testing Responsibility:**
  - Inference latency benchmarking ($< 50\text{ ms}$).
  - Label index validity (ensuring no out-of-range class indices).
  - Robustness to dark or missing frames.

---

### 2. `sih_depth`
- **Purpose:** Converts raw depth data (from stereo disparity or RGB-D sensors) into 3D metric representations, estimates ground plane elevation, calculates local surface slopes, and identifies structural hazards (drop-offs, steps, boulders).
- **Node(s):**
  - `depth_geometry_node`: Computes surface normals, detects geometric steps/inclines, and projects spatial hazards.
- **Inputs:**
  - `/camera/depth/image_raw` (`sensor_msgs/msg/Image`)
  - `/camera/camera_info` (`sensor_msgs/msg/CameraInfo`)
- **Outputs:**
  - `/depth/geometry_hazards` (`sensor_msgs/msg/Image`)
  - `/depth/pointcloud` (`sensor_msgs/msg/PointCloud2`)
  - `/depth/slope_grid` (`nav_msgs/msg/OccupancyGrid`)
- **Dependencies:** `rclpy`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `tf2_ros`, `tf2_geometry_msgs`
- **Testing Responsibility:**
  - Rejection of invalid depth values (NaN, infinity, zero-range).
  - Step-height threshold validation on synthetic and recorded ramps.
  - Coordinate projection precision against known calibration matrices.

---

### 3. `sih_fusion`
- **Purpose:** Spatio-temporally synchronizes 2D visual traversability masks with 3D geometric terrain hazards to synthesize a single, unified 2.5D Traversability Costmap.
- **Node(s):**
  - `terrain_fusion_node`: Merges multi-modal inputs, computes weighted cost formulas, and inflates obstacle boundaries.
- **Inputs:**
  - `/perception/traversability_mask` (`sensor_msgs/msg/Image`)
  - `/perception/confidence` (`std_msgs/msg/Float32`)
  - `/depth/geometry_hazards` (`sensor_msgs/msg/Image`)
  - `/depth/slope_grid` (`nav_msgs/msg/OccupancyGrid`)
- **Outputs:**
  - `/costmap/traversability_grid` (`nav_msgs/msg/OccupancyGrid`)
  - `/costmap/fusion_confidence` (`std_msgs/msg/Float32`)
- **Dependencies:** `rclpy`, `sensor_msgs`, `nav_msgs`, `std_msgs`, `message_filters`
- **Testing Responsibility:**
  - Synchronization under message jitter and dropped frames.
  - Costmap bounds checking (cost values strictly within $[0, 100]$).
  - Geometric hazard priority rule (drop-offs must mark costmap cells as lethal regardless of visual appearance).

---

### 4. `sih_localization`
- **Purpose:** Tracks 6-DoF vehicle ego-motion in real time using Monocular/Stereo Visual Odometry (VO), publishing dynamic coordinate frames from `odom` to `base_link`.
- **Node(s):**
  - `visual_odometry_node`: Computes frame-to-frame feature displacement and vehicle pose.
  - `pose_diagnostics_node`: Evaluates feature match density and flags tracking degradation.
- **Inputs:**
  - `/camera/image_raw` (`sensor_msgs/msg/Image`)
  - `/camera/camera_info` (`sensor_msgs/msg/CameraInfo`)
- **Outputs:**
  - `/localization/odometry` (`nav_msgs/msg/Odometry`)
  - TF2 transform broadcast: `odom` $\rightarrow$ `base_link`
  - `/localization/status` (`std_msgs/msg/String`)
- **Dependencies:** `rclpy`, `nav_msgs`, `geometry_msgs`, `sensor_msgs`, `tf2_ros`
- **Testing Responsibility:**
  - Continuity validation: trajectory must not exhibit instantaneous teleportation jumps.
  - Quaternions must remain strictly normalized ($|q| = 1.0$).
  - Degradation detection under feature-poor or rapidly moving scenes.

---

### 5. `sih_navigation`
- **Purpose:** Generates collision-free path trajectories through the traversability costmap toward navigation waypoints, balancing path length against terrain cost.
- **Node(s):**
  - `local_planner_node`: Evaluates candidate trajectory rollouts (Lattice / Dynamic Window Approach) on the costmap and produces motion commands.
- **Inputs:**
  - `/costmap/traversability_grid` (`nav_msgs/msg/OccupancyGrid`)
  - `/localization/odometry` (`nav_msgs/msg/Odometry`)
  - `/ugv/goal_pose` (`geometry_msgs/msg/PoseStamped`)
- **Outputs:**
  - `/navigation/planned_path` (`nav_msgs/msg/Path`)
  - `/navigation/cmd_vel_raw` (`geometry_msgs/msg/Twist`)
- **Dependencies:** `rclpy`, `nav_msgs`, `geometry_msgs`, `std_msgs`
- **Testing Responsibility:**
  - Obstacle avoidance: planned path must not traverse cells with cost $\ge 80$.
  - Goal arrival behavior: velocities must decelerate smoothly within goal tolerance ($< 0.3\text{ m}$).
  - Planning loop deadline: cycle time must stay under $50\text{ ms}$ (20 Hz).

---

### 6. `sih_safety`
- **Purpose:** The final fail-safe supervisor for autonomous operations. Constantly monitors watchdog heartbeats, perception confidence levels, and localization status; applies dynamic velocity derating and triggers immediate emergency stops (E-STOP) when safety thresholds are breached.
- **Node(s):**
  - `safety_supervisor_node`: Evaluates safety metrics and gates raw velocity commands onto the final hardware `/cmd_vel` topic.
- **Inputs:**
  - `/navigation/cmd_vel_raw` (`geometry_msgs/msg/Twist`)
  - `/perception/confidence` (`std_msgs/msg/Float32`)
  - `/costmap/fusion_confidence` (`std_msgs/msg/Float32`)
  - `/localization/status` (`std_msgs/msg/String`)
  - `/ugv/heartbeat` (`std_msgs/msg/String`)
- **Outputs:**
  - `/cmd_vel` (`geometry_msgs/msg/Twist`): Final verified vehicle motion command.
  - `/safety/status` (`std_msgs/msg/String`): Diagnostic health status (`AUTONOMOUS_OK`, `CONFIDENCE_DERATED`, `ESTOP_ACTIVE`).
- **Dependencies:** `rclpy`, `geometry_msgs`, `std_msgs`, `diagnostic_msgs`
- **Testing Responsibility:**
  - Heartbeat deadman timer: command velocity must reset to zero within $200\text{ ms}$ of heartbeat loss.
  - Confidence derating: proportional speed reductions when confidence drops below $0.6$.
  - E-STOP priority: zero-velocity command overrides all planner inputs during obstacle alerts.

---

### 7. `sih_visualization`
- **Purpose:** Telemetry aggregator and bridge node. Subscribes to internal autonomy topics, prepares lightweight JSON/WebSocket streams for the web-based laptop dashboard (`localhost:5000`), and generates RViz2 visual marker arrays.
- **Node(s):**
  - `telemetry_bridge_node`: Serializes ROS 2 topics for dashboard display and publishes marker arrays.
- **Inputs:**
  - `/camera/image_raw` (`sensor_msgs/msg/Image`)
  - `/perception/traversability_mask` (`sensor_msgs/msg/Image`)
  - `/costmap/traversability_grid` (`nav_msgs/msg/OccupancyGrid`)
  - `/localization/odometry` (`nav_msgs/msg/Odometry`)
  - `/navigation/planned_path` (`nav_msgs/msg/Path`)
  - `/cmd_vel` (`geometry_msgs/msg/Twist`)
  - `/safety/status` (`std_msgs/msg/String`)
- **Outputs:**
  - Web dashboard telemetry stream (HTTP/WebSocket on `localhost:5000`).
  - `/visualization/markers` (`visualization_msgs/msg/MarkerArray`).
- **Dependencies:** `rclpy`, `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `visualization_msgs`, `std_msgs`
- **Testing Responsibility:**
  - Telemetry dispatch rate without blocking the ROS 2 executor.
  - JSON serialization schema integrity.

---

### 8. `sih_bringup`
- **Purpose:** System orchestration package containing launch descriptions and configuration parameters. Coordinates the startup sequence of all 7 functional packages. Does **not** include simulation plugins or Gazebo components.
- **Node(s):** None (Configuration and launch orchestration only).
- **Launch Configurations:**
  - `ugv_system.launch.py`: Full end-to-end autonomy pipeline.
  - `perception_stack.launch.py`: Starts camera, perception, depth, and fusion nodes.
  - `dataset_replay.launch.py`: Replays recorded outdoor rosbags while running downstream nodes.
- **Inputs / Outputs:** Manages parameter bindings (`config/ugv_params.yaml`) and namespace remappings.
- **Dependencies:** `rclpy`, `launch`, `launch_ros`, and all functional `sih_*` packages.
- **Testing Responsibility:**
  - Launch file syntax validation via dry-run execution.
  - Correct propagation of parameters from YAML files to child nodes.

---

## 3. Team Ownership Matrix

| Member | Assigned Packages | Focus Areas |
| :--- | :--- | :--- |
| **Member 1 (Perception & AI)** | `sih_perception`, `sih_depth` | Deep learning inference, semantic segmentation, depth geometry, slope hazard detection |
| **Member 2 (Localization & Robotics)** | `sih_localization` | Visual odometry, pose tracking, TF2 coordinate transforms (`odom` $\rightarrow$ `base_link`) |
| **Member 3 (Planning & Control)** | `sih_fusion`, `sih_navigation` | Multi-modal costmap synthesis, obstacle inflation, local path planning, trajectory generation |
| **Member 4 (Safety, Vis & Integration)** | `sih_safety`, `sih_visualization`, `sih_bringup` | Safety supervisor, heartbeat watchdog, dashboard telemetry bridge, master launch files |

---

## 4. Package Dependency Graph

```
                         [ sih_bringup ]
                                │
   ┌───────────────┬────────────┼────────────┬───────────────┐
   ▼               ▼            ▼            ▼               ▼
sih_perception  sih_depth  sih_localization sih_safety  sih_visualization
   │               │            │            ▲               ▲
   └───────┬───────┘            │            │               │
           ▼                    │            │               │
      sih_fusion                │            │               │
           │                    │            │               │
           └───────────┬────────┘            │               │
                       ▼                     │               │
                 sih_navigation ─────────────┘               │
                       │                                     │
                       └─────────────────────────────────────┘
```

---

## 5. Summary Topic Contract Matrix

| Package | Subscribed Topics | Published Topics | Key Message Types |
| :--- | :--- | :--- | :--- |
| `sih_perception` | `/camera/image_raw` | `/perception/traversability_mask`, `/perception/confidence` | `sensor_msgs/Image`, `std_msgs/Float32` |
| `sih_depth` | `/camera/depth/image_raw` | `/depth/geometry_hazards`, `/depth/slope_grid` | `sensor_msgs/Image`, `nav_msgs/OccupancyGrid` |
| `sih_fusion` | `/perception/*`, `/depth/*` | `/costmap/traversability_grid`, `/costmap/fusion_confidence` | `nav_msgs/OccupancyGrid`, `std_msgs/Float32` |
| `sih_localization` | `/camera/image_raw` | `/localization/odometry`, `/localization/status` | `nav_msgs/Odometry`, `std_msgs/String` |
| `sih_navigation` | `/costmap/*`, `/localization/odometry` | `/navigation/planned_path`, `/navigation/cmd_vel_raw` | `nav_msgs/Path`, `geometry_msgs/Twist` |
| `sih_safety` | `/navigation/cmd_vel_raw`, confidences | `/cmd_vel`, `/safety/status` | `geometry_msgs/Twist`, `std_msgs/String` |
| `sih_visualization` | All system topics | `/visualization/markers`, UI Socket | `visualization_msgs/MarkerArray`, `std_msgs/String` |
| `sih_bringup` | Launch arguments | Process lifecycles & configurations | Launch descriptions & YAML configs |
