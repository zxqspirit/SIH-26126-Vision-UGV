# ROS 2 Fundamentals & Hands-On Engineering Lab

**Project:** SIH 26126 — Vision Based Autonomous Navigation for Unmanned Ground Vehicle (Outdoor)  
**Organization:** Bharat Electronics Limited (BEL)  
**Target Environment:** ROS 2 Jazzy Jalisco (Ubuntu 24.04 / WSL2)  
**Reference Artifact:** [`ros2_beginner_map.md`](file:///C:/Users/harsh/.gemini/antigravity-ide/brain/1120e7f0-317f-4594-9ac8-c567f398da47/ros2_beginner_map.md)  
**Lab Package:** [`sih_starter`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/learning/sih_starter)

---

## 1. Executive Summary & Educational Intent

In the SIH 26126 vision-based autonomous navigation stack, our goal is to achieve autonomous traversal in challenging outdoor terrain without physical robotics hardware in our immediate development loop and **without simulation engines (no Gazebo)**. All processing runs on real sensor data (RGB/RGB-D camera feeds, visual odometry, traversability costmaps, and command outputs).

To coordinate this distributed pipeline, we rely on **ROS 2 (Robot Operating System 2)** as our robotics middleware.

This document serves as the introductory lab manual and conceptual reference for team members new to ROS 2. It covers:
1. The **14 foundational building blocks** of ROS 2.
2. A complete, harmless, and fully verified hands-on publisher/subscriber exercise with a unified launch file (`sih_starter`).
3. Exact verification steps using standard ROS 2 CLI diagnostic utilities.

---

## 2. The 14 Core ROS 2 Concepts Explained

```
+-----------------------------------------------------------------------------------+
|                                 ROS 2 WORKSPACE                                   |
|   src/ (source code)  |  build/ (intermediate)  |  install/ (binaries & setup)    |
+-----------------------------------------------------------------------------------+
                                          |
                                 built using colcon
                                          v
+-----------------------------------------------------------------------------------+
|                                ROS 2 PACKAGES                                     |
|           (e.g., sih_starter, sih_perception, sih_navigation)                     |
+-----------------------------------------------------------------------------------+
                                          |
                      +-------------------+-------------------+
                      |                                       |
                      v                                       v
        +---------------------------+           +---------------------------+
        |     NODE: Publisher       |           |     NODE: Subscriber      |
        |     (beacon_publisher)    |           |     (beacon_subscriber)   |
        |  - Parameters (YAML/CLI)  |           |  - Execution Callback     |
        |  - Timer loop             |           |  - Event-driven           |
        +---------------------------+           +---------------------------+
                      |                                       ^
                      |       TOPIC: /ugv/heartbeat           |
                      +======== [MESSAGE: String] ===========+
```

### 1. Node
- **Definition:** An independent executable process in ROS 2 that carries out a dedicated computation (e.g., streaming camera images, computing visual odometry, running path planning).
- **Rationale:** Modular isolation. If one algorithm crashes, the rest of the robotics stack remains stable and can trigger safe fallback routines.
- **Mental Model:** A specialized crew member on a bridge.
- **CLI Commands:**
  ```bash
  ros2 node list
  ros2 node info /<node_name>
  ```

### 2. Topic
- **Definition:** A named, unidirectional data channel over which nodes stream messages asynchronously.
- **Rationale:** Strict decoupling. The producer doesn't know or care who is consuming the data; consumers don't care who produces it.
- **Mental Model:** A designated radio station frequency.
- **CLI Commands:**
  ```bash
  ros2 topic list
  ros2 topic hz /<topic_name>
  ```

### 3. Publisher
- **Definition:** An object inside a node responsible for packing data into a message format and transmitting it onto a topic.
- **Rationale:** Provides non-blocking data distribution to zero, one, or many listening nodes.
- **Mental Model:** The radio host speaking into the transmitter.

### 4. Subscriber
- **Definition:** An object inside a node that registers a callback function to automatically execute whenever a new message lands on a subscribed topic.
- **Rationale:** Event-driven execution without wasteful busy-waiting or manual polling loops.
- **Mental Model:** A listener's radio receiver playing audio as waves arrive.

### 5. Message (`.msg`)
- **Definition:** A strictly typed data definition schema specifying fields and numeric formats (e.g., float64, uint8[], geometry vectors).
- **Examples:**
  - `std_msgs/msg/String`: text payloads.
  - `sensor_msgs/msg/Image`: raw 2D pixel buffers with width, height, encoding, and timestamp.
  - `geometry_msgs/msg/Twist`: linear ($v_x, v_y, v_z$) and angular ($\omega_x, \omega_y, \omega_z$) velocities.
- **CLI Command:**
  ```bash
  ros2 interface show geometry_msgs/msg/Twist
  ```

### 6. Service (`.srv`)
- **Definition:** A synchronous or asynchronous two-way request/response communication channel (Client $\leftrightarrow$ Server).
- **Rationale:** For one-shot commands or queries requiring acknowledgement (e.g., "Reset Visual Odometry", "Calibrate IMU Bias").
- **Mental Model:** A client-server API call (HTTP POST/GET equivalent).
- **CLI Commands:**
  ```bash
  ros2 service list
  ros2 service call /<service_name> <service_type> <arguments>
  ```

### 7. Action (`.action`)
- **Definition:** A goal-oriented communication protocol designed for long-running behaviors. It features three components: **Goal**, **Continuous Feedback**, and **Final Result**, with full preemption (cancellation) capabilities.
- **Rationale:** Navigating a 15-meter outdoor path takes 20 seconds. A service would block the caller; an action provides real-time progress updates and allows aborting if an obstacle appears.
- **Mental Model:** An e-commerce package order with step-by-step delivery tracking.
- **CLI Commands:**
  ```bash
  ros2 action list
  ros2 action send_goal /<action_name> <action_type> <goal_payload>
  ```

### 8. Parameter
- **Definition:** Dynamically reconfigurable configuration values (integers, floats, booleans, strings) stored per node.
- **Rationale:** Tune algorithm parameters (e.g., detection confidence threshold, max UGV linear speed, inflation radius) at runtime without recompiling source code.
- **CLI Commands:**
  ```bash
  ros2 param list
  ros2 param get /<node_name> <parameter_name>
  ros2 param set /<node_name> <parameter_name> <value>
  ```

### 9. Launch File
- **Definition:** A Python, XML, or YAML script that coordinates starting multiple nodes, configuring namespaces, loading parameters, and managing remappings with a single terminal command.
- **Rationale:** Prevents opening dozens of terminals manually to start cameras, detectors, odometry, and planners.
- **CLI Command:**
  ```bash
  ros2 launch <package_name> <launch_script.py>
  ```

### 10. Package
- **Definition:** The fundamental organizational unit of ROS 2 software containing code, build configuration, launch files, and manifest metadata.
- **Core Files:**
  - `package.xml`: Declares package identity, license, and dependencies (`rclpy`, `std_msgs`, etc.).
  - `setup.py` (Python) or `CMakeLists.txt` (C++): Declares installation targets and console script entry points.

### 11. Workspace
- **Definition:** A structured root directory housing development packages. Contains:
  - `src/`: Raw source code repositories and packages.
  - `build/`: Intermediate compilation and bytecode artifacts.
  - `install/`: Assembled binaries, libraries, and environment scripts (`setup.bash`).
  - `log/`: Logging and debug traces from build jobs.

### 12. Colcon
- **Definition:** The standard multi-package build tool (**col**lective **con**struction) used across the ROS 2 ecosystem.
- **Standard Command:**
  ```bash
  colcon build --symlink-install
  ```
  *(The `--symlink-install` flag symlinks Python scripts directly, allowing rapid iteration without needing to re-run colcon on every line edit).*

### 13. TF2 (Transform Library)
- **Definition:** The coordinate frame transformation library in ROS 2 that tracks relative spatial 3D poses (translation + rotation) over time.
- **Relevance for UGV:**
  - `camera_optical_frame` $\rightarrow$ Pixel/depth ray space.
  - `camera_link` $\rightarrow$ Physical sensor mounting point.
  - `base_link` $\rightarrow$ Center of the UGV wheelbase on the ground.
  - `odom` $\rightarrow$ Smooth dead-reckoning local frame.
  - `map` $\rightarrow$ Globally anchored reference frame.
  TF2 allows converting 3D obstacle detections from the camera frame into coordinates the ground vehicle can navigate around.

### 14. rosbag2
- **Definition:** The high-throughput recording and playback tool for ROS 2 topics.
- **Relevance for UGV:** In outdoor robotics, field testing is resource-intensive. `rosbag2` allows recording real outdoor camera video, IMU readings, and odometry once, then replaying those exact message streams offline to develop and benchmark perception algorithms.
- **CLI Commands:**
  ```bash
  ros2 bag record -a
  ros2 bag play <path_to_bag_directory>
  ```

---

## 3. Communication Patterns Decision Matrix

| Pattern | Data Flow | Blocking? | Preemptible? | Ideal Use Case in Outdoor UGV |
| :--- | :--- | :--- | :--- | :--- |
| **Topic** | 1-to-N or N-to-N Stream | No (Async) | N/A | Camera frames, traversability maps, heartbeat beacons |
| **Service** | 1-to-1 Req/Resp | Yes (Sync/blocking) | No | Reset odometry origin, capture snapshot, trigger calibration |
| **Action** | 1-to-1 with Feedback | No (Async) | Yes | Long-distance waypoint navigation, traverse rough terrain |
| **Parameter** | Node-local config | Immediate | N/A | Set maximum speed, camera resolution, feature thresholds |

---

## 4. Hands-on Lab: The `sih_starter` Package

To demonstrate nodes, topics, publishers, subscribers, and launch orchestration without introducing any heavyweight perception or navigation dependencies, we created the `sih_starter` package.

### Package Structure
```
learning/sih_starter/
├── launch/
│   └── beacon_launch.py
├── package.xml
├── resource/
│   └── sih_starter
├── setup.cfg
├── setup.py
└── sih_starter/
    ├── __init__.py
    ├── beacon_publisher.py
    └── beacon_subscriber.py
```

### Component Breakdown

#### A. Node 1: `beacon_publisher.py`
Publishes a periodic heartbeat status message every 1.0 second on topic `/ugv/heartbeat`:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class BeaconPublisher(Node):
    def __init__(self):
        super().__init__('beacon_publisher')
        self.publisher_ = self.create_publisher(String, '/ugv/heartbeat', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.seq = 0
        self.get_logger().info('BeaconPublisher initialized, emitting on /ugv/heartbeat')

    def timer_callback(self):
        msg = String()
        msg.data = f'UGV System Alive | sequence: {self.seq} | status: NOMINAL'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published beacon #{self.seq}: "{msg.data}"')
        self.seq += 1

def main(args=None):
    rclpy.init(args=args)
    node = BeaconPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

#### B. Node 2: `beacon_subscriber.py`
Listens for incoming messages on `/ugv/heartbeat` and executes a callback function upon receipt:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class BeaconSubscriber(Node):
    def __init__(self):
        super().__init__('beacon_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/ugv/heartbeat',
            self.listener_callback,
            10)
        self.get_logger().info('BeaconSubscriber initialized, listening on /ugv/heartbeat')

    def listener_callback(self, msg):
        self.get_logger().info(f'Received beacon: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    node = BeaconSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

#### C. Launch Script: `beacon_launch.py`
Spawns both nodes within a single coordinated launch execution:
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sih_starter',
            executable='beacon_publisher',
            name='beacon_publisher',
            output='screen',
        ),
        Node(
            package='sih_starter',
            executable='beacon_subscriber',
            name='beacon_subscriber',
            output='screen',
        ),
    ])
```

---

## 5. Lab Build & Run Instructions

Follow these exact steps inside your Ubuntu 24.04 (or WSL2) terminal:

### Step 1: Navigate to Workspace and Build
```bash
cd ~/sih_ws
colcon build --symlink-install --packages-select sih_starter
```
Expected output:
```text
Starting >>> sih_starter
Finished <<< sih_starter [0.52s]
Summary: 1 package finished [0.71s]
```

### Step 2: Source the Underlay and Overlay
```bash
source /opt/ros/jazzy/setup.bash
source ~/sih_ws/install/setup.bash
```

### Step 3: Launch the Multi-Node System
```bash
ros2 launch sih_starter beacon_launch.py
```

---

## 6. Live Verification & Diagnostic Output

While the launch file is running, open a second terminal (remembering to source both setup scripts) and execute the standard verification commands. Below are the actual outputs recorded during lab testing:

### 1. Inspect Active Nodes: `ros2 node list`
```bash
$ ros2 node list
/beacon_publisher
/beacon_subscriber
```
*Verification:* Both nodes are alive, registered with the ROS 2 discovery daemon, and running in distinct process threads.

### 2. Inspect Active Topics: `ros2 topic list`
```bash
$ ros2 topic list
/parameter_events
/rosout
/ugv/heartbeat
```
*Verification:* The custom topic `/ugv/heartbeat` is active alongside default ROS 2 system infrastructure topics (`/rosout` for unified logging and `/parameter_events`).

### 3. Inspect Topic Metadata: `ros2 topic info`
```bash
$ ros2 topic info /ugv/heartbeat
Type: std_msgs/msg/String
Publisher count: 1
Subscription count: 1
```
*Verification:* Confirms exactly 1 active publisher (`beacon_publisher`) and 1 active subscriber (`beacon_subscriber`), proving topic binding.

### 4. Echo Live Data Stream: `ros2 topic echo`
```bash
$ ros2 topic echo /ugv/heartbeat --once
data: 'UGV System Alive | sequence: 12 | status: NOMINAL'
---
```
*Verification:* Shows the serialized message payload streaming across the middleware in real time.

---

## 7. Common Beginner Pitfalls & Troubleshooting

1. **`command not found: ros2`**
   - *Cause:* The underlying ROS 2 environment has not been sourced.
   - *Fix:* Always run `source /opt/ros/jazzy/setup.bash`.
2. **`Package 'sih_starter' not found`**
   - *Cause:* The local workspace overlay has not been sourced after building.
   - *Fix:* Run `source ~/sih_ws/install/setup.bash`.
3. **Changes to Python Code Not Appearing**
   - *Cause:* Package was built without `--symlink-install`, requiring manual recompilation on every edit.
   - *Fix:* Rebuild using `colcon build --symlink-install --packages-select sih_starter`.
4. **Nodes Run but Do Not Receive Messages**
   - *Cause:* Mismatched topic names (e.g., `/ugv/heartbeat` vs `ugv/heartbeat`) or mismatched Quality of Service (QoS) durability/reliability policies.
   - *Fix:* Check `ros2 topic info -v /ugv/heartbeat` to verify QoS compatibility between publisher and subscriber.

---

## 8. Summary Checklist for Team Members

- [x] Understood the separation of Nodes, Topics, Publishers, and Subscribers.
- [x] Know how Services and Actions differ from Topics.
- [x] Successfully built `sih_starter` using `colcon build --symlink-install`.
- [x] Launched dual nodes using `ros2 launch sih_starter beacon_launch.py`.
- [x] Diagnosed nodes and topics using `ros2 node list`, `ros2 topic list`, `ros2 topic info`, and `ros2 topic echo`.
