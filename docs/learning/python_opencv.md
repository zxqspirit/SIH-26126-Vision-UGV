# Python & OpenCV Learning Lab: Computer Vision Fundamentals

**Target Audience:** Engineering team members learning Computer Vision fundamentals for outdoor UGV applications.  
**Scope:** Core Python, Virtual Environments, NumPy, and OpenCV operations.  
**Executable Example:** [`learning/cv_basics.py`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/learning/cv_basics.py)

---

## 1. Virtual Environments (`venv`)

A virtual environment isolates project dependencies so package versions don't conflict with system tools.

```bash
# 1. Create a virtual environment named .venv
python -m venv .venv

# 2. Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / WSL:
source .venv/bin/activate

# 3. Install core computer-vision libraries
pip install numpy opencv-python-headless
```

---

## 2. Python Architecture: Functions, Modules & Classes

### A. Functions
Reusable blocks of logic with typed inputs and outputs.
```python
def compute_aspect_ratio(width: int, height: int) -> float:
    """Return width-to-height ratio."""
    if height == 0:
        raise ValueError("Height cannot be zero.")
    return width / height
```

### B. Modules
A **module** is simply a `.py` file containing functions and classes. A **package** is a directory containing an `__init__.py` file.
```python
# Importing from another module
from src.interfaces.types import CameraIntrinsics
```

### C. Classes
Object-oriented blueprints encapsulating state (attributes) and behaviors (methods).
```python
class CameraFeed:
    """Manages an image sensor resolution and frame counter."""
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.frame_count = 0

    def record_frame(self) -> int:
        self.frame_count += 1
        return self.frame_count
```

---

## 3. NumPy Fundamentals for Computer Vision

In Python, digital images are stored as **NumPy n-dimensional arrays** (`ndarray`):

```python
import numpy as np

# Create an empty black canvas: shape (H, W, C) = (480, 640, 3), 8-bit unsigned integer [0..255]
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Array properties
print(canvas.shape)  # (480, 640, 3) -> (Height, Width, Channels)
print(canvas.dtype)  # uint8 (0 to 255)
print(canvas.size)   # 480 * 640 * 3 = 921,600 values
```

### Key Array Indexing Rules:
- Coordinates in NumPy: `image[row, col]` = `image[y, x]`
- Channel 0 = Blue, Channel 1 = Green, Channel 2 = Red (in OpenCV)

---

## 4. Digital Images & the BGR vs. RGB Difference

Standard graphics systems (Matplotlib, web browsers, PIL) store colors as **RGB** (Red, Green, Blue).  
**OpenCV defaults to BGR** (Blue, Green, Red) due to historical camera hardware standards.

```python
import cv2

# 1. Load image from disk (returns BGR format)
img_bgr = cv2.imread("image.jpg")

# 2. Convert between color spaces
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # For display in browser/matplotlib
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) # For edge detection and feature tracking
```

---

## 5. Geometric Operations: Resizing & Cropping

### A. Resizing
Downscaling reduces memory footprint and enables real-time mobile execution (>10 FPS).
```python
# Note: cv2.resize takes (width, height), NOT (height, width)!
target_w, target_h = 320, 240
resized_img = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
```

### B. Cropping (Slicing)
Cropping uses standard NumPy array slicing: `image[y_start:y_end, x_start:x_end]`.
```python
# In UGV vision, the ground plane lies in the lower half of the frame:
h, w = img_bgr.shape[:2]
ground_region = img_bgr[int(h * 0.45):h, :]  # Crop lower 55% of image
```

---

## 6. Processing Video Sequences (Frames)

A video is simply a sequential stream of image frames captured at a constant frequency (e.g. 10–30 Hz).

```python
# Reading from a video file or camera device
cap = cv2.VideoCapture("outdoor_drive.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of stream

    # Process individual frame...
    h, w = frame.shape[:2]

cap.release()
```

---

## 7. Binary Masks & Transparent Blending

A **mask** is a single-channel binary image (`uint8`, values `0` or `255`) where `255` indicates pixels of interest (e.g. traversable ground) and `0` indicates background.

```python
# 1. Initialize a black mask
mask = np.zeros((240, 320), dtype=np.uint8)

# 2. Draw a polygon corridor (e.g., path in front of the vehicle)
polygon = np.array([[120, 100], [200, 100], [270, 235], [50, 235]], dtype=np.int32)
cv2.fillPoly(mask, [polygon], color=255)

# 3. Create a colored highlight overlay (Green in BGR = [0, 220, 50])
color_tint = np.zeros_like(resized_img)
color_tint[mask == 255] = [0, 220, 50]

# 4. Transparent alpha blending: result = (1 - alpha) * original + alpha * tint
alpha = 0.40
blended = cv2.addWeighted(color_tint, alpha, resized_img, 1.0, 0.0)
```

---

## 8. Drawing Primitives & Annotations

OpenCV provides drawing primitives to visualize bounding boxes, headings, and system telemetry:

```python
# 1. Bounding Rectangle: cv2.rectangle(img, pt1, pt2, color_bgr, thickness)
cv2.rectangle(blended, (25, 140), (70, 190), (0, 50, 230), 2)

# 2. Polylines Outline: cv2.polylines(img, [pts], isClosed, color_bgr, thickness)
cv2.polylines(blended, [polygon], isClosed=True, color=(255, 200, 0), thickness=2)

# 3. Text Overlay: cv2.putText(img, text, org, font, scale, color_bgr, thickness)
cv2.putText(blended, "TRAVERSABLE CORRIDOR", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
```

---

## 9. Running the Executable Learning Lab

A non-production learning script is located at [`learning/cv_basics.py`](file:///C:/Users/harsh/SIH-26126-Vision-UGV/learning/cv_basics.py).

To execute and verify:
```bash
python learning/cv_basics.py
```

### Verified Execution Output:
```
============================================================
STEP 1: Load Real Outdoor Image
============================================================
Successfully loaded image from: datasets/processed/scenario_1_open_path/rgb/frame_0000.jpg

============================================================
STEP 2: Inspect Dimensions & Channels (NumPy Array)
============================================================
Height (rows):       480 px
Width (columns):     640 px
Channels (BGR):      3
Data type:           uint8 (values in [0..255])
Total pixel count:   307,200

============================================================
STEP 3: Resize & Crop Image
============================================================
Resized image dimensions: 320x240 (WxH)
Ground crop shape:        (132, 320, 3) (rows 108 to 240)

============================================================
STEP 4: Create a Simple Mask
============================================================
Created trapezoidal corridor mask.
Mask shape: (240, 320), Masked pixels: 17574 (22.9% of frame)

============================================================
STEP 5: Overlay Mask with Transparency & Drawing Primitives
============================================================
Blended semi-transparent traversability mask.
Added polyline boundary, bounding rectangle, and text overlay.

============================================================
STEP 6: Save Result to Disk
============================================================
Saved verified annotated image to: learning/output/outdoor_traversability_demo.jpg
OpenCV learning demonstration completed successfully!
```
The output image is stored in `learning/output/outdoor_traversability_demo.jpg`.
