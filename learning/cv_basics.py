"""Beginner Computer Vision Learning Script: Python & OpenCV Fundamentals.

This is a non-production educational example that demonstrates:
1. Loading an outdoor image from disk
2. Inspecting dimensions and channel structure
3. Resizing the image
4. Creating a binary region-of-interest (ROI) mask
5. Blending a colored mask overlay using cv2.addWeighted
6. Drawing text and bounding shapes using OpenCV drawing primitives
7. Saving the annotated result to disk

Run directly with:
    python learning/cv_basics.py
"""

from __future__ import annotations

import os
import cv2
import numpy as np


def run_cv_basics(
    input_image_path: str = "datasets/processed/scenario_1_open_path/rgb/frame_0000.jpg",
    output_dir: str = "learning/output",
) -> str:
    """Execute step-by-step OpenCV fundamentals on an outdoor image."""
    print("=" * 60)
    print("STEP 1: Load Real Outdoor Image")
    print("=" * 60)
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    # OpenCV loads images as BGR uint8 NumPy array
    img_bgr = cv2.imread(input_image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to decode image from: {input_image_path}")

    print(f"Successfully loaded image from: {input_image_path}")

    print("\n" + "=" * 60)
    print("STEP 2: Inspect Dimensions & Channels (NumPy Array)")
    print("=" * 60)
    h, w, c = img_bgr.shape
    dtype = img_bgr.dtype
    total_pixels = h * w
    print(f"Height (rows):       {h} px")
    print(f"Width (columns):     {w} px")
    print(f"Channels (BGR):      {c}")
    print(f"Data type:           {dtype} (values in [0..255])")
    print(f"Total pixel count:   {total_pixels:,}")

    print("\n" + "=" * 60)
    print("STEP 3: Resize & Crop Image")
    print("=" * 60)
    # Resize to standard lower resolution for fast edge compute (e.g. 320x240)
    target_w, target_h = 320, 240
    resized_img = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    print(f"Resized image dimensions: {resized_img.shape[1]}x{resized_img.shape[0]} (WxH)")

    # Crop ground region (lower 55% of the image)
    crop_top = int(target_h * 0.45)
    ground_crop = resized_img[crop_top:target_h, :]
    print(f"Ground crop shape:        {ground_crop.shape} (rows {crop_top} to {target_h})")

    print("\n" + "=" * 60)
    print("STEP 4: Create a Simple Mask")
    print("=" * 60)
    # A mask is a 2D single-channel binary image (uint8, 0 or 255)
    # We create a trapezoidal mask representing the forward traversable path
    mask = np.zeros((target_h, target_w), dtype=np.uint8)

    # Define polygon vertices for the central ground corridor:
    # [top-left, top-right, bottom-right, bottom-left]
    path_polygon = np.array([
        [int(target_w * 0.38), int(target_h * 0.50)],  # Distant path horizon left
        [int(target_w * 0.62), int(target_h * 0.50)],  # Distant path horizon right
        [int(target_w * 0.85), int(target_h * 0.98)],  # Close foreground right
        [int(target_w * 0.15), int(target_h * 0.98)],  # Close foreground left
    ], dtype=np.int32)

    cv2.fillPoly(mask, [path_polygon], color=255)
    masked_pixel_count = np.sum(mask == 255)
    coverage_pct = (masked_pixel_count / (target_w * target_h)) * 100.0
    print(f"Created trapezoidal corridor mask.")
    print(f"Mask shape: {mask.shape}, Masked pixels: {masked_pixel_count} ({coverage_pct:.1f}% of frame)")

    print("\n" + "=" * 60)
    print("STEP 5: Overlay Mask with Transparency & Drawing Primitives")
    print("=" * 60)
    # Create colored overlay canvas (Green = BGR [0, 255, 0])
    color_tint = np.zeros_like(resized_img)
    color_tint[mask == 255] = [0, 220, 50]  # Vibrant green tint for traversable zone

    # Blend original image with green tint: output = (1 - alpha)*orig + alpha*tint
    alpha = 0.40
    annotated = cv2.addWeighted(color_tint, alpha, resized_img, 1.0, 0.0)

    # Step 5b: Draw annotations using OpenCV primitives
    # 1. Draw corridor boundary outline (Cyan BGR: [255, 200, 0], thickness 2)
    cv2.polylines(annotated, [path_polygon], isClosed=True, color=(255, 200, 0), thickness=2)

    # 2. Draw a simulated obstacle warning box on the left margin (Red BGR: [0, 0, 255])
    obs_x1, obs_y1 = int(target_w * 0.08), int(target_h * 0.60)
    obs_x2, obs_y2 = int(target_w * 0.22), int(target_h * 0.80)
    cv2.rectangle(annotated, (obs_x1, obs_y1), (obs_x2, obs_y2), color=(0, 50, 230), thickness=2)
    cv2.putText(annotated, "BUSH", (obs_x1, obs_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 50, 230), 1)

    # 3. Draw telemetry text header
    cv2.putText(annotated, "CV LEARNING LAB: TRAVERSABLE CORRIDOR", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(annotated, f"Corridor Coverage: {coverage_pct:.1f}%", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

    print("Blended semi-transparent traversability mask.")
    print("Added polyline boundary, bounding rectangle, and text overlay.")

    print("\n" + "=" * 60)
    print("STEP 6: Save Result to Disk")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "outdoor_traversability_demo.jpg")
    success = cv2.imwrite(out_file, annotated)
    if not success:
        raise IOError(f"Failed to write image to: {out_file}")

    print(f"Saved verified annotated image to: {out_file}")
    print("OpenCV learning demonstration completed successfully!")
    return out_file


if __name__ == "__main__":
    run_cv_basics()
