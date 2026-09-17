"""
docs/learning/lab1_basics.py

Tiny non-production learning example for SIH 26126.
Goal:
  1. create a simple artificial outdoor-style image (sky, ground, path, object)
  2. print dimensions
  3. resize it
  4. create a simple mask
  5. overlay the mask
  6. save the result

This file is learning code. It is intentionally separate from production code in src/.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Location for learning artifacts
# ---------------------------------------------------------------------------
LEARNING_DIR = Path(__file__).resolve().parent
IMAGE_PATH = LEARNING_DIR / "outdoor_scene.png"


def create_synthetic_outdoor_image(path: Path) -> Path:
    """Create a simple artificial outdoor-style image for offline learning."""
    height, width = 480, 640
    channels = 3

    # Start with sky gradient (top) -> ground (bottom)
    img = np.zeros((height, width, channels), dtype=np.uint8)

    for y in range(height):
        # Sky region 0..60% : light blue gradient
        if y < int(height * 0.6):
            t = y / (height * 0.6)
            b = int(135 + t * 20)
            g = int(180 - t * 30)
            r = int(220 - t * 40)
        else:
            # Ground region 60..100% : brownish soil / grass gradient
            t = (y - height * 0.6) / (height * 0.4)
            b = int(60 + t * 10)
            g = int(90 - t * 20)
            r = int(50 + t * 20)

        img[y, :] = [b, g, r]

    # Add a few "grass / path" texture marks
    rng = np.random.default_rng(42)
    for _ in range(500):
        y = int(rng.integers(height // 2, height))
        x = int(rng.integers(0, width))
        radius = int(rng.integers(1, 4))
        color = [int(rng.integers(80, 140)), int(rng.integers(110, 160)), int(rng.integers(40, 80))]
        cv2.circle(img, (x, y), radius, color, -1)

    # Add a "path" - lighter strip going into the distance
    path_center = width // 2
    path_base_width = 120
    for y in range(int(height * 0.6), height):
        frac = (y - height * 0.6) / (height * 0.4)
        half_width = int(path_base_width * (1.0 - frac * 0.85))
        x_start = path_center - half_width
        x_end = path_center + half_width
        if x_start < 0:
            x_start = 0
        if x_end > width:
            x_end = width
        lighter = [int(180 + 20 * (1 - frac)), int(170 + 20 * (1 - frac)), int(140 + 20 * (1 - frac))]
        img[y, x_start:x_end] = lighter

    # Add a "rock / obstacle" upper-right
    rock_center = (int(width * 0.75), int(height * 0.45))
    rock_radius = int(height * 0.08)
    rock_color = [int(120), int(115), int(110)]
    cv2.circle(img, rock_center, rock_radius, rock_color, -1)
    cv2.circle(img, rock_center, rock_radius, [int(60), int(60), int(60)], 2)

    # Add a small "bush" lower-left
    bush_center = (int(width * 0.2), int(height * 0.8))
    bush_color = [int(90), int(140), int(55)]
    cv2.circle(img, bush_center, int(height * 0.06), bush_color, -1)
    cv2.circle(img, (bush_center[0] - 8, bush_center[1] - 4), int(height * 0.035), bush_color, -1)
    cv2.circle(img, (bush_center[0] + 8, bush_center[1] - 2), int(height * 0.03), bush_color, -1)

    cv2.imwrite(str(path), img)
    print(f"Created synthetic outdoor image: {path}")
    return path


def load_image(path: Path) -> np.ndarray:
    """Load an image with OpenCV (BGR order)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"OpenCV could not read image: {path}")
    return img


def print_image_info(img: np.ndarray, label: str) -> None:
    """Print image dimensions and dtype."""
    if img.ndim == 2:
        height, width = img.shape
        channels = 1
    elif img.ndim == 3:
        height, width, channels = img.shape
    else:
        height, width, channels = img.shape[:2], 0, -1

    print(f"\n{label}")
    print(f"  shape   : {img.shape}")
    print(f"  dtype   : {img.dtype}")
    print(f"  height  : {height} px")
    print(f"  width   : {width} px")
    print(f"  channels: {channels}")
    print(f"  size    : {img.size} pixels")


def resize_image(img: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Resize by scale factor."""
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def make_rectangular_mask(height: int, width: int) -> np.ndarray:
    """Create a simple rectangular region-of-interest mask.

    White (255) inside the region, black (0) outside.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    margin_x = width // 4
    margin_y = height // 4
    mask[margin_y : height - margin_y, margin_x : width - margin_x] = 255
    return mask


def make_brightness_mask(img: np.ndarray, factor: float = 1.05) -> np.ndarray:
    """Create a simple pixel-based mask from brightness.

    Brighter than mean * factor => white.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    threshold = mean_val * factor
    mask = (gray > threshold).astype(np.uint8) * 255
    return mask


def overlay_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    """Overlay a single-channel mask on an image with transparency.

    mask: 0/255 single-channel image
    color_bgr: BGR color for the overlay
    alpha: transparency of the overlay where mask is white
    """
    overlay = img.copy()
    colored_mask = np.zeros_like(img)
    colored_mask[:, :] = color_bgr
    binary = (mask > 0).astype(np.uint8)
    blended = cv2.addWeighted(overlay, 1.0 - alpha, colored_mask, alpha, 0)
    out = np.where(binary[:, :, None] == 1, blended, img)
    return out.astype(np.uint8)


def draw_mask_outline(img: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    """Draw contour outlines of a mask on the image."""
    out = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color_bgr, thickness=2)
    return out


def main() -> None:
    print("=" * 70)
    print("Python/OpenCV Learning Lab 1 — outdoor scene mask overlay")
    print("=" * 70)

    # 1. Load / create a simple outdoor-style image
    if not IMAGE_PATH.exists():
        create_synthetic_outdoor_image(IMAGE_PATH)

    img = load_image(IMAGE_PATH)
    print_image_info(img, "Original image")

    # 2. Resize
    resized = resize_image(img, scale=0.5)
    print_image_info(resized, "Resized image (0.5x)")
    resized_path = LEARNING_DIR / "outdoor_scene_resized.png"
    cv2.imwrite(str(resized_path), resized)
    print(f"\nSaved resized image to: {resized_path}")

    # 3. Create masks
    rect_mask = make_rectangular_mask(resized.shape[0], resized.shape[1])
    bright_mask = make_brightness_mask(resized)

    rect_count = int(np.count_nonzero(rect_mask))
    bright_count = int(np.count_nonzero(bright_mask))
    total = resized.shape[0] * resized.shape[1]
    print(f"\nRectangular mask pixels: {rect_count} / {total} ({100 * rect_count / total:.1f}%)")
    print(f"Brightness mask pixels : {bright_count} / {total} ({100 * bright_count / total:.1f}%)")

    # 4. Overlay masks
    overlay_rect = overlay_mask(resized, rect_mask, color_bgr=(0, 255, 0), alpha=0.30)
    overlay_bright = overlay_mask(resized, bright_mask, color_bgr=(255, 0, 0), alpha=0.30)
    overlay_both = overlay_mask(overlay_rect, bright_mask, color_bgr=(255, 0, 0), alpha=0.30)

    # Add outlines for clarity
    overlay_rect = draw_mask_outline(overlay_rect, rect_mask, color_bgr=(0, 255, 0))
    overlay_bright = draw_mask_outline(overlay_bright, bright_mask, color_bgr=(255, 0, 0))

    # 5. Save results
    rect_path = LEARNING_DIR / "outdoor_scene_rect_mask.png"
    bright_path = LEARNING_DIR / "outdoor_scene_bright_mask.png"
    both_path = LEARNING_DIR / "outdoor_scene_mask_overlay.png"

    cv2.imwrite(str(rect_path), overlay_rect)
    cv2.imwrite(str(bright_path), overlay_bright)
    cv2.imwrite(str(both_path), overlay_both)

    print("\nSaved results:")
    print(f"  {rect_path}")
    print(f"  {bright_path}")
    print(f"  {both_path}")

    print("\nLegend:")
    print("  green  = rectangular region-of-interest mask")
    print("  red    = brightness-based mask (above-mean * 1.05)")
    print("  outlines show mask boundaries")
    print("\nDone.")


if __name__ == "__main__":
    main()
