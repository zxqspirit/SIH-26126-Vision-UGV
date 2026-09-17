"""Deterministic classical perception fallback for outdoor traversability.

Uses color spaces (HSV, Excess Green), texture energy, and adaptive spatial priors
to classify outdoor terrain into traversable paths vs non-traversable vegetation/obstacles.
Ensures real-time, deterministic execution with zero heavy external dependencies.
"""

from __future__ import annotations

import time
from typing import Tuple
import numpy as np
import cv2

from ..interfaces.types import TerrainClass, SemanticResult


class ColorTexturePerception:
    """Fast, deterministic classical perception for outdoor ground/path segmentation."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 480),
        green_threshold: float = 25.0,
        texture_threshold: float = 35.0,
        horizon_fraction: float = 0.34,  # Camera tilted 12 deg downward -> horizon is ~34% down
    ) -> None:
        self.target_width, self.target_height = target_size
        self.green_threshold = green_threshold
        self.texture_threshold = texture_threshold
        self.horizon_fraction = horizon_fraction

    def process(self, rgb_image: np.ndarray) -> SemanticResult:
        """Process an RGB image and return semantic traversability and terrain classes."""
        start_time = time.perf_counter()

        h, w = rgb_image.shape[:2]
        if (w, h) != (self.target_width, self.target_height):
            img = cv2.resize(rgb_image, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
            resized = True
        else:
            img = rgb_image
            resized = False

        img_float = img.astype(np.float32)
        r = img_float[:, :, 0]
        g = img_float[:, :, 1]
        b = img_float[:, :, 2]

        # 1. Excess Green index: ExG = 2*G - R - B
        exg = 2.0 * g - r - b

        # 2. Convert to HSV for soil/dirt and brightness analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)

        # 3. High-frequency texture energy
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_mag = cv2.magnitude(sobel_x, sobel_y)
        edge_blur = cv2.GaussianBlur(edge_mag, (15, 15), 0)

        # 4. Perspective / horizon prior
        y_grid = np.linspace(0.0, 1.0, self.target_height)[:, np.newaxis]
        is_ground_zone = np.broadcast_to(y_grid >= self.horizon_fraction, (self.target_height, self.target_width))
        is_sky = np.broadcast_to(y_grid < self.horizon_fraction, (self.target_height, self.target_width))

        # 5. Class identification
        terrain_map = np.zeros((self.target_height, self.target_width), dtype=np.int32)
        traversability = np.zeros((self.target_height, self.target_width), dtype=np.float32)

        is_green = exg > self.green_threshold
        is_smooth = edge_blur < self.texture_threshold

        # Dirt / gravel / asphalt path
        is_dirt_path = (~is_green) & is_smooth & (val > 40) & (val < 220) & is_ground_zone
        terrain_map[is_dirt_path] = TerrainClass.TRAVERSABLE_DIRT
        traversability[is_dirt_path] = 0.95

        # Paved / smooth pathway
        is_paved = (~is_green) & (edge_blur < 20.0) & (sat < 50.0) & is_ground_zone
        terrain_map[is_paved] = TerrainClass.PAVED_ROAD
        traversability[is_paved] = 1.00

        # Low grass
        is_low_grass = is_green & (edge_blur < 45.0) & is_ground_zone
        terrain_map[is_low_grass] = TerrainClass.LOW_GRASS
        traversability[is_low_grass] = 0.80

        # Gravel / rough terrain
        is_gravel = (~is_green) & (edge_blur >= self.texture_threshold) & (edge_blur < 70.0) & is_ground_zone
        terrain_map[is_gravel] = TerrainClass.GRAVEL
        traversability[is_gravel] = 0.65

        # High vegetation / thick bush
        is_bush = is_green & (edge_blur >= 45.0) & is_ground_zone
        terrain_map[is_bush] = TerrainClass.HIGH_VEGETATION
        traversability[is_bush] = 0.15

        # Solid obstacles
        is_solid_obstacle = (edge_blur >= 70.0) & is_ground_zone
        terrain_map[is_solid_obstacle] = TerrainClass.OBSTACLE_SOLID
        traversability[is_solid_obstacle] = 0.00

        # Sky / Horizon
        terrain_map[is_sky] = TerrainClass.UNKNOWN
        traversability[is_sky] = 0.00

        # 6. Smooth traversability
        traversability = cv2.GaussianBlur(traversability, (9, 9), 0)
        traversability = np.clip(traversability, 0.0, 1.0)

        # Resize back if needed
        if resized:
            traversability = cv2.resize(traversability, (w, h), interpolation=cv2.INTER_LINEAR)
            terrain_map = cv2.resize(terrain_map, (w, h), interpolation=cv2.INTER_NEAREST)

        # 7. Compute perception confidence C_perc
        ground_val = val[is_ground_zone]
        mean_val = float(np.mean(ground_val)) if len(ground_val) > 0 else 128.0
        val_confidence = 1.0 - abs(mean_val - 128.0) / 128.0
        ground_trav = traversability[is_ground_zone]
        has_path = float(np.mean(ground_trav > 0.6))
        overall_confidence = float(np.clip(0.4 * val_confidence + 0.6 * (0.5 + 0.5 * has_path), 0.25, 0.96))

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SemanticResult(
            traversability_mask=traversability,
            terrain_class_map=terrain_map,
            confidence=overall_confidence,
            latency_ms=latency_ms,
            metadata={
                "method": "color_texture_fallback",
                "mean_val": mean_val,
                "has_path": has_path,
            }
        )
