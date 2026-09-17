"""PyTorch Dataset module for SIH 26126 outdoor terrain semantic segmentation.

Loads paired RGB images and 8-bit single-channel grayscale masks with deterministic
augmentation pipelines, spatial transformations, and tensor normalization.
"""

from __future__ import annotations

import glob
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet statistics for standard transfer learning normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TerrainSegmentationDataset(Dataset):
    """Dataset for outdoor terrain segmentation."""

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        target_size: Tuple[int, int] = (512, 512),
        augment: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.split = split.lower()
        self.target_size = target_size  # (H, W)
        self.augment = augment and (self.split == "train")

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.split_dir = os.path.join(self.dataset_dir, self.split)
        self.rgb_dir = os.path.join(self.split_dir, "rgb")
        self.mask_dir = os.path.join(self.split_dir, "masks")

        if not os.path.isdir(self.rgb_dir) or not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Split directories not found under {self.split_dir}")

        rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, "*.jpg")))
        self.samples: List[Tuple[str, str]] = []

        for rgb_path in rgb_files:
            basename = os.path.splitext(os.path.basename(rgb_path))[0]
            mask_path = os.path.join(self.mask_dir, f"{basename}.png")
            if os.path.isfile(mask_path):
                self.samples.append((rgb_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No valid image-mask pairs found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentations(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies consistent spatial transforms and photometric jitter."""
        # 1. Random horizontal flip
        if random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            mask = cv2.flip(mask, 1)

        # 2. Photometric jitter (RGB only)
        if random.random() > 0.3:
            # Random brightness
            alpha = random.uniform(0.85, 1.15)
            # Random contrast
            beta = random.uniform(-15.0, 15.0)
            rgb = np.clip(alpha * rgb + beta, 0, 255).astype(np.uint8)

        # 3. Random scale and crop
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            h, w = rgb.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return rgb, mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rgb_path, mask_path = self.samples[idx]

        # Read RGB
        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"Failed to read image: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Read Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to read mask: {mask_path}")

        # Augmentations (training only)
        if self.augment:
            rgb, mask = self._apply_augmentations(rgb, mask)

        # Resize to standardized target size
        th, tw = self.target_size
        rgb_resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Normalize RGB to [0, 1] then ImageNet mean/std
        rgb_norm = rgb_resized.astype(np.float32) / 255.0
        rgb_norm = (rgb_norm - IMAGENET_MEAN) / IMAGENET_STD

        # Convert to tensors: Image (C, H, W), Mask (H, W)
        image_tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_resized).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "rgb_path": rgb_path,
            "mask_path": mask_path,
            "sample_id": os.path.splitext(os.path.basename(rgb_path))[0],
        }
