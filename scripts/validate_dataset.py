#!/usr/bin/env python3
"""Dataset validation and class frequency analyzer utility for outdoor terrain segmentation.

Enforces dataset integrity without modifying or rewriting annotations:
1. Detects missing mask or RGB pairs.
2. Detects dimension mismatches between images and masks.
3. Detects invalid class IDs (labels outside valid range [0, 8]).
4. Detects corrupt images or unreadable byte streams.
5. Performs dataset-wide class-frequency distribution analysis.

Safety Guarantee: Strictly read-only. Never modifies or overwrites user data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# Class labels matching TerrainClass in src/interfaces/types.py
CLASS_LABELS: Dict[int, str] = {
    0: "UNKNOWN",
    1: "PAVED_ROAD",
    2: "TRAVERSABLE_DIRT",
    3: "LOW_GRASS",
    4: "GRAVEL",
    5: "HIGH_VEGETATION",
    6: "OBSTACLE_SOLID",
    7: "WATER_PUDDLE",
    8: "DYNAMIC_OBSTACLE",
}

VALID_CLASS_IDS: Set[int] = set(CLASS_LABELS.keys())


@dataclass
class ValidationError:
    """Records a single dataset integrity issue."""
    file_path: str
    error_type: str  # 'MISSING_PAIR', 'CORRUPT_IMAGE', 'DIMENSION_MISMATCH', 'INVALID_CLASS_ID'
    details: str


@dataclass
class ValidationReport:
    """Consolidated summary of dataset health and class distribution."""
    dataset_dir: str
    total_rgb_files: int = 0
    total_mask_files: int = 0
    valid_pairs: int = 0
    errors: List[ValidationError] = field(default_factory=list)
    class_pixel_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_pixels_evaluated: int = 0

    @property
    def is_clean(self) -> bool:
        return len(self.errors) == 0

    def print_summary(self) -> None:
        """Prints a structured ASCII health and frequency report."""
        print("\n" + "=" * 80)
        print(f" DATASET VALIDATION REPORT: {self.dataset_dir}")
        print("=" * 80)
        print(f"Total RGB Images Found   : {self.total_rgb_files}")
        print(f"Total Mask Images Found  : {self.total_mask_files}")
        print(f"Valid Paired Frames      : {self.valid_pairs}")
        print(f"Total Errors Detected    : {len(self.errors)}")
        print("-" * 80)

        if self.errors:
            print("ERROR LOG:")
            for i, err in enumerate(self.errors, 1):
                print(f"  [{i:02d}] {err.error_type:<20} | {os.path.basename(err.file_path)} : {err.details}")
            print("-" * 80)
        else:
            print("STATUS: PASSED - Zero integrity violations detected.")
            print("-" * 80)

        # Class Frequency Breakdown
        if self.total_pixels_evaluated > 0:
            print("CLASS FREQUENCY DISTRIBUTION:")
            print(f"  {'ID':<4} {'Class Name':<20} {'Pixel Count':<16} {'Percentage':<12} {'Health':<10}")
            print("  " + "-" * 66)
            for cid in sorted(CLASS_LABELS.keys()):
                cname = CLASS_LABELS[cid]
                count = self.class_pixel_counts[cid]
                pct = (count / self.total_pixels_evaluated) * 100.0
                if count == 0:
                    health = "STARVED (0%)"
                elif pct < 0.5:
                    health = "LOW (<0.5%)"
                else:
                    health = "NOMINAL"
                print(f"  [{cid}]  {cname:<20} {count:<16,d} {pct:6.2f}%      {health}")
            print("=" * 80 + "\n")


def natural_sort_key(s: str) -> List[int]:
    """Natural alphanumeric sort key."""
    return [int(c) if c.isdigit() else 0 for c in re.split(r'(\d+)', s)]


def find_files_by_extensions(folder: str, extensions: Tuple[str, ...]) -> List[str]:
    """Finds all matching files in folder with natural sorting."""
    files: Set[str] = set()
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if any(fname.lower().endswith(ext.lower()) for ext in extensions):
                files.add(os.path.join(folder, fname))
    return sorted(files, key=natural_sort_key)


def validate_image_integrity(file_path: str) -> Tuple[bool, Optional[Tuple[int, int]], str]:
    """Verifies image file header and readability using PIL and OpenCV.
    
    Returns:
        (is_valid, (width, height), error_message)
    """
    if not os.path.exists(file_path):
        return False, None, "File does not exist"
    if os.path.getsize(file_path) == 0:
        return False, None, "File is 0 bytes (empty)"

    # Step 1: PIL verify
    try:
        with Image.open(file_path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        return False, None, f"PIL verification failed: {e}"

    # Step 2: OpenCV full decode test
    mat = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if mat is None:
        return False, None, "OpenCV failed to decode image buffer"

    h, w = mat.shape[:2]
    return True, (w, h), ""


def validate_dataset(dataset_dir: str) -> ValidationReport:
    """Validates RGB and mask dataset integrity in strict read-only mode."""
    report = ValidationReport(dataset_dir=dataset_dir)

    rgb_dir = os.path.join(dataset_dir, "rgb")
    mask_dir = os.path.join(dataset_dir, "masks")

    # Fallback to single folder if subdirectories do not exist
    if not os.path.isdir(rgb_dir):
        rgb_dir = dataset_dir
    if not os.path.isdir(mask_dir):
        mask_dir = dataset_dir

    rgb_files = find_files_by_extensions(rgb_dir, (".png", ".jpg", ".jpeg"))
    # Filter out files in masks folder if flat structure
    if rgb_dir == mask_dir:
        rgb_files = [f for f in rgb_files if "_mask" not in os.path.basename(f).lower()]

    mask_files = find_files_by_extensions(mask_dir, (".png", ".npy"))
    if rgb_dir == mask_dir:
        mask_files = [f for f in mask_files if "_mask" in os.path.basename(f).lower() or f.endswith(".npy")]

    report.total_rgb_files = len(rgb_files)
    report.total_mask_files = len(mask_files)

    # Index masks by file stem
    mask_map: Dict[str, str] = {}
    for m in mask_files:
        stem = os.path.splitext(os.path.basename(m))[0]
        # normalize stem: remove '_mask' suffix if present
        norm_stem = re.sub(r'(_mask|_label)$', '', stem, flags=re.IGNORECASE)
        mask_map[norm_stem] = m

    # 1. Check for missing mask pairs and corrupt RGBs
    evaluated_pairs: List[Tuple[str, str]] = []
    for rgb_path in rgb_files:
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        norm_stem = re.sub(r'(_mask|_label)$', '', stem, flags=re.IGNORECASE)

        # Check RGB integrity
        is_valid_rgb, rgb_dims, err_msg = validate_image_integrity(rgb_path)
        if not is_valid_rgb:
            report.errors.append(ValidationError(
                file_path=rgb_path,
                error_type="CORRUPT_IMAGE",
                details=f"Corrupt RGB file: {err_msg}",
            ))
            continue

        # Check for matching mask
        if norm_stem not in mask_map:
            report.errors.append(ValidationError(
                file_path=rgb_path,
                error_type="MISSING_PAIR",
                details=f"No corresponding mask found for stem '{norm_stem}'",
            ))
            continue

        mask_path = mask_map[norm_stem]
        evaluated_pairs.append((rgb_path, mask_path))

    # Check for orphan masks (masks without corresponding RGB)
    rgb_stems = {re.sub(r'(_mask|_label)$', '', os.path.splitext(os.path.basename(f))[0], flags=re.IGNORECASE) for f in rgb_files}
    for m_stem, m_path in mask_map.items():
        if m_stem not in rgb_stems:
            report.errors.append(ValidationError(
                file_path=m_path,
                error_type="MISSING_PAIR",
                details=f"Orphan mask found without corresponding RGB image: '{m_stem}'",
            ))

    # 2. Inspect paired masks: dimensions, corruptions, class IDs, class frequencies
    for rgb_path, mask_path in evaluated_pairs:
        # Read RGB dimensions
        rgb_mat = cv2.imread(rgb_path)
        rgb_h, rgb_w = rgb_mat.shape[:2]

        # Read Mask
        if mask_path.endswith(".npy"):
            try:
                mask_mat = np.load(mask_path)
            except Exception as e:
                report.errors.append(ValidationError(
                    file_path=mask_path,
                    error_type="CORRUPT_IMAGE",
                    details=f"Failed to load numpy mask: {e}",
                ))
                continue
        else:
            is_valid_mask, mask_dims, err_msg = validate_image_integrity(mask_path)
            if not is_valid_mask:
                report.errors.append(ValidationError(
                    file_path=mask_path,
                    error_type="CORRUPT_IMAGE",
                    details=f"Corrupt mask image: {err_msg}",
                ))
                continue
            mask_mat = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        mask_h, mask_w = mask_mat.shape[:2]

        # Dimension Check
        if (rgb_w, rgb_h) != (mask_w, mask_h):
            report.errors.append(ValidationError(
                file_path=mask_path,
                error_type="DIMENSION_MISMATCH",
                details=f"Mask size ({mask_w}x{mask_h}) does not match RGB size ({rgb_w}x{rgb_h})",
            ))
            continue

        # Class ID Bounds Check
        unique_ids = np.unique(mask_mat)
        invalid_ids = [int(cid) for cid in unique_ids if cid not in VALID_CLASS_IDS]
        if invalid_ids:
            report.errors.append(ValidationError(
                file_path=mask_path,
                error_type="INVALID_CLASS_ID",
                details=f"Mask contains invalid class IDs outside [0, 8]: {invalid_ids}",
            ))

        # Class Frequency Accumulation
        report.valid_pairs += 1
        for cid in unique_ids:
            if cid in VALID_CLASS_IDS:
                count = int(np.count_nonzero(mask_mat == cid))
                report.class_pixel_counts[int(cid)] += count
                report.total_pixels_evaluated += count

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dataset Validator & Class Frequency Analyzer for SIH 26126 UGV Terrain Segmentation"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing rgb/ and masks/ subdirectories",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Error: Target directory does not exist: {args.dataset_dir}", file=sys.stderr)
        return 1

    report = validate_dataset(args.dataset_dir)
    report.print_summary()

    return 0 if report.is_clean else 2


if __name__ == "__main__":
    sys.exit(main())
