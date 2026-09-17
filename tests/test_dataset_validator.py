"""Unit tests for dataset validator and class frequency analyzer."""

import hashlib
import os
import shutil
import tempfile
import cv2
import numpy as np
import pytest

from scripts.validate_dataset import validate_dataset


@pytest.fixture
def temp_dataset_dir():
    temp_dir = tempfile.mkdtemp(prefix="sih_test_dataset_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_valid_dataset(temp_dataset_dir):
    """Verifies that clean matching RGB and mask pairs pass with zero errors."""
    rgb_dir = os.path.join(temp_dataset_dir, "rgb")
    mask_dir = os.path.join(temp_dataset_dir, "masks")
    os.makedirs(rgb_dir)
    os.makedirs(mask_dir)

    # Create two synthetic frames: 64x64
    rgb_img = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb_img[:] = [100, 150, 200]

    # Mask with classes 1 (PAVED_ROAD), 2 (DIRT), 3 (GRASS)
    mask_img = np.zeros((64, 64), dtype=np.uint8)
    mask_img[:20, :] = 1
    mask_img[20:40, :] = 2
    mask_img[40:, :] = 3

    cv2.imwrite(os.path.join(rgb_dir, "frame_0001.png"), rgb_img)
    cv2.imwrite(os.path.join(mask_dir, "frame_0001.png"), mask_img)

    report = validate_dataset(temp_dataset_dir)
    assert report.is_clean
    assert report.valid_pairs == 1
    assert report.total_rgb_files == 1
    assert report.total_mask_files == 1
    assert report.class_pixel_counts[1] == 20 * 64
    assert report.class_pixel_counts[2] == 20 * 64
    assert report.class_pixel_counts[3] == 24 * 64


def test_missing_mask_detected(temp_dataset_dir):
    """Verifies detection of RGB frame lacking corresponding mask."""
    rgb_dir = os.path.join(temp_dataset_dir, "rgb")
    mask_dir = os.path.join(temp_dataset_dir, "masks")
    os.makedirs(rgb_dir)
    os.makedirs(mask_dir)

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(rgb_dir, "frame_missing.png"), rgb)

    report = validate_dataset(temp_dataset_dir)
    assert not report.is_clean
    assert any(err.error_type == "MISSING_PAIR" for err in report.errors)


def test_dimension_mismatch_detected(temp_dataset_dir):
    """Verifies detection of size mismatch between RGB and mask."""
    rgb_dir = os.path.join(temp_dataset_dir, "rgb")
    mask_dir = os.path.join(temp_dataset_dir, "masks")
    os.makedirs(rgb_dir)
    os.makedirs(mask_dir)

    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)  # Mismatched size

    cv2.imwrite(os.path.join(rgb_dir, "frame_0001.png"), rgb)
    cv2.imwrite(os.path.join(mask_dir, "frame_0001.png"), mask)

    report = validate_dataset(temp_dataset_dir)
    assert not report.is_clean
    assert any(err.error_type == "DIMENSION_MISMATCH" for err in report.errors)


def test_invalid_class_id_detected(temp_dataset_dir):
    """Verifies detection of pixel labels outside the valid range [0, 8]."""
    rgb_dir = os.path.join(temp_dataset_dir, "rgb")
    mask_dir = os.path.join(temp_dataset_dir, "masks")
    os.makedirs(rgb_dir)
    os.makedirs(mask_dir)

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0, 0] = 99  # Invalid class ID outside [0, 8]

    cv2.imwrite(os.path.join(rgb_dir, "frame_0001.png"), rgb)
    cv2.imwrite(os.path.join(mask_dir, "frame_0001.png"), mask)

    report = validate_dataset(temp_dataset_dir)
    assert not report.is_clean
    assert any(err.error_type == "INVALID_CLASS_ID" for err in report.errors)


def test_corrupt_image_detected(temp_dataset_dir):
    """Verifies detection of truncated/corrupted image headers."""
    rgb_dir = os.path.join(temp_dataset_dir, "rgb")
    mask_dir = os.path.join(temp_dataset_dir, "masks")
    os.makedirs(rgb_dir)
    os.makedirs(mask_dir)

    # Write truncated garbage bytes
    with open(os.path.join(rgb_dir, "frame_corrupt.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRgarbagebytesdeadbeef")

    report = validate_dataset(temp_dataset_dir)
    assert not report.is_clean
    assert any(err.error_type == "CORRUPT_IMAGE" for err in report.errors)


def test_read_only_guarantee(temp_dataset_dir):
    """Verifies that validating a dataset does not alter or rewrite human files."""
    rgb_dir = os.path.join(temp_dataset_dir, "rgb")
    mask_dir = os.path.join(temp_dataset_dir, "masks")
    os.makedirs(rgb_dir)
    os.makedirs(mask_dir)

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    rgb_path = os.path.join(rgb_dir, "frame_0001.png")
    mask_path = os.path.join(mask_dir, "frame_0001.png")

    cv2.imwrite(rgb_path, rgb)
    cv2.imwrite(mask_path, mask)

    def file_hash(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    rgb_hash_before = file_hash(rgb_path)
    mask_hash_before = file_hash(mask_path)

    # Run validation
    validate_dataset(temp_dataset_dir)

    assert file_hash(rgb_path) == rgb_hash_before
    assert file_hash(mask_path) == mask_hash_before
