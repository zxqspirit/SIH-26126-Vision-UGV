"""Standalone and ROS 2-compatible Perception Inference Engine for SIH 26126.

Integrates the trained MobileNetV3-Large + LR-ASPP semantic segmentation model,
converting raw RGB optical imagery into:
1. Dense 9-class segmentation masks (mono8: 0..8).
2. Traversability cost maps (mono8: 0..100).
3. Perception confidence scalars ([0.0, 1.0]).
4. Colorized visual overlays (rgb8).

Supports both PyTorch (.pth) and ONNX (.onnx) backends with automatic device detection,
frame-accurate timestamping, and comprehensive fault-tolerant error handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F

from src.perception.models import create_lraspp_mobilenet_v3

logger = logging.getLogger("sih_perception")

# 9-Class Ontology Color Palette [R, G, B]
PALETTE_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (128, 128, 128),  # UNKNOWN (Grey)
    1: (80, 80, 80),     # PAVED_ROAD (Dark Grey)
    2: (160, 100, 50),   # TRAVERSABLE_DIRT (Brown)
    3: (0, 200, 0),      # LOW_GRASS (Bright Green)
    4: (180, 180, 120),  # GRAVEL (Beige)
    5: (0, 90, 0),       # HIGH_VEGETATION (Forest Green)
    6: (220, 20, 20),    # OBSTACLE_SOLID (Red)
    7: (0, 150, 255),    # WATER_PUDDLE (Cyan)
    8: (255, 100, 0),    # DYNAMIC_OBSTACLE (Orange)
}

# Standardized Traversability Cost Lookup Table [0..100]
TERRAIN_COST_TABLE: np.ndarray = np.array([
    50,   # 0: UNKNOWN (Cautious neutral cost)
    5,    # 1: PAVED_ROAD (Optimal highway/asphalt)
    15,   # 2: TRAVERSABLE_DIRT (Standard nominal dirt path)
    20,   # 3: LOW_GRASS (Safe low ground cover)
    25,   # 4: GRAVEL (Loose pebble terrain)
    80,   # 5: HIGH_VEGETATION (Non-traversable tall brush)
    100,  # 6: OBSTACLE_SOLID (Lethal collision hazard)
    90,   # 7: WATER_PUDDLE (Near-lethal submergence/slip risk)
    100,  # 8: DYNAMIC_OBSTACLE (Lethal dynamic object)
], dtype=np.uint8)

# Standard ImageNet normalization coefficients
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class PerceptionOutput:
    """Strongly-typed perception output contract."""
    segmentation_mask: np.ndarray  # Shape (H, W), uint8 in [0..8]
    terrain_cost: np.ndarray  # Shape (H, W), uint8 in [0..100]
    confidence: float  # Scalar in [0.0, 1.0]
    colored_mask: np.ndarray  # Shape (H, W, 3), uint8 in RGB
    timestamp: float  # Preserved from source camera packet (seconds)
    latency_ms: float  # End-to-end execution time in milliseconds
    is_valid: bool  # False if frame was degraded/corrupted
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class PerceptionInferenceEngine:
    """Production perception engine supporting PyTorch and ONNX inference."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_backend: str = "pytorch",
        input_size: Tuple[int, int] = (512, 512),
        device: str = "auto",
        confidence_threshold: float = 0.60,
    ) -> None:
        self.input_width, self.input_height = input_size
        self.model_backend = model_backend.lower()
        self.confidence_threshold = confidence_threshold

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Default model path if not specified
        if model_path is None:
            default_pth = os.path.join(project_root, "models", "checkpoints", "production_v1", "checkpoint_best_mIoU.pth")
            default_onnx = os.path.join(project_root, "models", "checkpoints", "production_v1", "mobilenetv3_lraspp_best.onnx")
            if self.model_backend == "onnx" and os.path.isfile(default_onnx):
                model_path = default_onnx
            elif os.path.isfile(default_pth):
                model_path = default_pth
            else:
                model_path = ""

        self.model_path = model_path
        self.torch_model: Optional[torch.nn.Module] = None
        self.ort_session = None

        # Telemetry metrics
        self.total_frames_processed = 0
        self.total_latency_ms = 0.0
        self.latencies_window: List[float] = []

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Loads and prepares model weights."""
        if not self.model_path or not os.path.isfile(self.model_path):
            logger.warning(
                f"Model path '{self.model_path}' not found. Initializing untrained MobileNetV3-Large + LR-ASPP."
            )
            self.torch_model = create_lraspp_mobilenet_v3(num_classes=9, pretrained_backbone=False).to(self.device)
            self.torch_model.eval()
            return

        if self.model_backend == "onnx" or self.model_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
                self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
                self.model_backend = "onnx"
                logger.info(f"Loaded ONNX model '{self.model_path}' with providers: {self.ort_session.get_providers()}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX session: {e}. Falling back to PyTorch.")
                self.model_backend = "pytorch"

        if self.model_backend == "pytorch":
            self.torch_model = create_lraspp_mobilenet_v3(num_classes=9, pretrained_backbone=False).to(self.device)
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.torch_model.load_state_dict(state_dict)
                logger.info(f"Loaded PyTorch checkpoint '{self.model_path}' onto {self.device}")
            except Exception as e:
                logger.error(f"Error loading checkpoint '{self.model_path}': {e}. Using base initialization.")

            self.torch_model.eval()

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Converts single-channel class ID mask to RGB visualization."""
        h, w = mask.shape
        rgb_color = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id, color in PALETTE_RGB.items():
            rgb_color[mask == cls_id] = color
        return rgb_color

    def _create_fallback_output(
        self,
        height: int,
        width: int,
        timestamp: float,
        reason: str,
    ) -> PerceptionOutput:
        """Generates safe degraded output when incoming frame is invalid."""
        safe_mask = np.zeros((height, width), dtype=np.uint8)  # Class 0: UNKNOWN
        safe_cost = np.full((height, width), 50, dtype=np.uint8)  # Cost 50: Neutral/Caution
        safe_color = self._colorize_mask(safe_mask)

        return PerceptionOutput(
            segmentation_mask=safe_mask,
            terrain_cost=safe_cost,
            confidence=0.0,
            colored_mask=safe_color,
            timestamp=timestamp,
            latency_ms=0.0,
            is_valid=False,
            diagnostics={"error": reason, "status": "DEGRADED_FALLBACK"},
        )

    def process_frame(
        self,
        rgb_image: Optional[np.ndarray],
        timestamp: float = 0.0,
    ) -> PerceptionOutput:
        """Processes raw RGB camera frame through segmentation and cost mapping."""
        t_start = time.perf_counter()

        # 1. Invalid Frame Guard
        if rgb_image is None or rgb_image.size == 0:
            return self._create_fallback_output(480, 640, timestamp, "NULL_OR_EMPTY_FRAME")

        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            return self._create_fallback_output(480, 640, timestamp, f"INVALID_SHAPE_{rgb_image.shape}")

        orig_h, orig_w = rgb_image.shape[:2]

        try:
            # 2. Preprocessing & Standardization
            t_pre_start = time.perf_counter()
            resized = cv2.resize(rgb_image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            norm_img = resized.astype(np.float32) / 255.0
            norm_img = (norm_img - IMAGENET_MEAN) / IMAGENET_STD

            # Shape: (1, 3, H, W)
            tensor_input = np.transpose(norm_img, (2, 0, 1))
            tensor_input = np.expand_dims(tensor_input, axis=0)
            t_pre = (time.perf_counter() - t_pre_start) * 1000.0

            # 3. Model Inference
            t_infer_start = time.perf_counter()
            if self.model_backend == "onnx" and self.ort_session is not None:
                input_name = self.ort_session.get_inputs()[0].name
                outputs = self.ort_session.run(None, {input_name: tensor_input})
                logits_np = outputs[0][0]  # Shape: (9, H, W)
                # Compute softmax and argmax on numpy
                exp_logits = np.exp(logits_np - np.max(logits_np, axis=0, keepdims=True))
                probs_np = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
                pred_map = np.argmax(probs_np, axis=0).astype(np.uint8)
                max_probs = np.max(probs_np, axis=0)
            else:
                torch_in = torch.from_numpy(tensor_input).to(self.device)
                with torch.no_grad():
                    out = self.torch_model(torch_in)
                    logits = out["out"] if isinstance(out, dict) else out
                    probs = F.softmax(logits, dim=1)[0]
                    pred = logits.argmax(dim=1)[0]

                pred_map = pred.cpu().numpy().astype(np.uint8)
                max_probs = probs.max(dim=0)[0].cpu().numpy()

            t_infer = (time.perf_counter() - t_infer_start) * 1000.0

            # 4. Post-processing & Cost Mapping
            t_post_start = time.perf_counter()

            # Resize predicted class map back to native camera resolution
            if (orig_h, orig_w) != (self.input_height, self.input_width):
                full_class_map = cv2.resize(pred_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            else:
                full_class_map = pred_map

            # Deterministic Traversability Cost Lookup
            # Ensures all indices are within valid range [0, 8]
            safe_indices = np.clip(full_class_map, 0, 8)
            terrain_cost_map = TERRAIN_COST_TABLE[safe_indices]

            # Confidence Metric Calculation
            # Average peak probability penalized by fraction of UNKNOWN pixels
            raw_confidence = float(np.mean(max_probs))
            unknown_fraction = float(np.mean(pred_map == 0))
            adjusted_confidence = float(np.clip(raw_confidence * (1.0 - 0.7 * unknown_fraction), 0.0, 1.0))

            # Visual overlay
            color_mask = self._colorize_mask(full_class_map)
            t_post = (time.perf_counter() - t_post_start) * 1000.0

            total_latency = (time.perf_counter() - t_start) * 1000.0

            # Update telemetry
            self.total_frames_processed += 1
            self.total_latency_ms += total_latency
            self.latencies_window.append(total_latency)
            if len(self.latencies_window) > 100:
                self.latencies_window.pop(0)

            rolling_fps = 1000.0 / float(np.mean(self.latencies_window)) if self.latencies_window else 0.0

            return PerceptionOutput(
                segmentation_mask=full_class_map,
                terrain_cost=terrain_cost_map,
                confidence=adjusted_confidence,
                colored_mask=color_mask,
                timestamp=timestamp,
                latency_ms=total_latency,
                is_valid=True,
                diagnostics={
                    "pre_ms": t_pre,
                    "infer_ms": t_infer,
                    "post_ms": t_post,
                    "rolling_fps": rolling_fps,
                    "unknown_pct": unknown_fraction * 100.0,
                    "backend": self.model_backend,
                },
            )

        except Exception as e:
            logger.exception(f"Unexpected error during frame inference: {e}")
            return self._create_fallback_output(orig_h, orig_w, timestamp, f"EXCEPTION_{e}")

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Returns lifetime operational telemetry statistics."""
        mean_lat = self.total_latency_ms / max(1, self.total_frames_processed)
        p95_lat = float(np.percentile(self.latencies_window, 95)) if self.latencies_window else 0.0
        fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        return {
            "total_frames": self.total_frames_processed,
            "mean_latency_ms": mean_lat,
            "p95_latency_ms": p95_lat,
            "continuous_fps": fps,
            "backend": self.model_backend,
            "device": str(self.device),
        }

