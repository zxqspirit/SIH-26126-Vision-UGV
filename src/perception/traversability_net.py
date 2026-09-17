"""Lightweight CNN model runner for outdoor terrain traversability estimation.

Designed for real-time mobile inference (<30ms on CPU/edge GPU).
Supports ONNX Runtime execution with fallback to classical perception
when model weights are not loaded or runtime is unavailable.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple
import numpy as np
import cv2

from ..interfaces.types import SemanticResult
from .color_texture_fallback import ColorTexturePerception

logger = logging.getLogger(__name__)


class TraversabilityNet:
    """Lightweight CNN traversability segmentation engine."""

    def __init__(
        self,
        onnx_model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (640, 360),
        use_gpu: bool = True,
    ) -> None:
        self.onnx_model_path = onnx_model_path
        self.input_width, self.input_height = input_size
        self.use_gpu = use_gpu
        self.session = None
        self.fallback = ColorTexturePerception(target_size=(640, 480))

        if onnx_model_path and os.path.exists(onnx_model_path):
            self._load_onnx_model(onnx_model_path)
        else:
            logger.info("No ONNX model path specified or file not found. Using classical perception fallback.")

    def _load_onnx_model(self, model_path: str) -> None:
        """Load ONNX Runtime session if onnxruntime is available."""
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            logger.info("Loaded ONNX model: %s with providers: %s", model_path, self.session.get_providers())
        except Exception as e:
            logger.warning("Failed to initialize ONNX Runtime session: %s. Using classical fallback.", e)
            self.session = None

    def infer(self, rgb_image: np.ndarray) -> SemanticResult:
        """Run inference on RGB image."""
        if self.session is None:
            return self.fallback.process(rgb_image)

        start_time = time.perf_counter()
        orig_h, orig_w = rgb_image.shape[:2]

        # Preprocessing: resize and normalize
        resized = cv2.resize(rgb_image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32)

        # Forward pass
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: blob})
        logits = outputs[0][0]  # Shape (C, H, W)

        # Softmax over channels
        exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

        # Channel 0: Traversable ground probability
        traversability_small = probs[0]
        class_map_small = np.argmax(probs, axis=0).astype(np.int32)

        # Resize back to original dimensions
        traversability = cv2.resize(traversability_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        class_map = cv2.resize(class_map_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Compute confidence from entropy
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-7, 1.0)), axis=0)
        mean_entropy = float(np.mean(entropy))
        confidence = float(np.clip(1.0 - (mean_entropy / 1.5), 0.2, 0.98))

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SemanticResult(
            traversability_mask=traversability,
            terrain_class_map=class_map,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={"model": "onnx_cnn", "mean_entropy": mean_entropy}
        )
