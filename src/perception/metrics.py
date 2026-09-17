"""Evaluation metrics and runtime telemetry for SIH 26126 semantic segmentation.

Enforces strict empirical integrity (zero metric fabrication).
Computes exact per-class confusion matrices, IoU, mIoU, precision, recall,
throughput (FPS), and process memory footprints.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates predictions and calculates standard semantic segmentation metrics."""

    def __init__(self, num_classes: int = 9, class_names: Optional[Dict[int, str]] = None) -> None:
        self.num_classes = num_classes
        self.class_names = class_names or {i: f"CLASS_{i}" for i in range(num_classes)}
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.latencies_ms: List[float] = []

    def reset(self) -> None:
        """Resets accumulated confusion matrix and latency logs."""
        self.confusion_matrix.fill(0)
        self.latencies_ms.clear()

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        preds: (B, H, W) or (B, C, H, W) tensor
        targets: (B, H, W) ground-truth labels
        latency_ms: elapsed inference time for this batch in ms
        """
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)

        preds_np = preds.detach().cpu().numpy().astype(np.int64).flatten()
        targets_np = targets.detach().cpu().numpy().astype(np.int64).flatten()

        # Mask valid pixels in range [0, num_classes - 1]
        valid_mask = (targets_np >= 0) & (targets_np < self.num_classes) & (preds_np >= 0) & (preds_np < self.num_classes)
        t_valid = targets_np[valid_mask]
        p_valid = preds_np[valid_mask]

        # Accumulate bincount for 2D confusion matrix
        indices = self.num_classes * t_valid + p_valid
        bincount = np.bincount(indices, minlength=self.num_classes**2)
        self.confusion_matrix += bincount.reshape((self.num_classes, self.num_classes))

        if latency_ms is not None:
            self.latencies_ms.append(latency_ms)

    def compute(self) -> Dict[str, Any]:
        """Calculates exact metrics from accumulated confusion matrix."""
        cm = self.confusion_matrix
        total_pixels = cm.sum()

        per_class: Dict[str, Dict[str, float]] = {}
        ious: List[float] = []
        precisions: List[float] = []
        recalls: List[float] = []
        class_weights: List[float] = []

        for c in range(self.num_classes):
            c_name = self.class_names.get(c, f"CLASS_{c}")
            tp = float(cm[c, c])
            fp = float(cm[:, c].sum() - tp)
            fn = float(cm[c, :].sum() - tp)
            total_ground_truth = float(cm[c, :].sum())

            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            per_class[c_name] = {
                "class_id": c,
                "iou": iou,
                "precision": prec,
                "recall": rec,
                "support_pixels": int(total_ground_truth),
            }

            if total_ground_truth > 0:
                ious.append(iou)
                precisions.append(prec)
                recalls.append(rec)
                class_weights.append(total_ground_truth / total_pixels if total_pixels > 0 else 0.0)

        miou = float(np.mean(ious)) if ious else 0.0
        mprec = float(np.mean(precisions)) if precisions else 0.0
        mrec = float(np.mean(recalls)) if recalls else 0.0
        fwiou = float(np.sum(np.array(class_weights) * np.array(ious))) if ious else 0.0

        # Latency statistics
        if self.latencies_ms:
            mean_lat = float(np.mean(self.latencies_ms))
            median_lat = float(np.median(self.latencies_ms))
            p95_lat = float(np.percentile(self.latencies_ms, 95))
            fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
        else:
            mean_lat = median_lat = p95_lat = fps = 0.0

        # Memory telemetry
        ram_mb = 0.0
        try:
            import psutil
            process = psutil.Process(os.getpid())
            ram_mb = process.memory_info().rss / (1024 * 1024)
        except Exception:
            pass

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return {
            "mIoU": miou,
            "mPrecision": mprec,
            "mRecall": mrec,
            "FWIoU": fwiou,
            "per_class": per_class,
            "latency": {
                "mean_ms": mean_lat,
                "median_ms": median_lat,
                "p95_ms": p95_lat,
                "fps": fps,
            },
            "telemetry": {
                "process_ram_mb": ram_mb,
                "gpu_vram_mb": vram_mb,
            },
            "confusion_matrix": cm.tolist(),
        }
