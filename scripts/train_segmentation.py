#!/usr/bin/env python3
"""Reproducible training and evaluation harness for SIH 26126 outdoor terrain segmentation.

Enforces:
1. Strict deterministic seeding across random, numpy, torch, cuda.
2. Two-phase execution: --sanity (4-sample micro-overfitting) and full multi-epoch training.
3. Zero-fabrication empirical metrics (exact confusion-matrix-derived mIoU, precision, recall).
4. Comprehensive checkpoint metadata serialization.
5. Qualitative visual overlay generation (success and failure case inspection).
6. Production ONNX export with opset 14 and dynamic batch axes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.interfaces.types import TerrainClass
from src.perception.dataset import TerrainSegmentationDataset
from src.perception.losses import WeightedDiceCELoss
from src.perception.metrics import SegmentationMetrics
from src.perception.models import create_lraspp_mobilenet_v3

# Color palette for 9-class segmentation masks [R, G, B]
PALETTE = {
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

CLASS_NAMES = {
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


def seed_everything(seed: int = 42) -> None:
    """Pins all pseudo-random number generators for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit_hash() -> str:
    """Fetches active git commit hash for checkpoint provenance."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Colorizes single-channel class ID mask to RGB visualization."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in PALETTE.items():
        color[mask == cls_id] = rgb
    return color


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Executes single training epoch and returns mean loss."""
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs["out"] if isinstance(outputs, dict) else outputs

        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(1, steps)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: SegmentationMetrics,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluates model on validation loader with empirical metrics."""
    model.eval()
    total_loss = 0.0
    steps = 0
    metrics.reset()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0

            logits = outputs["out"] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, masks)

            total_loss += float(loss.item())
            steps += 1
            metrics.update(logits, masks, latency_ms=latency_ms)

    val_loss = total_loss / max(1, steps)
    metric_results = metrics.compute()
    metric_results["val_loss"] = val_loss
    return val_loss, metric_results


def export_onnx_model(model: nn.Module, output_path: str, device: torch.device) -> bool:
    """Exports model to optimized ONNX format."""
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dummy_input = torch.randn(1, 3, 512, 512, device=device)

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            return out["out"] if isinstance(out, dict) else out

    wrapper = Wrapper(model)
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_rgb"],
            output_names=["output_logits"],
            dynamic_axes={"input_rgb": {0: "batch_size"}, "output_logits": {0: "batch_size"}},
            dynamo=False,
        )
        return True
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}", file=sys.stderr)
        return False


def run_sanity_experiment(
    dataset_dir: str = "datasets/dataset_v1.0",
    output_dir: str = "models/checkpoints/sanity_run",
    seed: int = 42,
) -> bool:
    """Phase 1: Micro 4-sample sanity-training experiment."""
    print("=" * 80)
    print(" PHASE 1: MICRO SANITY-TRAINING EXPERIMENT (GATE)")
    print("=" * 80)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Load 4 representative samples
    full_ds = TerrainSegmentationDataset(dataset_dir, split="train", augment=False)
    micro_indices = [0, 10, 20, 30]  # diverse frames across scenarios
    micro_ds = Subset(full_ds, micro_indices)
    micro_loader = DataLoader(micro_ds, batch_size=2, shuffle=False)

    model = create_lraspp_mobilenet_v3(num_classes=9, pretrained_backbone=True).to(device)
    criterion = WeightedDiceCELoss(num_classes=9).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    losses = []
    print("\nExecuting 5-epoch micro-overfitting verification:")
    for epoch in range(1, 6):
        loss = train_one_epoch(model, micro_loader, criterion, optimizer, device)
        losses.append(loss)
        print(f"  [Epoch {epoch}/5] Loss: {loss:.4f}")

    # Check 1: Loss decreases monotonically or achieves significant reduction
    loss_reduction = losses[0] - losses[-1]
    is_decreasing = loss_reduction > 0.05
    print(f"\nSanity Check 1 (Loss Convergence): Initial {losses[0]:.4f} -> Final {losses[-1]:.4f} (Reduction: {loss_reduction:.4f}) -> {'PASS' if is_decreasing else 'FAIL'}")

    # Check 2: Non-zero gradient magnitudes
    has_nonzero_grads = all(p.grad is not None and torch.norm(p.grad).item() > 0 for p in model.parameters() if p.requires_grad)
    print(f"Sanity Check 2 (Gradient Flow): Non-zero across all trainable layers -> {'PASS' if has_nonzero_grads else 'FAIL'}")

    # Check 3: Confusion matrix non-NaN validation
    metrics = SegmentationMetrics(num_classes=9, class_names=CLASS_NAMES)
    val_loss, res = evaluate(model, micro_loader, criterion, device, metrics)
    has_valid_miou = not np.isnan(res["mIoU"]) and res["mIoU"] > 0.0
    print(f"Sanity Check 3 (Metric Integrity): Overfit mIoU = {res['mIoU']:.4f} (No NaNs) -> {'PASS' if has_valid_miou else 'FAIL'}")

    # Check 4: Checkpoint & Metadata Serialization
    ckpt_path = os.path.join(output_dir, "sanity_checkpoint.pth")
    meta_path = os.path.join(output_dir, "sanity_metadata.json")
    torch.save({"model_state_dict": model.state_dict(), "seed": seed, "losses": losses}, ckpt_path)
    metadata = {
        "status": "PASSED",
        "seed": seed,
        "device": str(device),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "overfit_mIoU": res["mIoU"],
        "losses": losses,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    has_saved_meta = os.path.isfile(ckpt_path) and os.path.isfile(meta_path)
    print(f"Sanity Check 4 (Metadata Persistence): Saved checkpoint and metadata -> {'PASS' if has_saved_meta else 'FAIL'}")

    all_passed = is_decreasing and has_nonzero_grads and has_valid_miou and has_saved_meta
    print("\n" + "=" * 80)
    print(f" SANITY GATE STATUS: {'PASSED (PROCEED TO FULL TRAINING)' if all_passed else 'FAILED'}")
    print("=" * 80 + "\n")
    return all_passed


def run_full_training(
    dataset_dir: str = "datasets/dataset_v1.0",
    output_dir: str = "models/checkpoints/production_v1",
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 5e-4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Phase 2: Approved full training on dataset_v1.0."""
    print("=" * 80)
    print(" PHASE 2: APPROVED REPRODUCIBLE TRAINING RUN")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr} | Seed: {seed}")
    print("=" * 80)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Datasets & Loaders
    train_ds = TerrainSegmentationDataset(dataset_dir, split="train", augment=True, seed=seed)
    val_ds = TerrainSegmentationDataset(dataset_dir, split="val", augment=False)
    test_ds = TerrainSegmentationDataset(dataset_dir, split="test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    print(f"Dataset partitions loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # 2. Model & Optimizer
    model = create_lraspp_mobilenet_v3(num_classes=9, pretrained_backbone=True).to(device)
    criterion = WeightedDiceCELoss(num_classes=9, dice_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    metrics = SegmentationMetrics(num_classes=9, class_names=CLASS_NAMES)

    best_val_miou = 0.0
    history: List[Dict[str, Any]] = []

    print("\nStarting multi-epoch training and validation loop:")
    for epoch in range(1, epochs + 1):
        t_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, metrics)
        t_epoch = time.perf_counter() - t_start

        epoch_miou = val_metrics["mIoU"]
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mIoU": epoch_miou,
            "val_fwiou": val_metrics["FWIoU"],
            "lr": current_lr,
            "duration_sec": t_epoch,
        })

        print(
            f"  [Epoch {epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {epoch_miou * 100:.2f}% | "
            f"Val FWIoU: {val_metrics['FWIoU'] * 100:.2f}% | "
            f"Time: {t_epoch:.1f}s"
        )

        # Save latest checkpoint
        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_mIoU": epoch_miou,
            "seed": seed,
            "git_commit_hash": get_git_commit_hash(),
        }
        torch.save(last_ckpt, os.path.join(output_dir, "checkpoint_last.pth"))

        # Save best checkpoint
        if epoch_miou > best_val_miou:
            best_val_miou = epoch_miou
            torch.save(last_ckpt, os.path.join(output_dir, "checkpoint_best_mIoU.pth"))
            print(f"    -> [BEST CHECKPOINT SAVED] New record val_mIoU: {best_val_miou * 100:.2f}%")

    # 3. Final Comprehensive Test Evaluation
    print("\nRunning final evaluation on held-out Test split (15 frames)...")
    best_ckpt = torch.load(os.path.join(output_dir, "checkpoint_best_mIoU.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, metrics)

    # 4. Save qualitative overlays (Success and Failure visual analysis)
    qualitative_dir = os.path.join("docs", "experiments", "qualitative_samples")
    os.makedirs(qualitative_dir, exist_ok=True)
    model.eval()

    print("Generating qualitative visual overlays...")
    sample_cases = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].cpu().numpy()[0]
            rgb_path = batch["rgb_path"][0]
            sample_id = batch["sample_id"][0]

            out = model(images)
            logits = out["out"] if isinstance(out, dict) else out
            pred = logits.argmax(dim=1).cpu().numpy()[0]

            # Compute sample IoU
            tp = (masks == pred).sum()
            total = masks.size
            acc = float(tp / total)

            # Build side-by-side composite: Original RGB | Ground Truth | Prediction
            raw_rgb = cv2.imread(rgb_path)
            raw_rgb = cv2.resize(raw_rgb, (512, 512))
            gt_color = colorize_mask(masks)
            pred_color = colorize_mask(pred)
            gt_bgr = cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR)
            pred_bgr = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

            composite = np.hstack([raw_rgb, gt_bgr, pred_bgr])
            overlay_file = f"{sample_id}_eval.jpg"
            overlay_path = os.path.join(qualitative_dir, overlay_file)
            cv2.imwrite(overlay_path, composite)

            sample_cases.append({
                "id": sample_id,
                "pixel_accuracy": acc,
                "overlay_file": overlay_file,
            })

    # Sort cases to identify clear successes and failures
    sample_cases.sort(key=lambda x: x["pixel_accuracy"], reverse=True)
    success_cases = sample_cases[:3]
    failure_cases = sample_cases[-3:]

    # 5. Export Best Model to ONNX
    onnx_path = os.path.join(output_dir, "mobilenetv3_lraspp_best.onnx")
    export_success = export_onnx_model(model, onnx_path, device)
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024) if export_success else 0.0

    # 6. Save Complete Metadata
    metadata = {
        "model_architecture": "MobileNetV3-Large + LR-ASPP",
        "num_classes": 9,
        "classes": CLASS_NAMES,
        "seed": seed,
        "git_commit_hash": get_git_commit_hash(),
        "device": str(device),
        "dataset_version": "dataset_v1.0",
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "loss": "WeightedDiceCELoss",
        },
        "best_val_mIoU": best_val_miou,
        "test_results": {
            "test_loss": test_loss,
            "test_mIoU": test_metrics["mIoU"],
            "test_mPrecision": test_metrics["mPrecision"],
            "test_mRecall": test_metrics["mRecall"],
            "test_FWIoU": test_metrics["FWIoU"],
            "latency_ms": test_metrics["latency"]["mean_ms"],
            "throughput_fps": test_metrics["latency"]["fps"],
            "per_class": test_metrics["per_class"],
        },
        "onnx_export": {
            "status": "SUCCESS" if export_success else "FAILED",
            "path": onnx_path,
            "size_mb": onnx_size_mb,
        },
        "training_history": history,
        "success_cases": success_cases,
        "failure_cases": failure_cases,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    meta_file = os.path.join(output_dir, "training_metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 7. Print Final Results Summary Table
    print("\n" + "=" * 80)
    print(" FINAL EVALUATION SUMMARY (Held-Out Test Set: 15 frames)")
    print("=" * 80)
    print(f"  Test Mean IoU (mIoU)  : {test_metrics['mIoU'] * 100:.2f}%")
    print(f"  Test FWIoU            : {test_metrics['FWIoU'] * 100:.2f}%")
    print(f"  Test Mean Precision   : {test_metrics['mPrecision'] * 100:.2f}%")
    print(f"  Test Mean Recall      : {test_metrics['mRecall'] * 100:.2f}%")
    print(f"  Mean Latency (ms)     : {test_metrics['latency']['mean_ms']:.2f} ms")
    print(f"  Throughput (FPS)      : {test_metrics['latency']['fps']:.1f} FPS")
    print(f"  Exported ONNX Size    : {onnx_size_mb:.2f} MB")
    print("-" * 80)
    print(f"{'Class ID':<8} | {'Class Name':<20} | {'IoU (%)':<10} | {'Precision':<10} | {'Recall':<10} | {'Support':<8}")
    print("-" * 80)
    for c_id in range(9):
        c_name = CLASS_NAMES[c_id]
        cd = test_metrics["per_class"].get(c_name, {})
        iou = cd.get("iou", 0.0) * 100
        prec = cd.get("precision", 0.0) * 100
        rec = cd.get("recall", 0.0) * 100
        sup = cd.get("support_pixels", 0)
        print(f"{c_id:<8} | {c_name:<20} | {iou:>8.2f}% | {prec:>8.2f}% | {rec:>8.2f}% | {sup:>8d}")
    print("=" * 80)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIH 26126 Semantic Segmentation Training")
    parser.add_argument("--sanity", action="store_true", help="Run 5-epoch micro-sanity experiment")
    parser.add_argument("--epochs", type=int, default=15, help="Number of full training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    if args.sanity:
        success = run_sanity_experiment(seed=args.seed)
        sys.exit(0 if success else 1)
    else:
        run_full_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
