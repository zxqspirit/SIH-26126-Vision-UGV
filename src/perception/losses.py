"""Loss functions for SIH 26126 outdoor terrain semantic segmentation.

Combines class-weighted Cross-Entropy with multi-class Soft Dice Loss to ensure robust
gradients on sparse safety-critical terrain categories (water puddles, obstacles).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Multi-class Soft Dice Loss."""

    def __init__(self, num_classes: int = 9, smooth: float = 1.0, ignore_index: Optional[int] = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W) integer class IDs
        """
        probs = F.softmax(logits, dim=1)
        b, c, h, w = probs.shape

        # One-hot encode targets: (B, C, H, W)
        target_one_hot = F.one_hot(targets.clamp(0, self.num_classes - 1), num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        dice_per_class = []
        for cls_idx in range(self.num_classes):
            if self.ignore_index is not None and cls_idx == self.ignore_index:
                continue

            p = probs[:, cls_idx].contiguous().view(-1)
            t = target_one_hot[:, cls_idx].contiguous().view(-1)

            intersection = (p * t).sum()
            denominator = p.sum() + t.sum()

            dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            dice_per_class.append(dice)

        mean_dice = torch.stack(dice_per_class).mean()
        return 1.0 - mean_dice


class WeightedDiceCELoss(nn.Module):
    """Composite Cross-Entropy + Dice Loss."""

    def __init__(
        self,
        num_classes: int = 9,
        class_weights: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = SoftDiceLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + self.dice_weight * dice_loss
