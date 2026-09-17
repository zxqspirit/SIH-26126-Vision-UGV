"""MobileNetV3-Large with Lite Reduced Atrous Spatial Pyramid Pooling (LR-ASPP).

Primary Champion architecture for SIH 26126 lightweight terrain segmentation.
Reduces decoder compute by 30-34% compared to standard ASPP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights


def create_lraspp_mobilenet_v3(
    num_classes: int = 9,
    pretrained_backbone: bool = True,
) -> nn.Module:
    """Builds MobileNetV3-Large + LR-ASPP configured for num_classes."""
    if pretrained_backbone:
        try:
            weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
            model = lraspp_mobilenet_v3_large(weights=weights)
            # Replace final classifier layer with custom num_classes
            in_channels = model.classifier.low_classifier.in_channels
            high_channels = model.classifier.high_classifier.in_channels
            model.classifier.low_classifier = nn.Conv2d(in_channels, num_classes, 1)
            model.classifier.high_classifier = nn.Conv2d(high_channels, num_classes, 1)
            return model
        except Exception:
            # Offline or unauthenticated fallback
            return lraspp_mobilenet_v3_large(weights=None, num_classes=num_classes)
    else:
        return lraspp_mobilenet_v3_large(weights=None, num_classes=num_classes)
