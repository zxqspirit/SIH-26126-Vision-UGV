"""BiSeNet V2 (Bilateral Segmentation Network).

Challenger architecture for SIH 26126 lightweight terrain segmentation.
Features a two-pathway design:
- Detail Branch: wide channels and shallow depth for high-res edge boundaries.
- Semantic Branch: deep, narrow trunk for contextual representation.
- Bilateral Guided Aggregation: fuses spatial details and contextual features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Standard Convolution + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DetailBranch(nn.Module):
    """Shallow, wide-channel path to capture high-resolution spatial boundaries (1/8 scale)."""

    def __init__(self) -> None:
        super().__init__()
        # Stage 1 (1/2 scale)
        self.s1 = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2),
            ConvBNReLU(32, 32, kernel_size=3, stride=1),
        )
        # Stage 2 (1/4 scale)
        self.s2 = nn.Sequential(
            ConvBNReLU(32, 64, kernel_size=3, stride=2),
            ConvBNReLU(64, 64, kernel_size=3, stride=1),
        )
        # Stage 3 (1/8 scale)
        self.s3 = nn.Sequential(
            ConvBNReLU(64, 128, kernel_size=3, stride=2),
            ConvBNReLU(128, 128, kernel_size=3, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        return x


class StemBlock(nn.Module):
    """Fast downsampling stem for Semantic Branch (1/4 scale)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_in = ConvBNReLU(3, 16, kernel_size=3, stride=2)
        self.left = ConvBNReLU(16, 8, kernel_size=1, stride=1, padding=0)
        self.left_conv = ConvBNReLU(8, 16, kernel_size=3, stride=2)
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(32, 16, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        left = self.left_conv(self.left(x))
        right = self.right(x)
        out = torch.cat([left, right], dim=1)
        return self.fuse(out)


class GatherExpansionBlock(nn.Module):
    """Depthwise separable inverted bottleneck for Semantic Branch."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: int = 6) -> None:
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio

        self.conv1 = ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=1)
        if stride == 2:
            self.dwconv = nn.Sequential(
                ConvBNReLU(in_channels, hidden_dim, kernel_size=3, stride=2, groups=in_channels),
                ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=1, groups=hidden_dim),
            )
            self.shortcut = nn.Sequential(
                ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.dwconv = ConvBNReLU(in_channels, hidden_dim, kernel_size=3, stride=1, groups=in_channels)
            self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.dwconv(out)
        out = self.conv_out(out)
        return self.relu(out + res)


class ContextEmbeddingBlock(nn.Module):
    """Captures global contextual receptive field using global average pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True), nn.Sigmoid())
        self.fuse = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv(self.gap(x))
        out = x * attn
        return self.fuse(out)


class BilateralGuidedAggregation(nn.Module):
    """Fuses Detail Branch and Semantic Branch representations."""

    def __init__(self) -> None:
        super().__init__()
        # Detail path
        self.detail_conv1 = ConvBNReLU(128, 128, kernel_size=3, stride=1)
        self.detail_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        # Semantic path
        self.semantic_conv1 = ConvBNReLU(128, 128, kernel_size=3, stride=1)
        self.semantic_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
        )

        self.fuse = ConvBNReLU(128, 128, kernel_size=3, stride=1)

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        # Upsample semantic to detail resolution (4x)
        semantic_up = F.interpolate(semantic, size=detail.shape[2:], mode="bilinear", align_corners=False)

        d1 = self.detail_conv1(detail)
        d2 = self.detail_conv2(detail)

        s1 = self.semantic_conv1(semantic_up)
        s2 = self.semantic_conv2(semantic)
        s2_up = F.interpolate(s2, size=detail.shape[2:], mode="bilinear", align_corners=False)

        # Cross-guidance multiplication
        branch1 = d1 * s2_up
        s2_resized = F.interpolate(s2, size=d2.shape[2:], mode='bilinear', align_corners=False); branch2 = F.interpolate(d2 * s2_resized, size=detail.shape[2:], mode='bilinear', align_corners=False)

        return self.fuse(branch1 + branch2)


class SegmentHead(nn.Module):
    """Final classification head upsampling logits to original image resolution."""

    def __init__(self, in_channels: int, num_classes: int, scale_factor: int = 8) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.cls_conv = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        logits = self.cls_conv(x)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


class BiSeNetV2(nn.Module):
    """Complete BiSeNet V2 Architecture."""

    def __init__(self, num_classes: int = 9) -> None:
        super().__init__()
        self.num_classes = num_classes

        # 1. Detail Branch
        self.detail_branch = DetailBranch()

        # 2. Semantic Branch
        self.stem = StemBlock()
        self.stage3 = nn.Sequential(
            GatherExpansionBlock(16, 32, stride=2),
            GatherExpansionBlock(32, 32, stride=1),
        )
        self.stage4 = nn.Sequential(
            GatherExpansionBlock(32, 64, stride=2),
            GatherExpansionBlock(64, 64, stride=1),
        )
        self.stage5 = nn.Sequential(
            GatherExpansionBlock(64, 128, stride=2),
            GatherExpansionBlock(128, 128, stride=1),
            ContextEmbeddingBlock(128, 128),
        )

        # 3. Bilateral Guided Aggregation
        self.bga = BilateralGuidedAggregation()

        # 4. Segment Head
        self.head = SegmentHead(128, num_classes, scale_factor=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Detail features
        detail_feat = self.detail_branch(x)

        # Semantic features
        sem = self.stem(x)
        sem = self.stage3(sem)
        sem = self.stage4(sem)
        sem = self.stage5(sem)

        # Bilateral aggregation
        fused = self.bga(detail_feat, sem)

        # Segmentation logits
        out = self.head(fused, target_size=input_size)
        return out


def create_bisenet_v2(num_classes: int = 9) -> nn.Module:
    """Factory creating BiSeNet V2."""
    return BiSeNetV2(num_classes=num_classes)
