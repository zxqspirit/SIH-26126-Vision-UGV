# Lightweight Semantic Segmentation Hardware Benchmark Report

**Project:** SIH 26126 — Vision Based Autonomous Navigation for UGV
**Compute Device:** `cpu` (Platform: `win32`)
**Standard Input Dimensions:** $512 \times 512 \times 3$

## 1. Measured Performance Results

| Architecture | Parameters (M) | Compute (GFLOPs) | Mean Latency (ms) | p95 Latency (ms) | Throughput (FPS) | ONNX Size (MB) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV3-Large + LR-ASPP** | 3.22 M | 3.93 G | 50.04 ms | 53.00 ms | **20.0 FPS** | 12.28 MB | `READY` |
| **BiSeNet V2** | 1.69 M | 11.18 G | 63.26 ms | 70.76 ms | **15.8 FPS** | 5.89 MB | `READY` |

## 2. Engineering Evaluation & Selection

1. **MobileNetV3-Large + LR-ASPP:** Selected as the **Primary Production Champion** due to official PyTorch torchvision maintenance, ImageNet pre-training availability, and flawless ONNX deployment.
2. **BiSeNet V2:** Retained as the **High-Detail Challenger** for complex off-road boundary testing.
