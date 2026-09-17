"""Perception engine orchestrator."""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np

from ..interfaces.types import SensorFrame, SemanticResult
from .traversability_net import TraversabilityNet

logger = logging.getLogger(__name__)


class PerceptionEngine:
    """Perception orchestrator: maps SensorFrame to SemanticResult."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.net = TraversabilityNet(onnx_model_path=model_path)

    def process_frame(self, frame: SensorFrame) -> SemanticResult:
        """Process incoming sensor frame."""
        return self.net.infer(frame.rgb)
