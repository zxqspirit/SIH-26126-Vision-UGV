"""Structured Safety Decision Audit Logger.

Enforces deterministic traceability for every decision:
- timestamp
- confidence
- reason
- action
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from ..interfaces.types import SafetyDecisionLog, SafetyState


class SafetyDecisionLogger:
    """Logs and indexes every safety gate arbitration decision for verification."""

    def __init__(
        self,
        max_memory_records: int = 1000,
        log_file_path: Optional[str | Path] = None,
    ) -> None:
        self.max_records = max_memory_records
        self.records: Deque[SafetyDecisionLog] = deque(maxlen=max_memory_records)
        self.log_file_path: Optional[Path] = Path(log_file_path) if log_file_path else None

        if self.log_file_path:
            self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        timestamp: float,
        confidence: float,
        reason: str,
        action: str,
        state: SafetyState | str = "HIGH",
        signals: Optional[Dict[str, float]] = None,
        nominal_v: float = 0.0,
        recommended_v: float = 0.0,
        recommended_w: float = 0.0,
        clearance_margin_m: float = 0.35,
        reobserve_active: bool = False,
    ) -> SafetyDecisionLog:
        """Record an arbitration decision cycle."""
        state_str = state.value if isinstance(state, SafetyState) else str(state)
        entry = SafetyDecisionLog(
            timestamp=float(timestamp),
            confidence=float(confidence),
            reason=str(reason),
            action=str(action),
            state=state_str,
            signals=signals or {},
            nominal_v=float(nominal_v),
            recommended_v=float(recommended_v),
            recommended_w=float(recommended_w),
            clearance_margin_m=float(clearance_margin_m),
            reobserve_active=bool(reobserve_active),
        )

        self.records.append(entry)

        if self.log_file_path:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry.to_dict()) + "\n")
            except Exception:
                pass  # Graceful fallback in constrained environments

        return entry

    def get_latest(self) -> Optional[SafetyDecisionLog]:
        """Return the most recent log record."""
        return self.records[-1] if len(self.records) > 0 else None

    def get_all(self) -> List[SafetyDecisionLog]:
        """Return all in-memory records."""
        return list(self.records)

    def get_by_state(self, state: SafetyState | str) -> List[SafetyDecisionLog]:
        """Filter records by safety state."""
        state_str = state.value if isinstance(state, SafetyState) else str(state)
        return [r for r in self.records if r.state == state_str]

    def clear(self) -> None:
        """Clear the in-memory buffer."""
        self.records.clear()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute aggregated statistics over recorded decisions."""
        if not self.records:
            return {"total_records": 0}

        confidences = [r.confidence for r in self.records]
        state_counts = {}
        action_counts = {}

        for r in self.records:
            state_counts[r.state] = state_counts.get(r.state, 0) + 1
            action_counts[r.action] = action_counts.get(r.action, 0) + 1

        return {
            "total_records": len(self.records),
            "mean_confidence": round(sum(confidences) / len(confidences), 4),
            "min_confidence": round(min(confidences), 4),
            "max_confidence": round(max(confidences), 4),
            "state_distribution": state_counts,
            "action_distribution": action_counts,
        }
