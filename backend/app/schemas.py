from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: dict[str, tuple[int, int]] | None = None
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox
        return {
            "bbox": list(self.bbox), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": max(0, x2 - x1), "height": max(0, y2 - y1),
            "center_x": (x1 + x2) / 2, "center_y": (y1 + y2) / 2,
            "confidence": self.confidence, "landmarks": self.landmarks,
            "timestamp": self.timestamp,
        }


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1
    misses: int = 0
    state: str = "ACTIVE"
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def to_dict(self) -> dict[str, Any]:
        x1, y1, x2, y2 = self.bbox
        cx, cy = self.center
        return {
            "track_id": self.track_id, "bbox": list(self.bbox),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "center_x": cx, "center_y": cy, "confidence": self.confidence,
            "age": self.hits, "misses": self.misses, "last_seen": self.last_seen,
            "state": self.state, "trajectory": [list(p) for p in self.trajectory],
        }


def dataclass_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
