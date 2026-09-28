from __future__ import annotations

import time
from dataclasses import dataclass

from app.schemas import Track


@dataclass
class TargetEvent:
    type: str
    severity: str
    description: str
    target_id: int | None
    timestamp: float


class TargetManager:
    def __init__(self, lost_timeout: float = 2.0) -> None:
        self.lost_timeout = lost_timeout
        self.target_id: int | None = None
        self.state = "NO_TARGET"
        self.lost_at: float | None = None
        self.last_event: TargetEvent | None = None

    def _transition(self, state: str, event_type: str, description: str, severity: str = "info") -> list[TargetEvent]:
        if state == self.state:
            return []
        self.state = state
        event = TargetEvent(event_type, severity, description, self.target_id, time.time())
        self.last_event = event
        return [event]

    def select(self, track_id: int, tracks: list[Track]) -> tuple[bool, list[TargetEvent]]:
        if not any(t.track_id == track_id and t.state == "ACTIVE" for t in tracks):
            return False, []
        self.target_id = track_id
        self.lost_at = None
        return True, self._transition("TARGET_LOCKED", "TARGET_ACQUIRED", f"Target #{track_id} locked.")

    def select_at(self, x: float, y: float, width: int, height: int, tracks: list[Track]) -> tuple[bool, list[TargetEvent]]:
        px, py = x * width, y * height
        candidates = [t for t in tracks if t.state == "ACTIVE" and t.bbox[0] <= px <= t.bbox[2] and t.bbox[1] <= py <= t.bbox[3]]
        if not candidates:
            return False, []
        candidates.sort(key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]), reverse=True)
        return self.select(candidates[0].track_id, tracks)

    def release(self) -> list[TargetEvent]:
        old_id = self.target_id
        self.target_id = None
        self.lost_at = None
        return self._transition("TARGET_RELEASED", "TARGET_RELEASED", f"Target #{old_id} released.")

    def update(self, tracks: list[Track], now: float | None = None) -> list[TargetEvent]:
        now = now if now is not None else time.time()
        if self.target_id is None:
            return []
        target = next((t for t in tracks if t.track_id == self.target_id and t.state == "ACTIVE"), None)
        if target:
            was_missing = self.lost_at is not None
            self.lost_at = None
            if was_missing:
                return self._transition("TARGET_REACQUIRED", "TARGET_REACQUIRED", f"Target #{self.target_id} reacquired.")
            if self.state not in {"TARGET_LOCKED", "TARGET_TRACKING", "TARGET_REACQUIRED"}:
                return self._transition("TARGET_LOCKED", "TARGET_REACQUIRED", f"Target #{self.target_id} reacquired.")
            if self.state == "TARGET_LOCKED":
                return self._transition("TARGET_TRACKING", "TARGET_TRACKING", f"Tracking target #{self.target_id}.")
            if self.state == "TARGET_REACQUIRED":
                return self._transition("TARGET_TRACKING", "TARGET_TRACKING", f"Tracking target #{self.target_id}.")
            return []
        if self.lost_at is None:
            self.lost_at = now
            return self._transition("TARGET_OCCLUDED", "TARGET_TEMPORARILY_LOST", f"Target #{self.target_id} temporarily occluded.", "warning")
        if now - self.lost_at >= self.lost_timeout:
            return self._transition("TARGET_LOST", "TARGET_LOST", f"Target #{self.target_id} lost; target lock retained.", "warning")
        if self.state == "TARGET_OCCLUDED":
            return self._transition("TARGET_SEARCHING", "TARGET_SEARCHING", f"Searching for target #{self.target_id}.", "warning")
        return []

    def status(self) -> dict:
        return {"target_id": self.target_id, "state": self.state, "lost_at": self.lost_at}
