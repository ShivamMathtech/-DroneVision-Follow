from __future__ import annotations

import math
from collections import deque


class MotionAnalyzer:
    def __init__(self, history_seconds: float = 60.0) -> None:
        self.history = deque(maxlen=3000)
        self.history_seconds = history_seconds
        self.previous: tuple[int, float, float, float, float] | None = None
        self.previous_time: float | None = None

    def update(self, target, frame_width: int, frame_height: int, now: float,
               distance_enabled: bool = True, face_width_meters: float = 0.16,
               focal_px: float = 900.0) -> dict:
        if target is None:
            self.previous = None
            return {"target_id": None, "speed": None, "acceleration": None, "direction": "NO TARGET",
                    "error_x": None, "error_y": None, "normalized_x": None, "normalized_y": None,
                    "estimated_distance": None, "trajectory": list(self.history)}
        cx, cy = target.center
        nx, ny = cx / max(frame_width, 1), cy / max(frame_height, 1)
        error_x, error_y = nx - 0.5, ny - 0.5
        speed = acceleration = velocity_x = velocity_y = 0.0
        direction = "STATIONARY"
        previous = self.previous
        if previous and previous[0] == target.track_id and self.previous_time is not None:
            _, px, py, pvx, pvy = previous
            dt = max(0.001, now - self.previous_time)
            velocity_x, velocity_y = (cx - px) / dt, (cy - py) / dt
            speed = math.hypot(velocity_x, velocity_y)
            acceleration = math.hypot(velocity_x - pvx, velocity_y - pvy) / dt
            if speed < 8:
                direction = "STATIONARY"
            else:
                horizontal = "RIGHT" if velocity_x > 0 else "LEFT"
                vertical = "DOWN" if velocity_y > 0 else "UP"
                if abs(velocity_x) > abs(velocity_y) * 1.5:
                    direction = horizontal
                elif abs(velocity_y) > abs(velocity_x) * 1.5:
                    direction = vertical
                else:
                    direction = vertical + "-" + horizontal
        self.previous_time = now
        self.previous = (target.track_id, cx, cy, velocity_x, velocity_y)
        bbox_width = max(1, target.bbox[2] - target.bbox[0])
        distance = focal_px * face_width_meters / bbox_width if distance_enabled else None
        point = {"timestamp": now, "x": cx, "y": cy, "normalized_x": nx, "normalized_y": ny}
        self.history.append(point)
        cutoff = now - self.history_seconds
        while self.history and self.history[0]["timestamp"] < cutoff:
            self.history.popleft()
        return {
            "target_id": target.track_id, "center_x": cx, "center_y": cy,
            "normalized_x": nx, "normalized_y": ny, "error_x": error_x, "error_y": error_y,
            "velocity_x": velocity_x, "velocity_y": velocity_y, "speed": speed,
            "acceleration": acceleration, "direction": direction,
            "estimated_distance": distance, "distance_is_estimate": distance is not None,
            "trajectory": list(self.history),
        }

    def reset(self) -> None:
        self.history.clear()
        self.previous = None
        self.previous_time = None
