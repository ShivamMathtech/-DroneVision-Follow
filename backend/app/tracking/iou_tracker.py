from __future__ import annotations

import math
import time

from app.schemas import FaceDetection, Track


def intersection_over_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


class IoUTracker:
    """Greedy IoU plus center-distance association for a small number of face tracks."""

    def __init__(self, max_age: int = 20, max_center_distance: float = 100.0) -> None:
        self.max_age = max_age
        self.max_center_distance = max_center_distance
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: list[FaceDetection], now: float | None = None) -> list[Track]:
        now = now if now is not None else time.time()
        available_tracks = set(self.tracks)
        available_detections = set(range(len(detections)))
        candidates: list[tuple[float, int, int]] = []
        for track_id in available_tracks:
            track = self.tracks[track_id]
            tcx, tcy = track.center
            tw, th = track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1]
            for index, detection in enumerate(detections):
                dcx = (detection.bbox[0] + detection.bbox[2]) / 2
                dcy = (detection.bbox[1] + detection.bbox[3]) / 2
                distance = math.hypot(dcx - tcx, dcy - tcy)
                iou = intersection_over_union(track.bbox, detection.bbox)
                limit = max(self.max_center_distance, 0.8 * max(tw, th))
                if iou >= 0.05 or distance <= limit:
                    score = iou + max(0.0, 1.0 - distance / max(limit, 1.0)) * 0.25
                    candidates.append((score, track_id, index))
        candidates.sort(reverse=True)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_id, index in candidates:
            if track_id in matched_tracks or index in matched_detections:
                continue
            track = self.tracks[track_id]
            detection = detections[index]
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.last_seen = now
            track.hits += 1
            track.misses = 0
            track.state = "ACTIVE"
            track.trajectory.append((now, *track.center))
            track.trajectory = track.trajectory[-300:]
            matched_tracks.add(track_id)
            matched_detections.add(index)
        for track_id in list(self.tracks):
            if track_id not in matched_tracks:
                track = self.tracks[track_id]
                track.misses += 1
                track.state = "LOST" if track.misses > 0 else track.state
                if track.misses > self.max_age:
                    del self.tracks[track_id]
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track = Track(self.next_id, detection.bbox, detection.confidence, now, now)
            track.trajectory.append((now, *track.center))
            self.tracks[self.next_id] = track
            self.next_id += 1
        return list(self.tracks.values())

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1
