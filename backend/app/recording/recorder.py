from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2


TELEMETRY_FIELDS = [
    "timestamp", "frame_id", "fps", "inference_latency_ms", "detector_latency_ms",
    "tracker_latency_ms", "face_count", "target_id", "target_confidence",
    "x1", "y1", "x2", "y2", "target_center_x", "target_center_y",
    "normalized_x", "normalized_y", "error_x", "error_y", "velocity_x",
    "velocity_y", "speed", "acceleration", "direction", "estimated_distance",
    "tracking_status", "drone_mode",
]


class SessionRecorder:
    def __init__(self, sessions_dir: Path, metadata: dict) -> None:
        self.sessions_dir = sessions_dir
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.session_id = "session_" + stamp
        self.path = sessions_dir / self.session_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.metadata = {**metadata, "session_id": self.session_id, "start_time": datetime.now(timezone.utc).isoformat()}
        self.metadata_path = self.path / "metadata.json"
        self._write_json(self.metadata_path, self.metadata)
        self.telemetry_file = (self.path / "telemetry.csv").open("w", newline="", encoding="utf-8")
        self.telemetry_writer = csv.DictWriter(self.telemetry_file, fieldnames=TELEMETRY_FIELDS, extrasaction="ignore")
        self.telemetry_writer.writeheader()
        self.events_file = (self.path / "events.csv").open("w", newline="", encoding="utf-8")
        self.events_writer = csv.DictWriter(self.events_file, fieldnames=["timestamp", "type", "severity", "description", "target_id"])
        self.events_writer.writeheader()
        self.trajectory_file = (self.path / "trajectory.csv").open("w", newline="", encoding="utf-8")
        self.trajectory_writer = csv.DictWriter(self.trajectory_file, fieldnames=["timestamp", "x", "y", "normalized_x", "normalized_y"])
        self.trajectory_writer.writeheader()
        self.tracking_file = (self.path / "tracking.csv").open("w", newline="", encoding="utf-8")
        self.tracking_writer = csv.DictWriter(
            self.tracking_file,
            fieldnames=["timestamp", "frame_id", "track_id", "x1", "y1", "x2", "y2",
                        "center_x", "center_y", "confidence", "age", "misses", "state", "selected_target"],
        )
        self.tracking_writer.writeheader()
        self.video_raw = None
        self.video_annotated = None
        self.video_size = None
        self.lock = threading.Lock()
        self.started = time.time()
        self.rows = 0
        self.latencies: list[float] = []
        self.confidences: list[float] = []
        self.speeds: list[float] = []
        self.distances: list[float] = []
        self.target_losses = 0
        self.reacquisitions = 0
        self.event_rows: list[dict] = []
        self.closed = False

    @staticmethod
    def _write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

    def write_event(self, event: dict) -> None:
        with self.lock:
            if self.closed:
                return
            self.events_writer.writerow(event)
            self.event_rows.append(dict(event))
            self.events_file.flush()
            if event.get("type") == "TARGET_LOST":
                self.target_losses += 1
            if event.get("type") == "TARGET_REACQUIRED":
                self.reacquisitions += 1

    def write_frame(self, raw, annotated, telemetry: dict) -> None:
        with self.lock:
            if self.closed:
                return
            height, width = raw.shape[:2]
            if self.video_size != (width, height):
                self._close_video()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = max(1.0, float(self.metadata.get("fps", 25)))
                self.video_raw = cv2.VideoWriter(str(self.path / "raw_video.mp4"), fourcc, fps, (width, height))
                self.video_annotated = cv2.VideoWriter(str(self.path / "annotated_video.mp4"), fourcc, fps, (width, height))
                self.video_size = (width, height)
            if self.video_raw is not None and self.video_raw.isOpened():
                self.video_raw.write(raw)
            if self.video_annotated is not None and self.video_annotated.isOpened():
                self.video_annotated.write(annotated)
            row = {key: telemetry.get(key) for key in TELEMETRY_FIELDS}
            self.telemetry_writer.writerow(row)
            self.telemetry_file.flush()
            self.rows += 1
            target_id = telemetry.get("target_id")
            for track in telemetry.get("tracks", []):
                self.tracking_writer.writerow({
                    "timestamp": telemetry.get("timestamp"), "frame_id": telemetry.get("frame_id"),
                    "track_id": track.get("track_id"), "x1": track.get("x1"), "y1": track.get("y1"),
                    "x2": track.get("x2"), "y2": track.get("y2"),
                    "center_x": track.get("center_x"), "center_y": track.get("center_y"),
                    "confidence": track.get("confidence"), "age": track.get("age"),
                    "misses": track.get("misses"), "state": track.get("state"),
                    "selected_target": track.get("track_id") == target_id,
                })
            self.tracking_file.flush()
            latency = telemetry.get("inference_latency_ms")
            if isinstance(latency, (int, float)):
                self.latencies.append(float(latency))
            confidence = telemetry.get("target_confidence")
            if isinstance(confidence, (int, float)):
                self.confidences.append(float(confidence))
            speed = telemetry.get("speed")
            if isinstance(speed, (int, float)):
                self.speeds.append(float(speed))
            distance = telemetry.get("estimated_distance")
            if isinstance(distance, (int, float)):
                self.distances.append(float(distance))
            for point in telemetry.get("trajectory", [])[-1:]:
                self.trajectory_writer.writerow(point)
            self.trajectory_file.flush()

    def _close_video(self) -> None:
        for writer in (self.video_raw, self.video_annotated):
            if writer is not None:
                writer.release()
        self.video_raw = self.video_annotated = None

    def stop(self) -> dict:
        with self.lock:
            if self.closed:
                summary_path = self.path / "session_summary.json"
                return json.loads(summary_path.read_text()) if summary_path.exists() else {}
            self.closed = True
            self._close_video()
            self.telemetry_file.close()
            self.events_file.close()
            self.trajectory_file.close()
            self.tracking_file.close()
            end = datetime.now(timezone.utc).isoformat()
            duration = max(0.0, time.time() - self.started)
            summary = {
                "session_id": self.session_id, "start_time": self.metadata["start_time"], "end_time": end,
                "duration_seconds": duration, "frames_processed": self.rows,
                "average_fps": self.rows / duration if duration else 0.0,
                "average_latency_ms": sum(self.latencies) / len(self.latencies) if self.latencies else None,
                "maximum_latency_ms": max(self.latencies) if self.latencies else None,
                "p95_latency_ms": sorted(self.latencies)[min(len(self.latencies) - 1, int(len(self.latencies) * 0.95))] if self.latencies else None,
                "average_confidence": sum(self.confidences) / len(self.confidences) if self.confidences else None,
                "minimum_confidence": min(self.confidences) if self.confidences else None,
                "maximum_confidence": max(self.confidences) if self.confidences else None,
                "average_speed_px_s": sum(self.speeds) / len(self.speeds) if self.speeds else None,
                "maximum_speed_px_s": max(self.speeds) if self.speeds else None,
                "average_distance_estimate_m": sum(self.distances) / len(self.distances) if self.distances else None,
                "target_loss_count": self.target_losses, "reacquisition_count": self.reacquisitions,
            }
            self._write_json(self.path / "session_summary.json", summary)
            self._write_json(self.path / "events.json", {"events": self.event_rows})
            self._write_json(self.path / "performance_report.json", {
                "session_id": self.session_id, "frames_processed": self.rows,
                "average_fps": summary["average_fps"],
                "average_latency_ms": summary["average_latency_ms"],
                "p95_latency_ms": summary["p95_latency_ms"],
                "maximum_latency_ms": summary["maximum_latency_ms"],
            })
            self.metadata["end_time"] = end
            self._write_json(self.metadata_path, self.metadata)
            return summary
