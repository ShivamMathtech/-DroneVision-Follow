from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2

from app.analytics import MotionAnalyzer
from app.camera.sources import CameraSource, create_camera_source
from app.config import Settings, settings
from app.control.pid import PID
from app.control.safety import DesiredCommand, SafetyLayer
from app.detection.face import FaceDetector, HaarFaceDetector
from app.drone.interface import MockDrone
from app.recording.recorder import SessionRecorder
from app.schemas import Track
from app.tracking.iou_tracker import IoUTracker
from app.tracking.target_manager import TargetEvent, TargetManager


class VisionService:
    """Threaded bounded-queue pipeline. The UI receives only the newest processed frame."""

    def __init__(self, config: Settings = settings, detector: FaceDetector | None = None) -> None:
        self.config = config
        self.detector = detector or HaarFaceDetector()
        self.tracker = IoUTracker(max_age=config.tracker_max_age)
        self.target = TargetManager(lost_timeout=config.target_lost_seconds)
        self.motion = MotionAnalyzer()
        self.safety = SafetyLayer()
        self.drone = MockDrone(self.safety)
        self.yaw_pid = PID()
        self.pitch_pid = PID()
        self.source: CameraSource | None = None
        self.source_kind = config.camera_source
        self.source_uri = config.camera_uri
        self.running = threading.Event()
        self.tracking_enabled = True
        self.capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self.capture_thread: threading.Thread | None = None
        self.process_thread: threading.Thread | None = None
        self.lock = threading.RLock()
        self.latest_jpeg: bytes | None = None
        self.latest_telemetry: dict[str, Any] = {}
        self.latest_tracks: list[Track] = []
        self.telemetry_history: deque[dict] = deque(maxlen=5000)
        self.frame_id = 0
        self.sequence = 0
        self.events: deque[dict] = deque(maxlen=2000)
        self.recorder: SessionRecorder | None = None
        self.controller_enabled = False
        self.start_time: float | None = None
        self.process_times: deque[float] = deque(maxlen=60)
        self.process_stamps: deque[float] = deque(maxlen=60)
        self.capture_times: deque[float] = deque(maxlen=60)
        self.dropped_frames = 0
        self.capture_fault_reported = False

    def start(self, source_kind: str | None = None, uri: str | None = None) -> dict:
        if self.running.is_set():
            return self.status()
        self.source_kind = source_kind or self.source_kind
        self.source_uri = uri if uri is not None else self.source_uri
        self.source = create_camera_source(self.source_kind, self.source_uri, self.config.frame_width, self.config.frame_height)
        self.source.open()
        self.tracker.reset()
        self.target = TargetManager(lost_timeout=self.config.target_lost_seconds)
        self.motion.reset()
        self.capture_queue = queue.Queue(maxsize=2)
        self.latest_jpeg = None
        self.latest_telemetry = {}
        self.latest_tracks = []
        self.telemetry_history.clear()
        self.process_times.clear()
        self.process_stamps.clear()
        self.capture_times.clear()
        self.capture_fault_reported = False
        self.frame_id = 0
        self.start_time = time.time()
        self.running.set()
        self.capture_thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, name="vision-process", daemon=True)
        self.capture_thread.start()
        self.process_thread.start()
        self._event("SYSTEM_STARTED", "info", "Camera and vision pipeline started.")
        return self.status()

    def stop(self) -> dict:
        was_running = self.running.is_set()
        self.running.clear()
        if self.source:
            self.source.close()
            self.source = None
        for thread in (self.capture_thread, self.process_thread):
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        if self.recorder:
            self.recorder.stop()
            self.recorder = None
        self.controller_enabled = False
        self.safety.target_valid = False
        if was_running:
            self._event("SYSTEM_STOPPED", "info", "Camera and vision pipeline stopped.")
        return self.status()

    def _capture_loop(self) -> None:
        previous = 0.0
        while self.running.is_set() and self.source:
            start = time.perf_counter()
            ok, frame = self.source.read()
            if not ok or frame is None:
                if self.source_kind in {"video", "file", "replay"} and self.source and self.source.capture:
                    # Loop video files for live dashboard testing.
                    self.source.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.02)
                    continue
                if not self.capture_fault_reported:
                    self._event("CAMERA_DISCONNECTED", "error", "Frame acquisition failed; check source and network.")
                    self.capture_fault_reported = True
                time.sleep(0.25)
                continue
            self.capture_fault_reported = False
            now = time.time()
            if previous:
                self.capture_times.append(now - previous)
            previous = now
            item = (frame, now)
            try:
                self.capture_queue.put_nowait(item)
            except queue.Full:
                try:
                    self.capture_queue.get_nowait()
                    self.dropped_frames += 1
                except queue.Empty:
                    pass
                try:
                    self.capture_queue.put_nowait(item)
                except queue.Full:
                    self.dropped_frames += 1
            elapsed = time.perf_counter() - start
            delay = max(0.0, (1.0 / self.config.target_fps) - elapsed)
            if delay:
                time.sleep(delay)

    def _process_loop(self) -> None:
        while self.running.is_set():
            try:
                raw, captured_at = self.capture_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            began = time.perf_counter()
            self.frame_id += 1
            frame = raw
            if frame.shape[1] != self.config.frame_width or frame.shape[0] != self.config.frame_height:
                frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height), interpolation=cv2.INTER_AREA)
            height, width = frame.shape[:2]
            detector_ms = tracker_ms = 0.0
            if self.tracking_enabled:
                detector_start = time.perf_counter()
                detections = self.detector.detect(frame)
                detections = [d for d in detections if d.confidence >= self.config.detection_confidence]
                detector_ms = (time.perf_counter() - detector_start) * 1000
                tracker_start = time.perf_counter()
                tracks = self.tracker.update(detections, captured_at)
                tracker_ms = (time.perf_counter() - tracker_start) * 1000
                events = self.target.update(tracks, captured_at)
                for event in events:
                    self._event(event.type, event.severity, event.description, event.target_id, event.timestamp)
                target_track = next((t for t in tracks if t.track_id == self.target.target_id and t.state == "ACTIVE"), None)
            else:
                tracks = []
                target_track = None
            if len(tracks) > 1 and self.frame_id % 30 == 1:
                self._event("MULTIPLE_FACES_DETECTED", "info", f"{len([t for t in tracks if t.state == 'ACTIVE'])} tracks are visible.")
            metrics = self.motion.update(
                target_track, width, height, captured_at, self.config.distance_estimation,
                self.config.face_width_meters, self.config.camera_focal_px,
            )
            self.safety.target_valid = target_track is not None and target_track.confidence >= self.config.detection_confidence
            command = {"yaw_rate": 0.0, "pitch": 0.0, "roll": 0.0, "throttle": 0.0}
            if self.controller_enabled and metrics.get("error_x") is not None:
                desired = self.safety.apply(DesiredCommand(
                    yaw_rate=self.yaw_pid.update(float(metrics["error_x"])),
                    pitch=self.pitch_pid.update(float(metrics["error_y"])),
                ))
                reply = self.drone.send_desired_command(desired)
                command = reply.get("command", command)
            elapsed_ms = (time.perf_counter() - began) * 1000
            self.process_times.append(time.perf_counter() - began)
            self.process_stamps.append(time.perf_counter())
            elapsed_window = self.process_stamps[-1] - self.process_stamps[0] if len(self.process_stamps) > 1 else 0.0
            fps = (len(self.process_stamps) - 1) / elapsed_window if elapsed_window > 0 else 0.0
            box = target_track.bbox if target_track else None
            telemetry = {
                "timestamp": captured_at, "frame_id": self.frame_id, "fps": fps,
                "capture_fps": self._rate(self.capture_times), "inference_fps": fps,
                "inference_latency_ms": elapsed_ms, "detector_latency_ms": detector_ms,
                "tracker_latency_ms": tracker_ms, "face_count": len([t for t in tracks if t.state == "ACTIVE"]),
                "target_id": self.target.target_id, "target_confidence": target_track.confidence if target_track else None,
                "bbox": list(box) if box else None,
                "x1": box[0] if box else None, "y1": box[1] if box else None,
                "x2": box[2] if box else None, "y2": box[3] if box else None,
                "target_center_x": metrics.get("center_x"), "target_center_y": metrics.get("center_y"),
                "normalized_x": metrics.get("normalized_x"), "normalized_y": metrics.get("normalized_y"),
                "error_x": metrics.get("error_x"), "error_y": metrics.get("error_y"),
                "velocity_x": metrics.get("velocity_x"), "velocity_y": metrics.get("velocity_y"),
                "speed": metrics.get("speed"), "acceleration": metrics.get("acceleration"),
                "direction": metrics.get("direction"), "estimated_distance": metrics.get("estimated_distance"),
                "distance_is_estimate": metrics.get("distance_is_estimate", False),
                "tracking_status": self.target.state, "tracks": [t.to_dict() for t in tracks],
                "trajectory": metrics.get("trajectory", []), "drone": self.drone.telemetry(),
                "controller_enabled": self.controller_enabled, "desired_command": command,
                "dropped_frames": self.dropped_frames,
                "resolution": {"width": width, "height": height},
            }
            annotated = self._annotate(frame, tracks, target_track, telemetry)
            ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
            with self.lock:
                if ok:
                    self.latest_jpeg = encoded.tobytes()
                self.latest_telemetry = telemetry
                self.latest_tracks = tracks
                self.telemetry_history.append(telemetry)
                self.sequence += 1
            if self.recorder:
                try:
                    self.recorder.write_frame(frame, annotated, telemetry)
                except Exception as exc:
                    self._event("RECORDING_ERROR", "error", str(exc))

    @staticmethod
    def _rate(deltas: deque[float]) -> float:
        return len(deltas) / sum(deltas) if deltas and sum(deltas) else 0.0

    def _annotate(self, frame, tracks: list[Track], target: Track | None, telemetry: dict):
        image = frame.copy()
        height, width = image.shape[:2]
        cv2.drawMarker(image, (width // 2, height // 2), (100, 220, 255), cv2.MARKER_CROSS, 24, 2)
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            chosen = target is not None and track.track_id == target.track_id
            color = (60, 220, 130) if chosen else (255, 180, 50)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3 if chosen else 2)
            label = f"TARGET #{track.track_id}" if chosen else f"FACE #{track.track_id}"
            cv2.putText(image, f"{label}  {track.confidence:.0%}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        trajectory = telemetry.get("trajectory", [])
        for left, right in zip(trajectory[-60:-1], trajectory[-59:]):
            p1 = (int(left["x"]), int(left["y"]))
            p2 = (int(right["x"]), int(right["y"]))
            cv2.line(image, p1, p2, (40, 190, 255), 2)
        if target:
            cx, cy = map(int, target.center)
            cv2.circle(image, (cx, cy), 5, (30, 255, 255), -1)
        status = telemetry.get("tracking_status", "NO_TARGET")
        cv2.putText(image, f"{status}   {telemetry.get('fps', 0):.1f} FPS   {telemetry.get('inference_latency_ms', 0):.0f} ms",
                    (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 244, 250), 2)
        distance = telemetry.get("estimated_distance")
        if distance is not None and target:
            cv2.putText(image, f"Estimated range: {distance:.1f} m", (18, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 244, 250), 1)
        return image

    def _event(self, event_type: str, severity: str, description: str, target_id: int | None = None,
               timestamp: float | None = None) -> dict:
        event = {
            "timestamp": timestamp or time.time(), "type": event_type, "severity": severity,
            "description": description, "target_id": target_id,
        }
        with self.lock:
            self.events.append(event)
        if self.recorder:
            try:
                self.recorder.write_event(event)
            except Exception:
                pass
        return event

    def select_target(self, x: float, y: float) -> dict:
        telemetry = self.latest_telemetry
        resolution = telemetry.get("resolution", {"width": self.config.frame_width, "height": self.config.frame_height})
        ok, events = self.target.select_at(x, y, resolution["width"], resolution["height"], self.latest_tracks)
        for event in events:
            self._event(event.type, event.severity, event.description, event.target_id, event.timestamp)
        return {"selected": ok, **self.target.status()}

    def select_track(self, track_id: int) -> dict:
        ok, events = self.target.select(track_id, self.latest_tracks)
        for event in events:
            self._event(event.type, event.severity, event.description, event.target_id, event.timestamp)
        return {"selected": ok, **self.target.status()}

    def release_target(self) -> dict:
        for event in self.target.release():
            self._event(event.type, event.severity, event.description, event.target_id, event.timestamp)
        return self.target.status()

    def start_recording(self) -> dict:
        if self.recorder:
            return {"recording": True, "session_id": self.recorder.session_id}
        self.recorder = SessionRecorder(
            self.config.root / "sessions",
            {"camera_source": self.source_kind, "camera_uri": self.source_uri,
             "resolution": [self.config.frame_width, self.config.frame_height],
             "fps": self.config.target_fps, "detector": self.detector.__class__.__name__,
             "tracker": self.tracker.__class__.__name__, "mode": "SIMULATION"},
        )
        self._event("RECORDING_STARTED", "info", f"Recording {self.recorder.session_id}.")
        return {"recording": True, "session_id": self.recorder.session_id, "path": str(self.recorder.path)}

    def stop_recording(self) -> dict:
        if not self.recorder:
            return {"recording": False}
        recorder = self.recorder
        summary = recorder.stop()
        self.recorder = None
        self._event("RECORDING_STOPPED", "info", f"Recording {recorder.session_id} stopped.")
        return {"recording": False, "summary": summary, "path": str(recorder.path)}

    def status(self) -> dict:
        connected = bool(self.source and self.source.capture and self.source.capture.isOpened())
        return {
            "online": self.running.is_set(), "camera_connected": connected,
            "camera_source": self.source_kind, "camera_uri": self.source_uri,
            "vision_model": self.detector.__class__.__name__, "tracker": self.tracker.__class__.__name__,
            "tracking_enabled": self.tracking_enabled, "target": self.target.status(),
            "drone": self.drone.telemetry(), "safety": self.safety.status(),
            "recording": bool(self.recorder), "controller_enabled": self.controller_enabled,
            "dropped_frames": self.dropped_frames,
        }

    def latest(self) -> dict:
        with self.lock:
            return dict(self.latest_telemetry)

    def events_since(self, after: float = 0.0) -> list[dict]:
        with self.lock:
            return [event for event in self.events if event["timestamp"] > after]

    def history(self, limit: int = 500) -> list[dict]:
        with self.lock:
            return list(self.telemetry_history)[-max(1, min(limit, 5000)):]

    def safety_status(self) -> dict:
        return self.safety.status()

    def emergency_stop(self) -> dict:
        self.drone.emergency_stop()
        self.controller_enabled = False
        self._event("EMERGENCY_STOP", "warning", "Simulation command output latched to zero.")
        return self.safety.status()

    def set_controller(self, enabled: bool) -> dict:
        self.controller_enabled = enabled
        self.safety.target_valid = bool(enabled and self.target.target_id is not None)
        if not enabled:
            self.drone.send_desired_command(DesiredCommand())
        self._event("CONTROLLER_ENABLED" if enabled else "CONTROLLER_DISABLED", "info",
                    "Simulation-only desired command output " + ("enabled." if enabled else "disabled."))
        return {"enabled": self.controller_enabled, "safety": self.safety.status()}
