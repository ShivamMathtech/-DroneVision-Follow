from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2


class CameraSource(ABC):
    """Common interface for camera, file, and compatible network stream sources."""

    def __init__(self, uri: str, width: int = 1280, height: int = 720) -> None:
        self.uri = uri
        self.width = width
        self.height = height
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.capture = cv2.VideoCapture(self.source_value())
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError(f"Could not open camera source: {self.uri}")

    @abstractmethod
    def source_value(self) -> Any:
        raise NotImplementedError

    def read(self):
        if self.capture is None:
            return False, None
        return self.capture.read()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def status(self) -> dict[str, Any]:
        return {"connected": bool(self.capture and self.capture.isOpened()), "source": self.__class__.__name__, "uri": self.uri}


class WebcamSource(CameraSource):
    def source_value(self) -> int:
        try:
            return int(self.uri)
        except ValueError:
            return 0


class VideoFileSource(CameraSource):
    def source_value(self) -> str:
        path = Path(self.uri).expanduser()
        if not path.exists():
            raise RuntimeError(f"Video file does not exist: {path}")
        return str(path)


class StreamSource(CameraSource):
    """OpenCV backend adapter for RTSP, HTTP/MJPEG, UDP, or GStreamer URIs."""

    def source_value(self) -> str:
        if not self.uri:
            raise RuntimeError("A stream URI is required.")
        return self.uri


def create_camera_source(kind: str, uri: str, width: int, height: int) -> CameraSource:
    normalized = kind.strip().lower()
    if normalized in {"webcam", "camera", "usb"}:
        return WebcamSource(uri or "0", width, height)
    if normalized in {"video", "file", "replay"}:
        return VideoFileSource(uri, width, height)
    if normalized in {"rtsp", "http", "mjpeg", "udp", "gstreamer", "stream"}:
        return StreamSource(uri, width, height)
    if normalized in {"drone", "ros"}:
        raise RuntimeError("Drone SDK and ROS adapters are extension points in this release. Select a stream URI if your camera exposes RTSP/HTTP.")
    raise RuntimeError(f"Unsupported camera source: {kind}")
