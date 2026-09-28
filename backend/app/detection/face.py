from __future__ import annotations

import time
from abc import ABC, abstractmethod

import cv2

from app.schemas import FaceDetection


class FaceDetector(ABC):
    @abstractmethod
    def detect(self, frame) -> list[FaceDetection]:
        raise NotImplementedError


class HaarFaceDetector(FaceDetector):
    """Dependency-light baseline detector; replaceable with a neural detector adapter."""

    def __init__(self, scale_factor: float = 1.12, min_neighbors: int = 5, min_size: tuple[int, int] = (32, 32)) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"OpenCV face cascade could not be loaded: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame) -> list[FaceDetection]:
        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = self.cascade.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=self.min_size, flags=cv2.CASCADE_SCALE_IMAGE,
        )
        result = []
        for x, y, w, h in boxes:
            # Haar cascades do not produce calibrated probabilities. This is a ranked proxy.
            confidence = min(0.95, 0.55 + 0.08 * self.min_neighbors)
            result.append(FaceDetection((int(x), int(y), int(x + w), int(y + h)), confidence, timestamp=now))
        return result
