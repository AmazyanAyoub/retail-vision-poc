"""Utilities for ingesting video streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np


@dataclass
class FrameBatch:
    """Container for a single frame and its metadata."""

    frame_id: int
    timestamp: float
    frame: np.ndarray


class VideoStream:
    """Lightweight wrapper around cv2.VideoCapture for consistent ingestion."""

    def __init__(self, source: str | int | Path, stride: int = 1) -> None:
        self.source = str(source)
        self.stride = max(1, stride)
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video source: {self.source}")

        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Generator[FrameBatch, None, None]:
        frame_id = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            if frame_id % self.stride != 0:
                frame_id += 1
                continue

            timestamp = frame_id / self.fps if self.fps > 0 else 0.0
            yield FrameBatch(frame_id=frame_id, timestamp=timestamp, frame=frame)
            frame_id += 1

    def release(self) -> None:
        self.capture.release()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.release()
        except Exception:
            pass


def read_video(source: str | int | Path, stride: int = 1) -> Generator[FrameBatch, None, None]:
    """Convenience generator for one-off iteration over video frames."""
    stream = VideoStream(source, stride=stride)
    try:
        yield from stream
    finally:
        stream.release()
