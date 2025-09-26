"""
Aggregate per-frame analytics into session-level metrics.

🎯 Purpose

Collect data frame by frame (detections, tracks, faces).

Save small snapshots (FrameSnapshot).

At the end, calculate session-level metrics (how many people, dwell time, recognized visitors, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from modules.yolo_detector import Detection
from modules.tracker import Track
from modules.face_analytics import FaceDetection


@dataclass
class FrameSnapshot:
    frame_id: int
    timestamp: float
    detection_count: int
    track_count: int
    recognized_visitors: List[str] = field(default_factory=list)


@dataclass
class SessionSummary:
    total_frames: int
    duration_seconds: float
    unique_people: int
    avg_dwell_seconds: float
    dwell_seconds: Dict[int, float]
    recognized_visitors: Dict[str, int]
    timeline: List[FrameSnapshot]


class SessionAggregator:
    """
    Collects frame-by-frame outputs and produces summary analytics.
    """

    def __init__(self, fps: float) -> None:
        self.fps = fps if fps > 0 else 30.0
        self.timeline: List[FrameSnapshot] = []
        self.track_activity: Dict[int, int] = {}
        self.recognized_counter: Dict[str, int] = {}
        self.total_frames: int = 0
        self.last_timestamp: float = 0.0

    def update(
        self,
        frame_id: int,
        timestamp: float,
        detections: Sequence[Detection],
        tracks: Sequence[Track],
        faces: Sequence[FaceDetection],
    ) -> None:
        self.total_frames += 1
        self.last_timestamp = max(self.last_timestamp, timestamp)

        for track in tracks:
            self.track_activity[track.track_id] = self.track_activity.get(track.track_id, 0) + 1

        recognized_now: List[str] = []
        for face in faces:
            if face.identity:
                recognized_now.append(face.identity)
                self.recognized_counter[face.identity] = self.recognized_counter.get(face.identity, 0) + 1

        snapshot = FrameSnapshot(
            frame_id=frame_id,
            timestamp=timestamp,
            detection_count=len(detections),
            track_count=len(tracks),
            recognized_visitors=recognized_now,
        )
        self.timeline.append(snapshot)

    def finalize(self) -> SessionSummary:
        dwell_seconds = {track_id: frames / self.fps for track_id, frames in self.track_activity.items()}
        avg_dwell = float(np.mean(list(dwell_seconds.values()))) if dwell_seconds else 0.0
        duration_seconds = self.last_timestamp if self.last_timestamp > 0 else self.total_frames / self.fps

        return SessionSummary(
            total_frames=self.total_frames,
            duration_seconds=duration_seconds,
            unique_people=len(self.track_activity),
            avg_dwell_seconds=avg_dwell,
            dwell_seconds=dwell_seconds,
            recognized_visitors=dict(sorted(self.recognized_counter.items(), key=lambda kv: kv[1], reverse=True)),
            timeline=self.timeline,
        )
