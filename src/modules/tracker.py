"""
DeepSORT-based multi-object tracker for retail analytics.

Purpose:

This script keeps track of objects across frames.
YOLO gives you detections (where objects are in one frame).
DeepSORT links those detections across frames → so each person gets a track ID (same ID even as they move).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from modules.yolo_detector import Detection


@dataclass
class Track:
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    hits: int = 0
    age: int = 0
    time_since_update: int = 0


@dataclass
class TrackerConfig:
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7
    nms_max_overlap: float = 1.0
    embedder: str = "mobilenet"
    half: bool = False
    bgr: bool = True  # frames arrive in BGR if we pass stabilized_frame


class DeepSortTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracker = DeepSort(
            max_age=self.config.max_age,
            n_init=self.config.n_init,
            max_iou_distance=self.config.max_iou_distance,
            nms_max_overlap=self.config.nms_max_overlap,
            embedder=self.config.embedder,
            half=self.config.half,
            bgr=self.config.bgr,
        )

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> List[Track]:
        ds_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            ds_detections.append(([x1, y1, width, height], det.score, det.class_id))

        raw_tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        tracks: List[Track] = []
        for raw in raw_tracks:
            if not raw.is_confirmed():
                continue
            l, t, r, b = raw.to_ltrb()
            class_id = raw.get_det_class()
            confidence = getattr(raw, "det_confidence", None)
            tracks.append(
                Track(
                    track_id=raw.track_id,
                    bbox_xyxy=(float(l), float(t), float(r), float(b)),
                    score=float(confidence) if confidence is not None else 0.0,
                    class_id=int(class_id) if class_id is not None else None,
                    class_name=None,  # you can fill using YOLO label map if desired
                    hits=raw.hits,
                    age=raw.age,
                    time_since_update=raw.time_since_update,
                )
            )
        return tracks
