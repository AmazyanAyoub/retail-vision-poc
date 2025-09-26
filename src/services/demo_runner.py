"""
Utilities to run a video through the pipeline and return analytics artifacts.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.modules.video_processor.video_preprocessor import VideoPreprocessor
from modules.yolo_detector import YOLODetector, YOLODetectorConfig
from modules.tracker import DeepSortTracker, TrackerConfig
from modules.face_analytics import FaceAnalytics, FaceAnalyticsConfig
from modules.session_aggregator import SessionAggregator, SessionSummary


def run_session(
    video_path: Path,
    stride: int = 2,
    max_frames: Optional[int] = None,
    detector_cfg: Optional[YOLODetectorConfig] = None,
) -> Tuple[SessionSummary, pd.DataFrame]:
    preprocessor = VideoPreprocessor(source=video_path, stride=stride)
    fps = preprocessor.stream.fps if preprocessor.stream.fps > 0 else 30.0

    detector = YOLODetector(detector_cfg or YOLODetectorConfig(model_path="yolov8n.pt", confidence_threshold=0.5))
    tracker = DeepSortTracker(TrackerConfig())
    face_cfg = FaceAnalyticsConfig()
    face_analytics = FaceAnalytics(face_cfg)
    face_analytics.load_or_build_gallery()

    aggregator = SessionAggregator(fps=fps)
    frame_iter = preprocessor if max_frames is None else islice(preprocessor, max_frames)

    try:
        for frame in frame_iter:
            detections = detector(frame.processed_frame)
            tracks = tracker.update(detections, frame.raw_frame)
            faces = face_analytics.analyze_frame(frame.stabilized_frame)
            aggregator.update(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                detections=detections,
                tracks=tracks,
                faces=faces,
            )
    finally:
        preprocessor.release()

    summary = aggregator.finalize()
    timeline_df = pd.DataFrame([asdict(snapshot) for snapshot in summary.timeline])
    return summary, timeline_df
