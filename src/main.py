from __future__ import annotations

import sys
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from modules.video_processor.video_preprocessor import VideoPreprocessor, PreprocessedFrame
from modules.yolo_detector import YOLODetector, YOLODetectorConfig, Detection
from modules.tracker import DeepSortTracker, TrackerConfig, Track
from modules.face_analytics import FaceAnalytics, FaceAnalyticsConfig, FaceDetection
from modules.session_aggregator import SessionAggregator


def process_video(
    video_path: Path,
    max_frames: int | None = None,
    stride: int = 1,
) -> None:
    print(f"\n=== Processing {video_path.name} ===")
    if not video_path.exists():
        print(f"[warn] Missing file: {video_path}")
        return
    preprocessor = VideoPreprocessor(source=video_path, stride=stride)
    fps = preprocessor.stream.fps if preprocessor.stream.fps > 0 else 30.0
    aggregator = SessionAggregator(fps)

    detector = YOLODetector(
        YOLODetectorConfig(model_path="yolov8n.pt", confidence_threshold=0.5)
    )
    tracker = DeepSortTracker(TrackerConfig())
    face_analytics = FaceAnalytics(FaceAnalyticsConfig())
    face_analytics.load_or_build_gallery()

    unique_tracks: Set[int] = set()
    track_activity: Dict[int, int] = defaultdict(int)
    recognized_counter: Counter[str] = Counter()

    frame_iterator: Iterable[PreprocessedFrame] = preprocessor
    if max_frames is not None:
        frame_iterator = islice(frame_iterator, max_frames)

    try:
        for frame in frame_iterator:
            detections: List[Detection] = detector(frame.processed_frame)
            tracks: List[Track] = tracker.update(detections, frame.raw_frame)
            faces: List[FaceDetection] = face_analytics.analyze_frame(frame.stabilized_frame)

            aggregator.update(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                detections=detections,
                tracks=tracks,
                faces=faces,
            )

            for track in tracks:
                unique_tracks.add(track.track_id)
                track_activity[track.track_id] += 1

            for face in faces:
                if face.identity is not None:
                    recognized_counter[face.identity] += 1

            print(
                f"frame {frame.frame_id:04d} | det={len(detections):02d} "
                f"| tracks={len(tracks):02d} | faces={len(faces):02d}"
            )
    finally:
        preprocessor.release()

    # dwell_seconds = {tid: frames / fps for tid, frames in track_activity.items()}
    # total_unique = len(unique_tracks)
    # avg_dwell = np.mean(list(dwell_seconds.values())) if dwell_seconds else 0.0
    summary = aggregator.finalize()

    print("\n--- Session Summary ---")
    print(f"unique people tracked: {summary.unique_people}")
    print(f"avg dwell time (s): {summary.avg_dwell_seconds:.2f}")

    # if recognized_counter:
    #     print("recognized visitors:")
    #     for name, count in recognized_counter.most_common():
    #         print(f"  {name}: {count} appearances")
    # else:
        # print("recognized visitors: none")


def main() -> None:
    data_root = ROOT.parent / "data" / "raw" / "videos"
    videos = [
        data_root / "vtest.avi",
        data_root / "ucsd_short.avi",
    ]
    for video_path in videos:
        process_video(video_path, max_frames=120, stride=2)


if __name__ == "__main__":
    main()
