"""
Pipeline wrapper: VideoStream -> optional stabilization -> frame processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from modules.video_processor.video_stream import VideoStream
from modules.video_processor.frame_processor import FrameProcessor, FrameProcessorConfig
from modules.video_processor.motion_stabilizer import MotionStabilizer, MotionStabilizerConfig


@dataclass
class PreprocessedFrame:
    frame_id: int
    timestamp: float
    raw_frame: np.ndarray
    stabilized_frame: np.ndarray
    processed_frame: np.ndarray


class VideoPreprocessor:
    def __init__(
        self,
        source: str | int | Path,
        stride: int = 1,
        processor: FrameProcessor | None = None,
        stabilizer: MotionStabilizer | None = None,
    ) -> None:
        self.stream = VideoStream(source, stride=stride)
        self.processor = processor or FrameProcessor(FrameProcessorConfig())
        self.stabilizer = stabilizer or MotionStabilizer(MotionStabilizerConfig())

    def __iter__(self) -> Generator[PreprocessedFrame, None, None]:
        for batch in self.stream:
            stabilized = (
                self.stabilizer.stabilize(batch.frame)
                if self.stabilizer is not None
                else batch.frame
            )
            processed = self.processor.process(stabilized)
            yield PreprocessedFrame(
                frame_id=batch.frame_id,
                timestamp=batch.timestamp,
                raw_frame=batch.frame,
                stabilized_frame=stabilized,
                processed_frame=processed,
            )

    def release(self) -> None:
        self.stream.release()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.release()
        except Exception:
            pass
