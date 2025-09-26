"""
YOLO-based detector for person/retail analytics frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
from ultralytics import YOLO
import numpy as np


@dataclass
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: Optional[str] = None


@dataclass
class YOLODetectorConfig:
    model_path: str = "yolov8n.pt"          # swap for custom weights if you have them
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.5
    allowed_class_ids: Optional[Sequence[int]] = field(default_factory=lambda: [0])  # 0 == person in COCO
    device: Optional[str] = None  # "cuda", "cpu", or None to auto-select
    half_precision: bool = False  # set True if running on supported GPU


class YOLODetector:
    def __init__(self, config: YOLODetectorConfig | None = None) -> None:
        self.config = config or YOLODetectorConfig()
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(self.config.model_path)

        # Move model to the desired device; Ultralytics handles precision internally
        self.model.to(device)

        if self.config.half_precision and device.startswith("cuda"):
            self.model.model.half()  # type: ignore[attr-defined]

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        return self.detect(frame)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.shape[0] == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls.item())
                if (
                    self.config.allowed_class_ids is not None
                    and cls_id not in self.config.allowed_class_ids
                ):
                    continue

                xyxy = tuple(map(float, box.xyxy.squeeze().tolist()))  # (x1, y1, x2, y2)
                score = float(box.conf.item())
                class_name = (
                    result.names.get(cls_id) if hasattr(result, "names") else None
                )
                detections.append(
                    Detection(
                        bbox_xyxy=xyxy,
                        score=score,
                        class_id=cls_id,
                        class_name=class_name,
                    )
                )

        return detections