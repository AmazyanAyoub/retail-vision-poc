"""
Frame preprocessing utilities: resize, color conversion, and optional CLAHE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class FrameProcessorConfig:
    target_size: Tuple[int, int] = (640, 360)  # (width, height)
    convert_to_rgb: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    apply_clahe: bool = False


class FrameProcessor:
    def __init__(self, config: FrameProcessorConfig | None = None) -> None:
        self.config = config or FrameProcessorConfig()
        self._clahe = None
        if self.config.apply_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size,
            )

    def process(self, frame: np.ndarray) -> np.ndarray:
        # Ensure contiguous array in case upstream yields memory views
        processed = np.ascontiguousarray(frame)

        if self.config.target_size:
            processed = cv2.resize(
                processed,
                self.config.target_size,
                interpolation=cv2.INTER_LINEAR,
            )

        if self.config.convert_to_rgb:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        if self._clahe is not None:
            # Contrast Limited Adaptive Histogram Equalization.
            # It improves local contrast in images.
            # Apply CLAHE on the luminance channel in LAB space to avoid color shifts
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            lab = cv2.merge((l, a, b))
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return processed


# Big Picture Purpose:

# This code is a frame preprocessing pipeline.
# Whenever you load video frames or images, it:

# Resizes them,

# Converts colors to the right format,

# Optionally boosts contrast (CLAHE).

# What you can do with it:

# Preprocess video frames before feeding them into a deep learning model.

# Standardize all frames to the same size and format.

# Enhance visibility in low-light videos with CLAHE.

# Use it in video analytics, object detection, face recognition pipelines, etc.