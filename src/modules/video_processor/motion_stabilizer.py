"""
Simple motion stabilizer using optical flow + affine warp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MotionStabilizerConfig:
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: int = 30
    smoothing: float = 0.9  # exponential smoothing factor


class MotionStabilizer:
    def __init__(self, config: MotionStabilizerConfig | None = None) -> None:
        self.config = config or MotionStabilizerConfig()
        self.prev_gray: Optional[np.ndarray] = None
        self.transforms = np.eye(3, dtype=np.float32)

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
        )

        if prev_pts is None or len(prev_pts) < 4:
            self.prev_gray = gray
            return frame

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)
        if curr_pts is None:
            self.prev_gray = gray
            return frame

        idx = status.flatten() == 1
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        if len(prev_pts) < 4:
            self.prev_gray = gray
            return frame

        matrix, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        if matrix is None:
            self.prev_gray = gray
            return frame

        # Convert 2x3 affine to 3x3 homogeneous
        affine = np.vstack([matrix, [0, 0, 1]]).astype(np.float32)
        self.transforms = (
            self.config.smoothing * self.transforms
            + (1.0 - self.config.smoothing) * affine
        )

        stabilized = cv2.warpAffine(
            frame,
            self.transforms[:2],
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        self.prev_gray = gray
        return stabilized

# Big Picture Purpose:

# This code stabilizes shaky videos by:

    # Tracking how small features move between frames.

    # Estimating the overall camera motion.

    # Smoothing that motion.

    # Warping the new frame so it aligns better with the previous one.

# What you can do with it:

    # Make hand-held camera videos smoother.

    # Preprocess video for object detection or tracking (stable video = better results).

    # Use in drone or GoPro footage to reduce shakiness.