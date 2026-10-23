"""One source of truth for source-image ↔ model-image geometry.

Horizontal crop stays disabled: it removed ~145 px from both sides of the
960x502 Garmin input and materially reduced lane recall.

Sky crop, by contrast, is now on by default.  The lane engine was trained on
OpenLane framing, where the horizon sits near mid-frame; the Garmin clips put it
much lower, so a full-frame resize squeezes the whole road into the bottom of
the 480x360 input.  Trimming sky restores training-like framing.

The P matrix deliberately does not change with this crop: it describes the
model's own assumed OpenLane camera, not the source frame.  Source ↔ model
geometry is entirely this class's job, and source_to_model/model_to_source
already carry crop_y, so overlays and ground-contact points stay aligned.

Measured over 200-frame samples (scripts/debug/crop_sweep.py), frames with a
measured ego corridor:

    clip           0%      20%
    GRMN6694    80.5%    84.5%
    GRMN6695    57.0%    76.5%
    GRMN6700    31.5%    44.0%

Ego lane width held at ~3.2 m across the change, so the crop did not bias the
model's 3D regression.  Past ~25% it collapses toward 2.6-2.9 m, which is why
this is capped well below that.  Small crops are worse than none at all -
around 10% recall falls off a cliff on every clip tested (1.5-11.5%), so do not
treat this value as safe to nudge downward without re-running the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Fraction of source height trimmed from the top before the model resize.
SKY_CROP_FRAC = 0.20


@dataclass(frozen=True)
class CameraTransform:
    """Geometry of an aspect-preserving source → model image conversion."""

    source_width: int
    source_height: int
    model_width: int = 480
    model_height: int = 360
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_width: float = 0.0
    crop_height: float = 0.0

    @classmethod
    def for_frame(
        cls,
        frame: np.ndarray,
        model_size: tuple[int, int] = (480, 360),
        sky_crop_px: int | None = None,
    ) -> "CameraTransform":
        """Build the transform for a source frame.

        sky_crop_px defaults to SKY_CROP_FRAC of the frame height; pass 0 to
        disable the crop, or an explicit pixel count to override it.
        """
        h, w = frame.shape[:2]
        model_w, model_h = model_size
        if h <= 0 or w <= 0:
            raise ValueError("Cannot build a camera transform for an empty frame")

        if sky_crop_px is None:
            sky_crop_px = int(SKY_CROP_FRAC * h)
        sky_crop_px = max(0, min(int(sky_crop_px), h - 2))
        return cls(
            w, h, model_w, model_h,
            crop_x=0.0,
            crop_y=float(sky_crop_px),
            crop_width=float(w),
            crop_height=float(h - sky_crop_px),
        )

    @property
    def scale_x(self) -> float:
        return float(self.model_width) / self.crop_width

    @property
    def scale_y(self) -> float:
        return float(self.model_height) / self.crop_height

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Crop and resize a BGR source frame into the model's native geometry."""
        x0 = max(0, int(np.floor(self.crop_x)))
        y0 = max(0, int(np.floor(self.crop_y)))
        x1 = min(self.source_width, int(np.ceil(self.crop_x + self.crop_width)))
        y1 = min(self.source_height, int(np.ceil(self.crop_y + self.crop_height)))
        cropped = frame[y0:y1, x0:x1]
        if cropped.size == 0:
            raise ValueError("Camera crop is empty")
        return cv2.resize(cropped, (self.model_width, self.model_height), interpolation=cv2.INTER_AREA)

    def source_to_model(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = pts.copy()
        out[:, 0] = (pts[:, 0] - self.crop_x) * self.scale_x
        out[:, 1] = (pts[:, 1] - self.crop_y) * self.scale_y
        return out

    def model_to_source(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = pts.copy()
        out[:, 0] = pts[:, 0] / self.scale_x + self.crop_x
        out[:, 1] = pts[:, 1] / self.scale_y + self.crop_y
        return out

    def source_point_is_visible_to_model(self, u: float, v: float) -> bool:
        return (
            self.crop_x <= u <= self.crop_x + self.crop_width
            and self.crop_y <= v <= self.crop_y + self.crop_height
        )
