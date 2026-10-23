"""Validity gating for ground-plane ranging, and the MiDaS fallback rescaler.

Ground-plane back-projection is metric and scale-free, but it is only valid when
the bbox bottom edge really is the object's contact patch with the road plane the
calibration describes.  It silently is not, in four cases we actually hit:

  1. Elevated structures.  A truck on an overpass, an overhead gantry, a sign.
     Its base is *above* the horizon, so the ray never meets the road.  The old
     `project_2d_to_3d_ground` clamped the resulting negative range with
     `max(1.0, ...)` and reported 1.0 m -- an elevated 60 m truck rendered on the
     ego bumper.  `REJECT_ABOVE_HORIZON` is that bug's fix.

  2. Near-horizon rays.  At 8 px below the horizon on this camera the range is
     109 m and one pixel of jitter is 13 m.  The measurement is real but
     worthless; better to coast the track than to feed it in.

  3. Truncated boxes.  If the box is clipped by the frame bottom or by the sky
     crop, `y2` is the crop edge, not the tyres.

  4. Occluded bases.  A car whose wheels are hidden behind another car has a
     bottom edge partway up its body, so it ranges long.  We cannot always
     detect this geometrically, which is the one place a depth prior earns its
     keep -- see `MidasScaleFitter`.

The road-mask gate (`on_drivable`) is optional: pass a drivable-area mask and the
contact point must land on road. Without a mask the other gates still catch the
overpass case, because an elevated object's base is above the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

import numpy as np

from src.utils.ground_calib import GroundCalibration, bbox_ground_point


class GateResult(str, Enum):
    OK = "ok"
    ABOVE_HORIZON = "above_horizon"
    NEAR_HORIZON = "near_horizon"
    BEYOND_MAX = "beyond_max"
    TRUNCATED = "truncated"
    OFF_ROAD = "off_road"
    IMPLAUSIBLE_WIDTH = "implausible_width"


# A box whose bottom is within this many px of a frame/crop edge is truncated.
EDGE_TOUCH_PX = 3.0

# Cross-check: a ground-contacting vehicle's apparent width implies a physical
# width via w_m = w_px * y / f. If that is wildly off for the class, the contact
# assumption is wrong (usually an occluded base or a bad box).
CLASS_WIDTH_M = {
    "car": (1.4, 2.3),
    "motorcycle": (0.5, 1.2),
    "bike": (0.5, 1.2),
    "bus": (2.0, 3.1),
    "truck": (1.9, 3.2),
}
WIDTH_GATE_SLACK = 1.9   # generous: box width is noisy, we only want blunders


@dataclass
class GroundMeasurement:
    """One gated ground-plane measurement in the vehicle frame."""

    x_m: float
    y_m: float
    sigma_x_m: float
    sigma_y_m: float
    u: float
    v: float
    result: GateResult = GateResult.OK

    @property
    def valid(self) -> bool:
        return self.result is GateResult.OK

    @property
    def R(self) -> np.ndarray:
        return np.diag([self.sigma_x_m ** 2, self.sigma_y_m ** 2])


def _rejected(u, v, result) -> GroundMeasurement:
    return GroundMeasurement(float("nan"), float("nan"), float("inf"),
                             float("inf"), float(u), float(v), result)


def measure_ground(
    bbox: Sequence[float],
    calib: GroundCalibration,
    label: str = "car",
    frame_shape: Optional[Tuple[int, int]] = None,
    road_mask: Optional[np.ndarray] = None,
    valid_top_px: float = 0.0,
    min_below_horizon_px: float = 12.0,
) -> GroundMeasurement:
    """Range a detection from its bbox bottom-centre, with all gates applied.

    `min_below_horizon_px` is deliberately stricter than the calibration's own
    floor: 12 px is ~73 m on this camera, where 1 px of jitter is 6 m. Raise it
    to trade maximum range for stability.
    """
    u, v = bbox_ground_point(bbox)
    x1, _y1, x2, y2 = (float(c) for c in bbox)

    dv = v - calib.v_vp
    if dv <= 0.0:
        return _rejected(u, v, GateResult.ABOVE_HORIZON)
    if dv < float(min_below_horizon_px):
        return _rejected(u, v, GateResult.NEAR_HORIZON)

    if frame_shape is not None:
        h_img, w_img = frame_shape[:2]
        if y2 >= h_img - EDGE_TOUCH_PX or y2 <= float(valid_top_px) + EDGE_TOUCH_PX:
            return _rejected(u, v, GateResult.TRUNCATED)
        if x1 <= EDGE_TOUCH_PX and x2 >= w_img - EDGE_TOUCH_PX:
            return _rejected(u, v, GateResult.TRUNCATED)

    hit = calib.backproject_ground(u, v)
    if hit is None:
        return _rejected(u, v, GateResult.BEYOND_MAX)
    x_m, y_m = hit

    lo, hi = CLASS_WIDTH_M.get(str(label).lower(), (1.0, 3.2))
    w_m = abs(x2 - x1) * y_m / calib.f_px
    if not (lo / WIDTH_GATE_SLACK <= w_m <= hi * WIDTH_GATE_SLACK):
        return _rejected(u, v, GateResult.IMPLAUSIBLE_WIDTH)

    if road_mask is not None and not _on_mask(road_mask, u, v):
        return _rejected(u, v, GateResult.OFF_ROAD)

    return GroundMeasurement(
        x_m=x_m,
        y_m=y_m,
        sigma_x_m=calib.lateral_sigma_m(x_m, y_m),
        sigma_y_m=calib.range_sigma_m(y_m),
        u=u,
        v=v,
        result=GateResult.OK,
    )


def _on_mask(mask: np.ndarray, u: float, v: float, pad: int = 4) -> bool:
    """True if any pixel in a small patch under the contact point is drivable.

    A patch rather than a point because the contact line straddles the paint/
    shadow boundary and a single pixel is a coin flip there.
    """
    h, w = mask.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    u0, u1 = max(0, ui - pad), min(w, ui + pad + 1)
    v0, v1 = max(0, vi - pad), min(h, vi + pad + 1)
    if u0 >= u1 or v0 >= v1:
        return False
    return bool(np.any(mask[v0:v1, u0:u1]))


class MidasScaleFitter:
    """Per-frame least-squares rescale of relative depth onto geometric range.

    MiDaS output is relative inverse depth with an unknown per-frame affine
    transform.  Treating it as metres (the old
    `trt_depth_estimator.query_vehicle_depth` did exactly that, with a hardcoded
    `4300.0 / median_inv_depth`) has no defensible meaning.

    The only sound use is as an *interpolator between objects that do have valid
    ground contact*.  For inverse depth d and range y the correct model is affine
    in inverse space:

        1/y = a * d + b

    Fit (a, b) each frame by weighted least squares against the gated
    ground-plane measurements, then apply it to base-occluded objects only.  With
    fewer than `min_anchors` anchors, refuse to produce a range at all -- an
    unanchored fit is the scale drift this whole module exists to avoid.
    """

    def __init__(self, min_anchors: int = 3, ema_alpha: float = 0.3,
                 max_residual: float = 0.35):
        self.min_anchors = int(min_anchors)
        self.ema_alpha = float(ema_alpha)
        self.max_residual = float(max_residual)
        self._a: Optional[float] = None
        self._b: Optional[float] = None
        self.last_rmse: Optional[float] = None
        self.last_n: int = 0

    def fit(self, inv_depths: Sequence[float],
            ranges_m: Sequence[float],
            sigmas_m: Optional[Sequence[float]] = None) -> bool:
        d = np.asarray(inv_depths, dtype=np.float64)
        y = np.asarray(ranges_m, dtype=np.float64)
        ok = np.isfinite(d) & np.isfinite(y) & (y > 1.0)
        d, y = d[ok], y[ok]
        self.last_n = int(d.size)
        if d.size < self.min_anchors:
            return False
        target = 1.0 / y
        # Weight by measurement quality: a 70 m anchor is worth far less than a
        # 15 m one, and in inverse space its own sigma shrinks as 1/y^2.
        if sigmas_m is not None:
            s = np.asarray(sigmas_m, dtype=np.float64)[ok]
            w = 1.0 / np.maximum(1e-6, s / np.maximum(1.0, y) ** 2)
        else:
            w = np.ones_like(y)
        A = np.stack([d, np.ones_like(d)], axis=1)
        Aw, tw = A * w[:, None], target * w
        try:
            sol, *_ = np.linalg.lstsq(Aw, tw, rcond=None)
        except np.linalg.LinAlgError:
            return False
        a, b = float(sol[0]), float(sol[1])
        if not np.isfinite(a) or a <= 0.0:
            return False
        resid = target - (a * d + b)
        rmse = float(np.sqrt(np.mean(resid ** 2)) * np.mean(y) ** 2)
        self.last_rmse = rmse
        if self._a is None:
            self._a, self._b = a, b
        else:
            al = self.ema_alpha
            self._a = (1.0 - al) * self._a + al * a
            self._b = (1.0 - al) * self._b + al * b
        return True

    @property
    def ready(self) -> bool:
        return self._a is not None

    def range_m(self, inv_depth: float) -> Optional[float]:
        """Metric range for a base-occluded object, or None if unanchored."""
        if self._a is None:
            return None
        inv_y = self._a * float(inv_depth) + self._b
        if inv_y <= 1e-4:
            return None
        y = 1.0 / inv_y
        return float(y) if 1.0 < y < 200.0 else None

    def reset(self) -> None:
        self._a = self._b = None
        self.last_rmse = None
        self.last_n = 0
