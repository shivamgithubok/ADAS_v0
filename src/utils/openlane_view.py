"""Rebuild a dashcam frame so the road matches the OpenLane training camera.

The lane network outputs meters in the OpenLane camera, and those meters are
drawn with that same camera. On a wider, lower-horizon dashcam the meters miss
the paint. This view copies each OpenLane road pixel from the dashcam pixel
that sees the same ground point, runs the network on that picture, then maps
the drawn pixels back with the same table.

Object boxes stay on the original frame. Only the lane network uses this view.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.calibration import P_final
from src.utils.ground_calib import GroundCalibration

MODEL_W = 480
MODEL_H = 360
# Ground hits closer than this are under the hood or behind the camera.
MIN_GROUND_Y_M = 1.0


class OpenLaneView:
    """Source frame ↔ OpenLane 480×360 road view, via the flat-road plane."""

    def __init__(self, calib: GroundCalibration, P_matrix=None):
        self.calib = calib
        self.P = np.asarray(P_final if P_matrix is None else P_matrix, dtype=np.float64)
        self.model_width = MODEL_W
        self.model_height = MODEL_H
        self.source_width = int(calib.source_width)
        self.source_height = int(calib.source_height)
        # Duck-type the fields CameraTransform callers read.
        self.crop_x = 0.0
        self.crop_y = 0.0
        self.crop_width = float(self.source_width)
        self.crop_height = float(self.source_height)
        self._map_x, self._map_y = self._build_maps()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Sample the source frame into the OpenLane 480×360 view."""
        return cv2.remap(
            frame,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def model_to_source(self, points: np.ndarray) -> np.ndarray:
        """OpenLane pixels → the source pixels those samples came from."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        su = _bilinear(self._map_x, pts[:, 0], pts[:, 1])
        sv = _bilinear(self._map_y, pts[:, 0], pts[:, 1])
        return np.column_stack((su, sv))

    def source_to_model(self, points: np.ndarray) -> np.ndarray:
        """Source pixels on the road → OpenLane pixels of the same ground point."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = np.empty_like(pts)
        for i, (u, v) in enumerate(pts):
            hit = self.calib.backproject_ground(float(u), float(v))
            if hit is None:
                out[i, 0] = float(u) / max(1.0, self.source_width) * MODEL_W
                out[i, 1] = float(v) / max(1.0, self.calib.v_vp) * self._horizon_v
                continue
            x, y = hit
            uu, vv = _ground_to_model(self.P, x, y)
            out[i, 0] = uu
            out[i, 1] = vv
        return out

    def source_point_is_visible_to_model(self, u: float, v: float) -> bool:
        xy = self.source_to_model(np.array([[u, v]], dtype=np.float64))[0]
        return 0.0 <= xy[0] < MODEL_W and 0.0 <= xy[1] < MODEL_H

    @property
    def _horizon_v(self) -> float:
        col = self.P[:, 1]
        if abs(col[2]) < 1e-9:
            return 0.5 * MODEL_H
        return float(col[1] / col[2])

    def _build_maps(self):
        uu, vv = np.meshgrid(
            np.arange(MODEL_W, dtype=np.float64),
            np.arange(MODEL_H, dtype=np.float64),
        )
        x, y, ok = _model_pixels_to_ground(self.P, uu, vv)
        ok &= y > MIN_GROUND_Y_M

        src_u = np.empty((MODEL_H, MODEL_W), dtype=np.float32)
        src_v = np.empty((MODEL_H, MODEL_W), dtype=np.float32)

        # Sky and behind-camera pixels: stretch the dashcam sky into the
        # OpenLane sky so the network still sees a horizon, not a black band.
        horizon = self._horizon_v
        src_u[:] = (uu / MODEL_W * self.source_width).astype(np.float32)
        src_v[:] = (vv / max(horizon, 1.0) * float(self.calib.v_vp)).astype(np.float32)

        f = float(self.calib.f_px)
        hf = float(self.calib.hf)
        u_vp = float(self.calib.u_vp)
        v_vp = float(self.calib.v_vp)
        x_off = float(self.calib.lateral_mount_offset_m)
        road_u = u_vp + (x[ok] + x_off) * f / y[ok]
        road_v = v_vp + hf / y[ok]
        src_u[ok] = road_u.astype(np.float32)
        src_v[ok] = road_v.astype(np.float32)
        return src_u, src_v


def _model_pixels_to_ground(P, uu, vv):
    """Invert the ground-plane projection. z = 0."""
    a00 = P[0, 0] - uu * P[2, 0]
    a01 = P[0, 1] - uu * P[2, 1]
    a10 = P[1, 0] - vv * P[2, 0]
    a11 = P[1, 1] - vv * P[2, 1]
    b0 = uu * P[2, 3] - P[0, 3]
    b1 = vv * P[2, 3] - P[1, 3]
    det = a00 * a11 - a01 * a10
    ok = np.abs(det) > 1e-6
    x = np.zeros_like(uu)
    y = np.zeros_like(vv)
    x[ok] = (b0[ok] * a11[ok] - a01[ok] * b1[ok]) / det[ok]
    y[ok] = (a00[ok] * b1[ok] - b0[ok] * a10[ok]) / det[ok]
    return x, y, ok


def _ground_to_model(P, x, y, z=0.0):
    t = P @ np.array([float(x), float(y), float(z), 1.0], dtype=np.float64)
    w = t[2] if abs(t[2]) > 1e-8 else 1e-8
    return float(t[0] / w), float(t[1] / w)


def _bilinear(grid, u, v):
    h, w = grid.shape
    u = np.clip(u, 0.0, w - 1.001)
    v = np.clip(v, 0.0, h - 1.001)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    du = (u - u0).astype(np.float64)
    dv = (v - v0).astype(np.float64)
    u1 = np.clip(u0 + 1, 0, w - 1)
    v1 = np.clip(v0 + 1, 0, h - 1)
    return (
        grid[v0, u0] * (1.0 - du) * (1.0 - dv)
        + grid[v0, u1] * du * (1.0 - dv)
        + grid[v1, u0] * (1.0 - du) * dv
        + grid[v1, u1] * du * dv
    )
