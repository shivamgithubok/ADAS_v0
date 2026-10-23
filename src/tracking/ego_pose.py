"""Heading-up world frame: integrate ego pose and keep a short lane trail.

Live detections stay in the camera/ego frame (x right, y forward). This module
accumulates those points in a world frame using

    ψ += v · κ · dt          (or a GPS course when one exists)
    X += v · sin(ψ) · dt
    Y += v · cos(ψ) · dt

Render then maps world → current ego so the road you already drove swings
left/right when heading changes. Front-camera fusion is unchanged.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

# κ = 2 c2 at the bumper. Matches lane_frame.C2_MAX → radius >= 125 m.
KAPPA_MAX = 0.012
TRAIL_MAX_PTS = 480
TRAIL_MIN_STEP_M = 1.6
TRAIL_KEEP_M = 90.0


def wrap_pi(rad: float) -> float:
    return float((rad + math.pi) % (2.0 * math.pi) - math.pi)


def blend_angle(prev: float, new: float, alpha: float) -> float:
    d = wrap_pi(float(new) - float(prev))
    return wrap_pi(float(prev) + float(alpha) * d)


def kappa_from_corridor(left_3d, right_3d) -> Optional[float]:
    """Road curvature (1/m) at the ego from a left/right 3D pair."""
    if left_3d is None or right_3d is None:
        return None
    L = np.asarray(left_3d, dtype=np.float64)
    R = np.asarray(right_3d, dtype=np.float64)
    if L.ndim != 2 or R.ndim != 2 or L.shape[1] < 2 or R.shape[1] < 2:
        return None
    n = min(len(L), len(R))
    if n < 4:
        return None
    center = 0.5 * (L[:n] + R[:n])
    ys, xs = center[:, 1], center[:, 0]
    keep = (ys >= 0.0) & (ys <= 50.0)
    if int(keep.sum()) < 4:
        return None
    ys, xs = ys[keep], xs[keep]
    w = 1.0 / (1.0 + 0.05 * ys)
    try:
        c = np.polyfit(ys, xs, 3, w=w)[::-1]
    except Exception:
        return None
    if not np.all(np.isfinite(c)):
        return None
    return float(np.clip(2.0 * float(c[2]), -KAPPA_MAX, KAPPA_MAX))


class EgoPose:
    """Planar odometry. x right, y forward at t=0; ψ left-positive from +Y."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
        self.kappa = 0.0
        self.speed = 0.0
        self.distance = 0.0
        self.yaw_rate = 0.0  # rad/s, left-positive

    @property
    def yaw_deg(self) -> float:
        return float(math.degrees(self.psi))

    def reset(self) -> None:
        self.x = self.y = self.psi = self.kappa = self.speed = self.distance = 0.0
        self.yaw_rate = 0.0

    def update(
        self,
        dt: float,
        speed_mps: Optional[float],
        kappa: Optional[float] = None,
        course_rad: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        dt = float(np.clip(dt, 1e-4, 0.25))
        v = 0.0 if speed_mps is None else max(0.0, float(speed_mps))
        self.speed = v
        if kappa is not None and np.isfinite(kappa):
            self.kappa = float(np.clip(kappa, -KAPPA_MAX, KAPPA_MAX))
        else:
            self.kappa *= 0.90
            if abs(self.kappa) < 1e-4:
                self.kappa = 0.0
        self.yaw_rate = v * self.kappa
        if course_rad is not None and np.isfinite(course_rad):
            self.psi = blend_angle(self.psi, float(course_rad), 0.35)
        else:
            self.psi = wrap_pi(self.psi + v * self.kappa * dt)
        self.x += v * math.sin(self.psi) * dt
        self.y += v * math.cos(self.psi) * dt
        self.distance += v * dt
        return self.x, self.y, self.psi

    def to_world(self, x_ego, y_ego):
        x = np.asarray(x_ego, dtype=np.float64)
        y = np.asarray(y_ego, dtype=np.float64)
        c, s = math.cos(self.psi), math.sin(self.psi)
        return x * c + y * s + self.x, -x * s + y * c + self.y

    def to_ego(self, x_w, y_w):
        dx = np.asarray(x_w, dtype=np.float64) - self.x
        dy = np.asarray(y_w, dtype=np.float64) - self.y
        c, s = math.cos(self.psi), math.sin(self.psi)
        return dx * c - dy * s, dx * s + dy * c


class WorldRibbon:
    """Decaying left/right paint in the world frame (memory, not live 360°)."""

    def __init__(self, keep_m: float = TRAIL_KEEP_M):
        self.keep_m = float(keep_m)
        self._left: Deque[Tuple[float, float]] = deque(maxlen=TRAIL_MAX_PTS)
        self._right: Deque[Tuple[float, float]] = deque(maxlen=TRAIL_MAX_PTS)
        self._last_s = -1e9

    def reset(self) -> None:
        self._left.clear()
        self._right.clear()
        self._last_s = -1e9

    def ingest(self, pose: EgoPose, left_3d, right_3d) -> None:
        if left_3d is None or right_3d is None:
            return
        if pose.distance - self._last_s < TRAIL_MIN_STEP_M and self._left:
            return
        self._last_s = pose.distance
        L = np.asarray(left_3d, dtype=np.float64)
        R = np.asarray(right_3d, dtype=np.float64)
        if L.ndim != 2 or R.ndim != 2 or L.shape[1] < 2 or R.shape[1] < 2:
            return
        # Near-field only: far anchors are noisy and would smear the trail.
        for arr, buf in ((L, self._left), (R, self._right)):
            ys = arr[:, 1]
            m = (ys >= 0.0) & (ys <= 28.0)
            if int(m.sum()) < 2:
                continue
            wx, wy = pose.to_world(arr[m, 0], arr[m, 1])
            step = max(1, int(m.sum()) // 6)
            for x, y in zip(wx[::step], wy[::step]):
                buf.append((float(x), float(y)))
        self._prune(pose)

    def _prune(self, pose: EgoPose) -> None:
        lim2 = self.keep_m * self.keep_m

        def _keep(buf):
            return deque(
                ((x, y) for x, y in buf
                 if (x - pose.x) ** 2 + (y - pose.y) ** 2 <= lim2),
                maxlen=TRAIL_MAX_PTS,
            )

        self._left = _keep(self._left)
        self._right = _keep(self._right)

    def to_ego(self, pose: EgoPose) -> Tuple[np.ndarray, np.ndarray]:
        def _xy(buf: Sequence[Tuple[float, float]]) -> np.ndarray:
            if not buf:
                return np.zeros((0, 2), dtype=np.float64)
            w = np.asarray(buf, dtype=np.float64)
            xe, ye = pose.to_ego(w[:, 0], w[:, 1])
            return np.column_stack((xe, ye))

        return _xy(self._left), _xy(self._right)
