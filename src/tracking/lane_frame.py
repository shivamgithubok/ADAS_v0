"""Lane-anchored render frame: pin the lane, move the ego.

The BEV historically drew everything in the ego frame with ego locked at the
origin, so all ego motion appeared as the world sliding sideways, and geometry
was re-evaluated at fixed Y anchors every frame so the road never scrolled.

Here the smoothed ego corridor centerline is fitted as a cubic in the ego frame:

    X(Y) = c0 + c1*Y + c2*Y^2 + c3*Y^3

which separates ego pose from road shape:

    c0        ego lateral offset from lane center (m)
    atan(c1)  ego heading relative to the lane (rad)
    c2, c3    real road curvature

Rendering subtracts (c0 + c1*Y) so the lane is pinned to the canvas with only
curvature left, and the ego is drawn at (-c0, -atan(c1)).  Because the pose is
*measured* every frame rather than integrated, it cannot drift - unlike dead
reckoning from speed and yaw rate.

Longitudinal motion is the one quantity that must be integrated: `distance`
accumulates v*dt purely to scroll the dash pattern.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from src.inference import lane_filter_config as cfg

# Tuned by scripts/debug/bev_poc_tune.py over 1822 frames / 5 clips, scored on
# far-field calm + ego steadiness + fidelity against a zero-lag reference.
POSE_ALPHA = 0.20        # ego offset/heading responsiveness (~2 frames lag, 67 ms)
CURV_ALPHA = 0.08        # curvature: the far-field noise source, damp it hard
WIDTH_ALPHA = 0.25       # lane width responsiveness
HOLD_FRAMES = 90         # ~3 s of coast; the road must never blink out
FIT_MAX_Y_M = 60.0
# Physical highway limits. c2 ~ kappa/2, so |c2| <= 0.004 is radius >= 125 m.
C2_MAX = 0.004
C3_MAX = 2.0e-5

DASH_LEN_M = 3.2
DASH_GAP_M = 4.0
DASH_PERIOD_M = DASH_LEN_M + DASH_GAP_M

# The corridor *is* the ego lane by construction, so |offset| can never really
# exceed half a lane. A larger value means the ego pair latched onto a
# neighbouring lane; clamping keeps the car straddling the line (an honest
# "crossing" look) instead of flying off the canvas.
EGO_OFFSET_MARGIN_M = 0.35
EGO_YAW_MAX_DEG = 15.0
# BEV render: keep the ego car sprite on the lane centerline. Offset is still
# tracked internally for traffic/lane-frame math; only the drawn car is pinned.
# Without this, a lane change (or a wrong ego-pair latch) slides the car to the
# edge of the ribbon and looks like it "went into the next lane".
BEV_PIN_EGO_TO_LANE_CENTER = True
# If |c0| jumps by more than this, treat as a lane re-lock and snap pose so the
# EMA does not drag the car across the ribbon for several frames.
LANE_RELOCK_SNAP_M = 1.2


def road_shape(c1: float, c2: float, c3: float, y, y_max: float) -> np.ndarray:
    """Lateral road shape at range y; curvature continues linearly past y_max."""
    y = np.asarray(y, dtype=np.float64)
    yc = np.minimum(y, float(y_max))
    curv = c2 * yc ** 2 + c3 * yc ** 3 + (2.0 * c2 * yc + 3.0 * c3 * yc ** 2) * (y - yc)
    return c1 * y + curv


class LaneFrameModel:
    """Temporal ego-pose + road-shape estimate driving a lane-anchored render."""

    def __init__(
        self,
        pose_alpha: float = POSE_ALPHA,
        curv_alpha: float = CURV_ALPHA,
        width_alpha: float = WIDTH_ALPHA,
        hold_frames: int = HOLD_FRAMES,
    ) -> None:
        self.pose_alpha = float(pose_alpha)
        self.curv_alpha = float(curv_alpha)
        self.width_alpha = float(width_alpha)
        self.hold_frames = int(hold_frames)
        # Instance-level so scripts/debug/bev_poc_tune.py can sweep them.
        self.c2_max = C2_MAX
        self.c3_max = C3_MAX
        self.reset()

    def reset(self) -> None:
        self._c: Optional[np.ndarray] = None
        self._half_w = 0.5 * float(cfg.EGO_LANE_WIDTH_TARGET_M)
        self.distance = 0.0
        self.miss = 0
        self.valid = False
        self.relock_jump_m = 0.0
        self._shape_max_y: Optional[float] = None

    def set_world(self, state, speed_mps: Optional[float] = None,
                  dt: float = 1.0 / 30.0) -> bool:
        """Take pose and shape from a LaneWorldState instead of fitting a corridor.

        The world already tracks boundaries through relocks, so no EMA or relock
        snap is applied here: c0 is the ego-lane centre, c1..c3 the road shape.
        """
        v = 0.0 if speed_mps is None else max(0.0, float(speed_mps))
        self.distance += v * max(0.0, float(dt))
        self.relock_jump_m = 0.0
        if state is None or not state.valid:
            self.miss += 1
            self.valid = False
            self._c = None
            return False
        self._c = np.array(
            [state.ego_center, state.c1, state.c2, state.c3], dtype=np.float64
        )
        self._half_w = 0.5 * float(state.ego_width)
        self._shape_max_y = float(cfg.LW_SHAPE_MAX_Y_M)
        self.miss = self.miss + 1 if state.held else 0
        self.valid = True
        return True

    # ------------------------------------------------------------------ state
    @property
    def held(self) -> bool:
        """True when geometry is coasting rather than measured this frame."""
        return self.valid and self.miss > 0

    @property
    def half_width(self) -> float:
        return float(self._half_w)

    @property
    def lane_width(self) -> float:
        return 2.0 * float(self._half_w)

    @property
    def curvature(self) -> float:
        return 0.0 if self._c is None else float(self._c[2])

    # -------------------------------------------------------------------- fit
    @staticmethod
    def _fit_centerline(left, right) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if left is None or right is None:
            return None, None
        L = np.asarray(left, dtype=np.float64)
        R = np.asarray(right, dtype=np.float64)
        if L.ndim != 2 or R.ndim != 2 or L.shape[1] < 2 or R.shape[1] < 2:
            return None, None
        n = min(len(L), len(R))
        if n < 3:
            return None, None
        L, R = L[:n], R[:n]
        center = 0.5 * (L + R)
        ys, xs = center[:, 1], center[:, 0]
        keep = (ys >= 0.0) & (ys <= FIT_MAX_Y_M)
        if int(keep.sum()) < 3:
            return None, None
        ys, xs = ys[keep], xs[keep]
        # Weight the near field: that is what the driver actually looks at, and
        # far anchors carry most of the detector noise.
        w = 1.0 / (1.0 + 0.05 * ys)
        try:
            coeffs = np.polyfit(ys, xs, 3, w=w)[::-1]
        except Exception:
            return None, None
        if not np.all(np.isfinite(coeffs)):
            return None, None
        half_w = 0.5 * float(np.mean(np.abs(R[:, 0] - L[:, 0])))
        return np.asarray(coeffs, dtype=np.float64), half_w

    # ----------------------------------------------------------------- update
    def update(
        self,
        left_3d,
        right_3d,
        speed_mps: Optional[float] = None,
        dt: float = 1.0 / 30.0,
    ) -> bool:
        """Fold one frame in. Returns True while geometry is renderable."""
        v = 0.0 if speed_mps is None else max(0.0, float(speed_mps))
        self.distance += v * max(0.0, float(dt))

        c, half_w = self._fit_centerline(left_3d, right_3d)
        if c is None:
            self.miss += 1
            self.valid = self._c is not None and self.miss <= self.hold_frames
            if not self.valid:
                self._c = None
            return self.valid

        # Reject physically impossible curvature before it reaches the render.
        c[2] = float(np.clip(c[2], -self.c2_max, self.c2_max))
        c[3] = float(np.clip(c[3], -self.c3_max, self.c3_max))
        # Corridor width must stay one physical lane. extract_ego_corridor_3d
        # only clamps the upper bound, so a collapsed or inverted pair can
        # otherwise reach the renderer.
        half_w = float(np.clip(
            half_w,
            0.5 * float(cfg.EGO_LANE_WIDTH_MIN_M),
            0.5 * float(cfg.CORRIDOR_WIDTH_MAX_M),
        ))

        if self._c is None:
            self._c = c
            self._half_w = half_w
        else:
            # Lane change / pair re-lock: c0 jumps by ~one lane. Snap pose so the
            # car does not slide to the far edge of the ribbon during EMA catch-up.
            jump = float(c[0]) - float(self._c[0])
            if abs(jump) >= LANE_RELOCK_SNAP_M:
                # Lane change: the ego pair just locked onto the next lane.
                # Snap the road model, and report the jump so the car can
                # slide across instead of teleporting back to centre.
                self.relock_jump_m = jump
                self._c[0] = float(c[0])
                self._c[1] = float(c[1])
            else:
                self.relock_jump_m = 0.0
            ap, ac, aw = self.pose_alpha, self.curv_alpha, self.width_alpha
            self._c[0] = (1.0 - ap) * self._c[0] + ap * c[0]
            self._c[1] = (1.0 - ap) * self._c[1] + ap * c[1]
            self._c[2] = (1.0 - ac) * self._c[2] + ac * c[2]
            self._c[3] = (1.0 - ac) * self._c[3] + ac * c[3]
            self._half_w = (1.0 - aw) * self._half_w + aw * half_w

        self.miss = 0
        self.valid = True
        return True

    # ------------------------------------------------------------- geometry
    def lane_x(self, ys) -> np.ndarray:
        """Lane centerline in the lane frame: curvature only, ego pose removed."""
        y = np.asarray(ys, dtype=np.float64)
        if self._c is None:
            return np.zeros_like(y)
        if self._shape_max_y is not None:
            return road_shape(0.0, self._c[2], self._c[3], y, self._shape_max_y)
        return self._c[2] * y ** 2 + self._c[3] * y ** 3

    def centerline(self, y_max: float = FIT_MAX_Y_M, n: int = 48):
        """Uniformly sampled (x, y) centerline. Constant point count every frame,
        so segment pools no longer pop in and out as anchor visibility changes."""
        ys = np.linspace(0.0, float(y_max), int(n))
        return ys, self.lane_x(ys)

    def boundary(self, offset_m: float, y_max: float = FIT_MAX_Y_M, n: int = 48):
        """A lane-parallel boundary at a constant lateral offset from center."""
        ys, xs = self.centerline(y_max, n)
        return ys, xs + float(offset_m)

    def ego_pose(self) -> Tuple[float, float]:
        """(lateral offset m, heading rad) of ego within its lane, clamped.

        When BEV_PIN_EGO_TO_LANE_CENTER is on, returns (0, 0) so the car sprite
        stays in the middle of the locked lane ribbon.
        """
        if BEV_PIN_EGO_TO_LANE_CENTER:
            return 0.0, 0.0
        if self._c is None:
            return 0.0, 0.0
        lim = self._half_w + EGO_OFFSET_MARGIN_M
        offset = float(np.clip(-float(self._c[0]), -lim, lim))
        yaw_lim = math.radians(EGO_YAW_MAX_DEG)
        yaw = float(np.clip(-math.atan(float(self._c[1])), -yaw_lim, yaw_lim))
        return offset, yaw

    def raw_pose(self) -> Tuple[float, float]:
        """Measured (offset, heading) ignoring BEV_PIN_EGO_TO_LANE_CENTER.

        The cluster uses this only while a lane-departure state is active, so
        the ego visibly drifts toward the line it is about to cross.
        """
        if self._c is None:
            return 0.0, 0.0
        lim = self._half_w + EGO_OFFSET_MARGIN_M
        offset = float(np.clip(-float(self._c[0]), -lim, lim))
        yaw_lim = math.radians(EGO_YAW_MAX_DEG)
        yaw = float(np.clip(-math.atan(float(self._c[1])), -yaw_lim, yaw_lim))
        return offset, yaw

    @property
    def pose_saturated(self) -> bool:
        """True when the raw offset exceeded a lane half-width, i.e. the ego
        pair most likely latched onto a neighbouring lane."""
        if self._c is None:
            return False
        return abs(float(self._c[0])) > self._half_w + EGO_OFFSET_MARGIN_M

    def to_lane_frame(self, xs, ys) -> np.ndarray:
        """Map an ego-frame polyline's X into the lane frame."""
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        if self._c is None:
            return x
        return x - (self._c[0] + self._c[1] * y)

    def point_to_lane_frame(self, x: float, y: float) -> float:
        """Map a single ego-frame point's X into the lane frame."""
        if self._c is None:
            return float(x)
        return float(x) - (self._c[0] + self._c[1] * float(y))

    def dash_spans(
        self,
        y_max: float = FIT_MAX_Y_M,
        dash_len: float = DASH_LEN_M,
        period: float = DASH_PERIOD_M,
        max_dashes: int = 10,
        phase_m: Optional[float] = None,
    ) -> Sequence[Tuple[float, float]]:
        """(y_start, y_end) dash spans that scroll backwards as the ego advances.

        ``phase_m`` overrides the integrated odometry so the cluster can scroll
        the paint on a display timer (smooth motion between detection frames).
        """
        phase = (self.distance if phase_m is None else float(phase_m)) % period
        spans = []
        k = 0
        while len(spans) < max_dashes:
            s0 = k * period - phase
            k += 1
            s1 = s0 + dash_len
            if s1 <= 0.0:
                continue
            if s0 >= y_max:
                break
            spans.append((max(s0, 0.0), min(s1, y_max)))
        return spans
