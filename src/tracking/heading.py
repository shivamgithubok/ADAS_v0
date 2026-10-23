"""BEV traffic heading: lane tangent, plus world-velocity modes when ego speed exists.

Modes
  along         same-direction: face the lane tangent (Qt ~180°)
  oncoming      opposite: tangent flipped (Qt ~0°)
  cross_right   through-traffic toward +X (Qt ~+90°)
  cross_left    through-traffic toward −X (Qt ~−90°)

World velocity (only when ego speed is known, including a true 0):

    v_x = Kalman vx
    v_y = Kalman vy + v_ego     # relative range-rate plus ego forward speed

Without ego speed, world v_y is not recoverable. Fall back to the previous
rule: lane tangent, and a sticky oncoming bit on raw vy < −12 m/s. Crossing
is not attempted in that fallback — a closing lead looks the same as oncoming.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from src.tracking.lane_frame import FIT_MAX_Y_M

ALONG = "along"
ONCOMING = "oncoming"
CROSS_RIGHT = "cross_right"
CROSS_LEFT = "cross_left"
CROSS_MODES = (CROSS_LEFT, CROSS_RIGHT)

YAW_AWAY = 180.0
YAW_TOWARD = 0.0

# Trust motion only when |v_world| is a car, not Kalman birth noise.
MIN_WORLD_SPEED_MPS = 3.0
# Toward-camera in the world frame (~18 km/h). Catching a slow truck stays positive.
ONCOMING_WORLD_VY = -5.0
# Lane change is ~1–2 m/s lateral; a through-vehicle is several m/s and dominates.
CROSS_VX_MPS = 3.0
CROSS_FRAC = 0.50

ENTER_HITS = 4
LEAVE_HITS = 6
WARMUP_FRAMES = 5
YAW_ALPHA = 0.28

# Legacy (no ego speed): same-direction lead can close at ego speed, so this
# has to be a large number. Crossing is not classified in this path.
LEGACY_ONCOMING_VY = -12.0
LEGACY_ONCOMING_HITS = 3

TANGENT_Y_MAX = float(FIT_MAX_Y_M)


def wrap180(deg: float) -> float:
    a = (float(deg) + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def blend_yaw(prev: float, new: float, alpha: float) -> float:
    d = (float(new) - float(prev) + 180.0) % 360.0 - 180.0
    return wrap180(float(prev) + float(alpha) * d)


def world_velocity(vx: float, vy: float, ego_speed_mps: Optional[float]
                   ) -> Optional[Tuple[float, float]]:
    """(v_x, v_y) in the road plane, or None when ego speed is unknown."""
    if ego_speed_mps is None:
        return None
    return float(vx), float(vy) + float(ego_speed_mps)


def tangent_yaw_deg(lane_model, y: float, oncoming: bool) -> float:
    y = float(min(max(y, 1.0), TANGENT_Y_MAX))
    if lane_model is None:
        return YAW_TOWARD if oncoming else YAW_AWAY
    return float(lane_model.heading_yaw_deg(y, oncoming=oncoming))


def motion_yaw_deg(vx: float, vy_w: float) -> float:
    """Qt Y-euler from a world-frame road velocity (same atan2 as lane segments)."""
    if abs(vx) < 1e-6 and abs(vy_w) < 1e-6:
        return YAW_AWAY
    return wrap180(math.degrees(math.atan2(float(vx), -float(vy_w))))


def classify_world(vx: float, vy_w: float) -> Optional[str]:
    """One-frame motion class, or None if speed is too low to trust."""
    spd = math.hypot(float(vx), float(vy_w))
    if spd < MIN_WORLD_SPEED_MPS:
        return None
    lat_frac = abs(float(vx)) / spd
    if abs(float(vx)) >= CROSS_VX_MPS and lat_frac >= CROSS_FRAC:
        return CROSS_RIGHT if vx > 0.0 else CROSS_LEFT
    if vy_w <= ONCOMING_WORLD_VY:
        return ONCOMING
    return ALONG


def yaw_for_mode(mode: str, y: float, lane_model, vx: float, vy_w: float) -> float:
    if mode == CROSS_RIGHT:
        raw = motion_yaw_deg(vx, vy_w)
        return blend_yaw(raw, 90.0, 0.45)
    if mode == CROSS_LEFT:
        raw = motion_yaw_deg(vx, vy_w)
        return blend_yaw(raw, -90.0, 0.45)
    return tangent_yaw_deg(lane_model, y, oncoming=(mode == ONCOMING))


class HeadingEstimator:
    """Sticky per-track mode + wrap-aware yaw EMA."""

    def __init__(self):
        self._st: Dict[int, dict] = {}

    def drop(self, live_ids) -> None:
        live = set(int(t) for t in live_ids)
        self._st = {t: s for t, s in self._st.items() if t in live}

    def update(
        self,
        track_id: int,
        vx: float,
        vy: float,
        y: float,
        ego_speed_mps: Optional[float],
        lane_model,
    ) -> Tuple[float, str]:
        tid = int(track_id)
        st = self._st.get(tid)
        if st is None:
            st = {"mode": ALONG, "hits": 0, "yaw": YAW_AWAY, "age": 0}
            self._st[tid] = st
        st["age"] = int(st["age"]) + 1

        proposed = self._propose(st, vx, vy, ego_speed_mps)
        mode = self._stick(st, proposed, legacy=(ego_speed_mps is None))
        world = world_velocity(vx, vy, ego_speed_mps)
        vx_w, vy_w = (vx, vy) if world is None else world
        target = yaw_for_mode(mode, y, lane_model, vx_w, vy_w)
        d = (target - st["yaw"] + 180.0) % 360.0 - 180.0
        # Cardinal flips (along↔oncoming, along↔cross) snap; small tangent noise EMA's.
        alpha = 1.0 if abs(d) >= 80.0 else (YAW_ALPHA if st["age"] > 2 else 1.0)
        st["yaw"] = wrap180(st["yaw"] + alpha * d)
        st["mode"] = mode
        return float(st["yaw"]), str(mode)

    def _propose(self, st, vx, vy, ego_speed_mps) -> Optional[str]:
        world = world_velocity(vx, vy, ego_speed_mps)
        if world is None:
            meas = float(vy) < LEGACY_ONCOMING_VY
            if meas == (st["mode"] == ONCOMING):
                return st["mode"] if st["mode"] in (ALONG, ONCOMING) else ALONG
            return ONCOMING if meas else ALONG

        if st["age"] < WARMUP_FRAMES:
            return None
        return classify_world(world[0], world[1])

    @staticmethod
    def _stick(st, proposed: Optional[str], legacy: bool = False) -> str:
        if proposed is None:
            st["hits"] = 0
            return st["mode"]
        if proposed == st["mode"]:
            st["hits"] = 0
            return st["mode"]
        st["hits"] = int(st["hits"]) + 1
        if st["mode"] == ALONG:
            need = LEGACY_ONCOMING_HITS if legacy else ENTER_HITS
        elif proposed == ALONG:
            need = LEAVE_HITS
        else:
            need = ENTER_HITS
        if st["hits"] >= need:
            st["mode"] = proposed
            st["hits"] = 0
        return st["mode"]
