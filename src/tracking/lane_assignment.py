"""Assign objects to lanes, and place them in the lane's real curved geometry.

Two separate jobs that the current code conflates:

  1. *Classification*: which lane is this object in?  Must be answered by
     comparing the object's lateral offset to the lane boundaries **evaluated at
     the object's own longitudinal distance**.  `cipo_tracker._lane_rank` uses
     `get_lane_x_at_y(ego_left, 25.0)` -- a hardcoded 25 m -- for every object at
     every range (src/inference/cipo_tracker.py:271-272).  On a curve the
     boundary at 25 m and at 60 m differ by metres, so a car at 60 m is compared
     against geometry from the wrong place on the road.

  2. *Placement*: where do we draw it?  `bev_quick3d._traffic_payload` answers
     this with `x = float(lane_slot) * lane_w` (src/ui/bev_quick3d.py:523), which
     throws the measured lateral offset away entirely and puts every car on a
     straight, evenly spaced grid.  Worse,
     `_measured_lane_slot` returns 0 whenever `in_path` is true or `lane_rank`
     is <= 0 (src/ui/bev_quick3d.py:265-267), so any object the CIPO logic
     believes is in-path is pinned to x = 0 exactly.  Lane markings meanwhile
     *are* drawn curved from `lane_frame`, so objects and lanes are two
     independent overlays that disagree on a curve.

`place_in_lane` fixes the second by keeping the measured within-lane offset and
adding it to the lane centreline sampled at the object's own y.  Objects then
move with the road instead of sliding across it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Fraction of a lane width an object may sit outside a boundary and still be
# counted as that lane's occupant. Covers straddling during a lane change.
LANE_TOLERANCE_FRAC = 0.35

# Hysteresis on the discrete index. Entering ego is harder than staying aside.
ASSIGN_ENTER_HITS = 3
ASSIGN_EXIT_MISS = 5
ENTER_EGO_HITS = 3
LEAVE_EGO_HITS = 5
SIDE_CHANGE_HITS = 3


@dataclass
class LaneModel:
    """Ego-lane centreline as a cubic plus a set of parallel boundaries.

    `coeffs` are [c0, c1, c2, c3] for x(y) = c0 + c1 y + c2 y^2 + c3 y^3 in the
    ego frame -- the same parameterisation `src/tracking/lane_frame.py` already
    fits, so this can be built straight from a LaneFrameModel.
    """

    coeffs: np.ndarray
    lane_width_m: float
    n_left: int = 1
    n_right: int = 1

    @classmethod
    def from_lane_frame(cls, lf, n_left: int = 2, n_right: int = 2
                        ) -> Optional["LaneModel"]:
        if lf is None or not getattr(lf, "valid", False):
            return None
        c = getattr(lf, "_c", None)
        if c is None:
            return None
        return cls(np.asarray(c, dtype=np.float64).copy(),
                   float(lf.lane_width), int(n_left), int(n_right))

    def center_x(self, y) -> np.ndarray:
        """Ego-lane centre x at forward distance y. Full pose + curvature."""
        y = np.asarray(y, dtype=np.float64)
        c = self.coeffs
        return c[0] + c[1] * y + c[2] * y ** 2 + c[3] * y ** 3

    def shape_x(self, y) -> np.ndarray:
        """Curvature-only centre, with ego pose removed (the render frame)."""
        y = np.asarray(y, dtype=np.float64)
        c = self.coeffs
        return c[2] * y ** 2 + c[3] * y ** 3

    def boundary_x(self, index: int, y) -> np.ndarray:
        """x of the boundary `index` lane-widths right of the ego lane centre.

        index 0 is the ego lane's left boundary offset -0.5 w; use
        `lane_center_x` for lane centres.
        """
        return self.center_x(y) + (float(index) + 0.5) * self.lane_width_m

    def lane_center_x(self, lane_index: int, y) -> np.ndarray:
        """Centre of lane `lane_index` (0 = ego, -1 = left, +1 = right) at y."""
        return self.center_x(y) + float(lane_index) * self.lane_width_m

    def shape_tangent(self, y) -> float:
        """dx/dy of the render-frame centreline (pose removed, curvature only)."""
        y = float(y)
        c = self.coeffs
        return 2.0 * float(c[2]) * y + 3.0 * float(c[3]) * y * y

    def heading_yaw_deg(self, y, oncoming: bool = False) -> float:
        """Qt Y-euler for a vehicle sitting on the ribbon at forward distance y.

        Same convention as lane segments: atan2(dx, -dy). Straight same-direction
        traffic is 180° (faces −Z). Oncoming is that plus 180°.
        """
        xp = self.shape_tangent(y)
        if oncoming:
            return float(np.degrees(np.arctan2(-xp, 1.0)))
        yaw = float(np.degrees(np.arctan2(xp, -1.0)))
        # Left curves come back as ~-173°; wrap so they stay next to the 180° base.
        if yaw < 0.0:
            yaw += 360.0
        return yaw


def assign_lane(
    x_m: float,
    y_m: float,
    lane: Optional[LaneModel],
    max_index: int = 2,
) -> Tuple[Optional[int], float]:
    """Lane index and within-lane offset for an object at (x_m, y_m).

    Returns (lane_index, offset_m) where offset_m is the signed distance from
    that lane's centreline, or (None, x_m) when there is no lane model.

    The whole point: the boundaries are evaluated at `y_m`, the object's own
    distance, so a curving road no longer mis-sorts far objects.

    The index is clamped to +-max_index, so `offset_m` can exceed half a lane
    for something off the modelled road (a car on a slip road, or a bad
    measurement). Use `lane_index_raw` when you need to detect and drop those
    rather than dragging them onto the outermost modelled lane.
    """
    if lane is None:
        return None, float(x_m)
    raw = lane_index_raw(x_m, y_m, lane)
    idx = int(np.clip(raw, -max_index, max_index))
    offset = float(x_m) - float(lane.lane_center_x(idx, float(y_m)))
    return idx, offset


def lane_index_raw(x_m: float, y_m: float, lane: LaneModel) -> int:
    """Unclamped lane index. |index| > max_index means off the modelled road."""
    cx = float(lane.center_x(float(y_m)))
    w = max(1e-3, float(lane.lane_width_m))
    return int(np.floor((float(x_m) - cx) / w + 0.5))


def place_in_lane(
    x_m: float,
    y_m: float,
    lane: Optional[LaneModel],
    lane_index: Optional[int] = None,
    offset_clamp_m: float = 1.2,
    snap_strength: float = 0.0,
) -> Tuple[float, float]:
    """Render position (x, y) that follows the lane's actual curvature.

    Keeps the *measured* within-lane offset rather than quantising to the lane
    centre, so two cars abreast in the same lane stay distinguishable and a car
    drifting toward a boundary looks like it is drifting.

    `snap_strength` in [0, 1] optionally pulls toward the lane centre for
    cosmetic calm; 0 keeps the measurement, 1 reproduces the old hard snap.
    Use the Kalman filter for smoothing instead of raising this.
    """
    if lane is None:
        return float(x_m), float(y_m)
    if lane_index is None:
        lane_index, offset = assign_lane(x_m, y_m, lane)
    else:
        offset = float(x_m) - float(lane.lane_center_x(lane_index, float(y_m)))
    if lane_index is None:
        return float(x_m), float(y_m)
    # A clamped index leaves a large residual offset. Clamping that too would
    # silently move the object a whole lane; the caller should have dropped it.
    offset = float(np.clip(offset, -offset_clamp_m, offset_clamp_m))
    offset *= (1.0 - float(np.clip(snap_strength, 0.0, 1.0)))
    # Render frame is the lane frame: curvature only, ego pose removed, so the
    # ribbon stays pinned to the canvas exactly as the lane markings are drawn.
    x_render = float(lane.shape_x(float(y_m))) + lane_index * lane.lane_width_m + offset
    return x_render, float(y_m)


class LaneAssigner:
    """Per-track hysteresis over the discrete lane index."""

    def __init__(self, max_index: int = 2):
        self.max_index = int(max_index)
        self._state: dict = {}

    def update(self, track_id: int, x_m: float, y_m: float,
               lane: Optional[LaneModel]) -> Tuple[Optional[int], float]:
        idx, offset = assign_lane(x_m, y_m, lane, self.max_index)
        if idx is None:
            return None, offset
        return self.update_index(track_id, idx, offset)

    def update_index(self, track_id: int, idx: int, offset: float
                     ) -> Tuple[int, float]:
        """Sticky left/right; ego only after ENTER_EGO_HITS image frames."""
        idx = int(np.clip(idx, -self.max_index, self.max_index))
        st = self._state.get(track_id)
        if st is None:
            self._state[track_id] = {"idx": idx, "cand": idx, "hits": 0, "miss": 0}
            return idx, float(offset)
        if idx == st["idx"]:
            st["hits"], st["miss"], st["cand"] = 0, 0, idx
            return st["idx"], float(offset)
        if idx == st["cand"]:
            st["hits"] += 1
        else:
            st["cand"], st["hits"] = idx, 1
        need = ENTER_EGO_HITS
        if st["idx"] == 0 and idx != 0:
            need = LEAVE_EGO_HITS
        elif st["idx"] != 0 and idx == 0:
            need = ENTER_EGO_HITS
        else:
            need = SIDE_CHANGE_HITS
        if st["hits"] >= need:
            st["idx"], st["hits"] = idx, 0
        return int(st["idx"]), float(offset)

    def drop(self, live_ids) -> None:
        live = set(int(t) for t in live_ids)
        self._state = {k: v for k, v in self._state.items() if k in live}
