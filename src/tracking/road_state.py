"""Shared, fail-closed temporal road state for all ADAS entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.inference import lane_filter_config as cfg
from src.tracking.lane_association import LaneTrackerManager
from src.tracking.lane_world import LaneWorld, LaneWorldState
from src.utils.drivable_area import (
    CorridorEMA,
    extract_ego_corridor_3d,
    force_corridor_fixed_width,
)


@dataclass
class RoadState:
    # Immediate model output for visual continuity; never use this for CIPO.
    visual_lanes: list
    # Confirmed/predicted temporal tracks used for road-state decisions.
    lanes: list
    ego_left: Optional[np.ndarray]
    ego_right: Optional[np.ndarray]
    left_corridor_3d: Optional[np.ndarray]
    right_corridor_3d: Optional[np.ndarray]
    status: str
    age_frames: int
    source: str = "none"
    reconstructed_side: Optional[str] = None
    locked_width_m: Optional[float] = None
    reset_draw: bool = False
    # Ego-centred physical lane geometry shared by objects and the BEV.
    lane_world: Optional[LaneWorldState] = None
    lane_debug: dict = field(default_factory=dict)

    @property
    def has_valid_corridor(self) -> bool:
        return self.left_corridor_3d is not None and self.right_corridor_3d is not None

    @property
    def is_confirmed(self) -> bool:
        return self.status == "CONFIRMED"


class RoadStateEstimator:
    """Tracks lanes and exposes one validated ego-corridor snapshot per frame.

    The ego lane comes from ``LaneWorld``: boundary slots that keep their
    identity through detector relocks, so the corridor only moves to another
    lane on a geometric lane change. A boundary that is not measured this
    frame is drawn from the slot (PREDICTED, never CONFIRMED). Once the world
    has gone without lines past its hold, the state is UNKNOWN.
    """

    def __init__(self) -> None:
        self.lane_tracker = LaneTrackerManager(
            max_missed_frames=cfg.EKF_MAX_MISSED_FRAMES,
            dist_threshold=cfg.EKF_DIST_THRESHOLD_M,
            confirm_hits=cfg.EKF_CONFIRM_HITS,
            require_confirmed=cfg.EKF_REQUIRE_CONFIRMED,
        )
        self.lane_world = LaneWorld()
        self.corridor_ema = CorridorEMA()
        self._age_frames = 0
        self._lane_id: Optional[int] = None

    def reset(self) -> None:
        self.__init__()

    def update(self, raw_lanes: Optional[Sequence[np.ndarray]], dt: float, speed_mps: Optional[float] = None) -> RoadState:
        raw_lanes = [] if raw_lanes is None else list(raw_lanes)
        dt = max(1e-3, float(dt))
        tracked_lanes = list(
            self.lane_tracker.update(raw_lanes, dt=dt, speed_mps=speed_mps)
        )
        fresh = list(getattr(self.lane_tracker, "last_fresh", []))
        if len(fresh) != len(tracked_lanes):
            fresh = [bool(raw_lanes)] * len(tracked_lanes)
        if not raw_lanes:
            fresh = [False] * len(tracked_lanes)

        world = self.lane_world.update(
            tracked_lanes, fresh, dt=dt, speed_mps=speed_mps
        )
        matched = self.lane_world.matched_lines

        ego_left = ego_right = None
        reconstructed_side = None
        meas_l = world.valid and world.is_measured(-1)
        meas_r = world.valid and world.is_measured(1)
        if world.valid:
            ego_left = matched.get(-1) if meas_l else None
            ego_right = matched.get(1) if meas_r else None
            if ego_left is None:
                ego_left = world.boundary_polyline(-1)
                reconstructed_side = "left"
            if ego_right is None:
                ego_right = world.boundary_polyline(1)
                reconstructed_side = "both" if reconstructed_side else "right"

        confirmed_now = bool(meas_l and meas_r)
        if confirmed_now:
            self._age_frames = 0
        else:
            self._age_frames += 1

        can_emit = world.valid and (
            meas_l or meas_r or self._age_frames <= int(cfg.EGO_PAIR_HOLD_FRAMES)
        )

        reset_draw = world.valid and (
            self._lane_id is None or world.ego_lane_id != self._lane_id
        )
        self._lane_id = world.ego_lane_id if world.valid else None

        left = right = None
        width = world.ego_width if world.valid else None
        if can_emit:
            left, right = extract_ego_corridor_3d(
                tracked_lanes,
                ego_left=ego_left,
                ego_right=ego_right,
                lane_width_m=width,
            )
            if reset_draw:
                self.corridor_ema.reset()
            left, right = self.corridor_ema.update(left, right)
            if bool(getattr(cfg, "CORRIDOR_FORCE_FIXED_WIDTH", True)):
                left, right = force_corridor_fixed_width(left, right, width_m=width)
        else:
            ego_left = ego_right = None
            self.corridor_ema.reset()

        if not can_emit:
            status = "UNKNOWN"
            source = "none"
            reconstructed_side = None
        elif confirmed_now:
            status = "CONFIRMED"
            source = world.source
        else:
            status = "PREDICTED"
            source = world.source

        visual_lanes = list(tracked_lanes) if tracked_lanes else list(raw_lanes)

        return RoadState(
            visual_lanes=visual_lanes,
            lanes=tracked_lanes,
            ego_left=ego_left,
            ego_right=ego_right,
            left_corridor_3d=left,
            right_corridor_3d=right,
            status=status,
            age_frames=self._age_frames,
            source=source,
            reconstructed_side=reconstructed_side,
            locked_width_m=width,
            reset_draw=bool(reset_draw),
            lane_world=world,
            lane_debug=world.debug(),
        )
