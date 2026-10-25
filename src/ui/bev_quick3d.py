"""
Qt Quick 3D BEV viewport.

Embeds a View3D inside the existing QWidget cockpit via QQuickWidget.
Same public API as BEVWidget so main_window can swap backends.

The ego car is the origin. Geometry is drawn in the ego frame with the ego's
heading relative to the road removed (x - c1*y), so the road runs up the screen.
Lane lines sit at the physical boundary offsets tracked by ``LaneWorld``, and
each vehicle at its own measured lateral position, kept inside the lane it was
assigned to at its own range. Nothing is placed by ``lane_index * width``, so
a detector relock cannot move a car to another lane on screen. Everything is
sampled uniformly along Y with a constant point count, so the QML segment
pools never pop as anchor visibility changes.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time

import numpy as np

from PySide6.QtCore import QObject, QUrl, Qt, QTimer
from PySide6.QtQuickWidgets import QQuickWidget

from src.ui.car_assets import (
    EGO_AUDI,
    KIND_ASSETS,
    KIND_DODGE,
    KIND_MAX,
    KIND_SHC,
    KIND_SKODA,
    KIND_TESLA,
    TRAFFIC_DODGE,
    TRAFFIC_SHC,
    TRAFFIC_SKODA,
    TRAFFIC_TESLA,
    TRUCK_LABELS,
    asset_url,
)
from src.tracking.lane_frame import LaneFrameModel
from src.tracking.lane_assignment import LaneModel
from src.tracking.lane_world import SLOTS, LaneWorld, boundary_ids
from src.tracking.heading import CROSS_MODES, HeadingEstimator
from src.tracking.ego_pose import EgoPose
from src.utils.drivable_area import parse_lane_components
from src.adas.threat import CAUTION, WARNING, format_lead_tag

# Local copies — do not import bev_widget (QPainter + sprite pipeline).
BEV_MAX_DIST_M = 70.0
BEV_DISK_R_M = 70.0
BEV_MAX_LATERAL_M = 14.0
TURN_YAW_MAX_DEG = 14.0
TURN_YAW_GAIN = 0.45
TURN_YAW_ALPHA = 0.22
DEFAULT_VIEW_PITCH = 8.0
DEFAULT_VIEW_YAW = 0.0
DEFAULT_ZOOM = 1.08
DEFAULT_CALIB_PITCH = 0.0
DEFAULT_CALIB_H = 1.0

# Layer A: ego body keep-out (half sedan length ~2.3 m + bumper margin).
EGO_KEEP_OUT_M = 5.0
# Render-tick coast between ~12 Hz inference frames.
MAX_EXTRAPOLATE_S = 0.20

TRAFFIC_LANE_MAX = 1          # cars in the ego lane and one lane each side
DEFAULT_LANE_W_M = 3.7
SAME_LANE_GAP_M = 8.0         # min range gap for two cars in the same lane

# Uniform sampling of every lane-frame polyline. Constant across frames.
POLY_SAMPLES = 48
CORRIDOR_SEGS = 16
BOUNDARY_SEGS = 12
CORRIDOR_START_M = 5.0   # same near anchor as the lane lines, ahead of the ego body
CORRIDOR_DRAW_M = 30.0   # practical cluster lookahead

# Lane-slot presence hysteresis: how a neighbouring marking fades in/out.
SLOT_MATCH_M = 0.90
SLOT_ENTER_HITS = 3
SLOT_EXIT_MISS = 10
MAX_SLOT = 3  # boundaries per side beyond the ego pair

_QML_PATH = os.path.join(os.path.dirname(__file__), "qml", "BevScene.qml")
_ENV_MODES = ("auto", "day", "dusk", "night", "demo")


def smooth_measurement(prev, measured, alpha=0.28, max_step=0.55):
    """Ease a noisy (x, y) detection. Caps the step so a bad frame cannot jump."""
    if prev is None:
        return (float(measured[0]), float(measured[1]))
    px, py = float(prev[0]), float(prev[1])
    dx = float(measured[0]) - px
    dy = float(measured[1]) - py
    nx = px + float(alpha) * dx
    ny = py + float(alpha) * dy
    step = math.hypot(nx - px, ny - py)
    if step > float(max_step) and step > 1e-6:
        scale = float(max_step) / step
        nx = px + (nx - px) * scale
        ny = py + (ny - py) * scale
    return (nx, ny)


def ease_boundaries(disp, target, gain=0.3):
    """Ease displayed boundary offsets toward the tracked ones.

    Both dicts are keyed by road-fixed boundary id (``boundary_ids``). A
    boundary seen for the first time starts at its tracked position and one no
    longer tracked is dropped. Ids survive an ego lane change, so a line the
    car crosses keeps travelling across it instead of being renumbered.
    """
    out = {}
    for bid, x in target.items():
        prev = disp.get(bid)
        x = float(x)
        out[bid] = x if prev is None else float(prev) + float(gain) * (x - float(prev))
    return out


def lane_change_target(raw_offset, lane_left, lane_right):
    """Where the ego car should sit during a lane change or a departure.

    The measured offset is the lane change itself: the car slides across the
    fixed markings. A departure warning still pushes it clearly onto that line.
    """
    target = float(raw_offset)
    yaw = 0.0
    # A warning may push a small drift onto the line. It must not reverse a
    # slide that is already carrying the road the other way.
    if lane_left == "warning" and target < 0.4:
        target = min(target, -0.75)
        yaw = 8.0
    elif lane_right == "warning" and target > -0.4:
        target = max(target, 0.75)
        yaw = -8.0
    elif lane_left == "caution" and target < 0.4:
        target = min(target, -0.35)
        yaw = 4.0
    elif lane_right == "caution" and target > -0.4:
        target = max(target, 0.35)
        yaw = -4.0
    else:
        yaw = float(max(-10.0, min(10.0, -target * 4.0)))
    return float(max(-4.2, min(4.2, target))), yaw


class _SlotTracker:
    """Hysteretic presence for lane boundaries at fixed lane-width multiples.

    Replaces find_outer_lanes(), which re-picked the outermost lane purely by
    mean X every frame and so teleported the road edge by a whole lane width
    whenever a new far marking appeared.
    """

    def __init__(self):
        self._hits = {}
        self._on = {}

    def update(self, present_slots, all_slots):
        for s in all_slots:
            hit = s in present_slots
            h = self._hits.get(s, 0)
            if hit:
                h = min(SLOT_ENTER_HITS, h + 1) if h >= 0 else 1
                self._hits[s] = h
                if h >= SLOT_ENTER_HITS:
                    self._on[s] = SLOT_EXIT_MISS
            else:
                self._hits[s] = 0
                if s in self._on:
                    self._on[s] -= 1
                    if self._on[s] <= 0:
                        del self._on[s]
        return sorted(self._on.keys())

    def reset(self):
        self._hits.clear()
        self._on.clear()


class _BevDriver:
    """Feeds a BevScene.qml instance from perception output.

    Shared by the standalone QQuickWidget (classic cockpit) and the embedded
    scene inside Cluster.qml. Subclasses provide ``_root()``: the QML object
    whose properties this driver writes.
    """

    def _driver_init(self):
        self.proposals = []
        self.processed_objs = []
        self.cipo_status = "SAFE"
        self.left_3d = None
        self.right_3d = None
        self.cinematic_road = True
        self.show_lane_lines = False
        self.env_mode = "auto"
        self.lane_frame = LaneFrameModel()
        self._slots = _SlotTracker()
        self._tid_slot = {}   # track_id → (mesh_kind, mesh_pool_index)
        # Fallback when the caller does not hand over the pipeline's lane world.
        self._own_world = LaneWorld()
        self._world = None
        self._view_world = None
        self._bound_target = {}   # road-fixed boundary id -> offset at y=0 (m)
        self._bound_disp = {}     # same, eased for display
        self._heading = HeadingEstimator()
        self._kind_vote = {}  # tid -> {truck, hits}
        self._ego_pose = EgoPose()
        self._turn_yaw_deg = 0.0
        self._world_flash_deg = 0.0
        self._ego_speed_mps = None
        self._last_traffic_json = None
        self._last_corridor_json = None
        self._last_lane_json = None
        self._last_dash_json = None
        self._last_center_json = None
        self._last_edge_json = None
        self._traffic_seed = []
        self._traffic_t0 = 0.0
        self._threat = None
        self._lane_states = ("normal", "normal")
        self._units = "mph"
        self._ego_disp = 0.0
        self._ego_target = 0.0
        self._depart_yaw = 0.0
        self._depart_yaw_disp = 0.0
        # Paint scroll is independent of lane_frame.distance so the dashes can
        # move on the display timer. Unknown speed still crawls, so the road
        # does not look frozen while the speed readout says "No speed source".
        self._paint_m = 0.0
        self._paint_t = time.perf_counter()
        self._scene_paused = False
        self._track_smooth = {}
        self._track_motion = {}
        self._track_target = {}
        self._traffic_shown = []
        self._corridor_x = 0.0
        self._motion_t = time.perf_counter()
        self._mini_bev = "{}"
        self._last_chevron_json = None
        self._tag_seed = None

        self._push_camera(
            pitch=DEFAULT_VIEW_PITCH,
            yaw=DEFAULT_VIEW_YAW,
            zoom=DEFAULT_ZOOM,
            calib_pitch=DEFAULT_CALIB_PITCH,
            calib_h=DEFAULT_CALIB_H,
            pan_x=0.0,
            pan_y=0.0,
        )
        self._set("envMode", self.env_mode)
        self._set("diskRadiusM", float(BEV_DISK_R_M))
        self._set("debugMaxZ", float(BEV_MAX_DIST_M))
        self._set("trailJson", "[]")
        self._bind_ego_asset()
        self._extrap = QTimer(self._timer_parent())
        self._extrap.setInterval(33)
        self._extrap.timeout.connect(self._on_extrap_tick)
        self._extrap.start()

    def _timer_parent(self):
        return self

    def _root(self):
        raise NotImplementedError

    def set_units(self, units):
        self._units = "kmh" if str(units).lower() in ("kmh", "km/h", "kph") else "mph"

    def _set(self, name, value):
        root = self._root()
        if root is not None:
            root.setProperty(name, value)

    def _bind_ego_asset(self):
        asset = EGO_AUDI
        if not asset.exists():
            self._set("overlayHint", f"Phase 1 — missing {asset.filename}")
            print(f"[BEV] Ego GLB not found: {asset.path}")
            return
        url = QUrl.fromLocalFile(os.path.abspath(asset.path)).toString()
        self._set("egoGltf", url)
        self._set("egoScale", float(asset.scale))
        self._set("egoRotX", float(asset.rot_x))
        self._set("egoRotY", float(asset.rot_y))
        self._set("egoRotZ", float(asset.rot_z))
        self._set("egoY", float(asset.y))
        self._set("overlayHint", "Phase 4 — CIPO")
        print(f"[BEV] Ego GLB: {asset.path}")
        if TRAFFIC_SKODA.exists():
            self._set("skodaGltf", asset_url(TRAFFIC_SKODA))
            self._set("skodaScale", float(TRAFFIC_SKODA.scale))
            print(f"[BEV] Traffic Skoda: {TRAFFIC_SKODA.path}")
        if TRAFFIC_SHC.exists():
            self._set("shcGltf", asset_url(TRAFFIC_SHC))
            self._set("shcScale", float(TRAFFIC_SHC.scale))
            self._set("shcRotY", float(TRAFFIC_SHC.rot_y))
            print(f"[BEV] Traffic SHC: {TRAFFIC_SHC.path}")
        if TRAFFIC_DODGE.exists():
            self._set("dodgeGltf", asset_url(TRAFFIC_DODGE))
            self._set("dodgeScale", float(TRAFFIC_DODGE.scale))
            print(f"[BEV] Traffic truck: {TRAFFIC_DODGE.path}")
        else:
            print(f"[BEV] Traffic truck missing: {TRAFFIC_DODGE.path}")

    def _push_camera(self, pitch, yaw, zoom, calib_pitch, calib_h, pan_x, pan_y):
        self._set("pitchDeg", float(pitch))
        self._set("yawDeg", float(yaw))
        self._set("zoomFactor", float(zoom))
        self._set("calibPitch", float(calib_pitch))
        self._set("calibH", float(calib_h))
        self._set("panX", float(pan_x))
        self._set("panY", float(pan_y))

    def toggle_cinematic_road(self):
        self.cinematic_road = not self.cinematic_road
        self._set("cinematicRoad", self.cinematic_road)
        return self.cinematic_road

    def toggle_lane_lines(self):
        return self.set_lane_lines(not self.show_lane_lines)

    def set_lane_lines(self, on):
        self.show_lane_lines = bool(on)
        self._set("showLaneLines", self.show_lane_lines)
        return self.show_lane_lines

    def toggle_scenic_view(self):
        """Full scene (sky/grass/mountains) vs one-color road-only semantic."""
        root = self._root()
        cur = True
        if root is not None:
            val = root.property("scenicView")
            if val is not None:
                cur = bool(val)
        nxt = not cur
        self._set("scenicView", nxt)
        return nxt

    def cycle_env_mode(self):
        """Cycle BEV background: Auto → Day → Dusk → Night → Demo."""
        root = self._root()
        if root is not None:
            cur = root.property("envMode")
            if cur:
                self.env_mode = str(cur)
        try:
            i = _ENV_MODES.index(self.env_mode)
        except ValueError:
            i = -1
        self.env_mode = _ENV_MODES[(i + 1) % len(_ENV_MODES)]
        self._set("envMode", self.env_mode)
        return self.env_mode

    def set_env_mode(self, mode: str):
        mode = str(mode).lower().strip()
        if mode not in _ENV_MODES:
            raise ValueError(f"env mode must be one of {_ENV_MODES}")
        self.env_mode = mode
        self._set("envMode", self.env_mode)
        return self.env_mode

    def enable_cluster_chrome(self):
        """Hide QML controls; the dashboard cockpit supplies its own HUD."""
        self._set("clusterChrome", True)
        self.set_env_mode("day")
        self._set("scenicView", True)

    def set_calibration(self, pitch_deg, height_m):
        self._set("calibPitch", float(pitch_deg))
        self._set("calibH", float(height_m))

    def toggle_debug_metric(self):
        """Ego centreline at x=0, 10 m ruler ticks, per-object (x, z) in metres.

        Reads the same trafficJson the renderer consumes, so the printed numbers
        cannot drift from the drawn positions. Pair with
        scripts/debug/range_diagnostics.py to check BEV placement against the
        per-frame CSV.
        """
        root = self._root()
        cur = bool(root.property("debugMetric")) if root is not None else False
        self._set("debugMetric", not cur)
        self._set("debugMaxZ", float(BEV_MAX_DIST_M))
        return not cur

    def _bev_visible(self, obj) -> bool:
        z = float(obj.get("Z_3d", 0.0))
        x = float(obj.get("X_3d", 0.0))
        # Layer A: never draw into the ego GLB body.
        if not (EGO_KEEP_OUT_M < z <= BEV_MAX_DIST_M):
            return False
        if math.hypot(x, z) > BEV_DISK_R_M:
            return False
        lane = self._object_lane(obj)
        if lane is None:
            return abs(x) <= BEV_MAX_LATERAL_M
        return abs(int(lane)) <= TRAFFIC_LANE_MAX

    def _lane_width_m(self) -> float:
        if self.lane_frame.valid:
            return max(2.8, min(4.6, float(self.lane_frame.lane_width)))
        return DEFAULT_LANE_W_M

    def _object_lane(self, obj):
        """The tracker's lane label, else the lane at the object's own range."""
        idx = obj.get("lane_index")
        if idx is not None and int(idx) != 99:
            return int(idx)
        w = self._world
        if w is None or not w.valid:
            return None
        return int(w.lane_of(float(obj.get("X_3d", 0.0)), float(obj.get("Z_3d", 0.0))).index)

    def _render_x(self, x, z):
        """Ego-frame x with the ego heading removed, the frame every layer is drawn in."""
        if not self.lane_frame.valid or self.lane_frame._c is None:
            return float(x)
        return float(x) - float(self.lane_frame._c[1]) * float(z)

    def _sync_world(self, world):
        """Adopt this frame's lane world; boundary targets keyed by road-fixed id."""
        self._world = world
        if world is None or not world.valid:
            self._bound_target = {}
            self._bound_disp = {}
            self._view_world = None
            return
        ids = boundary_ids(world.ego_lane_id)
        self._bound_target = {ids[k]: world.offset(k) for k in SLOTS}
        if not self._bound_disp:
            self._bound_disp = dict(self._bound_target)
        self._refresh_view_world()

    def _refresh_view_world(self):
        """World as drawn: eased offsets, heading removed. Clamps use this one."""
        w = self._world
        if w is None or not w.valid:
            self._view_world = None
            return
        ids = boundary_ids(w.ego_lane_id)
        offs = tuple(
            float(self._bound_disp.get(ids[k], w.offset(k))) for k in SLOTS
        )
        self._view_world = dataclasses.replace(w, offsets=offs, c1=0.0)

    def _display_lines(self):
        """(x at y=0, pal) for the four drawn boundaries; pal 1/2 is the ego pair."""
        vw = self._view_world
        if vw is None:
            return []
        pal = {-2: 0, -1: 1, 1: 2, 2: 0}
        return [(vw.offset(k), pal[k]) for k in SLOTS]

    def _ego_lane_center(self):
        """Centre of the ego lane at the car, as drawn."""
        vw = self._view_world
        return 0.0 if vw is None else float(vw.ego_center)

    def _heading_yaw(self, obj, y, lane_model):
        """Lane tangent, or world-velocity class when ego speed is known."""
        return self._heading.update(
            int(obj.get("track_id", -1)),
            float(obj.get("vx", 0.0)),
            float(obj.get("vy", 0.0)),
            float(y),
            self._ego_speed_mps,
            lane_model,
        )

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _poly_segments(ys, xs, width, pal=None, max_segs=BOUNDARY_SEGS, radius=BEV_DISK_R_M,
                       y_min=0.0):
        """Segment rows from a uniformly sampled lane-frame polyline.

        ys/xs have a fixed length, so the emitted segment count is identical
        every frame and the QML pools never pop in or out. Live paint stays in
        the forward 70 m disk; sides and rear stay empty.
        """
        ys = np.asarray(ys, dtype=np.float64)
        xs = np.asarray(xs, dtype=np.float64)
        n = min(len(ys), len(xs))
        if n < 2:
            return []
        ys, xs = ys[:n], xs[:n]
        if radius is not None:
            r2 = float(radius) * float(radius)
            keep = (ys >= float(y_min)) & ((xs * xs + ys * ys) <= r2)
            if int(np.sum(keep)) < 2:
                return []
            ys, xs = ys[keep], xs[keep]
            n = len(ys)
            if n < 2:
                return []
        step = max(1, (n - 1) // max_segs)
        segs = []
        i = 0
        while i + step < n:
            x0, y0 = float(xs[i]), float(ys[i])
            x1, y1 = float(xs[i + step]), float(ys[i + step])
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length >= 0.35:
                # Qt yaw: +Y up, length along local Z after yaw. World Δx, Δz=-Δy.
                row = {
                    "x": round(0.5 * (x0 + x1), 2),
                    "z": round(-0.5 * (y0 + y1), 2),
                    "yaw": round(math.degrees(math.atan2(dx, -dy)), 1),
                    "len": round(length, 2),
                    "w": round(float(width), 2),
                }
                if pal is not None:
                    row["c"] = int(pal)
                segs.append(row)
            i += step
        return segs

    def _lane_frame_mean_xs(self, proposals):
        """Mean lateral X of every detected lane, expressed in the lane frame."""
        out = []
        if not proposals:
            return out
        for lane in proposals:
            try:
                xs, ys, _zs, vis = parse_lane_components(lane)
            except Exception:
                continue
            m = vis & (ys > 1.0) & (ys <= BEV_MAX_DIST_M)
            if int(np.sum(m)) < 2:
                continue
            lx = self.lane_frame.to_lane_frame(xs[m], ys[m])
            out.append(float(np.mean(lx)))
        return out

    def _active_slots(self, proposals):
        """Which lane boundaries exist, as signed lane-width multiples.

        Slot k>0 is the k-th boundary right of ego centre, k<0 to the left;
        |k|=1 is the ego lane's own marking.
        """
        half_w = self.lane_frame.half_width
        width = self.lane_frame.lane_width
        candidates = {}
        for k in range(1, MAX_SLOT + 2):
            off = half_w + (k - 1) * width
            candidates[k] = off
            candidates[-k] = -off

        means = self._lane_frame_mean_xs(proposals)
        present = set()
        for mx in means:
            best, best_d = None, SLOT_MATCH_M
            for k, off in candidates.items():
                d = abs(mx - off)
                if d < best_d:
                    best, best_d = k, d
            if best is not None:
                present.add(best)
        # Ego pair is implied by a valid corridor even if the paint is faint.
        if self.lane_frame.valid:
            present.update({-1, 1})
        return self._slots.update(present, list(candidates.keys())), candidates

    # -------------------------------------------------------------- payloads
    def _corridor_end_m(self):
        """How far the drivable band runs. No lead, no band."""
        if not self.lane_frame.valid:
            return None
        end = CORRIDOR_DRAW_M
        if self._threat is not None:
            lead = self._threat.lead
            if lead is None or lead.level <= 0:
                return None
            end = min(BEV_MAX_DIST_M, float(lead.range_m) - 1.2)
            if end < CORRIDOR_START_M + 1.5:
                return None
        return float(end)

    def _corridor_payload(self):
        """Ego-lane path ahead of the bumper — not a pad under the car.

        The band is broken into stripes that scroll toward the ego at the same
        rate as the lane dashes (``_paint_m``, integrated from ego speed). A
        solid ribbon would not show the speed.
        """
        end = self._corridor_end_m()
        if end is None:
            return []
        # Inside the ego lane, not edge to edge. 0.52 leaves a wider gap
        # before each lane line.
        width = float(np.clip(self.lane_frame.lane_width * 0.52, 1.4, 2.1))
        spans = self.lane_frame.dash_spans(
            end, dash_len=2.6, period=4.0, max_dashes=16, phase_m=self._paint_m,
        )
        rows = []
        for y0, y1 in spans:
            if y1 < CORRIDOR_START_M + 0.4:
                continue
            y0 = max(float(y0), CORRIDOR_START_M)
            if y1 - y0 < 0.45:
                continue
            ys = np.linspace(y0, y1, 3)
            xs = self.lane_frame.lane_x(ys) + float(self._corridor_x)
            rows.extend(self._poly_segments(
                ys, xs, width, max_segs=1, y_min=CORRIDOR_START_M,
            ))
            if len(rows) >= 16:
                break
        return rows[:16]

    def _center_payload(self):
        """Lane centerline, thinner than the corridor and on the same path."""
        if not self.lane_frame.valid:
            return []
        ys = np.linspace(CORRIDOR_START_M, CORRIDOR_DRAW_M, POLY_SAMPLES)
        xs = self.lane_frame.lane_x(ys) + float(self._corridor_x)
        rows = self._poly_segments(
            ys, xs, 0.08, max_segs=CORRIDOR_SEGS, y_min=CORRIDOR_START_M,
        )
        return rows[:18]

    def _paint_speed_mps(self):
        """Scroll rate for the lane paint. Real speed when we have it."""
        v = self._ego_speed_mps
        if v is None:
            return 12.0
        if float(v) < 0.6:
            return 0.0
        return float(v)

    def _advance_paint(self):
        now = time.perf_counter()
        dt = min(0.12, max(0.0, now - self._paint_t))
        self._paint_t = now
        v = self._paint_speed_mps()
        if v <= 0.0 or not self.lane_frame.valid:
            return
        self._paint_m += v * dt
        dashes = json.dumps(self._dash_payload(), separators=(",", ":"))
        if dashes != self._last_dash_json:
            self._last_dash_json = dashes
            self._set("dashJson", dashes)
        # The drivable corridor uses the same phase, so its stripes flow at
        # ego speed between detection frames, the same way the lane dashes do.
        corr = json.dumps(self._corridor_payload(), separators=(",", ":"))
        if corr != self._last_corridor_json:
            self._last_corridor_json = corr
            self._set("corridorJson", corr)

    def _ease_ego(self):
        """Hold the ego car at the origin; it yaws into a turn or departure."""
        self._ego_target = 0.0
        self._ego_disp += 0.14 * (float(self._ego_target) - self._ego_disp)
        if abs(self._ego_disp) < 0.004 and abs(self._ego_target) < 0.004:
            self._ego_disp = 0.0
        self._depart_yaw_disp += 0.14 * (float(self._depart_yaw) - self._depart_yaw_disp)
        if abs(self._depart_yaw_disp) < 0.05 and abs(self._depart_yaw) < 0.05:
            self._depart_yaw_disp = 0.0
        self._set("egoX", round(float(self._ego_disp), 3))
        self._set("egoYawDeg", round(-self._turn_yaw_deg + self._depart_yaw_disp, 2))

    def _ease_lanes(self):
        """Ease drawn boundaries toward the tracked ones; corridor to the ego lane.

        The lines are at physical offsets from the car, so ego lateral motion
        and a lane change show as the lines moving across a fixed car. After an
        ego lane change the corridor eases into the new lane.
        """
        self._bound_disp = ease_boundaries(self._bound_disp, self._bound_target)
        self._refresh_view_world()
        want = self._ego_lane_center()
        self._corridor_x += 0.22 * (float(want) - float(self._corridor_x))
        if abs(self._corridor_x - want) < 0.005:
            self._corridor_x = float(want)

    def _dash_payload(self):
        """Four lane markings scrolling toward the ego. Ego pair can turn red."""
        if not self.lane_frame.valid:
            return []
        lines = self._display_lines()
        spans = [
            (a, b) for a, b in self.lane_frame.dash_spans(
                BEV_MAX_DIST_M, phase_m=self._paint_m, max_dashes=5,
            )
            if b - a >= 0.45
        ]
        rows = []
        for x_off, pal in lines:
            side = "normal"
            if pal == 1:
                side = self._lane_states[0]
            elif pal == 2:
                side = self._lane_states[1]
            width = 0.22 if side in ("caution", "warning") else 0.13
            for span_y0, span_y1 in spans:
                ys = np.linspace(span_y0, span_y1, 3)
                cx = self.lane_frame.lane_x(ys) + float(x_off)
                rows.extend(self._poly_segments(
                    ys, cx, width, pal=pal, max_segs=1,
                ))
                if len(rows) >= 28:
                    return rows[:28]
        return rows[:28]

    def _edge_payload(self, slots, offsets):
        """Dashboard BEV shows only the ego lane — no outer road edges."""
        return []

    def _lane_payload(self, slots, offsets):
        """Dashboard BEV shows only the ego lane — no adjacent markings."""
        return []

    def _prefer_kind(self, obj):
        """Truck mesh only after a sticky vote — one YOLO 'truck' frame is not enough."""
        tid = int(obj.get("track_id", -1))
        label = str(obj.get("label", "car")).lower().strip()
        meas = label in TRUCK_LABELS
        st = self._kind_vote.get(tid)
        if st is None:
            st = {"truck": False, "hits": 0}
            self._kind_vote[tid] = st
        if meas == st["truck"]:
            st["hits"] = 0
        else:
            st["hits"] += 1
            need = 5 if meas else 3
            if st["hits"] >= need:
                st["truck"] = meas
                st["hits"] = 0
        if not st["truck"] or not TRAFFIC_DODGE.exists():
            return KIND_SKODA
        bbox = obj.get("bbox") or [0, 0, 1, 1]
        bw = abs(float(bbox[2]) - float(bbox[0]))
        z = max(1.0, float(obj.get("Z_3d", 1.0)))
        # Tiny far box: sedan YOLO-labelled truck. ~1.5 m wide at f≈700.
        if bw * z < 1100.0:
            return KIND_SKODA
        if KIND_MAX.get(KIND_DODGE, 0) <= 0:
            return KIND_SKODA
        return KIND_DODGE

    def _alloc_slot(self, tid, prefer_kind):
        """Keep (kind, slot) sticky per track_id so QML nodes do not swap cars."""
        existing = self._tid_slot.get(tid)
        if existing is not None:
            kind, slot = existing
            # Drop SHC/Tesla leftovers: those meshes were centimetre-scale specks.
            if kind in (KIND_SHC, KIND_TESLA) or KIND_MAX.get(kind, 0) <= slot:
                self._tid_slot.pop(tid, None)
            elif kind != prefer_kind:
                self._tid_slot.pop(tid, None)
            else:
                return existing
        taken = {k: set() for k in KIND_MAX}
        for k, s in self._tid_slot.values():
            taken.setdefault(k, set()).add(s)
        for kind in (prefer_kind, KIND_SKODA, KIND_DODGE):
            cap = int(KIND_MAX.get(kind, 0))
            if cap <= 0 or not KIND_ASSETS[kind].exists():
                continue
            used = taken.get(kind, set())
            for slot in range(cap):
                if slot not in used:
                    self._tid_slot[tid] = (kind, slot)
                    return kind, slot
        return None

    def _traffic_payload(self, objs):
        rows = []
        if not objs:
            self._tid_slot = {}
            self._heading.drop([])
            self._kind_vote = {}
            self._track_smooth = {}
            self._track_motion = {}
            self._track_target = {}
            return rows
        visible = [o for o in objs if self._bev_visible(o)]
        live = set()
        for obj in visible:
            tid = int(obj.get("track_id", -1))
            if tid > 0:
                live.add(tid)
        self._tid_slot = {tid: slot for tid, slot in self._tid_slot.items() if tid in live}
        self._heading.drop(live)
        self._kind_vote = {t: s for t, s in self._kind_vote.items() if t in live}
        self._track_smooth = {t: p for t, p in self._track_smooth.items() if t in live}
        self._track_motion = {t: p for t, p in self._track_motion.items() if t in live}
        self._track_target = {t: p for t, p in self._track_target.items() if t in live}

        cipo = self._pick_cipo(visible)
        cipo_tid = int(cipo.get("track_id", -999)) if cipo is not None else None
        lane_model = LaneModel.from_lane_frame(self.lane_frame, n_left=1, n_right=1)
        ordered = sorted(
            visible,
            key=lambda o: (
                int(o.get("track_id", -1)) != cipo_tid,
                int(o.get("track_id", -1)) not in self._tid_slot,
                float(o.get("Z_3d", 99.0)),
            ),
        )
        pending = []
        for obj in ordered:
            tid = int(obj.get("track_id", -1))
            if tid <= 0:
                continue
            z = float(obj["Z_3d"])
            if z <= EGO_KEEP_OUT_M:
                continue

            lane = self._object_lane(obj)
            if lane is not None and abs(int(lane)) > TRAFFIC_LANE_MAX:
                continue
            x_r = self._render_x(float(obj.get("X_3d", 0.0)), z)
            pending.append((obj, tid, 99 if lane is None else int(lane), z, x_r))

        pending = self._spread_same_lane(pending)

        for obj, tid, lane_slot, z, x_r in pending:
            yaw, mode = self._heading_yaw(obj, z, lane_model)
            x, y = float(x_r), float(z)
            if mode not in CROSS_MODES and lane_slot != 99:
                x = self._clamp_in_lane(x, y, lane_slot, obj)

            bound = self._alloc_slot(tid, self._prefer_kind(obj))
            if bound is None:
                continue
            tx, ty = float(x), float(y)
            self._track_target[tid] = (tx, ty)
            # First sighting is the detection itself. Smoothing only damps
            # later frames, so the mesh never drives out from the ego car.
            sx, sy = smooth_measurement(self._track_smooth.get(tid), (tx, ty))
            self._track_smooth[tid] = (sx, sy)
            self._track_motion[tid] = {"x": sx, "y": sy}
            kind, mesh_slot = bound
            asset = KIND_ASSETS[kind]
            thr = 0
            prim = False
            if self._threat is not None:
                thr = int(self._threat.levels.get(tid, 0))
                prim = (tid == self._threat.primary_tid)
            rows.append({
                "thr": thr,
                "prim": prim,
                "tid": tid,
                "kind": kind,
                "slot": mesh_slot,
                "lane": int(lane_slot),
                "posX": round(float(x), 2),
                "posY": round(float(asset.y), 2),
                "posZ": round(-float(y), 2),
                "x0": float(x),
                "y0": float(y),
                "vx": float(obj.get("vx", 0.0)),
                "vy": float(obj.get("vy", 0.0)),
                "yawDeg": round(float(yaw), 1),
                "head": mode,
            })
        kept = {int(r["tid"]) for r in rows}
        self._track_motion = {t: p for t, p in self._track_motion.items() if t in kept}
        self._track_target = {t: p for t, p in self._track_target.items() if t in kept}
        return rows

    def _clamp_in_lane(self, x, y, lane, obj):
        """Keep the drawn body inside its assigned lane, as drawn at range y.

        The measured position is kept wherever it already agrees with the
        label; it is only pulled in where the body would overhang a line,
        so the picture never contradicts the lane the object was assigned to.
        """
        vw = self._view_world
        if vw is None:
            return float(x)
        lo, hi = vw.lane_bounds(int(lane), float(y))
        half_body = 0.9 if str(obj.get("label", "car")).lower() not in TRUCK_LABELS else 1.25
        half = min(half_body, max(0.0, 0.5 * (hi - lo) - 0.15))
        return float(np.clip(float(x), lo + half, hi - half))

    @staticmethod
    def _spread_same_lane(pending):
        """Keep two cars in one lane from stacking: nearer stays, farther slides back."""
        if len(pending) < 2:
            return pending
        by_lane = {}
        for item in pending:
            by_lane.setdefault(item[2], []).append(item)
        out = []
        for _lane, items in by_lane.items():
            items.sort(key=lambda t: t[3])
            last_z = None
            for obj, tid, lane_slot, z, x in items:
                if last_z is not None and z < last_z + SAME_LANE_GAP_M:
                    z = last_z + SAME_LANE_GAP_M
                last_z = z
                out.append((obj, tid, lane_slot, z, x))
        return out

    def set_paused(self, paused):
        """Freeze paint scroll and traffic motion while the source is paused."""
        self._scene_paused = bool(paused)
        if not self._scene_paused:
            now = time.perf_counter()
            self._paint_t = now
            if self._traffic_t0 > 0.0:
                self._traffic_t0 = now

    def _compose_shown(self):
        shown = []
        for row in self._traffic_seed or []:
            tid = int(row.get("tid", -1))
            st = self._track_motion.get(tid)
            if st is None:
                continue
            item = dict(row)
            item["posX"] = round(float(st["x"]), 2)
            item["posZ"] = round(-float(st["y"]), 2)
            shown.append(item)
        return shown

    def _publish_traffic(self):
        rows = self._compose_shown()
        self._traffic_shown = rows
        payload = json.dumps(rows, separators=(",", ":"))
        if payload != self._last_traffic_json:
            self._last_traffic_json = payload
            self._set("trafficJson", payload)
        self._place_tag(rows)
        self._refresh_mini_bev()
        return rows

    def _publish_slid_geometry(self):
        corr = json.dumps(self._corridor_payload(), separators=(",", ":"))
        if corr != self._last_corridor_json:
            self._last_corridor_json = corr
            self._set("corridorJson", corr)
        dashes = json.dumps(self._dash_payload(), separators=(",", ":"))
        if dashes != self._last_dash_json:
            self._last_dash_json = dashes
            self._set("dashJson", dashes)
        center = json.dumps(self._center_payload(), separators=(",", ":"))
        if center != self._last_center_json:
            self._last_center_json = center
            self._set("centerJson", center)
        mode, chev = self._chevron_payload()
        self._set("chevronMode", int(mode))
        chev_json = json.dumps(chev, separators=(",", ":"))
        if chev_json != self._last_chevron_json:
            self._last_chevron_json = chev_json
            self._set("chevronJson", chev_json)
        self._set("cipoX", round(float(self._corridor_x), 3))
        self._refresh_mini_bev()

    def _on_extrap_tick(self):
        if self._scene_paused:
            return
        before = dict(self._bound_disp)
        before_c = float(self._corridor_x)
        self._ease_lanes()
        self._advance_paint()
        self._ease_ego()
        moved = abs(float(self._corridor_x) - before_c) > 0.012 or any(
            abs(float(x) - float(before.get(bid, x))) > 0.012
            for bid, x in self._bound_disp.items()
        )
        if moved:
            self._publish_slid_geometry()

    def mini_bev_json(self):
        return self._mini_bev

    def _refresh_mini_bev(self):
        lines = []
        if self.lane_frame.valid:
            lines = [round(float(x), 2) for x, _pal in self._display_lines()]
        cars = []
        for row in self._traffic_shown or []:
            y = -float(row.get("posZ", 0.0))
            # The mini map draws the lanes straight, so take the curve out of the cars too.
            bend = float(self.lane_frame.lane_x(np.array([y]))[0]) if self.lane_frame.valid else 0.0
            cars.append({
                "x": round(float(row.get("posX", 0.0)) - bend, 2),
                "y": round(y, 1),
                "t": int(row.get("thr", 0)),
            })
        payload = {
            "lanes": lines,
            "left": self._lane_states[0],
            "right": self._lane_states[1],
            "ego": 0.0,
            "cars": cars[:8],
            "w": round(self._lane_width_m(), 2),
        }
        self._mini_bev = json.dumps(payload, separators=(",", ":"))

    def _place_tag(self, rows):
        """Pin the distance tag over the lead vehicle's rendered position."""
        tag = self._tag_seed
        if tag is None or self._threat is None or self._threat.lead is None:
            self._set("tagVisible", False)
            return
        tid = int(self._threat.lead.track_id)
        row = next((r for r in rows if int(r.get("tid", -1)) == tid), None)
        if row is None:
            self._set("tagVisible", False)
            return
        truck = int(row.get("kind", 0)) == KIND_DODGE
        # Traffic lives inside worldRig, which yaws by worldYawDeg during the
        # turn flash; rotate the anchor the same way so the tag stays on the car.
        th = math.radians(float(self._world_flash_deg))
        x, z = float(row["posX"]), float(row["posZ"])
        self._set("tagX", x * math.cos(th) + z * math.sin(th))
        self._set("tagZ", -x * math.sin(th) + z * math.cos(th))
        self._set("tagH", 4.6 if truck else 2.1)
        self._set("tagText", tag["text"])
        self._set("tagSub", tag["sub"])
        self._set("tagLevel", int(tag["level"]))
        self._set("tagVisible", True)

    def _chevron_payload(self):
        """Chevrons on the gap band (spec UX-FCW-02, UX-GAP-03)."""
        lead = self._threat.lead if self._threat is not None else None
        if lead is None or not self.lane_frame.valid:
            return 0, []
        if lead.level >= WARNING:
            mode, step = 2, 2.8
        elif lead.level == CAUTION and lead.closing_fast:
            mode, step = 1, 3.2
        else:
            return 0, []
        y0 = CORRIDOR_START_M + 1.2
        y1 = float(lead.range_m) - 2.6
        rows = []
        y = y0
        i = 0
        while y <= y1 and i < 10:
            x = float(self.lane_frame.lane_x(np.array([y]))[0]) + float(self._corridor_x)
            xn = float(self.lane_frame.lane_x(np.array([y + 0.5]))[0]) + float(self._corridor_x)
            yaw = math.degrees(math.atan2(-(xn - x), 0.5))
            rows.append({
                "x": round(x, 2), "z": round(-y, 2), "yaw": round(yaw, 1),
                "w": 1.7, "len": 0.85, "back": mode == 2, "i": i,
            })
            y += step
            i += 1
        # Red chevrons sequence from the lead back toward the ego.
        if mode == 2:
            for k, r in enumerate(reversed(rows)):
                r["i"] = k
        return mode, rows

    def _pick_cipo(self, objs):
        vis = [o for o in objs if self._bev_visible(o)]
        if not vis:
            return None
        marked = [o for o in vis if o.get("is_cipo")]
        if marked:
            return min(marked, key=lambda o: float(o.get("Z_3d", 99.0)))
        path = [o for o in vis if o.get("in_path") and int(o.get("lane_index", 0)) == 0]
        if path:
            return min(path, key=lambda o: float(o.get("Z_3d", 99.0)))
        return None

    def _push_cipo(self, objs, status):
        cipo = self._pick_cipo(objs or [])
        self._set("cipoStatus", str(status or "SAFE"))
        if cipo is None:
            self._set("cipoVisible", False)
            self._set("cipoDist", 0.0)
            return
        z = float(cipo.get("Z_3d", 0.0))
        self._set("cipoDist", round(z, 1))
        # Keep HUD distance, but never park the CIPO beacon inside the ego body.
        if z <= EGO_KEEP_OUT_M:
            self._set("cipoVisible", False)
            return
        self._set("cipoVisible", True)
        # CIPO beacon rides the ego-lane centre.
        self._set("cipoX", round(float(self._corridor_x), 3))
        self._set("cipoZ", round(-z, 2))

    def _push_ego_pose(self):
        # The car stays at the origin; only the nose yaws toward a departure.
        raw = 0.0
        if self._world is not None and self._world.valid:
            raw = float(self._world.ego_offset_in_lane)
        left, right = self._lane_states
        _ignored, self._depart_yaw = lane_change_target(raw, left, right)
        self._ego_target = 0.0
        # Brief 3rd-person turn: yaw the ego GLB from v·κ, counter-yaw the
        # world a little, then both EMA back to heading-up (0). Do not apply
        # accumulated odometry or the chase cam just orbits.
        rate_deg = math.degrees(self._ego_pose.yaw_rate)
        target = float(np.clip(rate_deg * TURN_YAW_GAIN, -TURN_YAW_MAX_DEG, TURN_YAW_MAX_DEG))
        a = TURN_YAW_ALPHA
        self._turn_yaw_deg = (1.0 - a) * self._turn_yaw_deg + a * target
        if abs(self._turn_yaw_deg) < 0.12 and abs(target) < 0.12:
            self._turn_yaw_deg = 0.0
        ego_q = -self._turn_yaw_deg
        world_target = -0.35 * ego_q
        self._world_flash_deg = (1.0 - a) * self._world_flash_deg + a * world_target
        if abs(self._world_flash_deg) < 0.08 and abs(world_target) < 0.08:
            self._world_flash_deg = 0.0
        self._set("worldYawDeg", round(self._world_flash_deg, 2))
        self._set("laneValid", bool(self.lane_frame.valid))
        self._set("laneHeld", bool(self.lane_frame.held))
        self._ease_lanes()
        self._ease_ego()

    def update_bev_data(
        self,
        proposals,
        processed_objs=None,
        cipo_status="SAFE",
        left_3d=None,
        right_3d=None,
        speed_mps=None,
        dt=1.0 / 30.0,
        alerts=None,
        threat=None,
    ):
        self._threat = threat
        if threat is not None:
            self._lane_states = (threat.lane_left, threat.lane_right)
            self._set("laneLeftState", threat.lane_left)
            self._set("laneRightState", threat.lane_right)
            self._set("gapLevel", int(threat.lead.level) if threat.lead is not None else 0)
            self._tag_seed = format_lead_tag(threat.lead, self._units)
        else:
            self._lane_states = ("normal", "normal")
            self._tag_seed = None
        self.proposals = proposals if proposals is not None else []
        self.processed_objs = processed_objs if processed_objs is not None else []
        self.cipo_status = cipo_status
        self.left_3d = left_3d
        self.right_3d = right_3d

        self._ego_speed_mps = None if speed_mps is None else float(speed_mps)
        alerts = alerts or {}
        world = alerts.get("lane_world")
        if world is None:
            lines = [l for l in (left_3d, right_3d) if l is not None]
            world = self._own_world.update(lines, dt=dt, speed_mps=speed_mps)
        self.lane_frame.set_world(world, speed_mps=speed_mps, dt=dt)
        self._sync_world(world)
        kappa = (2.0 * float(self.lane_frame.curvature)) if self.lane_frame.valid else None
        self._ego_pose.update(dt, speed_mps, kappa=kappa)
        if not self.lane_frame.valid:
            self._slots.reset()
        slots, offsets = self._active_slots(self.proposals)

        self._push_ego_pose()
        self._push_cipo(self.processed_objs, cipo_status)
        self._set("ldwSide", str(alerts.get("ldw_side") or alerts.get("ldw") or ""))
        self._set("fcwLevel", str(alerts.get("fcw") or "OFF"))

        rows = self._traffic_payload(self.processed_objs)
        self._traffic_seed = rows
        self._traffic_t0 = time.perf_counter()
        self._publish_traffic()
        corr = json.dumps(self._corridor_payload(), separators=(",", ":"))
        if corr != self._last_corridor_json:
            self._last_corridor_json = corr
            self._set("corridorJson", corr)
        lanes = json.dumps(self._lane_payload(slots, offsets), separators=(",", ":"))
        if lanes != self._last_lane_json:
            self._last_lane_json = lanes
            self._set("laneJson", lanes)
        dashes = json.dumps(self._dash_payload(), separators=(",", ":"))
        if dashes != self._last_dash_json:
            self._last_dash_json = dashes
            self._set("dashJson", dashes)
        center = json.dumps(self._center_payload(), separators=(",", ":"))
        if center != self._last_center_json:
            self._last_center_json = center
            self._set("centerJson", center)
        edges = json.dumps(self._edge_payload(slots, offsets), separators=(",", ":"))
        if edges != self._last_edge_json:
            self._last_edge_json = edges
            self._set("edgeJson", edges)
        mode, chev = self._chevron_payload()
        self._set("chevronMode", int(mode))
        chev_json = json.dumps(chev, separators=(",", ":"))
        if chev_json != self._last_chevron_json:
            self._last_chevron_json = chev_json
            self._set("chevronJson", chev_json)
        self._refresh_mini_bev()

    def reset_view(self):
        self.set_dashboard_camera("chase")

    def set_dashboard_camera(self, mode):
        """Chase camera for ADAS View, steeper pitch for Bird's-Eye."""
        bird = str(mode) == "bird"
        self._push_camera(
            pitch=62.0 if bird else DEFAULT_VIEW_PITCH,
            yaw=DEFAULT_VIEW_YAW,
            zoom=0.82 if bird else DEFAULT_ZOOM,
            calib_pitch=DEFAULT_CALIB_PITCH,
            calib_h=DEFAULT_CALIB_H,
            pan_x=0.0,
            pan_y=0.0,
        )


class BevQuick3DWidget(_BevDriver, QQuickWidget):
    """Standalone BEV viewport (classic cockpit, ``--ui classic``)."""

    def __init__(self, parent=None):
        QQuickWidget.__init__(self, parent)
        self.setMinimumSize(320, 240)
        self.setResizeMode(QQuickWidget.SizeRootObjectToView)
        # Transparent so the QML 2D sky gradient behind View3D is visible.
        self.setClearColor(Qt.transparent)
        self.setAttribute(Qt.WA_AlwaysStackOnTop, False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setSource(QUrl.fromLocalFile(os.path.abspath(_QML_PATH)))
        if self.status() == QQuickWidget.Error:
            msgs = "; ".join(err.toString() for err in self.errors())
            raise RuntimeError(f"Failed to load BevScene.qml: {msgs}")
        self._driver_init()

    def _root(self):
        return self.rootObject()


# Cluster chase camera: low and close behind the ego, looking down the lane
# (spec UX-3D-01). Pitch/zoom are the BevScene camera rig's own parameters.
CLUSTER_VIEW_PITCH = 11.5
CLUSTER_ZOOM = 1.1
CLUSTER_CAM_H = 6.4      # rig units; camera height = camH / 1.6 m
CLUSTER_CAM_DIST = 13.5  # m behind the ego
CLUSTER_FOV_DEG = 34.0


class BevSceneDriver(_BevDriver, QObject):
    """Drives the BevScene embedded in Cluster.qml (objectName ``bevScene``)."""

    def __init__(self, scene_item, parent=None):
        QObject.__init__(self, parent)
        self._item = scene_item
        self._driver_init()
        self._set("clusterStyle", True)
        self._set("clusterChrome", True)
        self._set("interactive", False)
        self._set("scenicView", False)
        self.set_env_mode("night")
        self._set("camH", CLUSTER_CAM_H)
        self._set("camDist", CLUSTER_CAM_DIST)
        self._set("fovDeg", CLUSTER_FOV_DEG)
        self._push_camera(
            pitch=CLUSTER_VIEW_PITCH, yaw=0.0, zoom=CLUSTER_ZOOM,
            calib_pitch=0.0, calib_h=DEFAULT_CALIB_H, pan_x=0.0, pan_y=0.0,
        )

    def _root(self):
        return self._item

    def reset_view(self):
        self._push_camera(
            pitch=CLUSTER_VIEW_PITCH, yaw=0.0, zoom=CLUSTER_ZOOM,
            calib_pitch=0.0, calib_h=DEFAULT_CALIB_H, pan_x=0.0, pan_y=0.0,
        )


def create_bev_widget(backend: str = "quick3d", parent=None):
    """
    backend: 'quick3d' | 'painter'
    Falls back to QPainter BEVWidget if Quick 3D QML fails to load.
    """
    kind = (backend or "quick3d").strip().lower()
    if kind in ("quick3d", "qtquick3d", "3d"):
        try:
            widget = BevQuick3DWidget(parent)
            print("[BEV] Qt Quick 3D viewport (CIPO)")
            return widget
        except Exception as exc:
            print(f"[BEV] Quick 3D unavailable ({exc}); falling back to QPainter")
    from src.ui.bev_widget import BEVWidget
    print("[BEV] QPainter viewport")
    return BEVWidget(parent)
