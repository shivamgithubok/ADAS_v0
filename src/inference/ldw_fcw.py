"""Level-0 LDW / FCW. Warnings only — no steering or brake."""

from __future__ import annotations

import cv2
import numpy as np

from src.inference import lane_filter_config as cfg
from src.inference.postprocess import decode_lane_pixels
from src.utils.drivable_area import parse_lane_components

# LDW
LDW_LOOKAHEAD_M = 10.0
HALF_CAR_M = 0.90
LDW_ENTER_M = 0.35
LDW_EXIT_M = 0.52
LDW_HITS = 4
LDW_MIN_MPH = 25.0

# FCW (TTC)
FCW_TTC_S = 2.7
FCW_PLUS_TTC_S = 1.4
FCW_PLUS_RANGE_M = 8.0
FCW_MAX_RANGE_M = 50.0
FCW_MIN_CLOSE_MPS = 0.5
FCW_HITS = 2
FCW_MIN_MPH = 15.0
FCW_ID_HOLD = 2


def _x_at_y(lane, y_target):
    if lane is None:
        return None
    xs, ys, zs, vis = parse_lane_components(lane)
    if int(np.sum(vis)) < 2:
        return None
    order = np.argsort(ys[vis])
    y_valid = ys[vis][order]
    x_valid = xs[vis][order]
    y_t = float(np.clip(y_target, y_valid[0], y_valid[-1]))
    return float(np.interp(y_t, y_valid, x_valid))


def _mph(speed_mps):
    if speed_mps is None:
        return None
    return float(speed_mps) * 2.236936


class LaneDepartureWarning:
    def __init__(self):
        self.status = "OFF"
        self.side = None
        self.left_gap = None
        self.right_gap = None
        self.width_m = None
        self._hits = 0
        self._cand = None
        self._prev_left = None
        self._prev_right = None

    def snapshot(self):
        return {
            "status": self.status,
            "side": self.side,
            "left_gap": self.left_gap,
            "right_gap": self.right_gap,
            "width_m": self.width_m,
        }

    def _clear(self, status="OFF"):
        self.status = status
        self.side = None
        self._hits = 0
        self._cand = None
        return self.snapshot()

    def update(self, ego_left, ego_right, road_status, speed_mps):
        mph = _mph(speed_mps)
        ok_road = road_status == "CONFIRMED"
        if not ok_road or ego_left is None or ego_right is None:
            self.left_gap = self.right_gap = self.width_m = None
            return self._clear("OFF")
        if mph is None or mph < LDW_MIN_MPH:
            self.left_gap = self.right_gap = self.width_m = None
            return self._clear("OFF")

        xl = _x_at_y(ego_left, LDW_LOOKAHEAD_M)
        xr = _x_at_y(ego_right, LDW_LOOKAHEAD_M)
        if xl is None or xr is None or xr <= xl + 0.5:
            return self._clear("OFF")

        width = float(xr - xl)
        self.width_m = width
        wmin = float(getattr(cfg, "EGO_LANE_WIDTH_MIN_M", 2.8))
        wmax = float(getattr(cfg, "EGO_LANE_WIDTH_MAX_M", 4.8))
        if width < wmin or width > wmax:
            return self._clear("OFF")

        ego_x = float(getattr(cfg, "CAMERA_LATERAL_OFFSET_M", 0.0))
        left_gap = (ego_x - xl) - HALF_CAR_M
        right_gap = (xr - ego_x) - HALF_CAR_M
        self.left_gap = left_gap
        self.right_gap = right_gap

        if left_gap < 0.15 and right_gap < 0.15:
            return self._clear("OFF")

        toward_l = (
            self._prev_left is not None
            and (self._prev_left - left_gap) >= 0.012
        )
        toward_r = (
            self._prev_right is not None
            and (self._prev_right - right_gap) >= 0.012
        )
        self._prev_left = left_gap
        self._prev_right = right_gap

        cand = None
        if left_gap < LDW_ENTER_M and toward_l and left_gap <= right_gap - 0.08:
            cand = "LEFT"
        elif right_gap < LDW_ENTER_M and toward_r and right_gap <= left_gap - 0.08:
            cand = "RIGHT"

        if self.status in ("LEFT", "RIGHT"):
            gap = left_gap if self.status == "LEFT" else right_gap
            if gap > LDW_EXIT_M:
                return self._clear("WATCH" if min(left_gap, right_gap) < 0.7 else "OFF")
            self.side = self.status
            return self.snapshot()

        if cand is None:
            self._hits = 0
            self._cand = None
            self.status = "WATCH" if min(left_gap, right_gap) < 0.7 else "OFF"
            self.side = None
            return self.snapshot()

        if cand == self._cand:
            self._hits += 1
        else:
            self._cand = cand
            self._hits = 1
        if self._hits >= LDW_HITS:
            self.status = cand
            self.side = cand
        else:
            self.status = "WATCH"
            self.side = None
        return self.snapshot()


class ForwardCollisionWarning:
    def __init__(self):
        self.status = "OFF"
        self.ttc = None
        self.v_close = None
        self.range_m = None
        self._hits = 0
        self._cand = None
        self._prev_tid = None
        self._prev_z = None
        self._id_hold = 0

    def snapshot(self):
        return {
            "status": self.status,
            "ttc": self.ttc,
            "v_close": self.v_close,
            "range_m": self.range_m,
        }

    def _clear(self):
        self.status = "OFF"
        self.ttc = None
        self.v_close = None
        self._hits = 0
        self._cand = None
        return self.snapshot()

    def update(self, cipo_obj, cipo_status, speed_mps, dt=1.0 / 30.0):
        mph = _mph(speed_mps)
        if cipo_obj is None or not cipo_obj.get("in_path"):
            self._prev_tid = None
            self._prev_z = None
            self.range_m = None
            return self._clear()
        if cipo_status == "DEGRADED":
            return self._clear()
        # Unknown ego speed (no HUD JSON) must not disable FCW. Only skip when
        # we *know* the vehicle is below the moving threshold.
        if mph is not None and mph < FCW_MIN_MPH:
            return self._clear()

        z = float(cipo_obj.get("Z_3d", 99.0))
        self.range_m = z
        tid = int(cipo_obj.get("track_id", -1))
        if self._prev_tid is not None and tid > 0 and tid != self._prev_tid:
            self._id_hold = FCW_ID_HOLD
        self._prev_tid = tid if tid > 0 else self._prev_tid
        if self._id_hold > 0:
            self._id_hold -= 1
            self._prev_z = z
            return self._clear()

        vy = float(cipo_obj.get("vy", 0.0))
        v_close = -vy
        if dt and dt > 1e-3 and self._prev_z is not None:
            v_fd = (float(self._prev_z) - z) / float(dt)
            if v_fd > v_close:
                v_close = v_fd
        self._prev_z = z
        self.v_close = float(v_close)

        ttc = z / max(v_close, 1e-3) if v_close >= FCW_MIN_CLOSE_MPS else None
        self.ttc = float(ttc) if ttc is not None else None
        cand = None
        if z < FCW_PLUS_RANGE_M:
            cand = "FCW+"
        elif ttc is not None and ttc < FCW_PLUS_TTC_S:
            cand = "FCW+"
        elif ttc is not None and ttc < FCW_TTC_S:
            cand = "FCW"
        elif cipo_status == "DANGER" and z < 15.0 and v_close >= 0.15:
            cand = "FCW"
        elif z > FCW_MAX_RANGE_M or v_close < FCW_MIN_CLOSE_MPS:
            cand = None

        if cand is None:
            return self._clear()
        if cand == self._cand:
            self._hits += 1
        else:
            self._cand = cand
            self._hits = 1
        if self._hits >= FCW_HITS:
            self.status = cand
        else:
            self.status = "OFF"
        return self.snapshot()


OBJ_ADJ_RANGE_M = 12.0
OBJ_CROSS_RANGE_M = 8.0


def nearby_object_alert(objects, cipo_obj):
    """True when a close vehicle is beside us, not the FCW lead (CIPO).

    OBJ is the left-stack card under FCW. It fires on:
      • adjacent-lane vehicle (L±1) inside 12 m
      • out-of-path vehicle inside 8 m (cross / cut-in / parked-close)
    Ego-path CIPO is FCW's job, so that track is skipped here.
    """
    if not objects:
        return False
    cipo_tid = int(cipo_obj.get("track_id", -1)) if cipo_obj else -1
    for obj in objects:
        tid = int(obj.get("track_id", -1) or -1)
        if tid > 0 and tid == cipo_tid:
            continue
        z = float(obj.get("Z_3d", 99.0))
        if z <= 0.0:
            continue
        li = int(obj.get("lane_index", 99) or 99)
        if abs(li) == 1 and z < OBJ_ADJ_RANGE_M:
            return True
        if not obj.get("in_path") and z < OBJ_CROSS_RANGE_M:
            return True
    return False


class AdasWarningTracker:
    def __init__(self):
        self.ldw = LaneDepartureWarning()
        self.fcw = ForwardCollisionWarning()

    def update(self, ego_left, ego_right, road_status, cipo_obj, cipo_status, speed_mps, dt=1.0 / 30.0, objects=None):
        ldw = self.ldw.update(ego_left, ego_right, road_status, speed_mps)
        fcw = self.fcw.update(cipo_obj, cipo_status, speed_mps, dt=dt)
        obj_on = nearby_object_alert(objects, cipo_obj)
        prio = "none"
        if fcw["status"] == "FCW+":
            prio = "fcw+"
        elif fcw["status"] == "FCW":
            prio = "fcw"
        elif obj_on:
            prio = "obj"
        elif ldw["status"] in ("LEFT", "RIGHT"):
            prio = "ldw"
        return {
            "ldw": ldw["status"],
            "ldw_side": ldw["side"],
            "left_gap": ldw["left_gap"],
            "right_gap": ldw["right_gap"],
            "fcw": fcw["status"],
            "ttc": fcw["ttc"],
            "v_close": fcw["v_close"],
            "range_m": fcw["range_m"],
            "obj": obj_on,
            "priority": prio,
        }


def _lane_draw_pts(lane, P_matrix, frame_transform, w_img, h_img):
    if lane is None:
        return []
    pts = decode_lane_pixels(lane, P_matrix, flat_ground=False)
    model_pts = np.asarray([(u, v) for u, v in pts if 0 <= u < 480 and 0 <= v < 360])
    if model_pts.size == 0:
        return []
    if frame_transform is not None:
        target = frame_transform.model_to_source(model_pts)
        return [
            (int(round(u)), int(round(v)))
            for u, v in target
            if 0 <= u < w_img and 0 <= v < h_img
        ]
    sx, sy = w_img / 480.0, h_img / 360.0
    return [(int(u * sx), int(v * sy)) for u, v in model_pts]


def draw_adas_alerts(
    frame_bgr,
    alerts,
    ego_left=None,
    ego_right=None,
    P_matrix=None,
    frame_transform=None,
    cipo_obj=None,
):
    """Bottom-centre bar + departing lane paint. Does not cover LIMIT/EGO badges."""
    out = frame_bgr
    if not alerts:
        return out
    h, w = out.shape[:2]
    ldw = alerts.get("ldw")
    fcw = alerts.get("fcw")
    side = alerts.get("ldw_side")

    if ldw in ("LEFT", "RIGHT") and P_matrix is not None:
        lane = ego_left if ldw == "LEFT" else ego_right
        pts = _lane_draw_pts(lane, P_matrix, frame_transform, w, h)
        if len(pts) > 1:
            arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [arr], False, (0, 165, 255), 5, cv2.LINE_AA)

    if fcw == "FCW+" and cipo_obj is not None:
        x1, y1, x2, y2 = [int(v) for v in cipo_obj.get("bbox") or [0, 0, 0, 0]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)

    bar = None
    color = (0, 165, 255)
    if fcw == "FCW+":
        ttc = alerts.get("ttc")
        bar = "FCW+" if ttc is None else f"FCW+  {ttc:.1f}s"
        color = (0, 0, 220)
    elif fcw == "FCW":
        ttc = alerts.get("ttc")
        bar = "FCW" if ttc is None else f"FCW  {ttc:.1f}s"
        color = (0, 140, 255)
    elif ldw == "LEFT":
        bar = "LDW  <<"
    elif ldw == "RIGHT":
        bar = "LDW  >>"

    if bar:
        (tw, th), _ = cv2.getTextSize(bar, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
        x0 = (w - tw) // 2 - 18
        y0 = h - 62
        cv2.rectangle(out, (x0, y0), (x0 + tw + 36, y0 + 44), (16, 16, 20), -1)
        cv2.rectangle(out, (x0, y0), (x0 + tw + 36, y0 + 44), color, 2)
        cv2.putText(out, bar, (x0 + 18, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)
    return out
