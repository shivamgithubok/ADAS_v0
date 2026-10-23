import numpy as np
from src.utils.drivable_area import find_ego_lanes, parse_lane_components, STANDARD_LANE_WIDTH, pair_gap_m
from src.inference.postprocess import decode_lane_pixels
from src.utils.calibration import P_final
from src.inference import lane_filter_config as cfg
from src.utils.ground_calib import GroundCalibration
from src.utils.ground_gate import GateResult, measure_ground
from src.tracking.bev_kalman import BevTracker

# Single source of truth: cam_height=1.5m, pitch=-3°, then crop/resize to 480x360
# (same as src.utils.calibration.P_final — do not hardcode a divergent copy)
DEFAULT_P_MATRIX = np.asarray(P_final, dtype=np.float64)

ANCHOR_LEN = 20
ANCHOR_Y_STEPS = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100], dtype=np.float64)

# Constant-velocity coast when YOLO misses. ~0.5–1 s at 15 FPS.
COAST_FRAMES = 15
KILL_FRAMES = 22
OCCLUDE_IOU = 0.22
# Re-init if ByteTrack reused an ID on a different car.
REINIT_Z_M = 18.0
GATE_Z_M = 10.0
GATE_X_M = 4.5
Q_ACC_Z = 3.0
Q_ACC_X = 2.0
R_Z = 9.0
R_X = 2.25
VZ_MIN, VZ_MAX = -25.0, 12.0
VX_MIN, VX_MAX = -6.0, 6.0

_VEHICLE_WIDTH_M = {
    "car": 1.85,
    "motorcycle": 0.80,
    "bike": 0.80,
    "bus": 2.55,
    "truck": 2.55,
}

_STATUS_COLOR = {
    "DANGER": (0, 0, 255),
    "WARNING": (0, 165, 255),
    "SAFE": (0, 215, 255),
}


def _kf_predict(p, v, P00, P01, P11, dt, q_acc):
    dt = float(dt)
    p = p + v * dt
    P00n = P00 + 2.0 * dt * P01 + dt * dt * P11
    P01n = P01 + dt * P11
    P11n = P11
    q = q_acc * q_acc
    dt2 = dt * dt
    P00n += q * dt2 * dt2 / 4.0
    P01n += q * dt2 * dt / 2.0
    P11n += q * dt2
    return p, v, P00n, P01n, P11n


def _kf_update(p, v, P00, P01, P11, meas, R, gate):
    innov = float(meas) - p
    S = P00 + R
    if S <= 1e-9 or abs(innov) > gate:
        return p, v, P00, P01, P11, False
    k0 = P00 / S
    k1 = P01 / S
    p = p + k0 * innov
    v = v + k1 * innov
    P00n = (1.0 - k0) * P00
    P01n = (1.0 - k0) * P01
    P11n = P11 - k1 * P01
    return p, v, P00n, P01n, P11n, True


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter))


def _interval_overlap(a0, a1, b0, b1):
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    return max(0.0, hi - lo)


def _cfg(name, default):
    return getattr(cfg, name, default)


def _new_cv_state(x, z, track_id):
    return {
        "x": float(x),
        "z": float(z),
        "vx": 0.0,
        "vz": 0.0,
        "xp00": 25.0,
        "xp01": 0.0,
        "xp11": 40.0,
        "zp00": 25.0,
        "zp01": 0.0,
        "zp11": 40.0,
        "frame": 0,
        "miss": 0,
        "track_id": int(track_id),
        "bbox": [0, 0, 1, 1],
        "label": "car",
        "conf": 1.0,
        "u_model": 240.0,
        "v_model": 360.0,
        "in_path": False,
        "path_score": 0.0,
        "lane_rank": 1,
        "color": (255, 220, 0),
        "status": "OUT OF PATH",
        "range_gate": "ok",
    }


class CIPOTracker:
    def __init__(
        self,
        P_matrix=DEFAULT_P_MATRIX,
        danger_dist=15.0,
        warning_dist=30.0,
        ema_alpha=0.35,
        ground_calib=None,
    ):
        self.P = P_matrix
        self.danger_dist = danger_dist  # HUD / legacy mid-band
        self.warning_dist = warning_dist
        self.ema_alpha = ema_alpha       # unused; kept for caller compatibility
        self.ground_calib = ground_calib
        self._video_path = None
        self.bev = BevTracker()
        self.track_history = {}          # tid -> last packed extras (compat)
        self._hist_frame = 0
        self._hist_ttl = KILL_FRAMES
        self._inpath_state = {}         # tid -> {in_path, hits, miss, score}
        self._lane_last = {}            # tid -> (lane_index, offset_m, frame)
        self._cipo_tid = None
        self._status_band = "SAFE"
        self.last_cipo_status = "SAFE"

    def project_2d_to_3d_ground(self, u, v):
        """Legacy OpenLane-P ranger. Do not use for metric objects.

        Kept only so overlays that still project *lane* geometry through P stay
        consistent. Object range/lateral go through `ground_calib`.
        Invalid rays return None instead of clamping to 1.0 m.
        """
        P = self.P
        denom_y = (P[2, 1] * v - P[1, 1])
        if abs(denom_y) < 1e-4:
            return None
        Y = float(P[1, 3] / denom_y)
        if not np.isfinite(Y) or Y <= 0.0 or Y > 100.0:
            return None
        denom_x = P[0, 0]
        if abs(denom_x) < 1e-6:
            return None
        X = float((Y * (P[2, 1] * u - P[0, 1])) / denom_x)
        return X, Y

    def _ensure_calib_shape(self, h_img, w_img):
        calib = self.ground_calib
        if calib is None:
            self.ground_calib = GroundCalibration.for_video(self._video_path, (h_img, w_img))
            return
        if calib.source_width == w_img and calib.source_height == h_img:
            return
        self.ground_calib = calib.adapted_to(w_img, h_img)

    def _range_detection(self, bbox, label, frame_shape, u_model, v_model):
        """Metric (x, y) from bbox bottom-centre, or None if gated out."""
        if self.ground_calib is not None:
            gm = measure_ground(
                bbox, self.ground_calib, label=label, frame_shape=frame_shape
            )
            if not gm.valid:
                return None
            return gm.x_m, gm.y_m, gm.result.value
        hit = self.project_2d_to_3d_ground(u_model, v_model)
        if hit is None:
            return None
        return hit[0], hit[1], GateResult.OK.value

    def get_2d_lane_u_at_v(self, lane_proposal, v_target, slack_px=0.0, flat_ground=False):
        """
        Calculates the 2D projected u-pixel coordinate of a lane line at a specific v-pixel row.
        """
        if lane_proposal is None:
            return None
        pts_2d = decode_lane_pixels(lane_proposal, self.P, flat_ground=flat_ground)
        if len(pts_2d) < 2:
            return None

        us = [p[0] for p in pts_2d]
        vs = [p[1] for p in pts_2d]

        order = np.argsort(vs)
        vs = np.array(vs)[order]
        us = np.array(us)[order]

        slack = float(slack_px)
        if v_target < vs[0] - slack or v_target > vs[-1] + slack:
            return None

        u_interp = float(np.interp(np.clip(v_target, vs[0], vs[-1]), vs, us))
        return u_interp

    def get_lane_x_at_y(self, lane_proposal, y_target):
        """Interpolate 3D lateral X of a lane line at forward distance y_target (meters)."""
        if lane_proposal is None:
            return None
        xs, ys, zs, vis = parse_lane_components(lane_proposal, ANCHOR_LEN)
        if vis.sum() < 2:
            return None
        order = np.argsort(ys[vis])
        y_valid = ys[vis][order]
        x_valid = xs[vis][order]
        y_t = float(np.clip(y_target, y_valid[0], y_valid[-1]))
        return float(np.interp(y_t, y_valid, x_valid))

    def _ego_u_pair(self, v_model, ego_left, ego_right, left_3d=None, right_3d=None, slack_px=24.0):
        """Image u of the ego paint at this row. Prefer the smoothed corridor
        the overlay draws, on the ground plane (same as the box bottom)."""
        src_l = left_3d if left_3d is not None else ego_left
        src_r = right_3d if right_3d is not None else ego_right
        u_left = self.get_2d_lane_u_at_v(
            src_l, v_model, slack_px=slack_px, flat_ground=True
        ) if src_l is not None else None
        u_right = self.get_2d_lane_u_at_v(
            src_r, v_model, slack_px=slack_px, flat_ground=True
        ) if src_r is not None else None
        if u_left is None or u_right is None:
            u_left = self.get_2d_lane_u_at_v(
                ego_left, v_model, slack_px=slack_px, flat_ground=True
            ) if ego_left is not None else u_left
            u_right = self.get_2d_lane_u_at_v(
                ego_right, v_model, slack_px=slack_px, flat_ground=True
            ) if ego_right is not None else u_right
        if u_left is not None and u_right is not None and u_right < u_left:
            u_left, u_right = u_right, u_left
        return u_left, u_right

    def _x_at_y_pts(self, pts, y_target):
        if pts is None:
            return None
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return None
        order = np.argsort(arr[:, 1])
        ys = arr[order, 1]
        xs = arr[order, 0]
        y_t = float(np.clip(y_target, ys[0], ys[-1]))
        return float(np.interp(y_t, ys, xs))

    def _lane_aware_x(self, u_model, v_model, y_fwd, x_geom, proposals):
        """
        Snap lateral X to a detected-lane center.

        Using only the ego pair collapses every adjacent car into one slot
        (they overlap on the BEV). All lane lines at this depth are used.
        """
        if proposals is None or len(proposals) == 0:
            return x_geom

        samples = []
        for lane in proposals:
            u = self.get_2d_lane_u_at_v(lane, v_model)
            x = self.get_lane_x_at_y(lane, y_fwd)
            if u is None and x is None:
                continue
            samples.append((u, x))

        xs = sorted(x for _, x in samples if x is not None)
        uniq = []
        for x in xs:
            if not uniq or abs(x - uniq[-1]) > 0.8:
                uniq.append(x)
        xs = uniq
        if not xs:
            return x_geom

        if len(xs) == 1:
            w = STANDARD_LANE_WIDTH
            centers = [xs[0] - w, xs[0], xs[0] + w]
        else:
            widths = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            med_w = float(np.median(widths))
            med_w = max(2.8, min(4.6, med_w))
            centers = [xs[0] - 0.5 * med_w]
            for i in range(len(xs) - 1):
                centers.append(0.5 * (xs[i] + xs[i + 1]))
            centers.append(xs[-1] + 0.5 * med_w)

        with_u = [(u, x) for u, x in samples if u is not None and x is not None]
        with_u.sort(key=lambda t: t[0])
        snapped = None
        if len(with_u) >= 2:
            us = [t[0] for t in with_u]
            xsu = [t[1] for t in with_u]
            if u_model <= us[0]:
                snapped = centers[0]
            elif u_model >= us[-1]:
                snapped = centers[-1]
            else:
                for i in range(len(us) - 1):
                    if us[i] <= u_model <= us[i + 1]:
                        snapped = 0.5 * (xsu[i] + xsu[i + 1])
                        break
        if snapped is None:
            snapped = min(centers, key=lambda c: abs(c - x_geom))

        # Small within-lane offset from geometry, without collapsing lanes
        return snapped + 0.12 * float(np.clip(x_geom - snapped, -1.5, 1.5))

    def _lane_from_image(
        self, u_model, v_model, x_3d, y_fwd, ego_left, ego_right,
        u1=None, u2=None, left_3d=None, right_3d=None,
    ):
        """Signed lane index from the *image*, plus a small within-lane offset.

        Object metric X and OpenLane lane X are different cameras. The BEV
        ribbon is the lane net; the box the user sees is YOLO. Match those
        two, not X_3d vs the corridor.
        0 = ego, −1 left, +1 right.
        """
        lane_w = STANDARD_LANE_WIDTH
        x_left = self.get_lane_x_at_y(ego_left, y_fwd)
        x_right = self.get_lane_x_at_y(ego_right, y_fwd)
        if x_left is not None and x_right is not None:
            lane_w = float(np.clip(abs(x_right - x_left), 2.8, 4.6))

        u_left, u_right = self._ego_u_pair(
            v_model, ego_left, ego_right, left_3d, right_3d, slack_px=24.0
        )
        if u_left is not None and u_right is not None:
            if u_right < u_left:
                u_left, u_right = u_right, u_left
            width_px = max(18.0, float(u_right - u_left))
            pad = 0.22 * width_px
            # YOLO boxes sit a bit left of centre; paint u is noisy. Stay ego
            # unless the box is clearly outside the pair.
            overlap = 0.0
            if u1 is not None and u2 is not None and u2 > u1:
                overlap = _interval_overlap(u1, u2, u_left, u_right) / (u2 - u1)
            if overlap >= 0.45 or (u_left - pad <= u_model <= u_right + pad):
                idx = 0
                frac = (u_model - u_left) / width_px
            elif u_model < u_left - pad:
                n = int(np.floor((u_left - u_model) / width_px + 0.5))
                idx = -int(np.clip(max(1, n), 1, 2))
                lane_l = u_left + idx * width_px
                frac = (u_model - lane_l) / width_px
            else:
                n = int(np.floor((u_model - u_right) / width_px + 0.5))
                idx = int(np.clip(max(1, n), 1, 2))
                lane_l = u_right + (idx - 1) * width_px
                frac = (u_model - lane_l) / width_px
            offset = float(np.clip((frac - 0.5) * lane_w, -1.15, 1.15))
            return idx, offset

        # No ego paint at this v. Do not guess from metric X (it collapses
        # at range). Caller keeps the last image index if it has one.
        return None, 0.0

    def _lane_rank(self, u_model, v_model, x_3d, y_fwd, ego_left, ego_right):
        idx, _ = self._lane_from_image(
            u_model, v_model, x_3d, y_fwd, ego_left, ego_right
        )
        if idx is None:
            return 99
        return abs(int(idx))

    def _predict_cv(self, st, dt):
        dt = float(np.clip(dt, 0.01, 0.12))
        x, vx, xp00, xp01, xp11 = _kf_predict(
            st["x"], st["vx"], st["xp00"], st["xp01"], st["xp11"], dt, Q_ACC_X
        )
        z, vz, zp00, zp01, zp11 = _kf_predict(
            st["z"], st["vz"], st["zp00"], st["zp01"], st["zp11"], dt, Q_ACC_Z
        )
        st["x"] = x
        st["vx"] = float(np.clip(vx, VX_MIN, VX_MAX))
        st["xp00"] = max(1e-4, xp00)
        st["xp01"] = xp01
        st["xp11"] = max(1e-4, xp11)
        st["z"] = float(np.clip(z, 1.0, 90.0))
        st["vz"] = float(np.clip(vz, VZ_MIN, VZ_MAX))
        st["zp00"] = max(1e-4, zp00)
        st["zp01"] = zp01
        st["zp11"] = max(1e-4, zp11)

    def _update_cv(self, st, x_meas, z_meas):
        gate_z = max(GATE_Z_M, 0.20 * abs(st["z"]))
        x, vx, xp00, xp01, xp11, _ = _kf_update(
            st["x"], st["vx"], st["xp00"], st["xp01"], st["xp11"], x_meas, R_X, GATE_X_M
        )
        z, vz, zp00, zp01, zp11, _ = _kf_update(
            st["z"], st["vz"], st["zp00"], st["zp01"], st["zp11"], z_meas, R_Z, gate_z
        )
        st["x"] = x
        st["vx"] = float(np.clip(vx, VX_MIN, VX_MAX))
        st["xp00"] = max(1e-4, xp00)
        st["xp01"] = xp01
        st["xp11"] = max(1e-4, xp11)
        st["z"] = float(np.clip(z, 1.0, 90.0))
        st["vz"] = float(np.clip(vz, VZ_MIN, VZ_MAX))
        st["zp00"] = max(1e-4, zp00)
        st["zp01"] = zp01
        st["zp11"] = max(1e-4, zp11)

    def _vehicle_half_width(self, label, u1, u2, y_fwd, v_src=None):
        cls_w = _VEHICLE_WIDTH_M.get(str(label).lower(), 1.85)
        if u2 > u1 + 1.0 and y_fwd > 1.0:
            if self.ground_calib is not None:
                w = abs(u2 - u1) * float(y_fwd) / max(1.0, self.ground_calib.f_px)
            elif v_src is not None:
                hit_l = self.project_2d_to_3d_ground(u1, v_src)
                hit_r = self.project_2d_to_3d_ground(u2, v_src)
                w = abs(hit_r[0] - hit_l[0]) if hit_l and hit_r else None
            else:
                w = None
            if w is not None and 0.6 <= w <= 3.6:
                cls_w = 0.55 * w + 0.45 * cls_w
        return 0.5 * float(np.clip(cls_w, 0.6, 3.2))

    def _corridor_xs(self, y_fwd, ego_left, ego_right, left_3d, right_3d):
        xl = self._x_at_y_pts(left_3d, y_fwd)
        xr = self._x_at_y_pts(right_3d, y_fwd)
        if xl is None:
            xl = self.get_lane_x_at_y(ego_left, y_fwd)
        if xr is None:
            xr = self.get_lane_x_at_y(ego_right, y_fwd)
        if xl is None or xr is None:
            return None, None
        if xr < xl:
            xl, xr = xr, xl
        return xl, xr

    def _range_margin(self, z):
        base = float(_cfg("CIPO_X_MARGIN_M", 0.40))
        per_z = float(_cfg("CIPO_X_MARGIN_PER_Z", 0.02))
        return base + per_z * max(0.0, float(z))

    def _score_3d(self, x, z, half_w, xl, xr):
        m = self._range_margin(z)
        overlap = _interval_overlap(x - half_w, x + half_w, xl - m, xr + m)
        return float(overlap / max(1e-3, 2.0 * half_w))

    def _score_2d(self, u1, u2, v_model, ego_left, ego_right, left_3d=None, right_3d=None):
        slack = float(_cfg("CIPO_U_MARGIN_PX", 8.0))
        u_left, u_right = self._ego_u_pair(
            v_model, ego_left, ego_right, left_3d, right_3d, slack_px=20.0
        )
        if u_left is None or u_right is None:
            return None
        width_px = max(18.0, float(u_right - u_left))
        pad = max(slack, 0.22 * width_px)
        # Adjacent wide boxes overlap the ego ribbon; the box *centre* must
        # sit in the paint span (with paint-noise pad) or this is not occupancy.
        uc = 0.5 * (float(u1) + float(u2))
        if uc < (u_left - pad) or uc > (u_right + pad):
            overlap = _interval_overlap(u1, u2, u_left, u_right) / max(1.0, u2 - u1)
            if overlap < 0.45:
                return 0.0
        overlap = _interval_overlap(u1, u2, u_left - slack, u_right + slack)
        return float(overlap / max(1.0, u2 - u1))

    def _occupancy_score(
        self,
        x,
        z,
        half_w,
        u1,
        u2,
        v_model,
        ego_left,
        ego_right,
        left_3d,
        right_3d,
    ):
        # Prefer image-space occupancy: 3D corridor X is OpenLane-frame and
        # must not be mixed with Garmin-metric object X after P0.
        score_2d = self._score_2d(
            u1, u2, v_model, ego_left, ego_right, left_3d, right_3d
        )
        if score_2d is not None:
            return score_2d
        # Garmin: paint missing at this v. Collapsed X is not occupancy.
        if self.ground_calib is not None:
            return None
        xl, xr = self._corridor_xs(z, ego_left, ego_right, left_3d, right_3d)
        if xl is not None and xr is not None:
            return self._score_3d(x, z, half_w, xl, xr)
        half = 0.5 * STANDARD_LANE_WIDTH
        if ego_left is not None and ego_right is not None:
            gap = pair_gap_m(ego_left, ego_right, ANCHOR_LEN)
            if gap is not None:
                half = 0.5 * float(gap)
        return self._score_3d(x, z, half_w, -half, half)

    def _in_path_from_score(self, state_key, score):
        st = self._inpath_state.get(state_key)
        if st is None:
            st = {"in_path": False, "hits": 0, "miss": 0, "score": 0.0}
        enter_high = float(_cfg("CIPO_SCORE_ENTER_HIGH", 0.60))
        enter = float(_cfg("CIPO_SCORE_ENTER", 0.45))
        hold = float(_cfg("CIPO_SCORE_HOLD", 0.25))
        enter_hits = int(_cfg("CIPO_ENTER_HITS", 2))
        exit_miss = int(_cfg("CIPO_EXIT_MISS", 8))

        if score is None:
            if st["in_path"]:
                st["miss"] += 1
                if st["miss"] >= exit_miss:
                    st["in_path"] = False
                    st["hits"] = 0
            else:
                st["hits"] = 0
                st["miss"] += 1
        else:
            st["score"] = float(score)
            if st["in_path"]:
                if score >= hold:
                    st["miss"] = 0
                    st["hits"] += 1
                else:
                    st["miss"] += 1
                    st["hits"] = 0
                    if st["miss"] >= exit_miss:
                        st["in_path"] = False
            elif score >= enter_high:
                st["hits"] += 1
                st["miss"] = 0
                st["in_path"] = True
            elif score >= enter:
                st["hits"] += 1
                st["miss"] = 0
                if st["hits"] >= enter_hits:
                    st["in_path"] = True
            else:
                st["hits"] = 0
                st["miss"] += 1

        st["last_frame"] = self._hist_frame
        self._inpath_state[state_key] = st
        return bool(st["in_path"]), float(st.get("score", 0.0))

    def _adopt_inpath(self, src_key, dst_key):
        if src_key == dst_key or dst_key in self._inpath_state:
            return
        src = self._inpath_state.get(src_key)
        if src is None:
            return
        self._inpath_state[dst_key] = dict(src)

    def _style_for(self, in_path, z, lane_index=99):
        """Box color follows CIPO bands for ego-lane / in-path cars."""
        ego = bool(in_path) or int(lane_index) == 0
        if not ego:
            return "OUT OF PATH", (255, 220, 0)
        if z < float(_cfg("CIPO_DANGER_ENTER_M", 14.0)):
            return "DANGER", (0, 0, 255)
        if z < float(_cfg("CIPO_WARN_ENTER_M", 28.0)):
            return "WARNING", (0, 165, 255)
        return "IN PATH", (0, 215, 255)

    def _pack_obj(self, st, in_path, path_score=0.0, quality="confirmed"):
        x, z = float(st["x"]), float(st["z"])
        status, color = self._style_for(
            in_path, z, lane_index=int(st.get("lane_index", 99))
        )
        return {
            "bbox": list(st["bbox"]),
            "label": st["label"],
            "track_id": st["track_id"],
            "conf": st["conf"],
            "X_3d": x,
            "Z_3d": z,
            "Y_ground": 0.0,
            "range_gate": st.get("range_gate", "ok"),
            "in_path": in_path,
            "path_score": float(path_score),
            "status": status,
            "color": color,
            "is_cipo": False,
            "cipo_quality": quality,
            "lane_rank": int(st.get("lane_rank", 1)),
            "lane_index": int(st.get("lane_index", 0)),
            "lane_offset_m": float(st.get("lane_offset_m", 0.0)),
            "vx": float(st.get("vx", 0.0)),
            "vy": float(st.get("vy", 0.0)),
            "show_bev": True,
        }

    def _remap_occluded_id(self, track_id, bbox, seen_ids):
        """Keep the coasting track when ByteTrack issues a new id after overlap."""
        if track_id > 0 and track_id in self.track_history:
            return track_id
        best_id, best_iou = None, OCCLUDE_IOU
        for tid, st in self.track_history.items():
            if tid in seen_ids:
                continue
            iou = _bbox_iou(bbox, st.get("bbox") or [0, 0, 0, 0])
            if iou > best_iou:
                best_iou, best_id = iou, tid
        return int(best_id) if best_id is not None else track_id

    def _select_cipo(self, processed_objects):
        ego_objs = [
            obj for obj in processed_objects
            if int(obj.get("lane_index", 99)) == 0
        ]
        in_path_objs = [obj for obj in ego_objs if obj.get("in_path")]
        candidates = in_path_objs or ego_objs
        if not candidates:
            self._cipo_tid = None
            return None
        closest = min(candidates, key=lambda obj: obj["Z_3d"])
        stick = float(_cfg("CIPO_STICK_MARGIN_M", 5.0))
        current = None
        if self._cipo_tid is not None:
            for obj in candidates:
                if int(obj.get("track_id", -1)) == int(self._cipo_tid):
                    current = obj
                    break
        if current is None:
            chosen = closest
        elif float(closest["Z_3d"]) + stick < float(current["Z_3d"]):
            chosen = closest
        else:
            chosen = current
        tid = int(chosen.get("track_id", -1))
        self._cipo_tid = tid if tid > 0 else None
        return chosen

    def _update_status_band(self, cipo_obj, road_status, has_corridor):
        if cipo_obj is None:
            self._status_band = "SAFE"
            if road_status == "CONFIRMED" or (road_status == "PREDICTED" and has_corridor):
                self.last_cipo_status = "SAFE"
            else:
                self.last_cipo_status = "DEGRADED"
            return self.last_cipo_status

        z = float(cipo_obj["Z_3d"])
        danger_in = float(_cfg("CIPO_DANGER_ENTER_M", 14.0))
        danger_out = float(_cfg("CIPO_DANGER_EXIT_M", 17.0))
        warn_in = float(_cfg("CIPO_WARN_ENTER_M", 28.0))
        warn_out = float(_cfg("CIPO_WARN_EXIT_M", 32.0))
        band = self._status_band or "SAFE"
        if band == "SAFE":
            if z < danger_in:
                band = "DANGER"
            elif z < warn_in:
                band = "WARNING"
        elif band == "WARNING":
            if z < danger_in:
                band = "DANGER"
            elif z > warn_out:
                band = "SAFE"
        elif band == "DANGER":
            if z > danger_out:
                band = "WARNING" if z < warn_out else "SAFE"
        else:
            band = "DANGER" if z < danger_in else ("WARNING" if z < warn_in else "SAFE")
        self._status_band = band
        self.last_cipo_status = band
        return band

    def _mark_cipo(self, processed_objects, cipo_obj, status, quality):
        cipo_tid = int(cipo_obj.get("track_id", -999)) if cipo_obj is not None else None
        color = _STATUS_COLOR.get(status, (0, 215, 255))
        for obj in processed_objects:
            is_sel = (
                cipo_obj is not None
                and int(obj.get("track_id", -1)) == cipo_tid
                and bool(obj.get("in_path"))
            )
            obj["is_cipo"] = is_sel
            if is_sel:
                obj["status"] = status
                obj["color"] = color
                obj["cipo_quality"] = quality
                cipo_obj["is_cipo"] = True
                cipo_obj["status"] = status
                cipo_obj["color"] = color
                cipo_obj["cipo_quality"] = quality

    def process_detections(
        self,
        detections,
        lane_proposals,
        frame_size=(1080, 720),
        depth_map=None,
        depth_estimator=None,
        ego_left=None,
        ego_right=None,
        frame_transform=None,
        road_state_confirmed=True,
        road_status=None,
        left_corridor_3d=None,
        right_corridor_3d=None,
        dt=1.0 / 15.0,
        ego_speed_mps=None,
    ):
        """
        Range is gated ground-plane from the box bottom in *source* pixels.
        BEV Kalman [x,y,vx,vy] with R(d²/hf). OpenLane P stays for lanes.
        """
        processed_objects = []
        w_img, h_img = frame_size
        scale_u = 480.0 / float(w_img)
        scale_v = 360.0 / float(h_img)
        dt = float(np.clip(dt, 0.01, 0.25))
        self._ensure_calib_shape(h_img, w_img)
        calib = self.ground_calib

        if road_status is None:
            road_status = "CONFIRMED" if road_state_confirmed else "UNKNOWN"
        has_pair = ego_left is not None or ego_right is not None
        has_corr = left_corridor_3d is not None and right_corridor_3d is not None
        if (not has_pair) and (not has_corr) and road_status != "UNKNOWN":
            ego_left, ego_right = find_ego_lanes(lane_proposals, ANCHOR_LEN)
            has_pair = ego_left is not None or ego_right is not None
        has_corridor = has_pair or has_corr
        quality = "confirmed" if road_status == "CONFIRMED" else "probable"

        self._hist_frame += 1
        measurements = []
        extras = {}

        for det in detections or []:
            x1, y1, x2, y2 = det["bbox"]
            raw_tid = int(det.get("track_id", -1))
            u_img = (x1 + x2) / 2.0
            v_img = float(y2)

            if frame_transform is not None:
                uv = frame_transform.source_to_model(
                    np.array([[u_img, v_img], [x1, y2], [x2, y2]], dtype=np.float64)
                )
                u_model, v_model = uv[0]
                u1_model, u2_model = float(uv[1, 0]), float(uv[2, 0])
            else:
                u_model = u_img * scale_u
                v_model = v_img * scale_v
                u1_model, u2_model = float(x1) * scale_u, float(x2) * scale_u

            ranged = self._range_detection(
                bbox=[x1, y1, x2, y2],
                label=det.get("class", "car"),
                frame_shape=(h_img, w_img),
                u_model=u_model,
                v_model=v_model,
            )
            if ranged is None:
                continue
            X_meas, Y_meas, gate_name = ranged
            if calib is not None:
                R = calib.measurement_cov(X_meas, Y_meas)
            else:
                R = np.diag([2.25, 9.0])
            measurements.append({
                "x": X_meas,
                "y": Y_meas,
                "R": R,
                "label": det.get("class", "car"),
                "conf": det.get("conf", 1.0),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_id": raw_tid,
                "u_model": float(u_model),
                "v_model": float(v_model),
                "gate": gate_name,
                "u1_model": u1_model,
                "u2_model": u2_model,
            })
            extras[raw_tid] = (u1_model, u2_model, v_model)

        self.bev.update(measurements, dt, ego_speed_mps=ego_speed_mps)

        for tr in self.bev.renderable():
            x, z = float(tr.x[0]), float(tr.x[1])
            bbox = tr.bbox or [0, 0, 1, 1]
            u1_model, u2_model, v_model = extras.get(
                tr.track_id,
                (tr.u_model - 8.0, tr.u_model + 8.0, tr.v_model),
            )
            half_w = self._vehicle_half_width(
                tr.label, float(bbox[0]), float(bbox[2]), z
            )
            if has_corridor:
                score = self._occupancy_score(
                    x, z, half_w, u1_model, u2_model, v_model,
                    ego_left, ego_right, left_corridor_3d, right_corridor_3d,
                )
            else:
                score = None
            in_path, path_score = self._in_path_from_score(tr.track_id, score)
            lane_index, lane_offset = self._lane_from_image(
                tr.u_model, tr.v_model, x, z, ego_left, ego_right,
                u1_model, u2_model, left_corridor_3d, right_corridor_3d,
            )
            if lane_index is None:
                prev = self._lane_last.get(tr.track_id)
                if prev is not None:
                    lane_index, lane_offset = int(prev[0]), float(prev[1])
            if lane_index is None:
                lane_rank = 99
                packed_index = 99
                in_path_lane = False
            else:
                packed_index = int(lane_index)
                lane_rank = abs(packed_index)
                in_path_lane = packed_index == 0
                self._lane_last[tr.track_id] = (
                    packed_index, float(lane_offset), self._hist_frame,
                )
            # Occupancy may overlap a neighbour; CIPO only if the photo says ego.
            tr.in_path = bool(in_path) and in_path_lane
            # No paint / weak occupancy still leaves a centered ego-lane car
            # as in-path so the box follows CIPO (red / amber / cyan).
            if in_path_lane and not tr.in_path and (score is None or abs(x) <= 2.2):
                tr.in_path = True
                if path_score <= 0.0:
                    path_score = 0.45
            tr.path_score = path_score
            tr.lane_rank = lane_rank
            tr.lane_index = packed_index if packed_index != 99 else None
            st = {
                "x": x,
                "z": z,
                "vx": float(tr.x[2]),
                "vy": float(tr.x[3]),
                "bbox": list(bbox),
                "label": tr.label,
                "track_id": tr.track_id,
                "conf": tr.conf,
                "v_model": tr.v_model,
                "lane_rank": lane_rank,
                "lane_index": packed_index,
                "lane_offset_m": float(np.clip(lane_offset, -1.15, 1.15)),
                "range_gate": tr.range_gate,
            }
            processed_objects.append(self._pack_obj(st, tr.in_path, path_score, quality))

        self._inpath_state = {
            k: v for k, v in self._inpath_state.items()
            if int(v.get("last_frame", 0)) >= self._hist_frame - self._hist_ttl
        }
        self._lane_last = {
            k: v for k, v in self._lane_last.items()
            if int(v[2]) >= self._hist_frame - self._hist_ttl
        }

        cipo_obj = self._select_cipo(processed_objects)
        if cipo_obj is not None:
            cipo_obj["in_path"] = True
            for obj in processed_objects:
                if int(obj.get("track_id", -1)) == int(cipo_obj.get("track_id", -2)):
                    obj["in_path"] = True
                    break
        status = self._update_status_band(cipo_obj, road_status, has_corridor)
        self._mark_cipo(processed_objects, cipo_obj, status, quality)
        return processed_objects, cipo_obj
