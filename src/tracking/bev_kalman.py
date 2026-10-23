"""BEV object tracking: constant-velocity Kalman in the road plane.

Replaces the two scalar filters in `cipo_tracker._predict_cv` / `_update_cv`.
Those track x and z independently with *fixed* R (R_Z = 9.0 m^2, R_X = 2.25 m^2)
at every distance.  That is the wrong shape of noise for a monocular ranger: the
true range variance follows the d^2/h law,

    sigma_y = y^2 / (h*f) * sigma_v

which on this camera (h*f = 873) is 0.26 m at 15 m and 5.6 m at 70 m -- a factor
of 460 in variance.  A constant R therefore over-trusts far measurements (the
frame-to-frame jitter you see at range) and under-trusts near ones (the sluggish
response to a car cutting in).  `GroundCalibration.measurement_cov` supplies the
correct per-measurement R and this filter consumes it.

Differences from the code being replaced, all deliberate:

  * State is [x, y, vx, vy] with a shared 4x4 covariance, so the filter can
    represent the x/y correlation that ranging error actually induces (a row
    error moves x and y together, by x/y and 1 respectively).
  * `predict` takes ego motion. A static car at 40 m closes at the ego speed;
    without compensation the filter has to learn that as target velocity and
    lags every real manoeuvre. Speed comes from the same HUD OCR log the lane
    frame already uses.
  * Lifecycle is confirm-after-N-hits then coast-before-delete, so a single
    spurious detection never reaches the renderer and a brief occlusion does not
    delete a track.
  * Gating is chi-square on the innovation using S, not a fixed metre threshold.
    A 3-sigma gate at 15 m is ~1 m; at 70 m it is ~17 m. The old fixed
    `GATE_Z_M = 10.0` was simultaneously too loose near and far too tight far.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Process noise as an acceleration 1-sigma, m/s^2. Highway traffic rarely
# exceeds 3 m/s^2 longitudinally; lateral lane changes are slower than that.
Q_ACC_LON = 2.5
Q_ACC_LAT = 1.2

# Chi-square 2-DOF gate. 9.21 = 99%, 5.99 = 95%.
CHI2_GATE_2DOF = 16.0

# Lifecycle
CONFIRM_HITS = 2
COAST_FRAMES = 12          # ~0.8 s at 15 FPS: rides out an overtake occlusion
DELETE_FRAMES = 20

# Physical clamps. Closing faster than this is a detection error, not a car.
VY_MIN, VY_MAX = -45.0, 25.0
VX_MIN, VX_MAX = -8.0, 8.0
# Twin YOLO boxes that still got two ByteTrack IDs: same vehicle in BEV.
DEDUP_X_M = 1.8
DEDUP_Y_M = 3.2
DEDUP_BOX_IOU = 0.40
# ByteTrack id reuse (new car, old id). Not a χ² reject of the same car.
REINIT_Y_M = 18.0
REINIT_X_M = 3.0

# Initial uncertainty. Velocity is genuinely unknown on the first frame, so it
# must start large or the filter will not accept the second measurement.
P0_POS = 4.0
P0_VEL = 100.0

# Render-time extrapolation is clamped: a stale track must not slide across the
# BEV on the strength of a velocity estimate that is no longer being corrected.
MAX_EXTRAPOLATION_S = 0.20


class BevTrack:
    """One tracked object in the road plane. State [x, y, vx, vy], metres."""

    __slots__ = ("track_id", "x", "P", "hits", "misses", "age", "confirmed",
                 "label", "conf", "bbox", "last_meas_t", "last_result",
                 "lane_index", "in_path", "sigma_y",
                 "u_model", "v_model", "range_gate", "path_score", "lane_rank",
                 "reject_streak")

    def __init__(self, track_id: int, x_m: float, y_m: float,
                 R: np.ndarray, label: str = "car", conf: float = 1.0):
        self.track_id = int(track_id)
        self.x = np.array([float(x_m), float(y_m), 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([
            max(float(R[0, 0]), P0_POS),
            max(float(R[1, 1]), P0_POS),
            P0_VEL,
            P0_VEL,
        ]).astype(np.float64)
        self.hits = 1
        self.misses = 0
        self.age = 1
        self.confirmed = False
        self.label = str(label)
        self.conf = float(conf)
        self.bbox: Optional[List[float]] = None
        self.last_meas_t = 0.0
        self.last_result = "ok"
        self.lane_index: Optional[int] = None
        self.in_path = False
        self.sigma_y = float(math.sqrt(max(1e-6, float(R[1, 1]))))
        self.u_model = 240.0
        self.v_model = 360.0
        self.range_gate = "ok"
        self.path_score = 0.0
        self.lane_rank = 1
        self.reject_streak = 0

    # ------------------------------------------------------------- accessors
    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def alive(self) -> bool:
        return self.misses <= DELETE_FRAMES

    @property
    def renderable(self) -> bool:
        return self.confirmed and self.misses <= COAST_FRAMES

    # --------------------------------------------------------------- filter
    def predict(self, dt: float, ego_speed_mps: Optional[float] = None,
                ego_yaw_rate: float = 0.0) -> None:
        """Constant-velocity predict in the current ego frame.

        Geometric measurements are already in this frame, so ego translation
        must not be subtracted here — that double-counts closing rate (the
        filter also learns it in vy). Yaw-rate rotation is still applied
        because a heading change is not in the range measurement.
        """
        dt = float(np.clip(dt, 1e-3, 0.25))
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.x = F @ self.x
        _ = ego_speed_mps  # signature kept; measurements are already ego-frame

        if ego_yaw_rate:
            th = float(ego_yaw_rate) * dt
            c, s = math.cos(th), math.sin(th)
            px, py = self.x[0], self.x[1]
            self.x[0] = c * px + s * py
            self.x[1] = -s * px + c * py
            vx, vy = self.x[2], self.x[3]
            self.x[2] = c * vx + s * vy
            self.x[3] = -s * vx + c * vy

        # Continuous-acceleration process noise, per axis.
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        qx, qy = Q_ACC_LAT ** 2, Q_ACC_LON ** 2
        Q = np.zeros((4, 4), dtype=np.float64)
        Q[0, 0], Q[0, 2], Q[2, 0], Q[2, 2] = qx * dt4 / 4, qx * dt3 / 2, qx * dt3 / 2, qx * dt2
        Q[1, 1], Q[1, 3], Q[3, 1], Q[3, 3] = qy * dt4 / 4, qy * dt3 / 2, qy * dt3 / 2, qy * dt2
        self.P = F @ self.P @ F.T + Q

        self.x[2] = float(np.clip(self.x[2], VX_MIN, VX_MAX))
        self.x[3] = float(np.clip(self.x[3], VY_MIN, VY_MAX))
        self.x[1] = float(np.clip(self.x[1], 1.0, 110.0))
        self.age += 1

    def reinit(self, x_m: float, y_m: float, R: np.ndarray) -> None:
        """ByteTrack reused this id on a different object. Drop velocity."""
        R = np.asarray(R, dtype=np.float64)
        self.x = np.array([float(x_m), float(y_m), 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([
            max(float(R[0, 0]), P0_POS),
            max(float(R[1, 1]), P0_POS),
            P0_VEL,
            P0_VEL,
        ]).astype(np.float64)
        self.hits = CONFIRM_HITS
        self.misses = 0
        self.confirmed = True
        self.sigma_y = float(math.sqrt(max(1e-6, float(R[1, 1]))))
        self.reject_streak = 0

    def update(self, x_m: float, y_m: float, R: np.ndarray,
               gated: bool = True) -> bool:
        """Kalman update. Returns False if the meas is rejected.

        `gated=False` is for a ByteTrack ID match: the detector already said
        this is the same object, so a tight χ² gate must not spawn a twin
        track that then coasts the original into nonsense.
        """
        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        z = np.array([float(x_m), float(y_m)], dtype=np.float64)
        innov = z - H @ self.x
        S = H @ self.P @ H.T + np.asarray(R, dtype=np.float64)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        if gated and float(innov @ Sinv @ innov) > CHI2_GATE_2DOF:
            return False
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ innov
        I_KH = np.eye(4) - K @ H
        # Joseph form: stays positive-definite under the wide R range this
        # filter sees (0.07 m^2 near to 31 m^2 far).
        self.P = I_KH @ self.P @ I_KH.T + K @ np.asarray(R, dtype=np.float64) @ K.T
        self.x[2] = float(np.clip(self.x[2], VX_MIN, VX_MAX))
        self.x[3] = float(np.clip(self.x[3], VY_MIN, VY_MAX))
        self.hits += 1
        self.misses = 0
        self.sigma_y = float(math.sqrt(max(1e-6, float(R[1, 1]))))
        if self.hits >= CONFIRM_HITS:
            self.confirmed = True
        self.reject_streak = 0
        return True

    def mark_missed(self) -> None:
        self.misses += 1

    def extrapolate(self, dt: float) -> Tuple[float, float]:
        """Position for a render tick between inference frames.

        Clamped so a coasting track drifts at most MAX_EXTRAPOLATION_S worth --
        the renderer runs faster than inference and must not amplify a stale
        velocity into visible sliding.
        """
        t = float(np.clip(dt, 0.0, MAX_EXTRAPOLATION_S))
        return (float(self.x[0] + self.x[2] * t),
                float(self.x[1] + self.x[3] * t))


class BevTracker:
    """Greedy nearest-neighbour association in metric BEV space.

    Association is by Mahalanobis distance, not IoU: two cars one behind the
    other overlap heavily in the image but are far apart in BEV, which is the
    case that made `_remap_occluded_id` swap identities.

    Detector track IDs (ByteTrack) are used as a strong prior when present, and
    the BEV filter arbitrates when they are missing or reused.
    """

    def __init__(self):
        self.tracks: Dict[int, BevTrack] = {}
        self._next_id = 1
        self.frame = 0

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def predict(self, dt: float, ego_speed_mps: Optional[float] = None,
                ego_yaw_rate: float = 0.0) -> None:
        for tr in self.tracks.values():
            tr.predict(dt, ego_speed_mps, ego_yaw_rate)

    def update(
        self,
        measurements: Sequence[dict],
        dt: float,
        ego_speed_mps: Optional[float] = None,
        ego_yaw_rate: float = 0.0,
    ) -> List[BevTrack]:
        """Fold one inference frame in.

        Each measurement is {"x", "y", "R", "label", "conf", "bbox", "det_id"}.
        Rejected (gated-out) detections must simply be omitted: a track with no
        valid ground measurement should coast, not be corrected with a guess.
        """
        self.frame += 1
        self.predict(dt, ego_speed_mps, ego_yaw_rate)

        unmatched = list(range(len(measurements)))
        matched_tracks = set()

        # Pass 1: honour the detector's own track id where it maps to a live BEV
        # track and still passes the statistical gate.
        det_id_map = {tr.track_id: tr for tr in self.tracks.values()}
        for mi in list(unmatched):
            m = measurements[mi]
            did = int(m.get("det_id", -1) or -1)
            tr = det_id_map.get(did)
            if tr is None or did <= 0 or id(tr) in matched_tracks:
                continue
            dx = abs(float(m["x"]) - float(tr.x[0]))
            dy = abs(float(m["y"]) - float(tr.x[1]))
            # Neighbour swap: large X, similar Y. A wild range (overpass /
            # truncated box) is an outlier — coast, do not snap to it.
            id_reuse = (
                dx > REINIT_X_M
                and dy < REINIT_Y_M
                and dx > 2.0 * max(dy, 0.5)
            )
            if id_reuse:
                tr.reinit(m["x"], m["y"], m["R"])
                self._stamp(tr, m)
            elif tr.update(m["x"], m["y"], m["R"], gated=True):
                self._stamp(tr, m)
            else:
                tr.reject_streak += 1
                if tr.reject_streak >= 2:
                    tr.reinit(m["x"], m["y"], m["R"])
                    self._stamp(tr, m)
                else:
                    tr.misses = 0
            matched_tracks.add(id(tr))
            unmatched.remove(mi)

        # Pass 2: Mahalanobis-greedy for the rest.
        free = [tr for tr in self.tracks.values() if id(tr) not in matched_tracks]
        pairs = []
        for mi in unmatched:
            m = measurements[mi]
            for tr in free:
                d2 = self._maha(tr, m)
                if d2 <= CHI2_GATE_2DOF:
                    pairs.append((d2, mi, tr))
        pairs.sort(key=lambda t: t[0])
        used_m = set()
        for _d2, mi, tr in pairs:
            if mi in used_m or id(tr) in matched_tracks:
                continue
            m = measurements[mi]
            if tr.update(m["x"], m["y"], m["R"]):
                self._stamp(tr, m)
                matched_tracks.add(id(tr))
                used_m.add(mi)
        unmatched = [mi for mi in unmatched if mi not in used_m]

        # Births. Never mint a second track for a live detector id — that is
        # how a gated reject turned into a 3× range ghost.
        for mi in unmatched:
            m = measurements[mi]
            did = int(m.get("det_id", -1) or -1)
            if did > 0 and did in self.tracks:
                continue
            tid = did if did > 0 else self._new_id()
            tr = BevTrack(tid, m["x"], m["y"], np.asarray(m["R"]),
                          m.get("label", "car"), float(m.get("conf", 1.0)))
            self._stamp(tr, m)
            self.tracks[tid] = tr
            matched_tracks.add(id(tr))

        # Deaths
        for tr in self.tracks.values():
            if id(tr) not in matched_tracks:
                tr.mark_missed()
        self.tracks = {t: tr for t, tr in self.tracks.items() if tr.alive}
        self._drop_bev_twins()
        return [tr for tr in self.tracks.values() if tr.renderable]

    def _drop_bev_twins(self) -> None:
        """Keep one track when two IDs sit on the same vehicle."""
        live = [tr for tr in self.tracks.values() if tr.alive]
        if len(live) < 2:
            return
        live.sort(key=lambda t: (-int(t.hits), -int(t.age), int(t.track_id)))
        drop = set()
        for i, a in enumerate(live):
            if a.track_id in drop:
                continue
            for b in live[i + 1 :]:
                if b.track_id in drop:
                    continue
                near = (
                    abs(float(a.x[0]) - float(b.x[0])) < DEDUP_X_M
                    and abs(float(a.x[1]) - float(b.x[1])) < DEDUP_Y_M
                )
                overlap = (
                    a.bbox is not None
                    and b.bbox is not None
                    and self._bbox_iou(a.bbox, b.bbox) >= DEDUP_BOX_IOU
                )
                if near or overlap:
                    drop.add(b.track_id)
        if drop:
            self.tracks = {t: tr for t, tr in self.tracks.items() if t not in drop}

    @staticmethod
    def _bbox_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = aa + ba - inter
        return float(inter / denom) if denom > 0 else 0.0

    @staticmethod
    def _stamp(tr: BevTrack, m: dict) -> None:
        tr.label = str(m.get("label", tr.label))
        tr.conf = float(m.get("conf", tr.conf))
        tr.bbox = list(m["bbox"]) if m.get("bbox") is not None else tr.bbox
        if "u_model" in m:
            tr.u_model = float(m["u_model"])
        if "v_model" in m:
            tr.v_model = float(m["v_model"])
        if m.get("gate"):
            tr.range_gate = str(m["gate"])

    @staticmethod
    def _maha(tr: BevTrack, m: dict) -> float:
        innov = np.array([m["x"] - tr.x[0], m["y"] - tr.x[1]], dtype=np.float64)
        S = tr.P[:2, :2] + np.asarray(m["R"], dtype=np.float64)
        try:
            return float(innov @ np.linalg.inv(S) @ innov)
        except np.linalg.LinAlgError:
            return float("inf")

    def renderable(self) -> List[BevTrack]:
        return [tr for tr in self.tracks.values() if tr.renderable]

    def reset(self) -> None:
        self.tracks.clear()
        self.frame = 0
