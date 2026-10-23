"""Scripted perception for demos and bench testing (``--sim``).

Emits exactly what ``InferenceWorker`` emits, so the cluster, threat logic and
3D scene run the same code path as on the road. Only the perception is
synthetic: the LDW/FCW decisions come from the real ``AdasWarningTracker``.

The 60-second loop walks through every alert the cluster can show:

    0–2 s    cruise, car centred in its lane
    2–5 s    lane change to the right: lines hand off, corridor follows into the new lane
    5–8 s    settled in the new lane, corridor centred ahead
    8–10 s   return to the original lane
    10–14 s  lead brakes, closing fast (yellow, flowing chevrons)
    14–18 s  forward collision warning, then imminent (red, BRAKE)
    18–24 s  lead pulls away, alert clears
    24–32 s  vehicle in the right lane closes and edges toward us (yellow)
    32–42 s  drift left → lane departure warning (red, left line)
    42–50 s  drift right toward an occupied lane → "Vehicle alongside"
    50–60 s  speed limit drops 65 → 55 mph while still at 66 mph (red plate)

The camera tile shows a simple rendered road so the layout can be judged
without a video; it is labelled as simulated.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.inference.ldw_fcw import AdasWarningTracker

LOOP_S = 60.0
LANE_W = 3.7
MPH = 0.44704


def _smooth(t, t0, t1):
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    u = (t - t0) / (t1 - t0)
    return u * u * (3 - 2 * u)


class _Car:
    def __init__(self, tid, lane, z, label="car"):
        self.tid = tid
        self.lane = lane          # lane-centre index in lane widths
        self.lat = 0.0            # extra lateral offset (m)
        self.z = z
        self.vz = 0.0             # relative longitudinal speed (m/s, + = opening)
        self.vx = 0.0
        self.label = label

    def as_obj(self, ego_off):
        x = self.lane * LANE_W + self.lat - ego_off
        li = int(round((x) / LANE_W))
        return {
            "track_id": self.tid,
            "label": self.label,
            "X_3d": float(x),
            "Z_3d": float(self.z),
            "vx": float(self.vx),
            "vy": float(self.vz),
            "lane_index": li,
            "lane_rank": abs(li),
            "lane_offset_m": float(np.clip(x - li * LANE_W, -1.15, 1.15)),
            "in_path": li == 0,
            "is_cipo": False,
            "status": "IN PATH" if li == 0 else "OUT OF PATH",
            "bbox": [0, 0, 1, 1],
            "conf": 0.9,
        }


class SimulationWorker(QThread):
    frame_processed = Signal(np.ndarray, list, list, object, str, object, object, float, float, object, float, object)
    status_message = Signal(str)

    FPS = 15.0

    def __init__(self, parent=None, render_camera=True):
        super().__init__(parent)
        self.running = False
        self.paused = False
        self.show_lane_draw = True
        self.render_camera = bool(render_camera)
        self.video_path = "SIMULATION"
        self.live = False
        self.alerts = AdasWarningTracker()
        self.ground_calib = None
        self.t = 0.0
        self._reset_world()

    # ------------------------------------------------------------ world
    def _reset_world(self):
        self.lead = _Car(11, 0, 62.0)
        self.left = _Car(12, -1, 30.0)
        self.right = _Car(13, 1, 44.0, label="truck")
        self.right2 = _Car(14, 1, 70.0)
        self.ego_off = 0.0
        self.ego_mps = 65 * MPH
        self.limit = 65
        self.alerts = AdasWarningTracker()

    def _script(self, t, dt):
        lead, right, left = self.lead, self.right, self.left
        # Lead vehicle longitudinal behaviour
        if t < 10:
            lead.vz = (62.0 - lead.z) * 0.8
        elif t < 14:
            lead.vz = -7.0                         # brakes: closing fast
        elif t < 17.6:
            lead.vz = -7.0 if lead.z > 8.5 else 0.0
        elif t < 24:
            lead.vz = 8.0 if lead.z < 62.0 else 0.0  # pulls away
        else:
            lead.vz = (62.0 - lead.z) * 0.5
        lead.z = max(6.0, lead.z + lead.vz * dt)

        # Left car drifts back slowly, recycles far ahead.
        left.vz = -1.2
        left.z += left.vz * dt
        if left.z < 8.0:
            left.z = 80.0

        # Right lane: truck far, then a car closes and edges in (24–32 s),
        # sits alongside for the lane-change conflict (42–50 s).
        if 24 <= t < 32:
            target_z = 10.5
            right.label = "car"
            right.vz = (target_z - right.z) * 0.6
            right.lat = -0.55 * _smooth(t, 26, 29) * (1 - _smooth(t, 30, 32))
            right.vx = -0.8 if 26 <= t < 29 else (0.8 if 30 <= t < 32 else 0.0)
        elif 42 <= t < 50:
            right.label = "car"
            right.vz = (11.0 - right.z) * 0.8
            right.lat = 0.0
            right.vx = 0.0
        else:
            right.label = "truck" if right.z > 30 else right.label
            right.vz = (44.0 - right.z) * 0.4
            right.lat = 0.0
            right.vx = 0.0
        right.z += right.vz * dt

        # Ego lateral offset, + = the car moving into the lane on its right.
        # 2–6 s is a full lane change so the bird's-eye paint has to travel
        # one lane width. 32–50 s stays a small drift for the departure warning.
        if 2.0 <= t < 10.0:
            changing = _smooth(t, 2.2, 5.2)
            returning = _smooth(t, 7.6, 9.6)
            target = LANE_W * changing * (1.0 - returning)
        elif 32 <= t < 42:
            target = -0.85 * _smooth(t, 33, 36) * (1 - _smooth(t, 38.5, 41))
        elif 42 <= t < 50:
            target = 0.85 * _smooth(t, 43, 46) * (1 - _smooth(t, 47.5, 49.5))
        else:
            target = 0.0
        self.ego_off = target

        # Speed and posted limit
        if t < 50:
            self.limit = 65
            self.ego_mps = (64.0 + 2.0 * math.sin(t * 0.3)) * MPH
        else:
            self.limit = 55
            mph = 66.0 - 10.0 * _smooth(t, 54, 58)
            self.ego_mps = mph * MPH

    # ------------------------------------------------------------ lanes
    def _lanes(self, t):
        curv = 0.0012 * math.sin(t * 2 * math.pi / LOOP_S * 2)
        ys = np.linspace(3.0, 60.0, 20)
        shape = curv * ys ** 2
        mk = lambda off: np.stack([off + shape - self.ego_off, ys, np.zeros_like(ys)], axis=1)
        left = mk(-LANE_W / 2)
        right = mk(LANE_W / 2)
        far_left = mk(-1.5 * LANE_W)
        far_right = mk(1.5 * LANE_W)
        return left, right, [far_left, left, right, far_right]

    def _bracket_pair(self, proposals):
        """The two markings that contain the car, for the departure warning.

        The lane fit keeps one continuous pair so the paint can slide through
        the change. The warning has to follow the lane the car is actually in,
        or it stays latched on the lane that was left.
        """
        scored = []
        for lane in proposals:
            scored.append((float(np.median(lane[:, 0])), lane))
        scored.sort(key=lambda item: item[0])
        for i in range(len(scored) - 1):
            if scored[i][0] <= 0.0 <= scored[i + 1][0]:
                return scored[i][1], scored[i + 1][1]
        return proposals[1], proposals[2]

    # ------------------------------------------------------------ camera
    def _render_camera(self, objs, left, right):
        w, h = 640, 360
        img = np.zeros((h, w, 3), np.uint8)
        horizon = 150
        img[:horizon] = (40, 30, 28)
        cv2.rectangle(img, (0, horizon), (w, h), (34, 34, 36), -1)
        f, cam_h, cx = 520.0, 1.4, w / 2

        def proj(x, z):
            z = max(z, 0.5)
            return int(cx + f * x / z), int(horizon + f * cam_h / z)

        for lane in (left, right):
            pts = [proj(x, y) for x, y, _ in lane]
            for (a, b) in zip(pts[:-1], pts[1:]):
                cv2.line(img, a, b, (225, 225, 225), 2, cv2.LINE_AA)
        for o in sorted(objs, key=lambda o: -o["Z_3d"]):
            z = o["Z_3d"]
            if z < 3:
                continue
            truck = o["label"] == "truck"
            wm, hm = (2.5, 3.2) if truck else (1.85, 1.45)
            x0, yb = proj(o["X_3d"] - wm / 2, z)
            x1, _ = proj(o["X_3d"] + wm / 2, z)
            top = yb - int(f * hm / z)
            cv2.rectangle(img, (x0, top), (x1, yb), (70, 70, 78), -1)
            cv2.rectangle(img, (x0, top), (x1, yb), (150, 150, 160), 1)
            tl = max(2, (x1 - x0) // 6)
            cv2.rectangle(img, (x0 + 2, yb - (yb - top) // 2), (x0 + 2 + tl, yb - (yb - top) // 2 + 3), (40, 40, 220), -1)
            cv2.rectangle(img, (x1 - 2 - tl, yb - (yb - top) // 2), (x1 - 2, yb - (yb - top) // 2 + 3), (40, 40, 220), -1)
        cv2.putText(img, "SIMULATED CAMERA", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 160, 255), 1, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------ loop
    def step(self):
        """Advance the script one frame and emit it (also used headless)."""
        dt = 1.0 / self.FPS
        t0 = time.perf_counter()
        if self.t >= LOOP_S:
            self.t = 0.0
            self._reset_world()
        self._script(self.t, dt)
        left, right, proposals = self._lanes(self.t)
        cars = [self.lead, self.left, self.right, self.right2]
        objs = [c.as_obj(self.ego_off) for c in cars if 2.0 < c.z < 95.0]

        in_path = [o for o in objs if o["lane_index"] == 0]
        cipo = min(in_path, key=lambda o: o["Z_3d"]) if in_path else None
        if cipo is not None:
            z = cipo["Z_3d"]
            cipo_status = "DANGER" if z < 14 else ("WARNING" if z < 28 else "SAFE")
            cipo["is_cipo"] = True
            cipo["status"] = cipo_status
        else:
            cipo_status = "SAFE"

        warn_left, warn_right = self._bracket_pair(proposals)
        alerts = self.alerts.update(warn_left, warn_right, "CONFIRMED", cipo, cipo_status,
                                    self.ego_mps, dt=dt, objects=objs)
        alerts["isa"] = {"posted_mph": self.limit, "candidate_mph": self.limit,
                         "status": "CONFIRMED", "conf": 0.9}
        alerts["road_status"] = "CONFIRMED"
        alerts["caps"] = {"lanes": True, "vehicles": True, "signs": True}
        alerts["cipo_status"] = cipo_status
        alerts["n_objects"] = len(objs)
        alerts["lane_held"] = False
        alerts["speed_source"] = "simulated"

        frame = (self._render_camera(objs, left, right) if self.render_camera
                 else np.zeros((360, 640, 3), np.uint8))
        latency = (time.perf_counter() - t0) * 1000.0
        self.frame_processed.emit(
            frame, proposals, objs, cipo, cipo_status, left, right,
            float(self.FPS), float(latency), float(self.ego_mps), float(dt), alerts,
        )
        self.t += dt

    def run(self):
        self.running = True
        self.status_message.emit("Simulation running")
        dt = 1.0 / self.FPS
        next_t = time.perf_counter()
        while self.running:
            if self.paused:
                self.msleep(50)
                next_t = time.perf_counter()
                continue
            self.step()
            next_t += dt
            delay = next_t - time.perf_counter()
            if delay > 0:
                self.msleep(int(delay * 1000))
            else:
                next_t = time.perf_counter()

    def stop(self):
        self.running = False
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def set_lane_draw(self, on):
        self.show_lane_draw = bool(on)
        return self.show_lane_draw

    def set_object_calib(self, *_a):
        pass
