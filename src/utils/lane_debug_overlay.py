"""Front-view debug overlay for the lane world and object lane assignment.

Draws, in source pixels through the ground calibration:
  * every detected lane line this frame (thin yellow)
  * the four lane-world boundaries (ego pair green, outer grey; dashed when
    the slot was not measured this frame)
  * per object: track id, assigned lane, lateral x / range, and the reason
  * a header with ego lane id, confidence, width, offset, source, last event

Enable in the app with ADAS_LANE_DEBUG=1.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import cv2
import numpy as np

from src.utils.drivable_area import parse_lane_components

SLOT_COLORS = {-2: (150, 150, 150), -1: (60, 220, 60), 1: (60, 220, 60), 2: (150, 150, 150)}
LANE_COLORS = {0: (60, 220, 60), -1: (255, 170, 40), 1: (40, 170, 255)}
Y_SAMPLES = np.linspace(4.0, 60.0, 40)


def enabled() -> bool:
    return os.environ.get("ADAS_LANE_DEBUG", "0") not in ("", "0", "false", "False")


def _project(calib, xs, ys, shape):
    h, w = shape[:2]
    pts = []
    for x, y in zip(xs, ys):
        u, v = calib.project_ground(float(x), float(y))
        if -w <= u <= 2 * w and 0 <= v <= h:
            pts.append((int(round(u)), int(round(v))))
    return pts


def _polyline(img, pts, color, thick, dashed=False):
    if len(pts) < 2:
        return
    if not dashed:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thick, cv2.LINE_AA)
        return
    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, pts[i], pts[i + 1], color, thick, cv2.LINE_AA)


def _text(img, s, org, color=(255, 255, 255), scale=0.5):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_lane_world_debug(
    img: np.ndarray,
    world,
    calib,
    objects: Optional[Sequence[dict]] = None,
    lines: Optional[Sequence[np.ndarray]] = None,
    last_event: str = "",
) -> np.ndarray:
    if calib is None:
        return img
    for line in lines or []:
        xs, ys, _zs, vis = parse_lane_components(line)
        _polyline(img, _project(calib, xs[vis], ys[vis], img.shape), (0, 220, 255), 1)

    if world is not None and world.valid:
        for k in (-2, -1, 1, 2):
            xs = world.boundary_x(k, Y_SAMPLES)
            pts = _project(calib, xs, Y_SAMPLES, img.shape)
            _polyline(img, pts, SLOT_COLORS[k], 3 if abs(k) == 1 else 2,
                      dashed=not world.is_measured(k))
            if pts:
                _text(img, f"{k:+d}", (pts[0][0] - 10, pts[0][1] - 6), SLOT_COLORS[k], 0.45)
        d = world.debug()
        head = (f"lane id {d['ego_lane_id']}  conf {d['conf']:.2f}  w {d['ego_width']:.2f} m"
                f"  off {d['ego_offset']:+.2f} m  hdg {d['heading_deg']:+.1f}  {d['source']}"
                f"  [{d['measured']}]")
    else:
        head = "lane world: none"
    _text(img, head, (10, 22), (255, 255, 255), 0.55)
    if last_event:
        _text(img, last_event[:110], (10, 44), (120, 220, 255), 0.45)

    for obj in objects or []:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        li = obj.get("lane_index")
        li = None if li in (None, 99) else int(li)
        color = LANE_COLORS.get(li, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        lane = "?" if li is None else f"{li:+d}"
        _text(img, f"#{obj.get('track_id')} L{lane} x{float(obj.get('X_3d', 0.0)):+.1f}"
                   f" z{float(obj.get('Z_3d', 0.0)):.0f}", (x1, max(12, y1 - 18)), color, 0.45)
        reason = str(obj.get("lane_reason", ""))
        if reason:
            _text(img, reason[:48], (x1, max(24, y1 - 4)), color, 0.38)
    return img
